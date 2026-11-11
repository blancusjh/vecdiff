"""The transport leg on a lattice: any surface, any carrier, no symmetry assumed.

The output lattice is filled through the inverse of the ray map, so the
incident field is evaluated exactly at the preimages and never interpolated.
The carried plane components are re-expanded in the transmitted medium, which
restores the longitudinal component spectrally.
"""

import math
from dataclasses import replace

import numpy as np

from vecdiff._backend import bd, to_numpy
from vecdiff.fields import TransverseElectricField
from vecdiff.fields.angular_spectrum import angular_spectrum
from vecdiff.fields.lattice_spectrum import (
    READ,
    LatticeSpectrum,
    PlaneReader,
    default_block,
    in_blocks,
    lattice_points,
    lattice_spectrum,
    plane_blocks,
    row_blocks,
)
from vecdiff.fields.spectrum import PlaneSpectrum
from vecdiff.sampling import CartesianGrid
from vecdiff.transport.dyadic import check_window, ray_weights
from vecdiff.transport.rays import InterfaceMap
from vecdiff.transport.report import TransportReport

# Transverse band of the lattice leg's re-expansion, in units of n k0.
_SPECTRAL_CUTOFF = 0.995
# Largest share of the carried amplitude that may lie beyond a fold of the map,
# or on rays the output lattice cannot sample.
_FOLD_TOLERANCE = 1e-2
_SAMPLING_TOLERANCE = 1e-2
# Largest share of the carried amplitude the re-expansion's band may remove.
_BAND_TOLERANCE = 1e-2


class LatticeTransport:
    """Eikonal leg across one interface between copies of a reference plane.

    ``plane`` is the CartesianGrid of the incident field on the reference
    plane (its frame shares the surface orientation). The output lattice in
    the transmitted medium keeps that spacing and grows, up to ``max_growth``
    times the input size per axis, until it holds the image of every
    supported ray: refraction can spread the image beyond the input
    footprint, and cutting it would diffract spuriously. ``window`` = (start,
    end) rolls the carried field off with a raised cosine between those
    incidence angles in degrees; a callable window receives the RayMap of
    carried rays and returns their weights ray by ray (a reference fan).
    Hard-edged inputs are sensitive to this choice, which belongs with any
    result. The map and its inverse are prepared once per geometry and
    carrier and reused for every incident spectrum.

    The lattices are streamed: rays are traced, inverted and carried
    ``block_size`` points at a time, and only the preimages, the lit and
    supported masks and the carried field are kept, in host memory. A beam
    is read exactly at the preimages, and its leg returns a LatticeSpectrum,
    which holds lattices a PlaneSpectrum cannot.
    """

    def __init__(
        self,
        plane,
        interface,
        carrier,
        wavelength0,
        *,
        window=None,
        max_growth=4,
        seed_stride=None,
        block_size=None,
    ):
        if not isinstance(plane, CartesianGrid):
            raise TypeError("eikonal transport needs a CartesianGrid reference plane")

        surface = interface.surface
        frame = surface.frame

        if not bd.allclose(plane.frame.basis, frame.basis, atol=1e-12, rtol=0):
            raise ValueError("reference plane and surface must share an orientation")

        shift = (plane.frame.origin - frame.origin) @ frame.basis

        if bd.linalg.norm(shift[:2]) > 1e-12 * max(1.0, float(bd.max(bd.abs(plane.u)))):
            raise ValueError(
                "reference plane and surface must share a transverse origin"
            )

        block_size = default_block() if block_size is None else block_size
        if (
            isinstance(block_size, bool)
            or int(block_size) != block_size
            or block_size < 1
        ):
            raise ValueError("block_size must be a positive integer")

        self.window = check_window(window)
        self.plane, self.interface, self.carrier = plane, interface, carrier
        self.wavelength0 = float(wavelength0)
        self.block_size = int(block_size)
        self.map = InterfaceMap(
            surface,
            carrier,
            interface.minus.material.n,
            interface.plus.material.n,
            float(shift[2]),
        )
        family = self.map.family
        # A carrier's rays are labelled on this plane; refracted ones on the first.
        axes = (plane.u, plane.v) if family is None else family.axes
        width = len(axes[0])
        count = width * len(axes[1])
        valid = np.zeros(count, bool)
        image = np.full((count, 2), np.nan)
        starts = None if family is None else np.full((count, 2), np.nan)
        need = [0.0, 0.0]

        def label(rows):
            incoming = self.map.forward(lattice_points(axes[0], axes[1], rows))
            # Rays the window zeroes carry nothing and do not widen the lattice.
            carried = bd.flatnonzero(incoming.valid)
            if self.window is not None and len(carried):
                weight = ray_weights(self.window, incoming.subset(incoming.valid))
                carried = carried[weight > 0]
            # A ray the fan cuts is unsupported, like one the surface refuses.
            fanned = bd.zeros(len(incoming.valid), dtype=bool)
            fanned[carried] = True
            reach = (
                to_numpy(bd.max(bd.abs(incoming.output[carried]), axis=0))
                if len(carried)
                else (0.0, 0.0)
            )
            start = None if starts is None else to_numpy(incoming.start)
            return (
                rows,
                to_numpy(fanned),
                to_numpy(incoming.output),
                start,
                reach,
            )

        blocks = row_blocks(len(axes[1]), width, self.block_size)
        for rows, ok, output, start, reach in in_blocks(label, blocks):
            flat = slice(rows.start * width, rows.stop * width)
            valid[flat], image[flat] = ok, output
            if starts is not None:
                starts[flat] = start
            need = [max(a, float(b)) for a, b in zip(need, reach)]

        self.output_plane = self._cover(plane, need, max_growth)
        out = self.output_plane
        slack = 1e-6 * float(min(plane.u[1] - plane.u[0], plane.v[1] - plane.v[0]))
        bounds = [float(out.u[0]), float(out.u[-1]), float(out.v[0]), float(out.v[-1])]
        with np.errstate(invalid="ignore"):
            imaged = (
                (image[:, 0] >= bounds[0] - slack)
                & (image[:, 0] <= bounds[1] + slack)
                & (image[:, 1] >= bounds[2] - slack)
                & (image[:, 1] <= bounds[3] + slack)
            )
        del image
        # Supported rays of this plane's lattice, in host memory.
        self.support = valid & imaged
        if family is not None:
            self.support = self._covered(starts, valid)

        stride = seed_stride or max(1, int(np.ceil(max(len(a) for a in axes) / 256)))
        seeds = bd.stack(bd.meshgrid(axes[0][::stride], axes[1][::stride]), axis=-1)
        table = self.map.seed_table(seeds.reshape(-1, 2))
        scale = max(float(bd.max(bd.abs(out.u))), float(bd.max(bd.abs(out.v))))
        limit = 1e-12 * (scale + 1e-300)
        bounds = [
            float(plane.u[0]),
            float(plane.u[-1]),
            float(plane.v[0]),
            float(plane.v[-1]),
        ]
        size = math.prod(out.shape)
        lit = np.zeros(size, bool)
        preimage = np.zeros((size, 2))
        self.residual, top = 0.0, None

        def invert(rows):
            rho, found, residual = self.map.invert(
                lattice_points(out.u, out.v, rows), table, limit
            )
            rays = self.map.forward(rho[found])
            # Starts beyond the sampled input plane would read the periodic
            # continuation of the incident spectrum; they carry no field.
            start = rays.start
            inside = (
                rays.valid
                & (start[:, 0] >= bounds[0] - slack)
                & (start[:, 0] <= bounds[1] + slack)
                & (start[:, 1] >= bounds[2] - slack)
                & (start[:, 1] <= bounds[3] + slack)
            )
            found[bd.flatnonzero(found)[~inside]] = False
            worst = height = None
            if bool(found.any()):
                worst = float(bd.max(residual[found]))
                height = float(bd.max(rays.surface_height[inside]))
            return rows, to_numpy(found), to_numpy(rho), worst, height

        blocks = row_blocks(len(out.v), len(out.u), self.block_size)
        for rows, found, rho, worst, height in in_blocks(invert, blocks):
            flat = slice(rows.start * len(out.u), rows.stop * len(out.u))
            lit[flat], preimage[flat] = found, rho
            if worst is not None:
                self.residual = max(self.residual, worst)
                top = height if top is None else max(top, height)

        self.lit, self.preimage = lit, preimage
        self.boundary = 0.0 if top is None else max(0.0, top - self.map.offset)

    @property
    def rays(self):
        """RayMap of the carried rays, at the lit output points in order."""
        return self.map.forward(bd.asarray(self.preimage[self.lit]))

    def _covered(self, starts, valid):
        """Input-lattice points that a carried ray of the family starts near.

        The ray-validity of the label lattice, interpolated linearly over its
        starts, must be one: lattice points between rays that fail here, or
        that no ray of the family reaches, are unsupported.
        """
        from scipy.interpolate import griddata

        alive = np.isfinite(starts).all(axis=-1)
        lattice = np.stack(
            np.meshgrid(to_numpy(self.plane.u), to_numpy(self.plane.v)), axis=-1
        ).reshape(-1, 2)
        if alive.sum() < 3:
            return np.zeros(len(lattice), bool)
        value = griddata(
            starts[alive], valid[alive].astype(float), lattice, method="linear"
        )

        return np.nan_to_num(value) > 1 - 1e-9

    @staticmethod
    def _cover(plane, need, max_growth):
        """The input lattice, or one with its spacing and frame grown to reach ``need``.

        ``need`` is the largest |u| and |v| of the image to hold.
        """
        sizes = []
        for axis, reach in zip((plane.u, plane.v), need):
            step = float(axis[1] - axis[0])
            extent = float(bd.max(bd.abs(axis)))
            count = len(axis) if reach <= extent else 2 * int(np.ceil(reach / step)) + 4
            sizes.append(min(count, int(max_growth * len(axis))))
        if sizes == [len(plane.u), len(plane.v)]:
            return plane
        u = (bd.arange(sizes[0]) - sizes[0] // 2) * float(plane.u[1] - plane.u[0])
        v = (bd.arange(sizes[1]) - sizes[1] // 2) * float(plane.v[1] - plane.v[0])
        return CartesianGrid(u, v, plane.frame)

    def _reader(self, spectrum, method, **kwargs):
        """The incident field at points (x, y) of this plane, E (n, 3) global."""
        frame = self.map.surface.frame
        origin = frame.origin + self.map.offset * frame.normal
        axes = self.map.basis[:, :2]
        if isinstance(spectrum, LatticeSpectrum):
            blocks = spectrum.blocks
        elif hasattr(spectrum, "beam"):
            # The beam exactly, from plane waves that resolve the whole plane.
            plane = self.plane
            corners = bd.stack(
                [
                    origin + a * axes[:, 0] + b * axes[:, 1]
                    for a in (plane.u[0], plane.u[-1])
                    for b in (plane.v[0], plane.v[-1])
                ]
            )
            beam = spectrum.beam
            step = float(max(plane.u[1] - plane.u[0], plane.v[1] - plane.v[0]))
            reach = float(bd.max(bd.linalg.norm(corners - beam.frame.origin, axis=-1)))
            blocks = plane_blocks(beam.plane_waves(reach + step))
        else:
            blocks = plane_blocks(spectrum.expansion)

        return PlaneReader(blocks, origin, axes, method=method, **kwargs)

    def _transfer(self, r):
        """Per ray, the matrix from incident E (local axes) to the carried u, v components.

        g exp(i k0 Delta S) times the window and t_s s s + t_p p_t p_i + t_p s_t s_i
        (Eq. 4.3), of shape (n, 2, 3).
        """
        s = bd.cross(r.normal, r.incident)
        size = bd.linalg.norm(s, axis=-1, keepdims=True)
        fallback = bd.cross(r.incident, bd.asarray([0.0, 1.0, 0.0]))
        fallback = bd.where(
            bd.linalg.norm(fallback, axis=-1, keepdims=True) > 1e-6,
            fallback,
            bd.cross(r.incident, bd.asarray([1.0, 0.0, 0.0])),
        )
        s = bd.where(size > 1e-12, s, fallback)
        s = s / bd.linalg.norm(s, axis=-1, keepdims=True)
        n_i, n_t = self.map.n_incident, self.map.n_transmitted
        p_i, p_t = bd.cross(s, r.incident), bd.cross(s, r.transmitted)
        c_i, c_t = r.cos_incident, r.cos_transmitted
        t_s = 2 * n_i * c_i / (n_i * c_i + n_t * c_t)
        t_p = 2 * n_i * c_i / (n_t * c_i + n_i * c_t)
        k0 = 2 * bd.pi / self.wavelength0
        factor = self._window(r) * r.flux * bd.exp(1j * k0 * r.path)
        dyadic = (
            t_s[:, None, None] * s[:, :2, None] * s[:, None, :]
            + t_p[:, None, None] * p_t[:, :2, None] * p_i[:, None, :]
            + t_p[:, None, None] * r.transmitted[:, :2, None] * r.incident[:, None, :]
        )

        return factor[:, None, None] * dyadic

    def _carried(self, spectrum, method, **kwargs):
        """Carried u, v components on the output lattice, a host array (n, 2),
        and what was found on the way: the caustic index of the carried rays,
        the folded share and the undersampled share.

        The rays of one caustic index carry the field: those that hold most of
        it. Rays of another index lie beyond a fold of the map; they are
        zeroed, and a leg that would lose more than _FOLD_TOLERANCE of its
        amplitude that way is refused. So is a leg whose output lattice
        cannot sample more than _SAMPLING_TOLERANCE of what it carries.
        """
        if spectrum.expansion.region != self.interface.minus:
            raise ValueError(
                "incident spectrum must live in the interface's incident medium"
            )
        if method == "auto":
            import importlib.util

            method = "nufft" if importlib.util.find_spec("finufft") else "direct"

        read = self._reader(spectrum, method, **kwargs)
        basis = self.map.basis
        values = np.zeros((self.lit.size, 2), complex)
        lit = np.flatnonzero(self.lit)
        caustics = np.zeros(self.lit.size, np.int8)
        coarse = np.zeros(self.lit.size, bool)
        # The carried field turns at n_t k0 s_t across the output plane, and
        # its lattice holds half a wave per sample at most.
        out = self.output_plane
        nyquist = self.wavelength0 / (2 * self.map.n_transmitted)
        steps = bd.asarray([float(out.u[1] - out.u[0]), float(out.v[1] - out.v[0])])

        def geometry(index):
            rays = self.map.forward(bd.asarray(self.preimage[index]))
            caustics[index] = to_numpy(rays.caustics)
            coarse[index] = to_numpy(
                bd.any(bd.abs(rays.transmitted[:, :2]) * steps > nyquist, axis=-1)
            )
            return self._transfer(rays), rays.start

        # Incident sums cost most per call, so each reads many blocks' starts.
        for begin in range(0, len(lit), READ):
            index = lit[begin : begin + READ]
            parts = [
                index[part : part + self.block_size]
                for part in range(0, len(index), self.block_size)
            ]
            transfers, starts = zip(*in_blocks(geometry, parts))
            incident = read(bd.concatenate(starts)) @ basis
            carried = bd.einsum("nij,nj->ni", bd.concatenate(transfers), incident)
            values[index] = to_numpy(carried)

        power = np.sum(np.abs(values[lit]) ** 2, axis=-1)
        total = max(float(power.sum()), 1e-300)
        undersampled = float(np.sqrt(power[coarse[lit]].sum() / total))
        if undersampled > _SAMPLING_TOLERANCE:
            sine = self.wavelength0 / (2 * self.map.n_transmitted * float(steps.max()))
            raise ValueError(
                f"{undersampled:.1e} of the carried amplitude leaves the surface "
                "steeper than the output lattice can sample: refracted rays reach "
                f"sin(theta) > {sine:.2f} in the transmitted medium (index "
                f"{self.map.n_transmitted:g}), where the lattice spacing must stay "
                "below wavelength0 / (2 n sin(theta)). Refine the lattice"
            )
        found = {"caustics": 0, "folded": 0.0, "undersampled": undersampled}
        indices, member = np.unique(caustics[lit], return_inverse=True)
        if len(indices) < 2:
            found["caustics"] = int(indices[0]) if len(indices) else 0
            return values, found

        shares = np.bincount(member, weights=power)
        main = int(np.argmax(shares))
        folded = float(np.sqrt(1 - shares[main] / total))
        if folded > _FOLD_TOLERANCE:
            raise ValueError(
                f"{folded:.1e} of the carried amplitude reaches the output plane "
                "from beyond a caustic (caustic indices "
                f"{[int(i) for i in indices[shares > 0]]}): the ray map folds on "
                "this leg and one ray per point cannot carry the field. End the "
                "fan before the caustic with a window, or move the reference "
                "plane away from it"
            )
        values[lit[member != main]] = 0
        found.update(caustics=int(indices[main]), folded=folded)

        return values, found

    def _lost(self, spectrum, method, **kwargs):
        """Relative amplitude of the incident field on unsupported lattice points."""
        if isinstance(spectrum, PlaneSpectrum):
            lattice = spectrum.propagate(
                float(
                    (self.plane.frame.origin - spectrum.grid.frame.origin)
                    @ spectrum.grid.frame.normal
                )
            ).electric.values.reshape(-1, 3)
            power = to_numpy(bd.sum(bd.abs(lattice) ** 2, axis=-1))
            lost, total = power[~self.support].sum(), power.sum()
        else:
            if method == "auto":
                import importlib.util

                method = "nufft" if importlib.util.find_spec("finufft") else "direct"
            read = self._reader(spectrum, method, **kwargs)
            plane, lost, total = self.plane, 0.0, 0.0
            rows = row_blocks(len(plane.v), len(plane.u), READ)
            for block in rows:
                flat = slice(block.start * len(plane.u), block.stop * len(plane.u))
                field = read(lattice_points(plane.u, plane.v, block))
                power = to_numpy(bd.sum(bd.abs(field) ** 2, axis=-1))
                lost += power[~self.support[flat]].sum()
                total += power.sum()

        return float(np.sqrt(lost / max(float(total), 1e-300)))

    def carry(self, spectrum, *, method="auto", **kwargs):
        """Carried plane components on the output lattice and the TransportReport.

        The TransverseElectricField holds g exp(i k0 Delta S) T E at every
        lattice point with a supported preimage and zero elsewhere, before the
        spectral re-expansion that ``transmit`` performs.
        """
        values, found = self._carried(spectrum, method, **kwargs)
        field = TransverseElectricField(
            self.output_plane,
            self.wavelength0,
            bd.asarray(values.reshape(self.output_plane.shape + (2,))),
        )

        return field, self._report(spectrum, method, found, **kwargs)

    def _report(self, spectrum, method, found, band_removed=0.0, **kwargs):
        if band_removed > _BAND_TOLERANCE:
            raise ValueError(
                f"{band_removed:.1e} of the carried amplitude leaves the surface "
                f"within {100 * (1 - _SPECTRAL_CUTOFF):.1f} % of grazing the "
                "reference plane (sin(theta) > "
                f"{_SPECTRAL_CUTOFF} in the transmitted medium). The lattice leg "
                "does not re-expand those plane waves, whose normal component is "
                "unbounded on a lattice; they would be lost"
            )
        return TransportReport(
            self._lost(spectrum, method, **kwargs),
            int((~self.lit).sum()),
            self.residual,
            caustics=found["caustics"],
            folded_relative_amplitude=found["folded"],
            undersampled_relative_amplitude=found["undersampled"],
            band_removed_relative_amplitude=band_removed,
        )

    def _window(self, rays):
        """Ray weights: one, the window's callable, or a raised-cosine incidence roll-off."""
        return ray_weights(self.window, rays)

    def transmit(self, spectrum, *, method="auto", **kwargs):
        """(Spectrum in the transmitted medium on the plane, TransportReport).

        The carried components are re-expanded within the transverse band
        sin < _SPECTRAL_CUTOFF of the transmitted medium, as the manuscript
        does: nearer the light circle E_z = -kappa . E / k_z amplifies what the
        lattice leaves there (the edge of the carried support, for a hard
        pupil) a hundredfold and more. A sampled incident field (a
        PlaneSpectrum) gives a PlaneSpectrum; a beam or a LatticeSpectrum
        gives a LatticeSpectrum, transformed in place.
        """
        values, found = self._carried(spectrum, method, **kwargs)
        plus = self.interface.plus
        # Modal norm of the carried field: sum |c|² over the lattice's modes.
        total = max(float(np.sqrt(np.sum(np.abs(values) ** 2) / len(values))), 1e-300)

        if not isinstance(spectrum, PlaneSpectrum):
            outgoing = lattice_spectrum(
                self.output_plane,
                plus,
                self.wavelength0,
                values,
                cutoff=_SPECTRAL_CUTOFF,
                block=self.block_size,
            )
            report = self._report(
                spectrum, method, found, outgoing.band_removed_rms / total, **kwargs
            )
            return outgoing, report

        field = TransverseElectricField(
            self.output_plane,
            self.wavelength0,
            bd.asarray(values.reshape(self.output_plane.shape + (2,))),
        )
        outgoing = angular_spectrum(field, plus, evanescent="discard")
        k = 2 * bd.pi * plus.material.n / self.wavelength0
        transverse = (
            outgoing.expansion.wavevectors.real @ outgoing.grid.frame.basis[:, :2]
        )
        inside = bd.linalg.norm(transverse, axis=-1) <= _SPECTRAL_CUTOFF * k
        cut = outgoing.expansion.amplitudes[~inside] @ outgoing.grid.frame.basis[:, :2]
        report = self._report(
            spectrum, method, found, float(bd.linalg.norm(cut)) / total, **kwargs
        )
        amplitudes = bd.where(inside[:, None], outgoing.expansion.amplitudes, 0)
        outgoing = PlaneSpectrum(
            outgoing.grid, replace(outgoing.expansion, amplitudes=amplitudes)
        )

        return outgoing, report
