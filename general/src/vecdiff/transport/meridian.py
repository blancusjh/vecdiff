"""Axisymmetric eikonal transport: the map on one meridian, Hankel spectra per azimuthal order.

When the surfaces are surfaces of revolution about the incident field's axis
and the carrier is symmetric about it, the plane-surface-plane map is radial:
P at radius rho goes to P' at Phi(rho) on the same azimuth, and the carried
dyadic acts on the cylindrical components (E_rho, E_phi, E_z) by a matrix of
rho alone. Each azimuthal order m of the incident field is carried
separately along one meridian. On the output plane E_x + i E_y and
E_x - i E_y of order m are Hankel transforms of orders m + 1 and m - 1, and
E_z of order m follows from transversality. The leg is the lattice leg of
LatticeTransport on the meridian, with the same InterfaceMap and dyadic;
only the representation differs, and its cost grows linearly with the
radius instead of quadratically.

The transmitted spectrum is sampled in the polar angle theta of its plane
waves, kappa = k sin(theta): the obliquity 1/k_z of E_z and the phase k_z z
stay smooth up to grazing, so no spectral cutoff is imposed. Its panels
resolve every phase the representation can carry for the extent of the
carried field and of the region evaluated.
"""

from dataclasses import replace

import numpy as np

from vecdiff._backend import bd, to_numpy
from vecdiff.fields.beam_spectrum import BeamSpectrum
from vecdiff.fields.hankel_spectrum import (
    HankelSpectrum,
    check_coaxial,
    composite_legendre,
)
from vecdiff.geometry.frames import Frame
from vecdiff.sampling import PointGrid
from vecdiff.transport.dyadic import carried, encounter_indices, ray_weights
from vecdiff.transport.rays import InterfaceMap, RefractedRays
from vecdiff.transport.report import TransportReport

# Gauss-Legendre panels on the reference plane: 71 nodes per 15 wavelengths
# of the transmitted medium, and no wider than 10 lambda0 (index below 1.5):
# where the map compresses, the carried phase can outrun the medium's
# wavenumber (the meniscus's rim planes need it in air).
_PLANE_PANEL, _PLANE_NODES = 15.0, 71
# More plane nodes than this means an edge ray almost parallel to the plane.
_MAX_PLANE_NODES = 50_000_000
# A sampled incident field is read on rings of this many azimuths. Its weakest
# orders are dropped while together they hold at most _RING_FLOOR of it, and
# at most _MAX_ORDERS are kept; what is dropped counts as unsupported in the
# report, and beyond _RING_TOLERANCE the input is not axisymmetric at all.
_RING_SAMPLES, _MAX_ORDERS = 64, 16
_RING_FLOOR, _RING_TOLERANCE = 1e-5, 1e-2


def surfaces_of_revolution():
    from vecdiff.surfaces import (
        AxisymmetricSurface,
        CartesianOval,
        EvenAsphere,
        Plane,
        SphericalCap,
    )

    return (Plane, SphericalCap, EvenAsphere, AxisymmetricSurface, CartesianOval)


def of_revolution(surface):
    """Whether a surface, or the surface it is the reverse of, is one of revolution."""
    from vecdiff.surfaces import ReversedSurface

    if isinstance(surface, ReversedSurface):
        surface = surface.base
    return isinstance(surface, surfaces_of_revolution())


def turned_over(harmonics):
    """The harmonics of a field seen from the other side of its plane.

    In the reversed frame (u, -v, -n) the azimuth is -phi, the azimuthal unit
    vector -e_phi and the normal -n: order m becomes -m, with E_rho kept and
    E_phi, E_z negated.
    """
    return {-m: (rho, -phi, -z) for m, (rho, phi, z) in harmonics.items()}


def ring_harmonics(spectrum, plane, radii, measure, *, method="auto"):
    """{m: (E_rho, E_phi, E_z)} of a plane spectrum on rings, and the share left out.

    The rings are read at _RING_SAMPLES azimuths; ``measure`` is their
    rho d rho weight. The weakest orders are left out while their relative
    amplitude stays below _RING_FLOOR, and at most _MAX_ORDERS are kept; the
    relative amplitude of the rest is returned.
    """
    import importlib.util

    if method == "auto":
        method = "nufft" if importlib.util.find_spec("finufft") else "direct"

    count = _RING_SAMPLES
    phi = 2 * bd.pi * bd.arange(count) / count
    radial = bd.cos(phi)[:, None] * plane.u + bd.sin(phi)[:, None] * plane.v
    azimuthal = -bd.sin(phi)[:, None] * plane.u + bd.cos(phi)[:, None] * plane.v
    points = plane.origin + radii[:, None, None] * radial[None]
    E = spectrum.evaluate(PointGrid(points), method=method).electric.values
    components = bd.stack(
        (bd.sum(E * radial, -1), bd.sum(E * azimuthal, -1), E @ plane.normal), axis=-1
    )
    coefficients = bd.fft.fft(components, axis=1) / count
    m = np.rint(np.fft.fftfreq(count, 1 / count)).astype(int)
    energy = to_numpy(bd.sum(measure[:, None] * bd.sum(abs(coefficients) ** 2, -1), 0))
    rising = np.argsort(energy)
    dropped = np.cumsum(energy[rising]) <= _RING_FLOOR**2 * max(energy.sum(), 1e-300)
    keep = rising[~dropped][::-1][:_MAX_ORDERS]
    left = np.delete(energy, keep)
    share = float(np.sqrt(np.sum(left) / max(np.sum(energy), 1e-300)))

    return {
        int(m[j]): tuple(coefficients[:, j, axis] for axis in range(3)) for j in keep
    }, share


class MeridianTransport:
    """Eikonal leg across one surface of revolution, carried along a meridian.

    ``axis`` is the incident field's frame, whose normal is the symmetry
    axis; the reference planes sit ``offset`` from the surface vertex along
    it. ``window`` weights the carried rays as in LatticeTransport. The fan
    of carried rays ends where the surface stops supporting the carrier,
    where the window closes, or at ``reach`` (a sampled input's inscribed
    radius); the search for that edge starts from ``scale``. A surface that
    carries every ray, such as a plane under a collimated beam, leaves the
    edge to ``extent``, the radius where the incident field ends. The output plane
    is sampled by Gauss-Legendre panels on [0, Phi(edge)], and each node's
    preimage is found by Newton steps with the analytic D Phi. ``branch`` is
    that of the encounter, as for LatticeTransport: a reflected leg's
    HankelSpectrum lies on the same plane, in the incident medium,
    travelling back. After a reflection the axis is turned over (its frame
    reversed, u, -v, -n) so that it follows the light; the incident field's
    harmonics are turned over with it (``turned_over``).
    """

    def __init__(
        self,
        encounter,
        carrier,
        axis,
        wavelength0,
        *,
        offset=0.0,
        window=None,
        reach=None,
        scale=None,
        extent=None,
    ):
        branch, n_incident, n_far = encounter_indices(encounter)
        surface = encounter.surface
        wavelength0 = float(wavelength0)

        if not of_revolution(surface):
            raise ValueError(
                f"{type(surface).__name__} is not a surface of revolution; "
                "axisymmetric transport takes Plane, SphericalCap, EvenAsphere, "
                "AxisymmetricSurface or CartesianOval; others take the lattice "
                "(a sampled field, or a beam with lattice=)"
            )

        check_coaxial(surface.frame, axis, wavelength0, "the surface")
        # A surface of revolution turned about its axis is itself: every
        # surface takes the axis frame's axes, so all meridians coincide.
        surface = replace(surface, frame=Frame(surface.frame.origin, axis.basis))
        self.encounter, self.carrier, self.wavelength0 = encounter, carrier, wavelength0
        self.window, self.branch = window, branch
        self.aperture = encounter.aperture
        reflected = branch != "transmitted"
        self.outgoing = encounter.outgoing
        self.direction = -1 if reflected else 1
        self.map = InterfaceMap(
            surface,
            carrier,
            n_incident,
            n_far,
            offset,
            branch="reflected" if reflected else "transmitted",
        )
        self.plane = Frame(
            surface.frame.origin + float(offset) * axis.normal, axis.basis
        )
        scale = float(
            next((v for v in (scale, reach) if v is not None), 100 * wavelength0)
        )
        if not isinstance(carrier, RefractedRays):
            self._check_carrier(scale)
        self.edge = self._fan(reach, scale, extent)
        self.image = float(to_numpy(self._meridian(np.array([self.edge])).output)[0, 0])
        panel = _PLANE_PANEL * wavelength0 / max(self.outgoing.material.n, 1.5)
        if self.image / panel * _PLANE_NODES > _MAX_PLANE_NODES:
            raise ValueError(
                f"the carried fan's image reaches {self.image / wavelength0:.3g} "
                "wavelengths: its edge rays leave nearly parallel to the plane; "
                "bound the fan with a window"
            )
        self.radius, self.weights = composite_legendre(
            0.0, self.image, panel, _PLANE_NODES
        )
        self.preimage, self.residual = self._invert(self.radius)
        self.rays = self._meridian(self.preimage)
        heights = to_numpy(self.rays.surface_height)
        # The heights the surface spans where it carries light, in its frame.
        self.band = (float(heights.min()), float(heights.max()))
        self.boundary = float(max(0.0, heights.max() - self.map.offset))
        self._transfer = self._dyadic()

    def footprint(self, count=4096, azimuths=16):
        """Points on the output plane and directions (global) of carried rays,
        the meridian's turned to ``azimuths`` azimuths about the axis."""
        rays = self.rays
        keep = np.arange(
            0, len(rays.output), max(1, len(rays.output) * azimuths // count)
        )
        output = to_numpy(rays.output)[keep, 0]
        direction = to_numpy(rays.transmitted)[keep]
        frame = self.map.surface.frame
        basis, origin = to_numpy(frame.basis), to_numpy(frame.origin)
        points, directions = [], []
        for phi in 2 * np.pi * np.arange(azimuths) / azimuths:
            c, s = np.cos(phi), np.sin(phi)
            points.append(
                origin
                + np.c_[c * output, s * output, np.full(len(output), self.map.offset)]
                @ basis.T
            )
            turned = np.c_[
                c * direction[:, 0] - s * direction[:, 1],
                s * direction[:, 0] + c * direction[:, 1],
                direction[:, 2],
            ]
            directions.append(turned @ basis.T)
        return bd.asarray(np.concatenate(points)), bd.asarray(
            np.concatenate(directions)
        )

    def _meridian(self, radii):
        """RayMap at input radii along the surface frame's u axis."""
        radii = bd.asarray(radii, dtype=float)

        return self.map.forward(bd.stack((radii, bd.zeros_like(radii)), axis=-1))

    def _check_carrier(self, scale):
        """The carrier must turn with the azimuth and keep to the meridians."""
        radii = scale * np.array([0.05, 0.3, 0.7, 1.0])
        phi = np.array([0.0, 0.9, 2.3, 4.4])
        c, s = np.cos(phi), np.sin(phi)
        local = np.stack(
            (
                np.outer(radii, c),
                np.outer(radii, s),
                np.full((len(radii), len(phi)), self.map.offset),
            ),
            axis=-1,
        )
        frame = self.map.surface.frame
        world = frame.origin + bd.asarray(local) @ frame.basis.T
        direction = to_numpy(self.carrier.directions(world) @ frame.basis)
        jacobian = to_numpy(
            bd.einsum(
                "ji,abjk,kl->abil",
                frame.basis,
                self.carrier.jacobians(world),
                frame.basis,
            )
        )
        turn = np.zeros((len(phi), 3, 3))
        turn[:, 0, 0] = turn[:, 1, 1] = c
        turn[:, 0, 1], turn[:, 1, 0], turn[:, 2, 2] = -s, s, 1
        expected = np.einsum("bij,aj->abi", turn, direction[:, 0])
        turned = np.einsum("bij,ajk,blk->abil", turn, jacobian[:, 0], turn)
        size = max(float(np.abs(jacobian).max()), 1e-300)

        if (
            np.abs(direction - expected).max() > 1e-10
            or np.abs(direction[:, 0, 1]).max() > 1e-10
            or np.abs(jacobian - turned).max() > 1e-10 * size
        ):
            raise ValueError(
                "the carrier is not symmetric about the incident field's axis"
            )

    def _orientation(self):
        """+1 if this plane's v axis is that of the family's first plane, else -1."""
        first = self.map
        while first.family is not None:
            first = first.family.previous
        v = to_numpy(first.surface.frame.v) @ to_numpy(self.map.surface.frame.v)
        return 1.0 if v > 0 else -1.0

    def _supported(self, radii):
        """Rays the surface supports and the window still weights."""
        rays = self._meridian(radii)
        # The meridian carries the rays that met no caustic on this leg: its
        # map is inverted as an increasing function of the radius.
        start = to_numpy(rays.start_jacobian)
        # A label circle maps to a circle of this plane: radially outward, and
        # turning the same way, or the other way where this plane's axes are
        # those of the first plane reversed.
        valid = (
            to_numpy(rays.valid)
            & (to_numpy(rays.caustics) == 0)
            & (start[:, 0, 0] > 0)
            & (self._orientation() * start[:, 1, 1] > 0)
        )
        supported = valid.copy()

        if valid.any():
            weight = to_numpy(
                ray_weights(self.window, rays.subset(bd.asarray(valid)), self.aperture)
            )
            supported[valid] = weight > 0

        return supported

    def _fan(self, reach, scale, extent=None):
        """Input radius where the carried fan ends."""
        top = float(reach if reach is not None else 2 * scale)

        for _ in range(12):
            radii = np.linspace(0.0, top, 4097)
            supported = self._supported(radii)
            if not supported[0]:
                axial = self._meridian(radii[:1])
                if bool(axial.valid[0]) and int(axial.caustics[0]) != 0:
                    raise ValueError(
                        "the carrier focuses inside this leg; the meridian leg "
                        "carries no rays through a focus: pass lattice= to carry "
                        "them on a lattice, or move the reference plane past the focus"
                    )
                if bool(axial.valid[0]) and float(axial.start_jacobian[0, 0, 0]) < 0:
                    raise ValueError(
                        "the light focuses between the previous leg and this one: "
                        "its rays reach this plane across the axis, which the "
                        "meridian does not carry; pass lattice= to carry them on "
                        "a lattice"
                    )
                raise ValueError("this surface does not carry the carrier's axial ray")
            if not supported.all():
                first = int(np.argmin(supported))
                lo, hi = radii[first - 1], radii[first]
                # Multisection: 32 samples per round narrow the bracket 32 times.
                for _ in range(10):
                    samples = np.linspace(lo, hi, 34)[1:-1]
                    inside = self._supported(samples)
                    last = int(np.argmin(inside)) if not inside.all() else len(samples)
                    lo = samples[last - 1] if last else lo
                    hi = samples[last] if last < len(samples) else hi
                edge = lo
                break
            if reach is not None:
                edge = top
                break
            top *= 2
        else:
            # Every radius is carried (a plane, or a surface far wider than
            # the field): the fan ends where the incident field does.
            if extent is None:
                raise ValueError(
                    "the surface carries the field at every radius searched and "
                    "the incident field declares no extent; bound the fan with "
                    "a window (a callable of the RayMap, on its label radius)"
                )
            edge = float(extent)

        table = np.linspace(0.0, edge, 4097)
        rays = self._meridian(table)
        image, start = (to_numpy(a)[:, 0] for a in (rays.output, rays.start))

        if np.any(np.diff(image) <= 0) or np.any(np.diff(start) <= 0):
            raise ValueError(
                "the meridian map folds or crosses the axis inside the carried fan"
            )

        self._table = (table, image)

        return float(edge)

    def _invert(self, targets):
        """Input radii of output radii: table interpolation, then Newton steps."""
        table, image = self._table
        rho = np.interp(targets, image, table)
        limit = (
            8 * np.finfo(float).eps * max(float(np.max(targets, initial=0.0)), 1e-300)
        )
        active = np.ones(len(targets), bool)

        for _ in range(30):
            index = np.flatnonzero(active)
            if not len(index):
                break
            rays = self._meridian(rho[index])
            error = to_numpy(rays.output)[:, 0] - targets[index]
            slope = to_numpy(rays.jacobian)[:, 0, 0]
            step = np.where(slope > 0, error / np.where(slope > 0, slope, 1), 0)
            rho[index] = np.clip(rho[index] - step, 0.0, self.edge)
            active[index[~(np.abs(step) > limit)]] = False

        residual = np.abs(to_numpy(self._meridian(rho).output)[:, 0] - targets)

        return rho, float(residual.max(initial=0.0))

    def _dyadic(self):
        """Rows E'_rho, E'_phi of the carried dyadic acting on (E_rho, E_phi, E_z)."""
        rays = self.rays
        # On the meridian the TE axis n x s_i is the azimuthal direction; the
        # dyadic does not depend on its sign.
        s = bd.zeros(rays.incident.shape)
        s[:, 1] = 1.0
        k0 = 2 * bd.pi / self.wavelength0
        weight = (
            ray_weights(self.window, rays, self.aperture)
            * rays.flux
            * bd.exp(1j * k0 * rays.path)
        )
        columns = [
            carried(
                bd.broadcast_to(unit, rays.incident.shape),
                rays,
                s,
                self.map.n_incident,
                self.map.n_transmitted,
                self.branch,
            )
            for unit in bd.eye(3)
        ]

        return weight[:, None, None] * bd.stack(columns, axis=-1)[:, :2]

    def transmit(self, incident, *, method="auto"):
        """(HankelSpectrum of the outgoing region on the output plane, TransportReport).

        The outgoing region is the transmitted one, or for a reflected leg
        the incident one, the field travelling back.
        """
        from vecdiff.fields.spectrum import PlaneSpectrum

        if incident.region != self.encounter.incident:
            raise ValueError(
                "incident field must live in the encounter's incident region"
            )

        rays = self.rays
        radii = rays.start[:, 0]
        # Input-plane quadrature of the carried disk, d rho = (d rho / d label)
        # (d label / d rho') d rho', with labels on the family's first plane.
        measure = (
            bd.asarray(self.weights)
            * radii
            * rays.start_jacobian[:, 0, 0]
            / rays.jacobian[:, 0, 0]
        )

        if isinstance(incident, (BeamSpectrum, HankelSpectrum)):
            # Harmonics about the incident field's own axes, then about this
            # leg's, which follow the light and may be those turned over.
            own = incident.beam.frame if hasattr(incident, "beam") else incident.frame
            harmonics, power = incident.harmonics(
                Frame(self.plane.origin, own.basis), radii
            )
            if float(own.normal @ self.plane.normal) < 0:
                harmonics = turned_over(harmonics)
        elif isinstance(incident, PlaneSpectrum):
            harmonics, share = ring_harmonics(
                incident, self.plane, radii, measure, method=method
            )
            if share > _RING_TOLERANCE:
                raise ValueError(
                    "the sampled incident field is not a few azimuthal orders about "
                    f"the axis: {share:.1e} of it lies beyond {_MAX_ORDERS} orders"
                )
            lattice = incident.propagate(0.0).electric.values
            power = float(bd.sum(abs(lattice) ** 2)) * float(
                np.prod([float(d) for d in incident.grid.spacing])
            )
        else:
            raise TypeError(
                "axisymmetric transport takes a beam, a plane spectrum or a "
                "transmitted Hankel field"
            )

        held = (
            2
            * bd.pi
            * sum(
                bd.sum(measure * sum(abs(c) ** 2 for c in parts))
                for parts in harmonics.values()
            )
        )
        unsupported = float(np.sqrt(max(0.0, 1 - float(held) / max(power, 1e-300))))
        carried = {}

        for m, parts in harmonics.items():
            vector = bd.stack(parts, axis=-1)
            radial, azimuthal = (
                bd.sum(self._transfer[:, row] * vector, -1) for row in (0, 1)
            )
            carried[m] = (radial + 1j * azimuthal, radial - 1j * azimuthal)

        field = HankelSpectrum(
            self.plane,
            self.outgoing,
            self.wavelength0,
            bd.asarray(self.radius),
            bd.asarray(self.weights),
            carried,
            self.direction,
        )
        report = TransportReport(unsupported, 0, self.residual, tuple(sorted(carried)))

        return field, report
