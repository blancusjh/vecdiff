"""Axisymmetric eikonal transport: the map on one meridian, Hankel spectra per azimuthal order.

When the surfaces are surfaces of revolution about the incident field's axis
and the carrier is symmetric about it, the plane-surface-plane map is radial:
P at radius rho goes to P' at Phi(rho) on the same azimuth, and the carried
dyadic acts on the cylindrical components (E_rho, E_phi, E_z) by a matrix of
rho alone. Each azimuthal order m of the incident field is carried
separately along one meridian. On the output plane E_x + i E_y and
E_x - i E_y of order m are Hankel transforms of orders m + 1 and m - 1, and
E_z of order m follows from transversality. The leg is the lattice leg of
EikonalTransport on the meridian, with the same InterfaceMap and dyadic;
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

from vecdiff_general._backend import bd, to_numpy
from vecdiff_general.fields.electromagnetic_field import (
    ElectricField,
    ElectromagneticField,
)
from vecdiff_general.fourier.polar import hankel_sum
from vecdiff_general.geometry.frames import Frame
from vecdiff_general.media.medium import Z0
from vecdiff_general.sampling import PointGrid, PolarGrid

# Gauss-Legendre panels on the reference plane: 71 nodes per 15 transmitted
# wavelengths resolve the carried phase twice over.
_PLANE_PANEL, _PLANE_NODES = 15.0, 71
# Spectral panels: 64 nodes integrate exp(i w s) on [-1, 1] to 1e-13 for w <= 85.
_SPECTRAL_NODES, _SPECTRAL_PHASE = 64, 80.0
# A sampled incident field is read on rings of this many azimuths. Its weakest
# orders are dropped while together they hold at most _RING_FLOOR of it, and
# at most _MAX_ORDERS are kept; what is dropped counts as unsupported in the
# report, and beyond _RING_TOLERANCE the input is not axisymmetric at all.
_RING_SAMPLES, _MAX_ORDERS = 64, 16
_RING_FLOOR, _RING_TOLERANCE = 1e-5, 1e-2


def surfaces_of_revolution():
    from vecdiff_general.surfaces import (
        AxisymmetricSurface,
        CartesianOval,
        EvenAsphere,
        Plane,
        SphericalCap,
    )

    return (Plane, SphericalCap, EvenAsphere, AxisymmetricSurface, CartesianOval)


def composite_legendre(lo, hi, panel, count):
    """Gauss-Legendre nodes and weights on [lo, hi] in panels of width ``panel``."""
    x, w = np.polynomial.legendre.leggauss(count)
    edges = np.arange(lo, hi + panel, panel)
    edges[-1] = hi
    edges = np.unique(edges)
    half = np.diff(edges)[:, None] / 2
    middle = (edges[:-1, None] + edges[1:, None]) / 2

    return (middle + half * x).ravel(), (half * w).ravel()


def check_coaxial(frame, axis, wavelength, what):
    """Raise unless the frame's normal is the axis and its origin lies on it."""
    if not bd.allclose(frame.normal, axis.normal, rtol=0, atol=1e-12):
        raise ValueError(f"{what} is not perpendicular to the incident field's axis")

    offset = frame.origin - axis.origin
    transverse = offset - (offset @ axis.normal) * axis.normal
    scale = max(wavelength, float(bd.linalg.norm(offset)))

    if float(bd.linalg.norm(transverse)) > 1e-12 * scale:
        raise ValueError(f"{what} is not centred on the incident field's axis")


def footprint(beam, plane):
    """Radius of the beam's geometric cone or column on a plane across its axis."""
    depth = abs(float((plane.origin - beam.frame.origin) @ beam.frame.normal))
    slope = min(float(np.tan(beam._theta_max())), 1.0)

    return float(beam._extent() + depth * slope + 10 * beam.wavelength0)


class BeamField:
    """A beam as the first stage of an axisymmetric solution.

    ``grid`` is a polar chart of the beam's footprint on the first reference
    plane; it stands in for the lattice of a sampled input where targets and
    coverage are resolved.
    """

    def __init__(self, beam, chart, region=None):
        self.beam, self.grid = beam, chart
        self.region = region or beam.region
        self.wavelength0 = beam.wavelength0
        self.axis = beam.frame
        # One carrier object, so prepared legs are reused across calls.
        self.carrier = beam.carrier

    @property
    def expansion(self):
        return self

    def relabeled(self, region):
        return BeamField(self.beam, self.grid, region)

    def evaluate(self, grid, **kwargs):
        return self.beam.evaluate(grid, **kwargs)

    def electric(self, grid, **kwargs):
        return self.beam.evaluate(grid, **kwargs).electric

    def harmonics(self, plane, radii):
        """{m: (E_rho, E_phi, E_z)} at radii on a plane across the axis, and |E|² on it.

        The azimuth is measured from the beam frame's u axis. Each order is a
        Hankel sum over the polar nodes: the azimuthal integral of
        exp(i l phi) exp(i k rho sin(theta) cos(phi - psi)) is
        2 pi i^l J_l(k rho sin(theta)) exp(i l psi).
        """
        beam = self.beam
        depth = float((plane.origin - beam.frame.origin) @ beam.frame.normal)
        reach = float(np.hypot(float(bd.max(radii)), depth))
        theta, weight, density = beam._harmonics(reach)
        k = beam.wavenumber
        sine, cosine = bd.sin(theta), bd.cos(theta)
        phase = 2 * bd.pi * weight * bd.exp(1j * k * depth * cosine)
        columns, orders = [], []

        for m, (f_theta, f_phi) in density.items():
            columns += [
                1j ** (m + 1) * phase * (cosine * f_theta + 1j * f_phi),
                1j ** (m - 1) * phase * (cosine * f_theta - 1j * f_phi),
                1j**m * phase * (-sine * f_theta),
            ]
            orders += [m + 1, m - 1, m]

        # Parseval on the plane: integral |E|² = (2 pi / k)² integral |F|² / cos dOmega.
        power = (2 * bd.pi / k) ** 2 * sum(
            2 * bd.pi * bd.sum(weight * (abs(f) ** 2 + abs(g) ** 2) / cosine)
            for f, g in density.values()
        )

        if not columns:
            return {}, 0.0

        values = hankel_sum(bd.stack(columns, -1), k * sine, radii, order=orders)
        harmonics = {}

        for index, m in enumerate(density):
            plus, minus, axial = (values[:, 3 * index + j] for j in range(3))
            harmonics[m] = ((plus + minus) / 2, (plus - minus) / 2j, axial)

        return harmonics, float(power)


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


class AxialTransport:
    """Eikonal leg across one surface of revolution, carried along a meridian.

    ``axis`` is the incident field's frame, whose normal is the symmetry
    axis; the reference planes sit ``offset`` from the surface vertex along
    it. ``window`` weights the carried rays as in EikonalTransport. The fan
    of carried rays ends where the surface stops supporting the carrier,
    where the window closes, or at ``reach`` (a sampled input's inscribed
    radius); the search for that edge starts from ``scale``. The output plane
    is sampled by Gauss-Legendre panels on [0, Phi(edge)], and each node's
    preimage is found by Newton steps with the analytic D Phi.
    """

    def __init__(
        self,
        interface,
        carrier,
        axis,
        wavelength0,
        *,
        offset=0.0,
        window=None,
        reach=None,
        scale=None,
    ):
        from vecdiff_general.propagation.eikonal import InterfaceMap, RefractedRays

        surface = interface.surface
        wavelength0 = float(wavelength0)

        if not isinstance(surface, surfaces_of_revolution()):
            raise ValueError(
                f"{type(surface).__name__} is not a surface of revolution; "
                "axisymmetric transport takes Plane, SphericalCap, EvenAsphere, "
                "AxisymmetricSurface or CartesianOval"
            )

        check_coaxial(surface.frame, axis, wavelength0, "the surface")
        # A surface of revolution turned about its axis is itself: every
        # surface takes the axis frame's axes, so all meridians coincide.
        surface = replace(surface, frame=Frame(surface.frame.origin, axis.basis))
        self.interface, self.carrier, self.wavelength0 = interface, carrier, wavelength0
        self.window = window
        self.map = InterfaceMap(
            surface,
            carrier,
            interface.minus.material.n,
            interface.plus.material.n,
            offset,
        )
        self.plane = Frame(
            surface.frame.origin + float(offset) * axis.normal, axis.basis
        )
        scale = float(
            next((v for v in (scale, reach) if v is not None), 100 * wavelength0)
        )
        if not isinstance(carrier, RefractedRays):
            self._check_carrier(scale)
        self.edge = self._fan(reach, scale)
        self.image = float(to_numpy(self._meridian(np.array([self.edge])).output)[0, 0])
        panel = _PLANE_PANEL * wavelength0 / interface.plus.material.n
        self.radius, self.weights = composite_legendre(
            0.0, self.image, panel, _PLANE_NODES
        )
        self.preimage, self.residual = self._invert(self.radius)
        self.rays = self._meridian(self.preimage)
        heights = to_numpy(self.rays.surface_height) - self.map.offset
        self.boundary = float(max(0.0, heights.max(initial=0.0)))
        self._transfer = self._dyadic()

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

    def _supported(self, radii):
        """Rays the surface supports and the window still weights."""
        from vecdiff_general.propagation.eikonal import ray_weights

        rays = self._meridian(radii)
        valid = to_numpy(rays.valid)
        supported = valid.copy()

        if valid.any():
            weight = to_numpy(ray_weights(self.window, rays.subset(rays.valid)))
            supported[valid] = weight > 0

        return supported

    def _fan(self, reach, scale):
        """Input radius where the carried fan ends."""
        top = float(reach if reach is not None else 2 * scale)

        for _ in range(12):
            radii = np.linspace(0.0, top, 4097)
            supported = self._supported(radii)
            if not supported[0]:
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
            raise ValueError(
                "the surface carries the field at every radius searched; "
                "bound the fan with an incidence window"
            )

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
        from vecdiff_general.propagation.eikonal import carried, ray_weights

        rays = self.rays
        # On the meridian the TE axis n x s_i is the azimuthal direction; the
        # dyadic does not depend on its sign.
        s = bd.zeros(rays.incident.shape)
        s[:, 1] = 1.0
        k0 = 2 * bd.pi / self.wavelength0
        weight = (
            ray_weights(self.window, rays) * rays.flux * bd.exp(1j * k0 * rays.path)
        )
        columns = [
            carried(
                bd.broadcast_to(unit, rays.incident.shape),
                rays,
                s,
                self.map.n_incident,
                self.map.n_transmitted,
            )
            for unit in bd.eye(3)
        ]

        return weight[:, None, None] * bd.stack(columns, axis=-1)[:, :2]

    def transmit(self, incident, *, method="auto"):
        """(HankelField of the transmitted region on the output plane, EikonalReport)."""
        from vecdiff_general.fields.spectrum import PlaneSpectrum
        from vecdiff_general.propagation.eikonal import EikonalReport

        if incident.expansion.region != self.interface.minus:
            raise ValueError(
                "incident field must live in the interface's incident medium"
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

        if isinstance(incident, (BeamField, HankelField)):
            harmonics, power = incident.harmonics(self.plane, radii)
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

        field = HankelField(
            self.plane,
            self.interface.plus,
            self.wavelength0,
            bd.asarray(self.radius),
            bd.asarray(self.weights),
            carried,
        )
        report = EikonalReport(unsupported, 0, self.residual, tuple(sorted(carried)))

        return field, report


class HankelField:
    """Field of a homogeneous region from transverse components on a plane, by azimuthal order.

    ``harmonics`` maps m to (E_x + i E_y, E_x - i E_y) on the radial
    quadrature (radius, weights) of the plane through ``frame``: orders
    m + 1 and m - 1 of the azimuth about the frame's normal, measured from
    its u axis. The plane-wave spectrum is formed on first evaluation for the
    depth and radius asked, and reused for any request it resolves.
    """

    def __init__(self, frame, region, wavelength0, radius, weights, harmonics):
        self.frame, self.region, self.wavelength0 = frame, region, float(wavelength0)
        self.radius, self.weights, self.carried = radius, weights, harmonics
        self.wavenumber = 2 * np.pi * region.material.n / self.wavelength0
        self.extent = float(bd.max(radius))
        self.grid = PolarGrid(radius, 2 * bd.pi * bd.arange(4) / 4, frame, weights)
        self._cache = None

    @property
    def expansion(self):
        return self

    @property
    def axis(self):
        return self.frame

    def relabeled(self, region):
        clone = HankelField(
            self.frame,
            region,
            self.wavelength0,
            self.radius,
            self.weights,
            self.carried,
        )
        clone._cache = self._cache

        return clone

    def spectrum(self, depth, reach):
        """Polar nodes theta, weights and the order m + 1, m - 1 Hankel spectra.

        The panels resolve exp(i (kappa rho' + k_z z)) J(kappa r) for
        rho' <= extent, r <= reach and |z| <= depth.
        """
        k = self.wavenumber
        bound = k * np.hypot(self.extent + reach, depth)
        panels = max(1, int(np.ceil(bound * (np.pi / 2) / (2 * _SPECTRAL_PHASE))))

        if self._cache is not None and self._cache[0] >= panels:
            return self._cache[1]

        theta, w = composite_legendre(
            0.0, np.pi / 2, np.pi / 2 / panels, _SPECTRAL_NODES
        )
        theta, w = bd.asarray(theta), bd.asarray(w)
        kappa = k * bd.sin(theta)
        measure = self.radius * self.weights
        columns, orders = [], []

        for m, (plus, minus) in self.carried.items():
            columns += [measure * plus, measure * minus]
            orders += [m + 1, m - 1]

        H = hankel_sum(bd.stack(columns, -1), self.radius, kappa, order=orders)
        result = (theta, w, kappa, H[:, 0::2], H[:, 1::2])
        self._cache = (panels, result)

        return result

    def harmonics(self, plane, radii):
        """{m: (E_rho, E_phi, E_z)} at radii on a plane across the axis, and |E|² on it.

        The plane shares the frame's axes. Its power follows from Parseval,
        2 pi sum_m integral (|E+|² / 2 + |E-|² / 2 + |E_z|²) kappa d kappa.
        """
        check_coaxial(plane, self.frame, self.wavelength0, "the next plane")
        if not bd.allclose(plane.basis, self.frame.basis, rtol=0, atol=1e-12):
            raise ValueError("the next plane must share the field's axes")
        depth = float((plane.origin - self.frame.origin) @ self.frame.normal)
        radii = bd.asarray(radii, dtype=float)
        reach = float(bd.max(radii)) if len(radii) else 0.0
        theta, w, kappa, plus, minus = self.spectrum(abs(depth), reach)
        k = self.wavenumber
        sine, cosine = bd.sin(theta)[:, None], bd.cos(theta)[:, None]
        flat, steep = k**2 * sine * cosine * w[:, None], k**2 * sine**2 * w[:, None]
        phase = bd.exp(1j * k * cosine * depth)
        orders = list(self.carried)
        columns = [flat * phase * plus, flat * phase * minus]
        columns.append(0.5j * steep * phase * (plus - minus))
        bessel = [m + 1 for m in orders] + [m - 1 for m in orders] + orders
        S = hankel_sum(bd.concatenate(columns, -1), kappa, radii, order=bessel)
        count = len(orders)
        result = {}
        for index, m in enumerate(orders):
            upper, lower = S[:, index], S[:, count + index]
            axial = S[:, 2 * count + index]
            result[m] = ((upper + lower) / 2, (upper - lower) / 2j, axial)
        # |E_z|² kappa d kappa = (kappa / 2 k_z)² |E+ - E-|² kappa d kappa.
        axial = steep**2 / (4 * flat) * abs(plus - minus) ** 2
        power = (
            2 * bd.pi * bd.sum(flat * (abs(plus) ** 2 + abs(minus) ** 2) / 2 + axial)
        )

        return result, float(power)

    def _sums(self, points, magnetic, method):
        """Cartesian E (and Z0 H) at points, one plane of constant depth at a time."""
        local = to_numpy((bd.asarray(points) - self.frame.origin) @ self.frame.basis)
        radius = np.hypot(local[:, 0], local[:, 1])
        azimuth = np.arctan2(local[:, 1], local[:, 0])
        depth = local[:, 2]
        theta, w, kappa, plus, minus = self.spectrum(
            float(np.abs(depth).max(initial=0.0)), float(radius.max(initial=0.0))
        )
        k, k0 = self.wavenumber, 2 * np.pi / self.wavelength0
        sine, cosine = bd.sin(theta)[:, None], bd.cos(theta)[:, None]
        # kappa d kappa = k² sin cos d theta and (kappa / k_z) kappa d kappa = k² sin² d theta.
        flat, steep = k**2 * sine * cosine * w[:, None], k**2 * sine**2 * w[:, None]
        orders = list(self.carried)
        bessel = [m + 1 for m in orders] + [m - 1 for m in orders] + orders
        fields = 2 if magnetic else 1
        values = np.zeros((fields, len(radius), 3), complex)
        planes, index = np.unique(depth, return_inverse=True)
        summation = "direct" if method == "direct" else "auto"

        for p, z in enumerate(planes):
            rows = np.flatnonzero(index == p)
            phase = bd.exp(1j * k * cosine * z)
            e_plus, e_minus = flat * phase * plus, flat * phase * minus
            e_z = 0.5j * steep * phase * (plus - minus)
            columns = [e_plus, e_minus, e_z]
            if magnetic:
                # Z0 H = k x E / k0 for each plane wave, order by order.
                along = k * sine * e_z
                columns += [
                    (1j * k * cosine * e_plus + along) / k0,
                    (-1j * k * cosine * e_minus + along) / k0,
                    -k * sine * (e_plus + e_minus) / (2 * k0),
                ]
            S = to_numpy(
                hankel_sum(
                    bd.concatenate(columns, axis=-1),
                    kappa,
                    bd.asarray(radius[rows]),
                    order=bessel * fields,
                    method=summation,
                )
            ).reshape(len(rows), fields, 3, len(orders))
            turn = np.exp(1j * np.outer(azimuth[rows], orders))[:, None]
            spin = np.exp(1j * azimuth[rows])[:, None, None]
            upper = np.sum(S[:, :, 0] * turn * spin, -1)
            lower = np.sum(S[:, :, 1] * turn / spin, -1)
            axial = np.sum(S[:, :, 2] * turn, -1)
            values[:, rows] = np.stack(
                ((upper + lower) / 2, (upper - lower) / 2j, axial), -1
            ).transpose(1, 0, 2)

        return bd.asarray(values @ to_numpy(self.frame.basis).T)

    def electric(self, grid, *, method="auto", **kwargs):
        values = self._sums(bd.asarray(grid.points).reshape(-1, 3), False, method)[0]

        return ElectricField(grid, self.wavelength0, values.reshape(grid.shape + (3,)))

    def evaluate(self, grid, *, method="auto", **kwargs):
        """SI E and H; method "direct" sums the Hankel integrals densely."""
        E, H = self._sums(bd.asarray(grid.points).reshape(-1, 3), True, method)

        return ElectromagneticField(
            ElectricField(grid, self.wavelength0, E.reshape(grid.shape + (3,))),
            H.reshape(grid.shape + (3,)) / Z0,
        )
