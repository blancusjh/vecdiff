"""Finite planar entrance stop and spherical exit, with orders 0..N at the exit.

A sphere is invariant under rotations about its centre, so the local
inverse-wavenumber expansion for one plane wave at one surface point depends
only on the incidence angle and on the s/p decomposition of the amplitude.
``SphericalExitTable`` evaluates the general graph recurrence once, on
Chebyshev nodes in incidence angle, and rotates the result to every
(mode, point) pair. No sphere-specific scattering coefficient enters; the
table is the recurrence of ``taylor_interface`` sampled and interpolated.

``PolarTaylorLens`` illuminates the planar entrance of a plano-convex lens,
expands the internal field in a polar quadrature of plane waves, and applies
the table at each exit sample. The lens edge does not cut the field; only
declared stops do. Internal returns, sidewall reflection and components
beyond the admissible (non-critical) band are absent by construction.
"""

from time import perf_counter

import numpy as np
from scipy.special import j0, j1

from ...fields import ElectricField, ElectromagneticField
from ...media.medium import Z0
from ..high_frequency import _add_spectra, _chart_surface
from ..operators import SurfaceTransport
from .taylor_interface import interface_expansion

BRANCHES = ("incident", "reflected", "transmitted")


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def canonical_values(
    radius, wavelength0, n_incident, n_transmitted, theta, order, curvature=1
):
    """Recurrence fields at incidence angles theta on a sphere of radius R.

    curvature=+1: the centre lies on the incident side (e.g. a convex lens
    exit); curvature=-1: on the transmitted side (e.g. a convex entrance).
    Returns values[branch, N, field(E, Z0 H), pol(p, s), comp(n, s, t), angle]:
    unit transverse p or s amplitude, local frame, incident phase removed.
    """
    if curvature not in (1, -1):
        raise ValueError("curvature must be +1 or -1")
    theta = np.asarray(theta, float)
    u = np.sin(theta)
    xy = np.column_stack((u * radius, np.zeros_like(u)))
    normal = np.column_stack((curvature * u, np.zeros_like(u), np.cos(theta)))
    s_hat = np.array([0.0, float(curvature), 0.0])  # unit(d x n) with d = z
    t_hat = np.cross(s_hat, normal)
    frame = np.stack((normal, np.broadcast_to(s_hat, normal.shape), t_hat), -2)
    values = np.zeros((3, order + 1, 2, 2, 3, len(theta)), complex)
    k0 = 2 * np.pi / wavelength0
    p_hat = np.cross(s_hat, [0.0, 0.0, 1.0])
    for pol, amplitude in enumerate((tuple(p_hat), tuple(s_hat))):
        expansion = interface_expansion(
            lambda a, b: curvature * (-1 + (1 - a * a - b * b) ** 0.5),
            xy,
            n_incident=n_incident,
            n_transmitted=n_transmitted,
            direction=(0, 0, 1),
            amplitude=amplitude,
            order=order,
            length_scale=radius,
        )
        phase = np.exp(1j * k0 * n_incident * expansion.points[:, 2])[:, None]
        for b, branch in enumerate(BRANCHES):
            for n in range(order + 1):
                trace = expansion.trace(branch, wavelength0, order=n)
                for f, v in enumerate((trace.electric.values, Z0 * trace.magnetic)):
                    values[b, n, f, pol] = np.einsum("pcj,pj->cp", frame, v / phase)
    return values


def series_ratio(values, kR):
    """Largest |term_m| / max(|term_{m-1}|, nominal) over m = 1..N.

    values has cumulative orders on axis -4 followed by (field, pol, comp),
    for example one branch of ``canonical_values`` with its angle axis moved
    to the front. term_m is the order-m increment of E and Z0 H over all
    components and both polarizations. The floor |F_0| (kR)^-(m-1) is the
    nominal size of term m-1; it avoids dividing by a term that vanishes only
    by symmetry, such as the first correction at normal incidence. A value
    near or above one means the asymptotic series has passed its smallest term.
    """
    order = values.shape[-4] - 1
    axes = (-3, -2, -1)
    size = [np.abs(values[..., 0, :, :, :]).max(axes)]
    for m in range(1, order + 1):
        size.append(
            np.abs(values[..., m, :, :, :] - values[..., m - 1, :, :, :]).max(axes)
        )
    ratio = np.zeros_like(size[0])
    for m in range(1, order + 1):
        floor = size[0] * kR ** -(m - 1)
        ratio = np.maximum(ratio, size[m] / np.maximum(size[m - 1], floor))
    return ratio


def convergent_incidence(
    radius,
    wavelength0,
    n_incident,
    n_transmitted,
    order,
    threshold=0.5,
    samples=241,
    curvature=1,
):
    """Largest incidence angle up to which the exact recurrence series keeps
    series_ratio <= threshold, scanned up to 85 degrees of incidence or of
    transmission, whichever comes first.

    Evaluated directly from the recurrence (no interpolation): near the
    critical angle the higher terms grow too fast for a polynomial table.
    """
    limit = n_transmitted * np.sin(np.radians(85)) / n_incident
    top = np.radians(85) if limit >= np.sin(np.radians(85)) else np.arcsin(limit)
    theta = np.linspace(0, top, samples)
    values = canonical_values(
        radius, wavelength0, n_incident, n_transmitted, theta, order, curvature
    )[BRANCHES.index("transmitted")]
    kR = 2 * np.pi * n_transmitted * radius / wavelength0
    ratio = series_ratio(np.moveaxis(values, -1, 0), kR)
    bad = np.flatnonzero(ratio > threshold)
    return float(top if not len(bad) else theta[max(bad[0] - 1, 0)])


class SphericalExitTable:
    """Local E and Z0*H at a sphere versus incidence angle, for s and p input.

    radius is the physical curvature radius |R|. curvature=+1 (default) puts
    the centre on the incident side (exit of a convex lens); -1 puts it on the
    transmitted side (a convex entrance). normals passed to ``apply`` are the
    unit normals pointing into the transmitted medium.
    Values are stored in the local frame (normal, s, t = s x normal), without
    the incident phase, for unit transverse amplitude. Orders are cumulative:
    entry N is the field retained through order N, exactly as
    ``InterfaceExpansion.trace(order=N)``, including curl(A_N)/(ik)^(N+1).
    """

    def __init__(
        self,
        radius,
        wavelength0,
        n_incident,
        n_transmitted,
        max_angle,
        *,
        order=3,
        nodes=96,
        curvature=1,
    ):
        if not 0 < max_angle < np.pi / 2:
            raise ValueError("max_angle must lie strictly between 0 and pi/2")
        if n_incident * np.sin(max_angle) >= n_transmitted:
            raise ValueError("max_angle reaches the critical angle")
        self.curvature = curvature
        self.radius, self.wavelength0 = float(radius), float(wavelength0)
        self.indices, self.order, self.max_angle = (
            (n_incident, n_transmitted),
            order,
            max_angle,
        )
        # Chebyshev-Lobatto nodes on [0, max_angle].
        x = np.cos(np.pi * np.arange(nodes) / (nodes - 1))
        self.theta = 0.5 * max_angle * (1 - x)
        values = canonical_values(
            radius, wavelength0, n_incident, n_transmitted, self.theta, order, curvature
        )
        # Chebyshev coefficients along the last axis.
        self.coefficients = np.polynomial.chebyshev.chebfit(
            x, values.reshape(-1, nodes).T, nodes - 1
        ).T.reshape(values.shape)
        self.values = values
        tail = np.abs(self.coefficients[..., -4:]).max()
        self.interpolation_tail = float(tail / np.abs(self.coefficients).max())

    def _basis(self, theta):
        x = 1 - 2 * theta / self.max_angle
        return np.polynomial.chebyshev.chebvander(x, self.coefficients.shape[-1] - 1)

    def angular(self, theta, branch):
        """Table values of one branch at arbitrary angles, (..., N+1, 2, 2, 3)."""
        theta = np.asarray(theta, float)
        if np.any(theta < -1e-12) or np.any(theta > self.max_angle * (1 + 1e-12)):
            raise ValueError("incidence angle outside the tabulated non-critical band")
        c = self.coefficients[BRANCHES.index(branch)]
        values = self._basis(theta) @ c.reshape(-1, c.shape[-1]).T
        return values.reshape(theta.shape + c.shape[:-1])

    def series_ratio(self, theta, branch="transmitted"):
        """``series_ratio`` of interpolated table values at incidence theta."""
        kR = 2 * np.pi * self.indices[1] * self.radius / self.wavelength0
        return series_ratio(self.angular(theta, branch), kR)

    def apply(self, directions, amplitudes, normals, branch="transmitted"):
        """Unit-phase local fields for plane waves meeting the sphere.

        directions (..., 3) unit vectors in the incident medium, amplitudes
        (..., 3) transverse complex E, normals (..., 3) outward sphere normals.
        Returns (..., N+1, 2 fields, 3 Cartesian) for one branch, without phase.
        """
        d, n = np.broadcast_arrays(directions, normals)
        a = np.broadcast_to(amplitudes, d.shape)
        cosine = np.clip(np.sum(d * n, -1), -1, 1)
        theta = np.arccos(cosine)
        cross = np.cross(d, n)
        size = np.linalg.norm(cross, axis=-1)
        # Normal incidence: any s direction; the sphere is isotropic there.
        fallback = np.cross(
            n, np.where(abs(n[..., :1]) < 0.9, [1.0, 0, 0], [0, 1.0, 0])
        )
        s_hat = np.where((size > 1e-12)[..., None], cross, fallback)
        s_hat = _unit(s_hat)
        p_hat = np.cross(s_hat, d)
        t_hat = np.cross(s_hat, n)
        weights = np.stack((np.sum(a * p_hat, -1), np.sum(a * s_hat, -1)), -1)
        local = np.einsum("...nfpc,...p->...nfc", self.angular(theta, branch), weights)
        frame = np.stack((n, s_hat, t_hat), -2)
        return np.einsum("...nfc,...cj->...nfj", local, frame)


def flat_fresnel(directions, amplitudes, n1, n2):
    """Exact per-mode Fresnel interaction at a plane z = const, normal +z.

    Returns {branch: (E, Z0 H)} for unit phase at the plane, branches
    'incident', 'reflected', 'transmitted'. A flat interface has no curvature
    corrections, so this is exact at every order. Propagating incidence only.
    """
    d, a = np.broadcast_arrays(directions, amplitudes)
    z = np.array([0.0, 0.0, 1.0])
    c1 = d[..., 2].real
    s = np.cross(d, z)
    size = np.linalg.norm(s, axis=-1, keepdims=True)
    s = np.where(size > 1e-12, s / np.maximum(size, 1e-300), [0.0, 1.0, 0.0])
    c2 = np.sqrt(1 - (n1 / n2) ** 2 * (1 - c1**2) + 0j)
    dt = (n1 / n2) * (d - c1[..., None] * z) + c2[..., None] * z
    dr = d - 2 * c1[..., None] * z
    a_s, a_p = np.sum(a * s, -1), np.sum(a * np.cross(s, d), -1)
    ts = 2 * n1 * c1 / (n1 * c1 + n2 * c2)
    tp = 2 * n1 * c1 / (n2 * c1 + n1 * c2)
    rs = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
    rp = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
    et = (ts * a_s)[..., None] * s + (tp * a_p)[..., None] * np.cross(s, dt)
    er = (rs * a_s)[..., None] * s + (rp * a_p)[..., None] * np.cross(s, dr)
    return {
        "incident": (a, n1 * np.cross(d, a)),
        "reflected": (er, n1 * np.cross(dr, er)),
        "transmitted": (et, n2 * np.cross(dt, et)),
    }


def admissible_internal_sine(radius, aperture, n_glass, n_air, max_transmitted_angle):
    """Largest internal spectral sine keeping every exit encounter non-critical."""
    rim = np.arcsin(aperture / radius)
    incidence = np.arcsin(n_air * np.sin(max_transmitted_angle) / n_glass)
    return float(np.sin(max(incidence - rim, 0.0)))


def entrance_modes(
    aperture,
    wavelength0,
    n_air,
    n_glass,
    internal_sine,
    radial_modes,
    azimuthal_modes,
    amplitude=1.0,
):
    """Polar plane-wave quadrature of t * E0 * disk(aperture), x polarized.

    Returns wavevectors (M,3), transverse amplitudes (M,3) including the
    quadrature weight d^2 kappa/(2 pi)^2, and the retained power fraction of the
    ideal disk spectrum. t = 2 n_air/(n_air+n_glass) is normal-incidence Fresnel.
    """
    k = 2 * np.pi * n_glass / wavelength0
    kmax = k * internal_sine
    nodes, w = np.polynomial.legendre.leggauss(radial_modes)
    kappa, dk = (nodes + 1) * kmax / 2, w * kmax / 2
    alpha = 2 * np.pi * np.arange(azimuthal_modes) / azimuthal_modes
    kap, alp = np.meshgrid(kappa, alpha, indexing="ij")
    weight = (
        (dk[:, None] * np.ones_like(alp))
        * kap
        * (2 * np.pi / azimuthal_modes)
        / (4 * np.pi**2)
    )
    t = 2 * n_air / (n_air + n_glass)
    spectrum = t * amplitude * 2 * np.pi * aperture * j1(aperture * kap) / kap
    kx, ky = kap * np.cos(alp), kap * np.sin(alp)
    q = np.sqrt(k * k - kap * kap)
    vectors = np.stack((kx, ky, q), -1).reshape(-1, 3)
    amplitudes = (spectrum * weight)[..., None] * np.stack(
        (np.ones_like(kx), np.zeros_like(kx), -kx / q), -1
    )
    x = aperture * kmax
    retained = 1 - j1(x) ** 2 - j0(x) ** 2
    return vectors, amplitudes.reshape(-1, 3), float(retained)


def default_resolution(aperture, thickness, wavelength0, n_glass, internal_sine):
    """Quadrature sizes from the spectral bandwidth of the exit trace."""
    k = 2 * np.pi * n_glass / wavelength0
    kmax = k * internal_sine
    band = 2 * aperture + thickness * internal_sine / np.sqrt(1 - internal_sine**2)
    radial = int(np.ceil(0.7 * band * kmax)) + 16
    x = kmax * aperture
    azimuthal = 2 * int(np.ceil(x + 3 * x ** (1 / 3) + 12))
    return radial, azimuthal


class PolarTaylorLens:
    """Single forward pass through a planar entrance and a spherical exit.

    The lens edge (system.apertures, equal on both faces) never cuts the
    field. Inside the entrance edge the field enters the glass with the
    normal-incidence Fresnel coefficient t, so the internal field is the exact
    angular spectrum of t E0 on that disk. Outside the edge the plane wave
    continues in air; it is inverted exactly on the flat entrance chart and
    added to the outgoing spectrum. Internal light reaching the exit chart
    beyond the rim crosses its flat extension with exact flat Fresnel. Stops
    (system.stops) cut: an entrance stop limits both the glass disk and the
    light passing outside; an exit stop cuts everything on the exit chart
    beyond its radius.

    The internal spectrum is restricted to |kappa| <= n k0 internal_sine so
    that the local asymptotic series converges at every exit encounter: its
    retained terms must keep decreasing (``series_threshold``, see
    ``SphericalExitTable.series_ratio``). This is far stricter than the critical
    angle. ``max_transmitted_angle`` optionally caps the band further. Orders
    0..N at the exit use the same modes, points and Q^-1; order zero is the
    ordinary Fresnel interaction of each mode. Internal returns and sidewall
    reflection are absent.
    """

    def __init__(
        self,
        system,
        grid,
        wavelength0,
        *,
        amplitude=1.0,
        order=3,
        internal_sine=None,
        max_transmitted_angle=None,
        series_threshold=0.5,
        radial_modes=None,
        azimuthal_modes=None,
        table_nodes=96,
        max_sine=0.92,
        branches=("transmitted",),
        chunk=16,
    ):
        start = perf_counter()
        from ...surfaces import Plane, SphericalCap

        front, back = system.interfaces
        aperture = system.apertures[0]
        if (
            not isinstance(front.surface, Plane)
            or not isinstance(back.surface, SphericalCap)
            or back.surface.radius >= 0
            or aperture is None
            or aperture != system.apertures[1]
            or front.plus != back.minus
            or front.minus.material.n != back.plus.material.n
            or not np.array_equal(front.surface.frame.basis, np.eye(3))
            or not np.array_equal(back.surface.frame.basis, np.eye(3))
            or np.any(front.surface.frame.origin)
            or np.any(back.surface.frame.origin[:2])
        ):
            raise ValueError(
                "requires a coaxial plano-convex lens with its entrance at z=0"
            )
        n_air, n_glass = front.minus.material.n, front.plus.material.n
        radius, thickness = -back.surface.radius, back.surface.frame.origin[2]
        if radius <= aperture:
            raise ValueError("aperture must be smaller than the curvature radius")
        rim = np.arcsin(aperture / radius)
        incidence = (
            convergent_incidence(
                radius, wavelength0, n_glass, n_air, order, series_threshold
            )
            if order
            else np.arcsin(n_air * np.sin(np.radians(85)) / n_glass)
        )
        if max_transmitted_angle is not None:
            incidence = min(
                incidence, np.arcsin(n_air * np.sin(max_transmitted_angle) / n_glass)
            )
        self.series_limit = float(incidence)
        limit = float(np.sin(max(incidence - rim, 0.0)))
        if limit <= 0:
            raise ValueError("the rim tilt alone exceeds the convergent incidence band")
        if internal_sine is None:
            internal_sine = limit
        if not 0 < internal_sine <= limit * (1 + 1e-12):
            raise ValueError(
                f"internal_sine must lie in (0, {limit:.4f}]: larger bands meet the "
                "exit beyond the angle where the local series converges "
                "(approaching the critical/TIR regime)"
            )
        auto = default_resolution(
            aperture, thickness, wavelength0, n_glass, internal_sine
        )
        radial_modes = radial_modes or auto[0]
        azimuthal_modes = azimuthal_modes or auto[1]
        self.wavelength0, self.order = wavelength0, order
        self.radius, self.aperture, self.thickness = radius, aperture, thickness
        self.internal_sine = internal_sine
        self.resolution = (radial_modes, azimuthal_modes)
        entrance_stop, exit_stop = system.stops
        disk = aperture if entrance_stop is None else min(aperture, entrance_stop)
        self.stops = (entrance_stop, exit_stop)
        vectors, amplitudes, retained = entrance_modes(
            disk,
            wavelength0,
            n_air,
            n_glass,
            internal_sine,
            radial_modes,
            azimuthal_modes,
            amplitude,
        )
        self.retained_disk_power = retained
        self.max_incidence = float(np.arcsin(internal_sine) + rim)
        self.table = SphericalExitTable(
            radius,
            wavelength0,
            n_glass,
            n_air,
            min(self.max_incidence * (1 + 1e-9), np.pi / 2 - 1e-6),
            order=order,
            nodes=table_nodes,
        )
        self.transport = SurfaceTransport(
            grid,
            _chart_surface(back.surface, aperture),
            back.plus,
            wavelength0,
            max_sine=max_sine,
            tolerance=2e-12,
        )
        points = self.transport.grid.points.reshape(-1, 3)
        radial = np.linalg.norm(points[:, :2], axis=-1)
        inside = np.flatnonzero(radial <= aperture)
        outside = np.flatnonzero(radial > aperture)
        centre = np.array([0.0, 0.0, thickness - radius])
        k = 2 * np.pi * n_glass / wavelength0
        directions = vectors / k
        shape = self.transport.grid.shape
        self.traces = {}
        for branch in branches:
            values = np.zeros((order + 1, 2, len(points), 3), complex)
            for first in range(0, len(inside), chunk):
                ids = inside[first : first + chunk]
                p = points[ids]
                normals = _unit(p - centre)
                phase = np.exp(1j * p @ vectors.T)  # (P, M)
                local = self.table.apply(
                    directions[None], amplitudes[None], normals[:, None], branch
                )
                values[..., ids, :] = np.einsum("pm,pmnfj->nfpj", phase, local)
            # Beyond the rim the chart is flat at rim height: the internal
            # field crosses it with exact flat Fresnel (sidewall model).
            e, h = flat_fresnel(directions, amplitudes, n_glass, n_air)[branch]
            for first in range(0, len(outside), 4 * chunk):
                ids = outside[first : first + 4 * chunk]
                phase = np.exp(1j * points[ids] @ vectors.T)
                values[:, 0, ids] = (phase @ e)[None]
                values[:, 1, ids] = (phase @ h)[None]
            if exit_stop is not None:
                values[:, :, radial > exit_stop] = 0
            self.traces[branch] = tuple(
                ElectromagneticField(
                    ElectricField(
                        self.transport.grid,
                        wavelength0,
                        values[n, 0].reshape(shape + (3,)),
                    ),
                    values[n, 1].reshape(shape + (3,)) / Z0,
                )
                for n in range(order + 1)
            )
        # Light outside the entrance edge (and inside any entrance stop)
        # continues in air: exact inverse on the flat entrance chart, same
        # lattice and reference plane as the exit spectrum.
        self.outside = None
        entrance = SurfaceTransport(
            grid, Plane(), front.minus, wavelength0, max_sine=max_sine
        )
        rho = np.linalg.norm(entrance.xy, axis=-1)
        passing = rho > aperture
        if entrance_stop is not None:
            passing &= rho <= entrance_stop
        if np.any(passing):
            e = np.zeros(entrance.grid.shape + (3,), complex)
            h = np.zeros_like(e)
            e[..., 0] = amplitude
            h[..., 1] = n_air * amplitude / Z0
            self.outside, _ = entrance.inverse(
                ElectromagneticField(
                    ElectricField(entrance.grid, wavelength0, e * passing[..., None]),
                    h * passing[..., None],
                ),
                tolerance=1e-12,
            )
        self.assembly_seconds = perf_counter() - start

    def spectrum(self, order, **kwargs):
        """Outgoing spectrum: Q^-1 of the order-N exit trace plus outside light.

        With an exit stop, the light passing outside is evaluated on the exit
        chart, cut there together with the lens trace, and fitted once.
        """
        options = dict(tolerance=1e-10, max_iterations=500) | kwargs
        trace = self.traces["transmitted"][order]
        exit_stop = self.stops[1]
        if self.outside is not None and exit_stop is not None:
            passing = self.transport.forward(self.outside)
            keep = (
                np.linalg.norm(self.transport.grid.points[..., :2], axis=-1)
                <= exit_stop
            )[..., None]
            trace = ElectromagneticField(
                ElectricField(
                    trace.electric.grid,
                    trace.electric.wavelength0,
                    trace.electric.values + passing.electric.values * keep,
                ),
                trace.magnetic + passing.magnetic * keep,
            )
            return self.transport.inverse(trace, **options)
        spectrum, report = self.transport.inverse(trace, **options)
        if self.outside is not None:
            spectrum = _add_spectra(spectrum, self.outside)
        return spectrum, report
