"""Experimental phase-space fields: local plane-wave atoms and R, Q, T, Q^-1 on them.

A field on planes parallel to z = 0 is a sum of atoms
    E(x, z) = sum_a u_a(x - x_a; z) exp(i k_a . x),
each with a centre x_a, a reference height z_a, a carrier k_a (the local
transverse wavevector) and a smooth vector envelope u_a sampled on its own
small grid. Carriers are removed, so envelopes are sampled at their own
bandwidth, not at the Nyquist rate of the field; the Nyquist lattice of a
whole plane is never formed.

* R (homogeneous propagation) is exact per atom: with
  kz(k_a + xi) = kz(k_a) + v_a . xi + delta(xi), v_a = -k_a / kz(k_a), the
  constant is a phase, the linear term moves the centre along the atom's ray
  and delta is one small FFT of the envelope.
* Q maps a plane atom to a chart atom on the surface z = h(x): through the
  tangent plane of its footprint the tilt g enters exactly as type-3 NUFFT
  frequencies xi + (kz - kz0) g, and the curvature residual r through a
  low-rank Chebyshev (Jacobi-Anger) expansion. Chart atoms carry E and H and
  the tangential carrier q = k_a + kz0 g.
* T is the library's local Fresnel operator on the chart samples; phase
  matching leaves the chart carriers unchanged.
* Q^-1 radiates each chart atom (Love currents J = n x H, M = -n x E, Weyl
  expansion) into a plane atom at its footprint height, carrier k' solving
  k' + kz'(k') g = q. No iterative fit.

Every truncation is bounded by a tolerance; envelope grids must hold each atom
(edge energy is reported). Research prototype for docs/phase_space_operators.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import erf, gammaln, jv

from ..fourier.nufft import NUFFT
from ..media.medium import Z0


def _window_value(x, centre, plateau, sharpness=6.0):
    sigma = plateau / sharpness
    return 0.5 * (
        erf((x - centre + plateau / 2) / sigma)
        - erf((x - centre - plateau / 2) / sigma)
    )


def _chebyshev_rank(radius, tolerance):
    if radius <= 0:
        return 1
    rank = max(1, int(np.ceil(radius)))
    while True:
        log_bound = (
            rank * np.log(radius / 2) - gammaln(rank + 1) + radius**2 / (4 * (rank + 1))
        )
        ratio = radius / (2 * (rank + 1))
        if ratio < 1 and log_bound + np.log(2 / (1 - ratio)) < np.log(tolerance):
            return rank
        rank += 1


def _chebyshev(t, rank):
    polynomials = [np.ones_like(t)]
    if rank > 1:
        polynomials.append(t)
    for _ in range(2, rank):
        polynomials.append(2 * t * polynomials[-1] - polynomials[-2])
    return polynomials


def _nufft3(points, strengths, targets, sign, tolerance):
    """sum_j strengths[:, j] exp(sign i targets.points_j), all cores (FINUFFT type 3)."""
    import finufft

    strengths = np.ascontiguousarray(strengths.T)
    out = finufft.nufft2d3(
        np.ascontiguousarray(points[:, 0]),
        np.ascontiguousarray(points[:, 1]),
        strengths,
        np.ascontiguousarray(targets[:, 0]),
        np.ascontiguousarray(targets[:, 1]),
        isign=sign,
        eps=max(tolerance, 1e-14),
        nthreads=0,
    )
    return np.atleast_2d(out).T


def _grid(size, step):
    local = (np.arange(size) - size // 2) * step
    xi = 2 * np.pi * np.fft.fftfreq(size, step)
    return local, xi


def _spectrum(envelope):
    """Coefficients U(xi) with u(y) = sum U exp(i xi . y), y from the centre."""
    size = envelope.shape[0]
    return np.fft.fft2(np.fft.ifftshift(envelope, axes=(0, 1)), axes=(0, 1)) / size**2


def _envelope(spectrum):
    size = spectrum.shape[0]
    return np.fft.fftshift(np.fft.ifft2(spectrum * size**2, axes=(0, 1)), axes=(0, 1))


@dataclass(frozen=True, eq=False)
class AtomField:
    """Plane atoms: centres (A, 2), heights (A,), carriers (A, 2), envelopes (A, M, M, 3).

    Envelopes are sampled at ``step`` with the centre at index M // 2 and
    describe E on the plane z = heights[a]. ``wavenumber`` is n k0.
    """

    centres: np.ndarray
    heights: np.ndarray
    carriers: np.ndarray
    envelopes: np.ndarray
    step: float
    wavenumber: float

    @property
    def size(self):
        return self.envelopes.shape[1]

    @property
    def samples(self):
        return self.envelopes.shape[0] * self.size**2

    @classmethod
    def from_function(
        cls,
        field,
        gradient,
        wavenumber,
        *,
        extent,
        tile,
        step,
        size,
        height=0.0,
        sharpness=6.0,
        tolerance=1e-8,
    ):
        """Atoms of an analytic field on |x|, |y| <= extent at z = height.

        field(xy) -> (..., 3) complex E; gradient(xy) -> (..., 2) its local
        transverse wavevector. Tiles of plateau ``tile`` form an erf partition
        of unity; atoms below tolerance x the largest envelope are dropped.
        """
        count = int(np.ceil(2 * extent / tile)) + 1
        axis = (np.arange(count) - (count - 1) / 2) * tile
        cx, cy = np.meshgrid(axis, axis)
        centres = np.column_stack((cx.ravel(), cy.ravel()))
        carriers = gradient(centres)
        local, _ = _grid(size, step)
        LX, LY = np.meshgrid(local, local)
        envelopes = np.empty((len(centres), size, size, 3), complex)
        for a, (centre, carrier) in enumerate(zip(centres, carriers)):
            x, y = centre[0] + LX, centre[1] + LY
            window = _window_value(x, centre[0], tile, sharpness) * _window_value(
                y, centre[1], tile, sharpness
            )
            envelopes[a] = (
                field(np.stack((x, y), -1))
                * (window * np.exp(-1j * (carrier[0] * x + carrier[1] * y)))[..., None]
            )
        norms = np.linalg.norm(envelopes.reshape(len(centres), -1), axis=1)
        keep = norms > tolerance * norms.max()
        heights = np.full(keep.sum(), float(height))
        return cls(
            centres[keep], heights, carriers[keep], envelopes[keep], step, wavenumber
        )

    def propagate_to(self, z):
        """Exact homogeneous R of every atom to the plane z (carriers unchanged)."""
        k = self.wavenumber
        _, xi = _grid(self.size, self.step)
        XU, XV = np.meshgrid(xi, xi)
        distance = z - self.heights
        ku = self.carriers[:, 0][:, None, None] + XU[None]
        kv = self.carriers[:, 1][:, None, None] + XV[None]
        kz = np.sqrt((k**2 - ku**2 - kv**2).astype(complex))
        kz0 = np.sqrt(k**2 - np.sum(self.carriers**2, 1))
        velocity = -self.carriers / kz0[:, None]
        residual = (
            kz
            - kz0[:, None, None]
            - velocity[:, 0][:, None, None] * XU
            - velocity[:, 1][:, None, None] * XV
        )
        spectrum = np.fft.fft2(
            np.fft.ifftshift(self.envelopes, axes=(1, 2)), axes=(1, 2)
        )
        spectrum *= np.exp(1j * residual * distance[:, None, None])[..., None]
        envelopes = np.fft.fftshift(np.fft.ifft2(spectrum, axes=(1, 2)), axes=(1, 2))
        envelopes *= np.exp(1j * kz0 * distance)[:, None, None, None]
        centres = self.centres - velocity * distance[:, None]
        return AtomField(
            centres,
            np.full(len(centres), float(z)),
            self.carriers,
            envelopes,
            self.step,
            k,
        )

    def propagate(self, distance):
        """R over ``distance`` for atoms that share one plane."""
        return self.propagate_to(self.heights[0] + distance)

    def edge_fraction(self, width=2):
        """Envelope energy within ``width`` samples of the grid edge (aliasing check)."""
        e = np.sum(abs(self.envelopes) ** 2, axis=-1)
        edge = e.copy()
        edge[:, width:-width, width:-width] = 0
        return float(edge.sum() / e.sum())

    def evaluate(self, points, *, tolerance=1e-12):
        """E at transverse points (P, 2) on the atoms' common plane."""
        points = np.asarray(points, float)
        local, xi = _grid(self.size, self.step)
        half = -local[0]
        XU, XV = np.meshgrid(xi, xi)
        freq = np.column_stack((XU.ravel(), XV.ravel()))
        out = np.zeros((len(points), self.envelopes.shape[-1]), complex)
        for centre, carrier, envelope in zip(
            self.centres, self.carriers, self.envelopes
        ):
            rel = points - centre
            inside = np.all(np.abs(rel) <= half, axis=1)
            if not inside.any():
                continue
            coeff = _spectrum(envelope).reshape(freq.shape[0], -1)
            values = NUFFT(
                coeff, freq, rel[inside], sign=1, backend="nufft", tolerance=tolerance
            )
            out[inside] += values * np.exp(1j * points[inside] @ carrier)[:, None]
        return out


@dataclass(frozen=True, eq=False)
class ChartAtoms:
    """Atoms on a graph surface z = h(x): E and H envelopes on chart grids.

    centres (A, 2) chart coordinates of the footprints, carriers (A, 2) the
    tangential phase gradient q, electric/magnetic (A, M, M, 3) Cartesian
    envelopes: E(x) = electric(x - centre) exp(i q . x) on the surface.
    """

    centres: np.ndarray
    carriers: np.ndarray
    electric: np.ndarray
    magnetic: np.ndarray
    step: float

    @property
    def size(self):
        return self.electric.shape[1]

    def evaluate(self, points, *, tolerance=1e-12):
        """E and H at chart points (P, 2) on the surface."""
        values = AtomField(
            self.centres,
            np.zeros(len(self.centres)),
            self.carriers,
            np.concatenate((self.electric, self.magnetic), axis=-1),
            self.step,
            1.0,
        ).evaluate(points, tolerance=tolerance)
        return values[:, :3], values[:, 3:]


def _on_surface(surface, xy):
    """Chart points inside the surface's domain (a cap ends at its equator)."""
    radius = getattr(surface, "radius", None)
    if radius is None:
        return np.ones(xy.shape[:-1], bool)
    return np.hypot(xy[..., 0], xy[..., 1]) < 0.999 * abs(radius)


def _heights(surface, xy, valid):
    h = np.zeros(xy.shape[:-1])
    h[valid] = surface.height(xy[valid])
    return h


def _footprint(surface, start, direction, height0):
    """Chart point where the ray start + s (direction) meets z = h(x)."""
    point = start.copy()
    for _ in range(50):
        h = surface.height(point[None])[0]
        new = start + direction * (h - height0)
        if np.linalg.norm(new - point) < 1e-12 * (1 + np.linalg.norm(new)):
            return new
        point = new
    return point


def surface_transport(atoms, surface, index, *, size, step, tolerance=1e-8):
    """Q: plane atoms (medium ``index``) -> chart atoms with E and H on the surface."""
    k = atoms.wavenumber
    local, _ = _grid(size, step)
    LX, LY = np.meshgrid(local, local)
    local_c, xi = _grid(atoms.size, atoms.step)
    XU, XV = np.meshgrid(xi, xi)
    centres, carriers, electric, magnetic = [], [], [], []
    for x_a, z_a, kappa, u in zip(
        atoms.centres, atoms.heights, atoms.carriers, atoms.envelopes
    ):
        kz0 = np.sqrt(k**2 - kappa @ kappa)
        x_f = _footprint(surface, x_a, kappa / kz0, z_a)
        h0 = surface.height(x_f[None])[0]
        g = surface.gradient(x_f[None])[0]
        X, Y = x_f[0] + LX, x_f[1] + LY
        chart_xy = np.stack((X, Y), -1)
        valid = _on_surface(surface, chart_xy)
        r = _heights(surface, chart_xy, valid) - h0 - g[0] * LX - g[1] * LY
        r = np.where(valid, r, 0.0)
        r_mid = (r[valid].max() + r[valid].min()) / 2
        r_half = (r[valid].max() - r[valid].min()) / 2
        r = np.where(valid, r, r_mid)
        U = _spectrum(u)
        ku, kv = kappa[0] + XU, kappa[1] + XV
        kz = np.sqrt(np.maximum(k**2 - ku**2 - kv**2, 0))
        magnitude = np.linalg.norm(U, axis=-1)
        band = (magnitude > tolerance * magnitude.max()) & (ku**2 + kv**2 < k**2)
        khat = np.stack((ku[band], kv[band], kz[band]), -1) / k
        E_spec = U[band]
        H_spec = index / Z0 * np.cross(khat, E_spec)
        kzb = kz[band]
        carrier_z = (kzb.max() + kzb.min()) / 2
        rank = _chebyshev_rank((kzb.max() - kzb.min()) / 2 * r_half, tolerance)
        t = (r - r_mid) / r_half if r_half > 0 else np.zeros_like(r)
        phase = np.exp(
            1j * (XU[band] * (x_f[0] - x_a[0]) + XV[band] * (x_f[1] - x_a[1]))
            + 1j * kzb * (h0 - z_a + r_mid)
        )
        coefficients = [
            (1 if m == 0 else 2) * (1j) ** m * jv(m, (kzb - carrier_z) * r_half)
            for m in range(rank)
        ]
        sources = np.column_stack(
            (XU[band] + (kzb - kz0) * g[0], XV[band] + (kzb - kz0) * g[1])
        )
        strengths = np.concatenate(
            [
                np.concatenate((E_spec, H_spec), -1) * (phase * c)[:, None]
                for c in coefficients
            ],
            -1,
        )
        values = _nufft3(
            sources,
            strengths,
            np.column_stack((LX.ravel(), LY.ravel())),
            1,
            tolerance / 10,
        ).reshape(size, size, rank, 6)
        spatial = np.exp(1j * carrier_z * r_half * t) * np.exp(
            -1j * kz0 * (g[0] * x_f[0] + g[1] * x_f[1])
        )
        polynomials = _chebyshev(t, rank)
        chart = sum(
            (spatial * p)[..., None] * values[:, :, m]
            for m, p in enumerate(polynomials)
        )
        chart = chart * valid[..., None]
        centres.append(x_f)
        carriers.append(kappa + kz0 * g)
        electric.append(chart[..., :3])
        magnetic.append(chart[..., 3:])
    return ChartAtoms(
        np.array(centres),
        np.array(carriers),
        np.array(electric),
        np.array(magnetic),
        step,
    )


def _local_wavevector(chart, surface, k):
    """3D incident wavevector per chart sample from the tangential phase gradient."""
    local, _ = _grid(chart.size, chart.step)
    LX, LY = np.meshgrid(local, local)
    out = np.empty(chart.electric.shape, float)
    for a, (x_f, q, E) in enumerate(zip(chart.centres, chart.carriers, chart.electric)):
        dominant = E[..., np.argmax(np.sum(abs(E) ** 2, axis=(0, 1)))]
        # Exact derivative of the band-limited envelope (spectral, periodic grid).
        _, xi = _grid(chart.size, chart.step)
        spectrum = _spectrum(dominant[..., None])[..., 0]
        XU, XV = np.meshgrid(xi, xi)
        gx = _envelope((1j * XU * spectrum)[..., None])[..., 0]
        gy = _envelope((1j * XV * spectrum)[..., None])[..., 0]
        lit = abs(dominant) > 1e-6 * abs(dominant).max()
        with np.errstate(all="ignore"):
            wx = q[0] + np.where(lit, np.nan_to_num((gx / dominant).imag), 0)
            wy = q[1] + np.where(lit, np.nan_to_num((gy / dominant).imag), 0)
        # A dark sample's phase gradient is noise: use the atom's carrier there.
        beyond = wx**2 + wy**2 >= (0.999 * k) ** 2
        wx, wy = np.where(beyond, q[0], wx), np.where(beyond, q[1], wy)
        chart_xy = np.stack((x_f[0] + LX, x_f[1] + LY), -1)
        valid = _on_surface(surface, chart_xy)
        g = np.zeros(chart_xy.shape)
        g[valid] = surface.gradient(chart_xy[valid])
        wg = wx * g[..., 0] + wy * g[..., 1]
        gg = 1 + g[..., 0] ** 2 + g[..., 1] ** 2
        disc = np.maximum(wg**2 - gg * (wx**2 + wy**2 - k**2), 0)
        kz = (wg + np.sqrt(disc)) / gg
        out[a] = np.stack((wx - kz * g[..., 0], wy - kz * g[..., 1], kz), -1)
    return out


def transmit_chart(chart, surface, interface, wavelength0):
    """T: local Fresnel transmission of chart atoms (carriers unchanged)."""
    from ..fields import ElectricField, ElectromagneticField
    from ..sampling import SurfaceGrid
    from .operators import fresnel_transmission

    k = 2 * np.pi * interface.minus.material.n / wavelength0
    wavevectors = _local_wavevector(chart, surface, k)
    local, _ = _grid(chart.size, chart.step)
    LX, LY = np.meshgrid(local, local)
    electric, magnetic = np.empty_like(chart.electric), np.empty_like(chart.magnetic)
    for a, x_f in enumerate(chart.centres):
        xy = np.stack((x_f[0] + LX, x_f[1] + LY), -1)
        valid = _on_surface(surface, xy)
        grid = SurfaceGrid(
            surface.points(xy[valid]), surface.normals(xy[valid]), np.ones(valid.sum())
        )
        trace = ElectromagneticField(
            ElectricField(grid, wavelength0, chart.electric[a][valid]),
            chart.magnetic[a][valid],
        )
        directions = wavevectors[a][valid]
        directions = directions / np.linalg.norm(directions, axis=-1)[..., None]
        encounter = fresnel_transmission(trace, interface, directions)
        electric[a], magnetic[a] = 0, 0
        electric[a][valid] = encounter.transmitted.electric.values
        magnetic[a][valid] = encounter.transmitted.magnetic
    return ChartAtoms(chart.centres, chart.carriers, electric, magnetic, chart.step)


def _refracted_carrier(q, g, k):
    """k' with k' + kz'(k') g = q: the forward plane wave whose chart phase gradient is q."""
    qg, gg = q @ g, 1 + g @ g
    kz = (qg + np.sqrt(max(qg**2 - gg * (q @ q - k**2), 0.0))) / gg
    return q - kz * g


def radiate(chart, surface, index, wavelength0, *, size, step, tolerance=1e-8):
    """Q^-1: chart atoms (E, H after T) -> plane atoms at their footprint heights."""
    k = 2 * np.pi * index / wavelength0
    impedance = Z0 / index
    local_c, _ = _grid(chart.size, chart.step)
    LX, LY = np.meshgrid(local_c, local_c)
    _, xi = _grid(size, step)
    XU, XV = np.meshgrid(xi, xi)
    dxi = 2 * np.pi / (size * step)
    centres, heights, carriers, envelopes = [], [], [], []
    for x_f, q, E, H in zip(
        chart.centres, chart.carriers, chart.electric, chart.magnetic
    ):
        xy = np.stack((x_f[0] + LX, x_f[1] + LY), -1)
        valid = _on_surface(surface, xy)
        normals = np.zeros(xy.shape[:-1] + (3,))
        normals[valid] = surface.normals(xy[valid])
        area = np.zeros(xy.shape[:-1])
        area[valid] = surface.area_jacobian(xy[valid]) * chart.step**2
        J = np.cross(normals, H) * area[..., None]
        M = -np.cross(normals, E) * area[..., None]
        h0 = surface.height(x_f[None])[0]
        g = surface.gradient(x_f[None])[0]
        r = _heights(surface, xy, valid) - h0 - g[0] * LX - g[1] * LY
        r_mid = (r[valid].max() + r[valid].min()) / 2
        r_half = (r[valid].max() - r[valid].min()) / 2
        r = np.where(valid, r, r_mid)
        kappa = _refracted_carrier(q, g, k)
        kz0 = np.sqrt(k**2 - kappa @ kappa)
        ku, kv = kappa[0] + XU, kappa[1] + XV
        kz = np.sqrt(np.maximum(k**2 - ku**2 - kv**2, 0))
        # Significant band: chart-current frequencies p = xi + (kz - kz0) g.
        currents = np.concatenate((J, M), -1)
        chart_spectrum = np.linalg.norm(_spectrum(currents), axis=-1)
        chart_band = chart_spectrum > tolerance * chart_spectrum.max()
        for _ in range(2):
            chart_band = (
                chart_band | np.roll(chart_band, 1, 0) | np.roll(chart_band, -1, 0)
            )
            chart_band = (
                chart_band | np.roll(chart_band, 1, 1) | np.roll(chart_band, -1, 1)
            )
        pu, pv = XU + (kz - kz0) * g[0], XV + (kz - kz0) * g[1]
        cell = 2 * np.pi / (chart.size * chart.step)
        iu = np.rint(pu / cell).astype(int) % chart.size
        iv = np.rint(pv / cell).astype(int) % chart.size
        inside = (np.abs(pu) < np.pi / chart.step) & (np.abs(pv) < np.pi / chart.step)
        propagating = (ku**2 + kv**2 < k**2) & inside & chart_band[iv, iu]
        kzb = kz[propagating]
        carrier_z = (kzb.max() + kzb.min()) / 2
        rank = _chebyshev_rank((kzb.max() - kzb.min()) / 2 * r_half, tolerance)
        t = (r - r_mid) / r_half if r_half > 0 else np.zeros_like(r)
        spatial = np.exp(-1j * carrier_z * r_half * t)
        sources = np.concatenate(
            [
                (currents * (spatial * p)[..., None]).reshape(-1, 6)
                for p in _chebyshev(t, rank)
            ],
            -1,
        )
        targets = np.column_stack((pu[propagating], pv[propagating]))
        integrals = _nufft3(
            np.column_stack((LX.ravel(), LY.ravel())),
            sources,
            targets,
            -1,
            tolerance / 10,
        ).reshape(-1, rank, 6)
        coefficients = np.stack(
            [
                (1 if m == 0 else 2) * (-1j) ** m * jv(m, (kzb - carrier_z) * r_half)
                for m in range(rank)
            ],
            -1,
        )
        transform = (
            np.einsum("pm,pmc->pc", coefficients, integrals)
            * np.exp(-1j * kzb * r_mid)[:, None]
        )
        Jt, Mt = transform[:, :3], transform[:, 3:]
        khat = np.stack((ku[propagating], kv[propagating], kzb), -1) / k
        bracket = impedance * (Jt - khat * np.sum(khat * Jt, -1)[:, None]) - np.cross(
            khat, Mt
        )
        amplitude = -(k / (8 * np.pi**2 * kzb))[:, None] * bracket
        spectrum = np.zeros((size, size, 3), complex)
        # Chart carrier and footprint phase: exp(i kz0' g . x_f) (the xi . x_f parts cancel).
        spectrum[propagating] = amplitude * dxi**2 * np.exp(1j * kz0 * (g @ x_f))
        centres.append(x_f)
        heights.append(h0)
        carriers.append(kappa)
        envelopes.append(_envelope(spectrum))
    return AtomField(
        np.array(centres),
        np.array(heights),
        np.array(carriers),
        np.array(envelopes),
        step,
        k,
    )
