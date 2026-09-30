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

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft
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


def _to_grid(sources, strengths, size, step, sign, tolerance):
    """sum_j strengths[:, j] exp(sign i sources_j . y) on the atom grid y (FINUFFT type 1).

    The grid is y = (m - size // 2) step, so the frequencies enter as the
    points sources * step. Returns (size, size, S) indexed [y, x].
    """
    import finufft

    scaled = sources * step
    if np.abs(scaled).max() >= 3 * np.pi:
        local, _ = _grid(size, step)
        LX, LY = np.meshgrid(local, local)
        targets = np.column_stack((LX.ravel(), LY.ravel()))
        return _nufft3(sources, strengths, targets, sign, tolerance).reshape(
            size, size, -1
        )
    out = finufft.nufft2d1(
        np.ascontiguousarray(scaled[:, 1]),
        np.ascontiguousarray(scaled[:, 0]),
        np.ascontiguousarray(strengths.T),
        (size, size),
        isign=sign,
        eps=max(tolerance, 1e-14),
        nthreads=1,
        upsampfac=1.25,
    )
    return np.moveaxis(np.atleast_3d(out.reshape(-1, size, size)), 0, -1)


def _from_grid(values, step, targets, sign, tolerance):
    """sum_y values[y] exp(sign i targets . y) from the atom grid (FINUFFT type 2).

    values (size, size, S) indexed [y, x]; returns (T, S).
    """
    import finufft

    size = values.shape[0]
    scaled = targets * step
    if np.abs(scaled).max() >= 3 * np.pi:
        local, _ = _grid(size, step)
        LX, LY = np.meshgrid(local, local)
        points = np.column_stack((LX.ravel(), LY.ravel()))
        return _nufft3(points, values.reshape(size * size, -1), targets, sign, tolerance)
    out = finufft.nufft2d2(
        np.ascontiguousarray(scaled[:, 1]),
        np.ascontiguousarray(scaled[:, 0]),
        np.ascontiguousarray(np.moveaxis(values, -1, 0)),
        isign=sign,
        eps=max(tolerance, 1e-14),
        nthreads=1,
        upsampfac=1.25,
    )
    return np.atleast_2d(out).reshape(values.shape[-1], -1).T


def _map_atoms(work, items):
    """work(item) for every atom on a thread pool (FFTs and NUFFTs release the GIL)."""
    items = list(items)
    with ThreadPoolExecutor(min(len(items), os.cpu_count() or 1) or 1) as pool:
        return list(pool.map(work, items))


def _grid(size, step):
    local = (np.arange(size) - size // 2) * step
    xi = 2 * np.pi * np.fft.fftfreq(size, step)
    return local, xi


def _spectrum(envelope):
    """Coefficients U(xi) with u(y) = sum U exp(i xi . y), y from the centre."""
    size = envelope.shape[0]
    return sfft.fft2(sfft.ifftshift(envelope, axes=(0, 1)), axes=(0, 1)) / size**2


def _envelope(spectrum):
    size = spectrum.shape[0]
    return sfft.fftshift(sfft.ifft2(spectrum * size**2, axes=(0, 1)), axes=(0, 1))


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
        spectrum = sfft.fft2(
            sfft.ifftshift(self.envelopes, axes=(1, 2)), axes=(1, 2), workers=-1
        )
        spectrum *= np.exp(1j * residual * distance[:, None, None])[..., None]
        envelopes = sfft.fftshift(
            sfft.ifft2(spectrum, axes=(1, 2), workers=-1), axes=(1, 2)
        )
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
    tangential phase gradient q, curvatures (A, 2, 2) a quadratic chart phase
    C, electric/magnetic (A, M, M, 3) Cartesian envelopes:
    E(x) = electric(y) exp(i q . x + i y.C y / 2), y = x - centre, on the surface.
    The quadratic carrier removes the surface-curvature chirp from envelopes.
    """

    centres: np.ndarray
    carriers: np.ndarray
    electric: np.ndarray
    magnetic: np.ndarray
    step: float
    curvatures: np.ndarray

    @property
    def size(self):
        return self.electric.shape[1]

    def evaluate(self, points, *, tolerance=1e-12):
        """E and H at chart points (P, 2) on the surface."""
        points = np.asarray(points, float)
        local, xi = _grid(self.size, self.step)
        half = -local[0]
        XU, XV = np.meshgrid(xi, xi)
        freq = np.column_stack((XU.ravel(), XV.ravel()))
        out = np.zeros((len(points), 6), complex)
        for centre, carrier, curvature, E, H in zip(
            self.centres, self.carriers, self.curvatures, self.electric, self.magnetic
        ):
            rel = points - centre
            inside = np.all(np.abs(rel) <= half, axis=1)
            if not inside.any():
                continue
            coeff = _spectrum(np.concatenate((E, H), -1)).reshape(freq.shape[0], -1)
            values = NUFFT(
                coeff, freq, rel[inside], sign=1, backend="nufft", tolerance=tolerance
            )
            y = rel[inside]
            chirp = 0.5 * np.einsum("pi,ij,pj->p", y, curvature, y)
            out[inside] += (
                values * np.exp(1j * (points[inside] @ carrier + chirp))[:, None]
            )
        return out[:, :3], out[:, 3:]


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

    def work(atom):
        x_a, z_a, kappa, u = atom
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
        values = _to_grid(sources, strengths, size, step, 1, tolerance / 10).reshape(
            size, size, rank, 6
        )
        spatial = np.exp(1j * carrier_z * r_half * t) * np.exp(
            -1j * kz0 * (g[0] * x_f[0] + g[1] * x_f[1])
        )
        polynomials = _chebyshev(t, rank)
        chart = sum(
            (spatial * p)[..., None] * values[:, :, m]
            for m, p in enumerate(polynomials)
        )
        # Quadratic chart carrier: C = kz0 * Hessian(h) at the footprint, fitted
        # to the tangent-plane residual; removed analytically at the samples.
        basis = np.stack((0.5 * LX**2, LX * LY, 0.5 * LY**2, np.ones_like(LX)), -1)
        fit = np.linalg.lstsq(basis[valid], r[valid], rcond=None)[0]
        curvature = kz0 * np.array([[fit[0], fit[1]], [fit[1], fit[2]]])
        chirp = 0.5 * (
            curvature[0, 0] * LX**2
            + 2 * curvature[0, 1] * LX * LY
            + curvature[1, 1] * LY**2
        )
        chart = chart * (valid * np.exp(-1j * chirp))[..., None]
        return x_f, kappa + kz0 * g, curvature, chart

    results = _map_atoms(
        work, zip(atoms.centres, atoms.heights, atoms.carriers, atoms.envelopes)
    )
    charts = np.array([r[3] for r in results])
    return ChartAtoms(
        np.array([r[0] for r in results]),
        np.array([r[1] for r in results]),
        charts[..., :3],
        charts[..., 3:],
        step,
        np.array([r[2] for r in results]),
    )


def _chart_geometry(chart, surface):
    """Chart points of every atom, their validity and surface gradients (batched)."""
    local, _ = _grid(chart.size, chart.step)
    LX, LY = np.meshgrid(local, local)
    xy = np.stack(
        (chart.centres[:, 0, None, None] + LX, chart.centres[:, 1, None, None] + LY), -1
    )
    valid = _on_surface(surface, xy)
    gradient = np.zeros(xy.shape)
    gradient[valid] = surface.gradient(xy[valid])
    return xy, valid, gradient


def _local_wavevector(chart, k, gradient):
    """3D incident wavevector per chart sample from the tangential phase gradient.

    All atoms at once: the dominant component's envelope is differentiated
    exactly (spectrally, it is band-limited) and the tangential wavevector q +
    grad(arg) is lifted to 3D with the forward root on the local tangent plane.
    """
    atoms = np.arange(len(chart.centres))
    component = np.argmax(np.sum(abs(chart.electric) ** 2, axis=(1, 2)), axis=-1)
    dominant = chart.electric[atoms, :, :, component]
    _, xi = _grid(chart.size, chart.step)
    XU, XV = np.meshgrid(xi, xi)
    spectrum = sfft.fft2(sfft.ifftshift(dominant, axes=(1, 2)), axes=(1, 2), workers=-1)
    gx = sfft.fftshift(sfft.ifft2(1j * XU * spectrum, axes=(1, 2), workers=-1), axes=(1, 2))
    gy = sfft.fftshift(sfft.ifft2(1j * XV * spectrum, axes=(1, 2), workers=-1), axes=(1, 2))
    peak = abs(dominant).max(axis=(1, 2))[:, None, None]
    lit = abs(dominant) > 1e-6 * peak
    local, _ = _grid(chart.size, chart.step)
    LX, LY = np.meshgrid(local, local)
    C = chart.curvatures
    q = chart.carriers[:, None, None, :] + np.stack(
        (
            C[:, 0, 0, None, None] * LX + C[:, 0, 1, None, None] * LY,
            C[:, 1, 0, None, None] * LX + C[:, 1, 1, None, None] * LY,
        ),
        -1,
    )
    with np.errstate(all="ignore"):
        wx = q[..., 0] + np.where(lit, np.nan_to_num((gx / dominant).imag), 0)
        wy = q[..., 1] + np.where(lit, np.nan_to_num((gy / dominant).imag), 0)
    # A dark sample's phase gradient is noise: use the atom's carrier there.
    beyond = wx**2 + wy**2 >= (0.999 * k) ** 2
    wx = np.where(beyond, q[..., 0], wx)
    wy = np.where(beyond, q[..., 1], wy)
    wg = wx * gradient[..., 0] + wy * gradient[..., 1]
    gg = 1 + gradient[..., 0] ** 2 + gradient[..., 1] ** 2
    kz = (wg + np.sqrt(np.maximum(wg**2 - gg * (wx**2 + wy**2 - k**2), 0))) / gg
    return np.stack((wx - kz * gradient[..., 0], wy - kz * gradient[..., 1], kz), -1)


def transmit_chart(chart, surface, interface, wavelength0):
    """T: local Fresnel transmission of chart atoms (carriers unchanged), batched."""
    from ..fields import ElectricField, ElectromagneticField
    from ..sampling import SurfaceGrid
    from .operators import fresnel_transmission

    k = 2 * np.pi * interface.minus.material.n / wavelength0
    xy, valid, gradient = _chart_geometry(chart, surface)
    wavevectors = _local_wavevector(chart, k, gradient)[valid]
    directions = wavevectors / np.linalg.norm(wavevectors, axis=-1)[:, None]
    xy, E, H = xy[valid], chart.electric[valid], chart.magnetic[valid]

    def work(chunk):
        grid = SurfaceGrid(
            surface.points(xy[chunk]), surface.normals(xy[chunk]), np.ones(len(chunk))
        )
        trace = ElectromagneticField(
            ElectricField(grid, wavelength0, E[chunk]), H[chunk]
        )
        transmitted = fresnel_transmission(trace, interface, directions[chunk])
        return transmitted.transmitted.electric.values, transmitted.transmitted.magnetic

    chunks = np.array_split(np.arange(len(xy)), 4 * (os.cpu_count() or 1))
    results = _map_atoms(work, [c for c in chunks if len(c)])
    electric = np.zeros_like(chart.electric)
    magnetic = np.zeros_like(chart.magnetic)
    electric[valid] = np.concatenate([r[0] for r in results])
    magnetic[valid] = np.concatenate([r[1] for r in results])
    return ChartAtoms(
        chart.centres, chart.carriers, electric, magnetic, chart.step, chart.curvatures
    )


def _refracted_carrier(q, g, k):
    """k' with k' + kz'(k') g = q: the forward plane wave whose chart phase gradient is q."""
    qg, gg = q @ g, 1 + g @ g
    kz = (qg + np.sqrt(max(qg**2 - gg * (q @ q - k**2), 0.0))) / gg
    return q - kz * g


def _upsample(envelope, fine):
    """Exact band-limited interpolation of an envelope onto ``fine`` samples (same extent)."""
    size = envelope.shape[0]
    if fine == size:
        return envelope
    spectrum = np.fft.fftshift(_spectrum(envelope), axes=(0, 1))
    pad = (fine - size) // 2
    padded = np.zeros((fine, fine) + envelope.shape[2:], complex)
    padded[pad : pad + size, pad : pad + size] = spectrum
    return _envelope(np.fft.ifftshift(padded, axes=(0, 1)))


def _quadratic(C, X, Y):
    return 0.5 * (C[0, 0] * X**2 + 2 * C[0, 1] * X * Y + C[1, 1] * Y**2)


def radiate(chart, surface, index, wavelength0, *, size, step, tolerance=1e-8):
    """Q^-1: chart atoms (E, H after T) -> plane atoms at their footprint heights.

    The chart's quadratic phase C is continuous across the interface, while
    the radiation kernel's curvature is kz0' H in the onward medium: their
    difference D = C - kz0' H is the transmitted wavefront curvature relative
    to the surface. Envelopes are band-limited, so they are interpolated
    exactly (spectral zero padding) onto a grid fine enough for D before the
    currents are radiated.
    """
    k = 2 * np.pi * index / wavelength0
    impedance = Z0 / index
    _, xi = _grid(size, step)
    XU, XV = np.meshgrid(xi, xi)
    dxi = 2 * np.pi / (size * step)
    coarse, coarse_xi = _grid(chart.size, chart.step)
    CX, CY = np.meshgrid(coarse, coarse)
    fit_basis = np.stack((0.5 * CX**2, CX * CY, 0.5 * CY**2, np.ones_like(CX)), -1)

    def work(atom):
        x_f, q, curvature, E, H = atom
        h0 = surface.height(x_f[None])[0]
        g = surface.gradient(x_f[None])[0]
        kappa = _refracted_carrier(q, g, k)
        kz0 = np.sqrt(k**2 - kappa @ kappa)
        # Residual chirp D on the coarse grid decides the refinement.
        xy = np.stack((x_f[0] + CX, x_f[1] + CY), -1)
        valid = _on_surface(surface, xy)
        r = _heights(surface, xy, valid) - h0 - g[0] * CX - g[1] * CY
        fit = np.linalg.lstsq(fit_basis[valid], r[valid], rcond=None)[0]
        D = curvature - kz0 * np.array([[fit[0], fit[1]], [fit[1], fit[2]]])
        magnitude = np.linalg.norm(_spectrum(np.concatenate((E, H), -1)), axis=-1)
        XC, YC = np.meshgrid(coarse_xi, coarse_xi)
        occupied = magnitude > tolerance * magnitude.max()
        envelope_band = max(np.abs(XC[occupied]).max(), np.abs(YC[occupied]).max())
        band = envelope_band + np.abs(D).sum(axis=1).max() * abs(coarse[0])
        # Smallest even grid (same extent) whose Nyquist band holds 1.2 x band.
        fine_size = chart.size + 2 * max(
            0, int(np.ceil((1.2 * band * chart.step / np.pi - 1) * chart.size / 2))
        )
        fine_step = chart.step * chart.size / fine_size
        E, H = _upsample(E, fine_size), _upsample(H, fine_size)
        local, _ = _grid(fine_size, fine_step)
        LX, LY = np.meshgrid(local, local)
        xy = np.stack((x_f[0] + LX, x_f[1] + LY), -1)
        valid = _on_surface(surface, xy)
        normals = np.zeros(xy.shape[:-1] + (3,))
        normals[valid] = surface.normals(xy[valid])
        area = np.zeros(xy.shape[:-1])
        area[valid] = surface.area_jacobian(xy[valid]) * fine_step**2
        currents = (
            np.concatenate((np.cross(normals, H), -np.cross(normals, E)), -1)
            * area[..., None]
        )
        r = _heights(surface, xy, valid) - h0 - g[0] * LX - g[1] * LY
        r_mid = (r[valid].max() + r[valid].min()) / 2
        r_half = (r[valid].max() - r[valid].min()) / 2
        r = np.where(valid, r, r_mid)
        # Integrand without its linear phase: currents exp(i(C-part) - i kz0' r).
        compensated = (
            currents * np.exp(1j * (_quadratic(curvature, LX, LY) - kz0 * r))[..., None]
        )
        ku, kv = kappa[0] + XU, kappa[1] + XV
        kz = np.sqrt(np.maximum(k**2 - ku**2 - kv**2, 0))
        spectrum = np.linalg.norm(_spectrum(compensated), axis=-1)
        chart_band = spectrum > tolerance * spectrum.max()
        for _ in range(2):
            chart_band = (
                chart_band | np.roll(chart_band, 1, 0) | np.roll(chart_band, -1, 0)
            )
            chart_band = (
                chart_band | np.roll(chart_band, 1, 1) | np.roll(chart_band, -1, 1)
            )
        pu, pv = XU + (kz - kz0) * g[0], XV + (kz - kz0) * g[1]
        cell = 2 * np.pi / (fine_size * fine_step)
        iu = np.rint(pu / cell).astype(int) % fine_size
        iv = np.rint(pv / cell).astype(int) % fine_size
        inside = (np.abs(pu) < np.pi / fine_step) & (np.abs(pv) < np.pi / fine_step)
        propagating = (ku**2 + kv**2 < k**2) & inside & chart_band[iv, iu]
        kzb = kz[propagating]
        # exp(-i kz r) = exp(-i kz0 r) exp(-i (kz - kz0) r): the second factor is low rank.
        offset = kzb - kz0
        carrier_z = (offset.max() + offset.min()) / 2
        rank = _chebyshev_rank((offset.max() - offset.min()) / 2 * r_half, tolerance)
        t = (r - r_mid) / r_half if r_half > 0 else np.zeros_like(r)
        spatial = np.exp(-1j * carrier_z * r_half * t)
        sources = np.concatenate(
            [compensated * (spatial * p)[..., None] for p in _chebyshev(t, rank)], -1
        )
        targets = np.column_stack((pu[propagating], pv[propagating]))
        integrals = _from_grid(
            sources, fine_step, targets, -1, tolerance / 10
        ).reshape(-1, rank, 6)
        coefficients = np.stack(
            [
                (1 if m == 0 else 2) * (-1j) ** m * jv(m, (offset - carrier_z) * r_half)
                for m in range(rank)
            ],
            -1,
        )
        transform = (
            np.einsum("pm,pmc->pc", coefficients, integrals)
            * np.exp(-1j * (offset - carrier_z) * r_mid - 1j * carrier_z * r_mid)[
                :, None
            ]
        )
        Jt, Mt = transform[:, :3], transform[:, 3:]
        khat = np.stack((ku[propagating], kv[propagating], kzb), -1) / k
        bracket = impedance * (Jt - khat * np.sum(khat * Jt, -1)[:, None]) - np.cross(
            khat, Mt
        )
        amplitude = -(k / (8 * np.pi**2 * kzb))[:, None] * bracket
        spectrum_out = np.zeros((size, size, 3), complex)
        # Chart carrier and footprint phase: exp(i kz0' g . x_f) (the xi . x_f parts cancel).
        spectrum_out[propagating] = amplitude * dxi**2 * np.exp(1j * kz0 * (g @ x_f))
        return x_f, h0, kappa, _envelope(spectrum_out)

    results = _map_atoms(
        work,
        zip(
            chart.centres,
            chart.carriers,
            chart.curvatures,
            chart.electric,
            chart.magnetic,
        ),
    )
    return AtomField(
        np.array([r[0] for r in results]),
        np.array([r[1] for r in results]),
        np.array([r[2] for r in results]),
        np.array([r[3] for r in results]),
        step,
        k,
    )
