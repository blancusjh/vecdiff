"""Tiled Q and explicit Q^-1 between a plane lattice and a graph surface.

The global operators of SurfaceTransport separate exp(i kz h) with a Chebyshev
rank that grows with (angular band) x (height span of the whole surface); for a
millimetre sag that rank reaches thousands. Here both directions work tile by
tile, so the rank follows only the height left over after each tile's tangent
plane:

* Q (plane -> Sigma): an erf partition of unity splits the plane field into
  pieces. Each piece keeps its own significant angular band and is evaluated on
  its surface footprint as sum_k c(k) exp(i (k + kz g).x) exp(i kz r(x)): the
  tangent-plane tilt g enters exactly through type-3 NUFFT frequencies, and the
  curvature residual r through a Jacobi-Anger/Chebyshev expansion.
* Q^H (Sigma -> spectrum): each surface tile is radiated into its own band and
  returned to an equivalent plane field in a local box; the boxes are summed
  and one global FFT gives the spectrum.
* The explicit Q^-1 (``radiate``) uses both tangential traces (equivalence
  principle, Weyl expansion) and Q^H; it needs no iterative solve.

Every truncation (window tails, band masks, Chebyshev rank, NUFFT) is bounded by
``tolerance``. Boxes and footprints follow the pieces' ray displacement and
angular spread over the local height, so the cost is O(N log N) and does not
grow with the total sag. The cell is periodic, as for every lattice operator:
fields must vanish smoothly before its edges.
"""

import numpy as np
from scipy.special import erf, gammaln, jv

from ..fields import ElectricField, ElectromagneticField
from ..fields.spectrum import PlaneSpectrum, PlaneWaveExpansion
from ..fourier.cartesian import frequency_grid
from ..media.medium import Z0
from ..sampling import CartesianGrid, SurfaceGrid


def _chebyshev_rank(radius, tolerance):
    """Terms of exp(i a t), |a| <= radius, |t| <= 1, for error below tolerance."""
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


def _jacobi_anger(argument, rank, sign):
    """Coefficients of exp(sign i a t) = sum_m coef_m T_m(t)."""
    return [
        (1 if m == 0 else 2) * (sign * 1j) ** m * jv(m, argument) for m in range(rank)
    ]


def _window(n, tile, centre, tolerance):
    """erf partition of unity: plateau `tile` about `centre`, Gaussian spectrum."""
    sigma = tile / 6
    offset = (np.arange(n) - centre + n // 2) % n - n // 2
    weight = 0.5 * (erf((offset + tile / 2) / sigma) - erf((offset - tile / 2) / sigma))
    support = np.flatnonzero(weight > 1e-2 * tolerance)
    return support, offset[support], weight[support]


def _dilate(mask, cells):
    for _ in range(cells):
        mask = (
            mask
            | np.roll(mask, 1, 0)
            | np.roll(mask, -1, 0)
            | np.roll(mask, 1, 1)
            | np.roll(mask, -1, 1)
        )
    return mask


class TiledSurfaceTransport:
    """Tiled Q, Q^H and explicit Q^-1 on a CartesianGrid tangent plane."""

    def __init__(
        self,
        plane,
        surface,
        region,
        wavelength0,
        *,
        max_sine=0.85,
        tolerance=1e-8,
        tile=None,
    ):
        try:
            import finufft
        except ImportError as error:
            raise ImportError(
                "install the nufft extra to use tiled transport"
            ) from error
        self._nufft = finufft.nufft2d3
        if not isinstance(plane, CartesianGrid):
            raise TypeError("tiled transport requires a CartesianGrid tangent plane")
        if not (0 < max_sine < 1 and 0 < tolerance < 1):
            raise ValueError("max_sine and tolerance must lie in (0, 1)")
        if not np.allclose(plane.frame.basis, surface.frame.basis, atol=1e-12, rtol=0):
            raise ValueError("plane and surface must share a frame orientation")
        self.plane, self.surface, self.region = plane, surface, region
        self.wavelength0, self.max_sine, self.tolerance = (
            wavelength0,
            max_sine,
            tolerance,
        )
        local = (plane.points - surface.frame.origin) @ surface.frame.basis
        self.xy = local[..., :2]
        self.height = surface.height(self.xy) - local[..., 2]
        self.gradient = surface.gradient(self.xy)
        self.grid = SurfaceGrid(
            surface.points(self.xy),
            surface.normals(self.xy),
            plane.weights * surface.area_jacobian(self.xy),
        )
        self.k = 2 * np.pi * region.material.n / wavelength0
        self.shape = ny, nx = plane.shape
        self.du, self.dv = plane.spacing
        common = np.gcd(ny, nx)
        if tile is None:
            divisors = [d for d in range(8, common // 4 + 1) if common % d == 0]
            tile = min(divisors, key=lambda d: abs(d - 64), default=common)
        if ny % tile or nx % tile:
            raise ValueError("tile must divide both lattice dimensions")
        self.tile = int(tile)
        fx, fy = frequency_grid(plane.coordinates)
        self.ku, self.kv = 2 * np.pi * fx, 2 * np.pi * fy
        self.active = self.ku**2 + self.kv**2 <= (max_sine * self.k) ** 2
        self.kz = np.sqrt(np.maximum(self.k**2 - self.ku**2 - self.kv**2, 0))
        self.size = ny * nx
        self.stats = {}

    # ------------------------------------------------------------------ helpers
    def _tiles(self):
        ny, nx = self.shape
        for cy in range(0, ny, self.tile):
            iy, oy, wy = _window(ny, self.tile, cy, self.tolerance)
            for cx in range(0, nx, self.tile):
                ix, ox, wx = _window(nx, self.tile, cx, self.tolerance)
                yield cy, cx, iy, ix, oy, ox, wy[:, None] * wx[None, :]

    def _box_modes(self, size):
        ku = 2 * np.pi * np.fft.fftfreq(size, self.du)
        kv = 2 * np.pi * np.fft.fftfreq(size, self.dv)
        KV, KU = np.meshgrid(kv, ku, indexing="ij")
        active = KU**2 + KV**2 <= (self.max_sine * self.k) ** 2
        return KU, KV, active

    def _kz(self, KU, KV):
        return np.sqrt(np.maximum(self.k**2 - KU**2 - KV**2, 0))

    def _span(self, centre, half, n):
        """Footprint indices inside the cell; periodic images are not evaluated."""
        offsets = np.arange(-half, half + 1)
        index = centre + offsets
        keep = (index >= 0) & (index < n)
        if not keep.any():
            raise ValueError("a tile footprint leaves the cell; pad the cell")
        return index[keep], offsets[keep]

    # ------------------------------------------------------------------ forward
    def forward_field(self, values):
        """Plane field (ny, nx, c) -> the same field evaluated on the surface."""
        ny, nx = self.shape
        du, dv, k, tol = self.du, self.dv, self.k, self.tolerance
        out = np.zeros(values.shape, complex)
        ranks = []
        floor = tol * np.linalg.norm(values)
        for cy, cx, iy, ix, oy, ox, w in self._tiles():
            piece = values[np.ix_(iy, ix)] * w[..., None]
            if np.linalg.norm(piece) <= floor:
                continue
            half = max(abs(oy).max(), abs(ox).max()) + 1
            size = 2 * half + self.tile // 2
            for attempt in range(3):
                box = np.zeros((size, size, values.shape[-1]), complex)
                box[np.ix_(oy + size // 2, ox + size // 2)] = piece
                B = np.fft.fft2(box, axes=(0, 1)) / size**2
                KU, KV, active = self._box_modes(size)
                magnitude = np.where(active, np.linalg.norm(B, axis=-1), 0)
                band = _dilate(magnitude > tol * magnitude.max(), 1) & active
                KUs, KVs = KU[band], KV[band]
                KZs = self._kz(KUs, KVs)
                energy = magnitude[band] ** 2
                mean_u, mean_v = (
                    (energy * KUs).sum() / energy.sum(),
                    (energy * KVs).sum() / energy.sum(),
                )
                mean_z = np.sqrt(max(k**2 - mean_u**2 - mean_v**2, 1e-300))
                # Footprint centre along the energy-centroid ray (local height).
                shift = np.zeros(2)
                for _ in range(4):
                    jy = int(np.clip(round(cy + shift[1] / dv), 0, ny - 1))
                    jx = int(np.clip(round(cx + shift[0] / du), 0, nx - 1))
                    shift = np.array([mean_u, mean_v]) / mean_z * self.height[jy, jx]
                slope_u, slope_v = KUs / KZs, KVs / KZs
                reach = abs(self.height[jy, jx])
                for _ in range(2):
                    spread = max(
                        np.ptp(slope_u) * reach / du, np.ptp(slope_v) * reach / dv
                    )
                    fhalf = half + int(np.ceil(spread / 2)) + self.tile // 4
                    fy_idx, _ = self._span(jy, fhalf, ny)
                    fx_idx, _ = self._span(jx, fhalf, nx)
                    reach = np.abs(self.height[np.ix_(fy_idx, fx_idx)]).max()
                needed = 2 * (half + int(np.ceil(spread))) + self.tile // 2
                if needed <= size:
                    break
                size = needed + needed % 2
            fy_idx, fy_off = self._span(jy, fhalf, ny)
            fx_idx, fx_off = self._span(jx, fhalf, nx)
            FY = (jy - cy + fy_off) * dv
            FX = (jx - cx + fx_off) * du
            FXg, FYg = np.meshgrid(FX, FY)
            x0 = np.array([(jx - cx) * du, (jy - cy) * dv])
            G = self.gradient[jy, jx]
            h0 = self.height[jy, jx]
            H = self.height[np.ix_(fy_idx, fx_idx)]
            r = H - (h0 + G[0] * (FXg - x0[0]) + G[1] * (FYg - x0[1]))
            r_mid, r_half = (r.max() + r.min()) / 2, (r.max() - r.min()) / 2
            carrier = (KZs.max() + KZs.min()) / 2
            rank = _chebyshev_rank((KZs.max() - KZs.min()) / 2 * r_half, tol)
            t = (r - r_mid) / r_half if r_half > 0 else np.zeros_like(r)
            # Box index m sits at (m - size//2) d from the tile centre.
            base = (
                B[band]
                * np.exp(
                    1j * (KUs * (size // 2) * du + KVs * (size // 2) * dv)
                    + 1j * KZs * (h0 - G[0] * x0[0] - G[1] * x0[1] + r_mid)
                )[:, None]
            )
            coefficients = _jacobi_anger((KZs - carrier) * r_half, rank, 1)
            strengths = np.concatenate([(base * c[:, None]).T for c in coefficients])
            f = self._nufft(
                KUs + KZs * G[0],
                KVs + KZs * G[1],
                np.ascontiguousarray(strengths),
                FXg.ravel(),
                FYg.ravel(),
                isign=1,
                eps=tol / 10,
            ).reshape(rank, values.shape[-1], *FXg.shape)
            spatial = np.exp(1j * carrier * r_half * t)
            result = sum(spatial * p * f[m] for m, p in enumerate(_chebyshev(t, rank)))
            out[np.ix_(fy_idx, fx_idx)] += np.moveaxis(result, 0, -1)
            ranks.append(rank)
        self.stats["forward"] = {"tiles": len(ranks), "max_rank": max(ranks, default=0)}
        return out

    def forward(self, spectrum):
        """Q: evaluate a PlaneSpectrum on the tangent lattice onto Sigma (E and H)."""
        if spectrum.grid.shape != self.plane.shape or not np.allclose(
            spectrum.grid.points, self.plane.points, atol=1e-14, rtol=0
        ):
            raise ValueError("Q spectrum must use its declared tangent lattice")
        shape = self.plane.shape + (3,)
        basis = self.plane.frame.basis
        coefficients = (
            np.concatenate(
                (
                    spectrum.expansion.amplitudes.reshape(shape) @ basis,
                    spectrum.expansion.magnetic_amplitudes.reshape(shape) @ basis,
                ),
                axis=-1,
            )
            * self.size
        )
        if np.linalg.norm(coefficients[~self.active]) > 1e-9 * max(
            np.linalg.norm(coefficients), 1e-300
        ):
            raise ValueError(
                "Q input exceeds its declared angular band; increase max_sine"
            )
        field = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(coefficients, axes=(0, 1)), axes=(0, 1)),
            axes=(0, 1),
        )
        values = self.forward_field(field)
        return ElectromagneticField(
            ElectricField(self.grid, self.wavelength0, values[..., :3] @ basis.T),
            values[..., 3:] @ basis.T,
        )

    # ------------------------------------------------------------------ adjoint
    def adjoint_spectrum(self, values):
        """S(k) = sum_x values(x) exp(-i(k.x + kz(k) h(x))) on the centred lattice."""
        ny, nx = self.shape
        du, dv, k, tol = self.du, self.dv, self.k, self.tolerance
        components = values.shape[-1]
        equivalent = np.zeros(self.shape + (components,), complex)
        ranks = []
        floor = tol * np.linalg.norm(values)
        for cy, cx, iy, ix, oy, ox, w in self._tiles():
            piece = values[np.ix_(iy, ix)] * w[..., None]
            if np.linalg.norm(piece) <= floor:
                continue
            PX, PY = np.meshgrid(ox * du, oy * dv)
            H = self.height[np.ix_(iy, ix)]
            G = self.gradient[cy, cx]
            h0 = self.height[cy, cx]
            tilt = G[0] * PX + G[1] * PY
            r = H - h0 - tilt
            r_mid, r_half = (r.max() + r.min()) / 2, (r.max() - r.min()) / 2
            half = max(abs(oy).max(), abs(ox).max()) + 1
            size = 2 * half + self.tile // 2
            # Radiated band: demodulate the tile by its own height phase.
            KUe, KVe, active_e = self._box_modes(size)
            kz_guess = k
            for _ in range(4):
                box = np.zeros((size, size, components), complex)
                box[np.ix_(oy + size // 2, ox + size // 2)] = (
                    piece * np.exp(-1j * kz_guess * (H - h0))[..., None]
                )
                magnitude = np.linalg.norm(np.fft.fft2(box, axes=(0, 1)), axis=-1)
                energy = magnitude**2
                mean_u = (energy * KUe).sum() / energy.sum()
                mean_v = (energy * KVe).sum() / energy.sum()
                kz_guess = np.sqrt(max(k**2 - mean_u**2 - mean_v**2, 1e-300))
            band_e = _dilate(magnitude > tol * magnitude.max(), 2) & active_e
            slope_u = KUe[band_e] / self._kz(KUe[band_e], KVe[band_e])
            slope_v = KVe[band_e] / self._kz(KUe[band_e], KVe[band_e])
            reach = np.abs(H).max()
            spread = max(
                np.ptp(slope_u) * reach / du + np.abs(slope_u).max() * np.ptp(H) / du,
                np.ptp(slope_v) * reach / dv + np.abs(slope_v).max() * np.ptp(H) / dv,
            )
            out_size = 2 * (half + int(np.ceil(spread))) + self.tile // 2
            out_size += out_size % 2
            if out_size > min(ny, nx):
                raise ValueError(
                    "a tile's equivalent plane field exceeds the periodic cell; pad the cell"
                )
            # Plane-equivalent centre: back along the mean ray to the plane.
            shift = -np.array([mean_u, mean_v]) / kz_guess * h0
            sy, sx = int(round(shift[1] / dv)), int(round(shift[0] / du))
            KUb, KVb, active_b = self._box_modes(out_size)
            # Select box modes whose nearest estimation mode is in the band.
            iu = np.rint(KUb / (2 * np.pi / (size * du))).astype(int) % size
            iv = np.rint(KVb / (2 * np.pi / (size * dv))).astype(int) % size
            band = band_e[iv, iu] & active_b
            if not band.any():
                continue
            KUs, KVs = KUb[band], KVb[band]
            KZs = self._kz(KUs, KVs)
            carrier = (KZs.max() + KZs.min()) / 2
            rank = _chebyshev_rank((KZs.max() - KZs.min()) / 2 * r_half, tol)
            t = (r - r_mid) / r_half if r_half > 0 else np.zeros_like(r)
            spatial = np.exp(-1j * carrier * r_half * t)
            sources = np.concatenate(
                [
                    (piece * (spatial * p)[..., None]).reshape(-1, components).T
                    for p in _chebyshev(t, rank)
                ]
            )
            g = self._nufft(
                PX.ravel(),
                PY.ravel(),
                np.ascontiguousarray(sources),
                KUs + KZs * G[0],
                KVs + KZs * G[1],
                isign=-1,
                eps=tol / 10,
            ).reshape(rank, components, -1)
            coefficients = np.stack(_jacobi_anger((KZs - carrier) * r_half, rank, -1))
            relative = (
                np.einsum("rm,rcm->mc", coefficients, g)
                * np.exp(-1j * KZs * (h0 + r_mid))[:, None]
            )
            # Box sample m sits at (m - out_size//2 + shift) d from the tile centre.
            spectrum = np.zeros((out_size, out_size, components), complex)
            spectrum[band] = (
                relative
                * np.exp(
                    1j
                    * (
                        KUs * (sx - out_size // 2) * du
                        + KVs * (sy - out_size // 2) * dv
                    )
                )[:, None]
            )
            local = np.fft.ifft2(spectrum, axes=(0, 1))
            rows = (cy + sy - out_size // 2 + np.arange(out_size)) % ny
            cols = (cx + sx - out_size // 2 + np.arange(out_size)) % nx
            equivalent[np.ix_(rows, cols)] += local
            ranks.append(rank)
        self.stats["adjoint"] = {"tiles": len(ranks), "max_rank": max(ranks, default=0)}
        spectrum = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(equivalent, axes=(0, 1)), axes=(0, 1)),
            axes=(0, 1),
        )
        return spectrum * self.active[..., None]

    # ------------------------------------------------------------ explicit Q^-1
    def radiate(self, trace):
        """Explicit Q^-1 with the tiled Q^H; see radiated_spectrum."""
        return radiated_spectrum(self, trace, self.adjoint_spectrum)


def radiated_spectrum(transport, trace, adjoint):
    """Explicit Q^-1: outgoing PlaneSpectrum from tangential E and H on Sigma.

    Love's equivalence with J = n x H, M = -n x E and the Weyl expansion give,
    for every propagating mode above the surface,
    A(k) = -k/(8 pi^2 kz) [Z (I - k^ k^) J~ - k^ x M~],
    J~ = int_Sigma J exp(-i k.r) dS, evaluated by ``adjoint`` (Q^H on the
    lattice). Both traces are used; no least-squares fit or iteration occurs.
    The spectrum represents the field above the surface.
    """
    plane = transport.plane
    basis = plane.frame.basis
    normals = transport.grid.normals
    E, H = trace.electric.values, trace.magnetic
    J = np.cross(normals, H) @ basis
    M = -np.cross(normals, E) @ basis
    weights = transport.grid.weights[..., None]
    S = adjoint(np.concatenate((J * weights, M * weights), axis=-1))
    Jt, Mt = S[..., :3], S[..., 3:]
    k, ku, kv, active = transport.k, transport.ku, transport.kv, transport.active
    kz = np.where(active, transport.kz, 1.0)
    khat = np.stack((ku, kv, transport.kz), -1) / k
    impedance = Z0 / transport.region.material.n
    bracket = impedance * (Jt - khat * np.sum(khat * Jt, -1)[..., None]) - np.cross(
        khat, Mt
    )
    cell = transport.size * plane.weights.flat[0]
    amplitudes = -(k / (2 * cell * kz))[..., None] * bracket * active[..., None]
    wavevectors = (
        ku[..., None] * plane.frame.u
        + kv[..., None] * plane.frame.v
        + np.sqrt((k**2 - ku**2 - kv**2).astype(complex))[..., None]
        * plane.frame.normal
    )
    ny, nx = plane.shape
    expansion = PlaneWaveExpansion(
        wavevectors.reshape(-1, 3),
        (amplitudes @ basis.T).reshape(-1, 3),
        plane.points[ny // 2, nx // 2],
        transport.wavelength0,
        transport.region,
    )
    return PlaneSpectrum(plane, expansion)
