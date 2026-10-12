"""Surface trace transfer Q, least-squares inversion, and outgoing reconstruction."""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import scipy.fft
from scipy.fft import ifft2
from scipy.sparse.linalg import LinearOperator, lsmr
from scipy.special import gammaln, jv

from vecdiff_general.fields import ElectricField, ElectromagneticField
from vecdiff_general.fields.spectrum import PlaneSpectrum, PlaneWaveExpansion
from vecdiff_general.fourier.cartesian import FFT2, frequency_grid
from vecdiff_general.media.medium import Z0
from vecdiff_general.sampling import CartesianGrid, SurfaceGrid

_PRECOMPUTE_BYTES = 64 * 1024**2
_COMPACT_FACTOR_BYTES = 512 * 1024**2


@dataclass(frozen=True)
class InverseReport:
    """Relative surface L2 trace errors; these are not full boundary errors.

    Electric and magnetic residuals use orthogonal tangential projection and
    physical surface area. Only tangential E is fitted; the magnetic residual
    reports how well the fitted Maxwell modes also match the trace's H.
    """

    relative_residual: float
    iterations: int
    condition_estimate: float
    magnetic_relative_residual: float
    solver: str


class SurfaceTransport:
    """Angular-spectrum trace Q_{n,Sigma} and reconstruction on a surface chart.

    A Chebyshev/Bessel separation of exp(i kz h) uses only FFTs and pointwise
    products, with an explicitly bounded scalar kernel truncation error.
    Work is O(rank * N log N); rank grows with sag times longitudinal bandwidth.
    Only forward propagating modes with sin(theta) <= max_sine are retained.
    The finite FFT cell is periodic: callers must test padding and sampling.
    """

    def __init__(
        self, plane, surface, region, wavelength0, *, max_sine=0.85, tolerance=1e-10
    ):
        if not isinstance(plane, CartesianGrid):
            raise TypeError("Q requires a CartesianGrid tangent plane")
        if not (0 < max_sine < 1 and 0 < tolerance < 1):
            raise ValueError("max_sine and tolerance must lie in (0, 1)")
        if not np.allclose(plane.frame.basis, surface.frame.basis, atol=1e-12, rtol=0):
            raise ValueError("Q plane and surface must share a frame orientation")
        self.plane, self.surface, self.region = plane, surface, region
        self.wavelength0 = wavelength0
        local = (plane.points - surface.frame.origin) @ surface.frame.basis
        self.xy = local[..., :2]
        self.height = surface.height(self.xy) - local[..., 2]
        self.gradient = surface.gradient(self.xy)
        self._jacobian = np.sqrt(1 + np.sum(self.gradient**2, axis=-1))
        self.grid = SurfaceGrid(
            surface.points(self.xy),
            surface.normals(self.xy),
            plane.weights * surface.area_jacobian(self.xy),
        )
        self.k = 2 * np.pi * region.material.n / wavelength0
        fx, fy = frequency_grid(plane.coordinates)
        self.ku, self.kv = 2 * np.pi * fx, 2 * np.pi * fy
        self.active = self.ku**2 + self.kv**2 <= (max_sine * self.k) ** 2
        self.kz = np.sqrt(np.maximum(self.k**2 - self.ku**2 - self.kv**2, 0))
        self.kz_safe = np.where(self.active, self.kz, self.k)
        self.count = int(self.active.sum())
        self.size = int(np.prod(plane.shape))
        lo, hi = self.kz[self.active].min(), self.kz[self.active].max()
        carrier = (hi + lo) / 2
        midpoint = (self.height.max() + self.height.min()) / 2
        halfspan = (self.height.max() - self.height.min()) / 2
        argument = (self.kz - carrier) * halfspan
        radius = (hi - lo) * halfspan / 2
        # |J_m(x)| <= (|x|/2)^m / m! * exp(x²/(4(m+1))).
        # Past m>|x|, successive bounds decrease geometrically.
        from scipy.special import gammaln

        rank = max(1, int(np.ceil(radius)))
        while radius > 0:
            log_bound = (
                rank * np.log(radius / 2)
                - gammaln(rank + 1)
                + radius**2 / (4 * (rank + 1))
            )
            ratio = radius / (2 * (rank + 1))
            if ratio < 1 and log_bound + np.log(2 / (1 - ratio)) < np.log(tolerance):
                break
            rank += 1
        self.rank = rank
        t = (
            np.zeros_like(self.height)
            if halfspan == 0
            else (self.height - midpoint) / halfspan
        )
        carrier_phase = np.exp(1j * carrier * self.height)
        frequency_phase = np.exp(1j * (self.kz - carrier) * midpoint)
        # Radially symmetric FFT lattices repeat many kz values. Evaluate the
        # expensive Bessel factors only at distinct, active arguments, without
        # rounding or changing the bandwidth or separation rank.
        arguments, argument_ids = np.unique(argument[self.active], return_inverse=True)
        self._terms = None
        if 4 * rank * self.size * np.dtype(complex).itemsize > _PRECOMPUTE_BYTES:
            # Keep the same expansion and rank while streaming its dense terms.
            # Large charts otherwise retain four rank-sized collections of
            # dense arrays per Q, before any fields have been propagated.
            self._term_coordinate = np.fft.ifftshift(t)
            self._carrier_phase = np.fft.ifftshift(carrier_phase)
            self._frequency_phase = np.fft.ifftshift(frequency_phase)
            self._shifted_active = np.fft.ifftshift(self.active)
            ids = np.zeros(self.plane.shape, dtype=int)
            ids[self.active] = argument_ids
            self._argument_ids = np.fft.ifftshift(ids)[self._shifted_active]
            self._arguments = arguments
            self._compact_factors = (
                [None] * rank
                if rank * arguments.size * np.dtype(float).itemsize
                <= _COMPACT_FACTOR_BYTES
                else None
            )
            return
        self.spatial, self.spectral = [], []
        previous, current = np.ones_like(t), t
        for order in range(rank):
            polynomial = previous if order == 0 else current
            self.spatial.append(carrier_phase * polynomial)
            factor = np.zeros_like(frequency_phase)
            factor[self.active] = (
                (1 if order == 0 else 2)
                * 1j**order
                * jv(order, arguments)[argument_ids]
            )
            self.spectral.append(frequency_phase * factor)
            if order:
                previous, current = current, 2 * t * current - previous
        # The same terms in unshifted FFT order: each sum then needs one shift.
        self._terms = [
            (np.fft.ifftshift(weight), np.fft.ifftshift(factor))
            for weight, factor in zip(self.spatial, self.spectral)
        ]

    def _iter_terms(self):
        if self._terms is not None:
            yield from self._terms
            return
        previous, current = np.ones_like(self._term_coordinate), self._term_coordinate
        for order in range(self.rank):
            polynomial = previous if order == 0 else current
            bessel = (
                None if self._compact_factors is None else self._compact_factors[order]
            )
            if bessel is None:
                bessel = jv(order, self._arguments)
                if self._compact_factors is not None:
                    self._compact_factors[order] = bessel
            factor = np.zeros(self.plane.shape, complex)
            factor[self._shifted_active] = (
                (1 if order == 0 else 2) * 1j**order * bessel[self._argument_ids]
            )
            yield self._carrier_phase * polynomial, self._frequency_phase * factor
            if order:
                previous, current = (
                    current,
                    2 * self._term_coordinate * current - previous,
                )

    def _synthesize(self, coefficients):
        shifted = np.fft.ifftshift(coefficients, axes=(0, 1))
        total = sum(
            weight[..., None]
            * scipy.fft.ifft2(factor[..., None] * shifted, axes=(0, 1), workers=-1)
            for weight, factor in self._iter_terms()
        )
        return np.fft.fftshift(total, axes=(0, 1))

    def _adjoint(self, values):
        shifted = np.fft.ifftshift(values, axes=(0, 1))
        total = sum(
            factor.conj()[..., None]
            * scipy.fft.fft2(
                weight.conj()[..., None] * shifted, axes=(0, 1), workers=-1
            )
            for weight, factor in self._iter_terms()
        )
        return np.fft.fftshift(total, axes=(0, 1)) / self.size

    def _complete(self, transverse):
        longitudinal = (
            -(self.ku * transverse[..., 0] + self.kv * transverse[..., 1])
            / self.kz_safe
        )
        return np.concatenate((transverse, longitudinal[..., None]), axis=-1)

    def _unpack(self, vector):
        coefficients = np.zeros(self.plane.shape + (2,), complex)
        coefficients[self.active] = vector.reshape(-1, 2)
        return coefficients

    def _tangent(self, local):
        return local[..., :2] + self.gradient * local[..., 2, None]

    def _trace_weight(self, tangent):
        """sqrt(J) G^-1/2 for covariant chart components, G=I+grad(h)grad(h)^T.

        Its squared Euclidean norm is the physical integral of |E_t|^2,
        up to the uniform plane cell area, which cancels in the solve.
        The symmetric rank-one expression avoids a matrix factorization per node.
        """
        parallel = np.sum(self.gradient * tangent, axis=-1)
        return np.sqrt(self._jacobian)[..., None] * (
            tangent
            - self.gradient
            * (parallel / (self._jacobian * (self._jacobian + 1)))[..., None]
        )

    def _inverse_operator(self):
        """Weighted tangential E of the propagating modes, and its adjoint."""

        def matvec(vector):
            coefficients = self._complete(self._unpack(vector))
            return self._trace_weight(
                self._tangent(self._synthesize(coefficients))
            ).ravel()

        def rmatvec(vector):
            tangent = self._trace_weight(vector.reshape(self.plane.shape + (2,)))
            values = np.concatenate(
                (tangent, np.sum(self.gradient * tangent, axis=-1)[..., None]), axis=-1
            )
            coefficients = self._adjoint(values)
            result = (
                coefficients[..., :2]
                - np.stack((self.ku, self.kv), axis=-1)
                / self.kz_safe[..., None]
                * coefficients[..., 2, None]
            )
            return result[self.active].ravel()

        return LinearOperator(
            (2 * self.size, 2 * self.count),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=complex,
        )

    def _coefficients(self, spectrum):
        """Validate and normalize a spectrum on this transport's plane."""
        if (
            spectrum.expansion.region != self.region
            or spectrum.expansion.wavelength0 != self.wavelength0
        ):
            raise ValueError("Q spectrum material or wavelength does not match")
        if spectrum.grid.shape != self.plane.shape or not np.allclose(
            spectrum.grid.points, self.plane.points, atol=1e-14, rtol=0
        ):
            raise ValueError("Q spectrum must use its declared tangent lattice")
        occupied = np.any(spectrum.expansion.amplitudes != 0, axis=-1)
        normal_frequency = spectrum.expansion.wavevectors @ self.plane.frame.normal
        if np.any(normal_frequency[occupied].real <= 0):
            raise ValueError("Q requires forward propagating modes")
        coefficients = (
            spectrum.expansion.amplitudes.reshape(self.plane.shape + (3,)) * self.size
        )
        outside = np.linalg.norm(coefficients[~self.active])
        if outside > 1e-9 * max(np.linalg.norm(coefficients), 1e-300):
            raise ValueError(
                "Q input exceeds its declared angular band; increase max_sine"
            )
        return coefficients

    def electric_jet(self, spectrum):
        """E and its three Cartesian derivatives on Sigma in one batched Q.

        Derivatives act on the homogeneous incident field before restriction
        to the surface. No finite difference of wrapped surface phase is used.
        The last two gradient axes are (electric component, derivative axis).
        """
        coefficients = self._coefficients(spectrum)
        k = spectrum.expansion.wavevectors.reshape(self.plane.shape + (3,))
        derivative = 1j * coefficients[..., :, None] * k[..., None, :]
        values = self._synthesize(
            np.concatenate(
                (coefficients, derivative.reshape(self.plane.shape + (9,))), axis=-1
            )
        )
        return values[..., :3], values[..., 3:].reshape(self.plane.shape + (3, 3))

    def forward(self, spectrum):
        """Evaluate a compatible plane spectrum on Sigma, including E and H."""
        coefficients = self._coefficients(spectrum)
        electric = self._synthesize(coefficients)
        magnetic = self._synthesize(
            spectrum.expansion.magnetic_amplitudes.reshape(coefficients.shape)
            * self.size
        )
        return ElectromagneticField(
            ElectricField(self.grid, self.wavelength0, electric), magnetic
        )

    def radiate(self, trace):
        """Outgoing surface-equivalence reconstruction from tangential E/H.

        This is an inverse on compatible fields with sufficient surface coverage
        and quadrature, not the discrete least-squares pseudoinverse. Finite open
        charts, tapering and the angular cutoff introduce reconstruction error.
        No least-squares fit: see radiated_spectrum.
        """
        self._validate_trace(trace)
        return radiated_spectrum(
            self, trace, lambda v: self._adjoint(v) * self.size * self.active[..., None]
        )

    def _validate_trace(self, trace):
        if trace.electric.wavelength0 != self.wavelength0:
            raise ValueError("Q inverse wavelength does not match")
        if trace.electric.grid is not self.grid:
            if trace.electric.grid.shape != self.grid.shape or not np.allclose(
                trace.electric.grid.points, self.grid.points, atol=1e-14, rtol=0
            ):
                raise ValueError("Q inverse trace must lie on its surface chart")

    def inverse(self, trace, *, tolerance=1e-9, max_iterations=200):
        """Q^-1: fit physical surface-tangential E in L2(dS) with outgoing modes.

        The residual is returned, never silently interpreted as Maxwell boundary
        accuracy. Local Fresnel data on curved surfaces need not be the trace of
        an exact outgoing Maxwell field. E and H residuals are reported separately.
        Flat charts use an exact per-mode solve; curved charts use LSMR.
        """
        self._validate_trace(trace)
        if not 0 < tolerance < 1 or max_iterations < 1:
            raise ValueError(
                "inverse tolerance must lie in (0, 1) and iterations must be positive"
            )
        target_e = self._trace_weight(
            self._tangent(trace.electric.values @ self.plane.frame.basis)
        )
        target_h = self._trace_weight(
            self._tangent(trace.magnetic @ self.plane.frame.basis)
        )
        height = self.height.flat[0]
        flat = not np.any(self.gradient) and (
            self.k * np.max(abs(self.height - height)) <= 32 * np.finfo(float).eps
        )
        if flat:
            transformed = FFT2(target_e) * np.exp(-1j * self.kz * height)[..., None]
            vector = transformed[self.active].ravel()
            iterations, condition, solver = 0, 1.0, "fft"
        else:
            result = lsmr(
                self._inverse_operator(),
                target_e.ravel(),
                atol=tolerance,
                btol=tolerance,
                maxiter=max_iterations,
            )
            if result[1] not in (0, 1, 2, 4, 5):
                raise ValueError(
                    f"Q inverse did not converge (stop={result[1]}, iterations={result[2]}); refine the chart or angular band"
                )
            vector, iterations, condition, solver = (
                result[0],
                result[2],
                result[6],
                "lsmr",
            )
        coefficients = self._complete(self._unpack(vector)) @ self.plane.frame.basis.T
        wavevectors = (
            self.ku[..., None] * self.plane.frame.u
            + self.kv[..., None] * self.plane.frame.v
            + np.sqrt((self.k**2 - self.ku**2 - self.kv**2).astype(complex))[..., None]
            * self.plane.frame.normal
        )
        expansion = PlaneWaveExpansion(
            wavevectors.reshape(-1, 3),
            coefficients.reshape(-1, 3) / self.size,
            self.plane.points[self.plane.shape[0] // 2, self.plane.shape[1] // 2],
            self.wavelength0,
            self.region,
        )
        spectrum = PlaneSpectrum(self.plane, expansion)
        reconstructed = self.forward(spectrum)
        actual_e = self._trace_weight(
            self._tangent(reconstructed.electric.values @ self.plane.frame.basis)
        )
        actual_h = self._trace_weight(
            self._tangent(reconstructed.magnetic @ self.plane.frame.basis)
        )

        def residual(actual, desired):
            return float(
                np.linalg.norm(actual - desired) / max(np.linalg.norm(desired), 1e-300)
            )

        return spectrum, InverseReport(
            residual(actual_e, target_e),
            iterations,
            condition,
            residual(actual_h, target_h),
            solver,
        )


def radiated_spectrum(transport, trace, adjoint):
    """Surface-equivalence outgoing spectrum from tangential E and H on Sigma.

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
    weights = transport.grid.weights[..., None]
    k, ku, kv, active = transport.k, transport.ku, transport.kv, transport.active
    khat = [ku / k, kv / k, transport.kz / k]
    impedance = Z0 / transport.region.material.n
    # J = n x H first; its transverse-projected spectrum starts the amplitudes.
    amplitudes = adjoint((np.cross(normals, trace.magnetic) @ basis) * weights)
    projection = sum(khat[i] * amplitudes[..., i] for i in range(3))
    for i in range(3):
        amplitudes[..., i] -= khat[i] * projection
    amplitudes *= impedance
    del projection
    # M = -n x E: subtract k^ x M~.
    moment = adjoint((-np.cross(normals, trace.electric.values) @ basis) * weights)
    amplitudes[..., 0] -= khat[1] * moment[..., 2] - khat[2] * moment[..., 1]
    amplitudes[..., 1] -= khat[2] * moment[..., 0] - khat[0] * moment[..., 2]
    amplitudes[..., 2] -= khat[0] * moment[..., 1] - khat[1] * moment[..., 0]
    del moment
    cell = transport.size * plane.weights.flat[0]
    amplitudes *= (-(k / (2 * cell * np.where(active, transport.kz, 1.0))) * active)[
        ..., None
    ]
    # The radiation kernel is analytically transverse. Cancellation between
    # electric/magnetic currents can leave nearly empty modes with longitudinal
    # roundoff comparable to their remaining amplitude. Complete the final
    # local vector explicitly, so these valid fields pass per-mode Maxwell
    # validation without weakening that validation or dropping weak modes.
    amplitudes[..., 2] = -(
        ku * amplitudes[..., 0] + kv * amplitudes[..., 1]
    ) / np.where(active, transport.kz, 1.0)
    if not np.allclose(basis, np.eye(3)):
        amplitudes = amplitudes @ basis.T
    kz = np.sqrt((k**2 - ku**2 - kv**2).astype(complex))
    wavevectors = (
        ku[..., None] * plane.frame.u
        + kv[..., None] * plane.frame.v
        + kz[..., None] * plane.frame.normal
    )
    del kz
    ny, nx = plane.shape
    expansion = PlaneWaveExpansion(
        wavevectors.reshape(-1, 3),
        amplitudes.reshape(-1, 3),
        plane.points[ny // 2, nx // 2],
        transport.wavelength0,
        transport.region,
    )
    return PlaneSpectrum(plane, expansion)


class BandSynthesis:
    """Scalar Q on a fixed supported subset; works on any height-map chart.

    Coefficients outside the declared support must be zero. All support modes
    are propagated with their original kz, including the frequency carrier.
    Only the separable representation's bandwidth and rank change.
    """

    def __init__(self, transport, retained_ids, *, workers=1, tolerance=1e-12):
        self.shape = transport.plane.shape
        self.workers = workers
        self.ids = np.unique(retained_ids)
        rows, columns = np.unravel_index(self.ids, self.shape)
        self.shifted_ids = ((rows - self.shape[0] // 2) % self.shape[0]) * self.shape[
            1
        ] + (columns - self.shape[1] // 2) % self.shape[1]
        active = np.zeros(transport.size, dtype=bool)
        active[retained_ids] = True
        active = active.reshape(self.shape)
        kz = transport.kz[active]
        carrier = (kz.max() + kz.min()) / 2
        h = transport.height
        midpoint, halfspan = (h.max() + h.min()) / 2, (h.max() - h.min()) / 2
        radius = (kz.max() - kz.min()) * halfspan / 2
        rank = max(1, int(np.ceil(radius)))
        while radius > 0:
            log_bound = (
                rank * np.log(radius / 2)
                - gammaln(rank + 1)
                + radius**2 / (4 * (rank + 1))
            )
            ratio = radius / (2 * (rank + 1))
            if ratio < 1 and log_bound + np.log(2 / (1 - ratio)) < np.log(tolerance):
                break
            rank += 1
        self.rank = rank
        self.tail_bound = (
            0.0 if radius == 0 else float(np.exp(log_bound) * 2 / (1 - ratio))
        )
        # Evaluate special functions only once for exactly equal arguments.
        argument, inverse = np.unique((kz - carrier) * halfspan, return_inverse=True)
        frequency_phase = np.exp(1j * (kz - carrier) * midpoint)
        self.carrier = np.fft.ifftshift(np.exp(1j * carrier * h))
        t = np.zeros_like(h) if halfspan == 0 else (h - midpoint) / halfspan
        if rank * (transport.size * 8 + len(kz) * 16) > _PRECOMPUTE_BYTES:
            self.terms = None
            self._term_coordinate = np.fft.ifftshift(t)
            self._arguments, self._argument_ids = argument, inverse
            self._frequency_phase = frequency_phase
            self._compact_factors = (
                [jv(order, argument) for order in range(rank)]
                if rank * argument.size * 8 <= _COMPACT_FACTOR_BYTES
                else None
            )
            self.nbytes = sum(
                a.nbytes
                for a in (
                    self.carrier,
                    self._term_coordinate,
                    self._arguments,
                    self._argument_ids,
                    self._frequency_phase,
                )
            ) + sum(a.nbytes for a in (self._compact_factors or []))
            return
        previous, current = np.ones_like(t), t
        self.terms = []
        for order in range(rank):
            polynomial = previous if order == 0 else current
            spectral = (
                frequency_phase
                * (1 if order == 0 else 2)
                * 1j**order
                * jv(order, argument)[inverse]
            )
            self.terms.append((np.fft.ifftshift(polynomial), spectral))
            if order:
                previous, current = current, 2 * t * current - previous
        self.nbytes = self.carrier.nbytes + sum(
            a.nbytes + b.nbytes for a, b in self.terms
        )

    def _iter_terms(self):
        if self.terms is not None:
            yield from self.terms
            return
        previous, current = np.ones_like(self._term_coordinate), self._term_coordinate
        for order in range(self.rank):
            polynomial = previous if order == 0 else current
            bessel = (
                jv(order, self._arguments)
                if self._compact_factors is None
                else self._compact_factors[order]
            )
            spectral = (
                self._frequency_phase
                * (1 if order == 0 else 2)
                * 1j**order
                * bessel[self._argument_ids]
            )
            yield polynomial, spectral
            if order:
                previous, current = (
                    current,
                    2 * self._term_coordinate * current - previous,
                )

    def __call__(self, coefficients):
        # Contiguous spatial axes avoid strided, interleaved polarization FFTs.
        occupied = np.any(coefficients != 0, axis=(0, 1))
        if not np.any(occupied):
            return np.zeros_like(coefficients)
        # Compact plan factors retain every supported mode. Only the FFT
        # workspace is dense, with exact zeros outside that support.
        values = coefficients.reshape(-1, coefficients.shape[-1])[self.ids][
            :, occupied
        ].T
        total = np.zeros((int(occupied.sum()),) + self.shape, complex)
        for polynomial, factor in self._iter_terms():
            workspace = np.zeros_like(total)
            workspace.reshape(len(values), -1)[:, self.shifted_ids] = (
                values * factor[None, :]
            )
            wave = ifft2(
                workspace,
                axes=(-2, -1),
                workers=self.workers,
                overwrite_x=True,
            )
            total += polynomial[None, ...] * wave
        total *= self.carrier[None, ...]
        result = np.zeros_like(coefficients)
        result[..., occupied] = np.moveaxis(
            np.fft.fftshift(total, axes=(-2, -1)), 0, -1
        )
        return result


class BandPlanCache:
    """Cache plans for repeated fields on the same geometry and modal support."""

    def __init__(self, workers=1, cache_bytes=1024**3):
        self.workers = workers
        self.cache_bytes = cache_bytes
        self.plans = OrderedDict()
        self.cached_bytes = 0

    def plan(self, transport, retained_ids):
        key = (transport, retained_ids.tobytes())
        if key in self.plans:
            self.plans.move_to_end(key)
            return self.plans[key]
        plan = BandSynthesis(transport, retained_ids, workers=self.workers)
        while self.plans and self.cached_bytes + plan.nbytes > self.cache_bytes:
            _, old = self.plans.popitem(last=False)
            self.cached_bytes -= old.nbytes
        if plan.nbytes <= self.cache_bytes:
            self.plans[key] = plan
            self.cached_bytes += plan.nbytes
        return plan
