"""R, Q and T for a forward, locally eikonal dielectric optical path.

Q evaluates the homogeneous angular spectrum at z=h(u,v). Its inverse is a
band-limited least-squares solve of the tangential electric trace, NOT Q with
-h; the fitted modes are complete Maxwell fields. Fresnel T is unchanged.
"""

from dataclasses import dataclass

import numpy as np
import scipy.fft
from scipy.sparse.linalg import LinearOperator, lsmr
from scipy.special import jv

from ..fields import ElectricField, ElectromagneticField
from ..fields.spectrum import PlaneSpectrum, PlaneWaveExpansion
from ..fourier.cartesian import FFT2, frequency_grid
from ..interfaces.directions import reflected_direction, transmitted_direction
from ..interfaces.fresnel import (
    apply_fresnel,
    fresnel_reflection_matrix,
    fresnel_transmission_matrix,
)
from ..media.medium import Z0
from ..sampling import CartesianGrid, SurfaceGrid
from .propagation import propagate_spectrum


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


class PlanarASM:
    """R_xi: exact homogeneous dispersion, on a parallel FFT lattice."""

    def __call__(self, spectrum, distance):
        return propagate_spectrum(spectrum, distance, model="asm")

    def spectrum(self, spectrum, distance):
        """Keep the modal representation between consecutive R/Q operators."""
        if not np.isfinite(distance):
            raise ValueError("distance must be finite")
        displacement = distance * spectrum.grid.frame.normal
        grid = spectrum.grid.translated(displacement)
        old = spectrum.expansion
        from ..fourier.nufft import phase_factor

        amplitudes = np.zeros_like(old.amplitudes)
        occupied = np.any(old.amplitudes != 0, axis=-1)
        amplitudes[occupied] = (
            old.amplitudes[occupied]
            * phase_factor(old.wavevectors[occupied], displacement[None, :])[0, :, None]
        )
        return PlaneSpectrum(
            grid,
            PlaneWaveExpansion(
                old.wavevectors,
                amplitudes,
                grid.points[grid.shape[0] // 2, grid.shape[1] // 2],
                old.wavelength0,
                old.region,
            ),
        )


class SurfaceTransport:
    """Q_{n,Sigma} and its inverse on a Cartesian surface chart.

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
        self.spatial, self.spectral = [], []
        previous, current = np.ones_like(t), t
        for order in range(rank):
            polynomial = previous if order == 0 else current
            self.spatial.append(carrier_phase * polynomial)
            factor = (1 if order == 0 else 2) * 1j**order * jv(order, argument)
            self.spectral.append(self.active * frequency_phase * factor)
            if order:
                previous, current = current, 2 * t * current - previous
        # The same terms in unshifted FFT order: each sum then needs one shift.
        self._terms = [
            (np.fft.ifftshift(weight), np.fft.ifftshift(factor))
            for weight, factor in zip(self.spatial, self.spectral)
        ]

    def _synthesize(self, coefficients):
        shifted = np.fft.ifftshift(coefficients, axes=(0, 1))
        total = sum(
            weight[..., None]
            * scipy.fft.ifft2(factor[..., None] * shifted, axes=(0, 1), workers=-1)
            for weight, factor in self._terms
        )
        return np.fft.fftshift(total, axes=(0, 1))

    def _adjoint(self, values):
        shifted = np.fft.ifftshift(values, axes=(0, 1))
        total = sum(
            factor.conj()[..., None]
            * scipy.fft.fft2(
                weight.conj()[..., None] * shifted, axes=(0, 1), workers=-1
            )
            for weight, factor in self._terms
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

    def forward(self, spectrum):
        """Evaluate a compatible plane spectrum on Sigma, including E and H."""
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
        electric = self._synthesize(coefficients)
        magnetic = self._synthesize(
            spectrum.expansion.magnetic_amplitudes.reshape(coefficients.shape)
            * self.size
        )
        return ElectromagneticField(
            ElectricField(self.grid, self.wavelength0, electric), magnetic
        )

    def radiate(self, trace):
        """Explicit Q^-1 from tangential E and H with the exact global Q^H.

        No least-squares fit: see radiated_spectrum.
        """
        return radiated_spectrum(
            self, trace, lambda v: self._adjoint(v) * self.size * self.active[..., None]
        )

    def inverse(self, trace, *, tolerance=1e-9, max_iterations=200):
        """Q^-1: fit physical surface-tangential E in L2(dS) with outgoing modes.

        The residual is returned, never silently interpreted as Maxwell boundary
        accuracy. Local Fresnel data on curved surfaces need not be the trace of
        an exact outgoing Maxwell field. E and H residuals are reported separately.
        Flat charts use an exact per-mode solve; curved charts use LSMR.
        """
        if trace.electric.wavelength0 != self.wavelength0:
            raise ValueError("Q inverse wavelength does not match")
        if not 0 < tolerance < 1 or max_iterations < 1:
            raise ValueError(
                "inverse tolerance must lie in (0, 1) and iterations must be positive"
            )
        if trace.electric.grid is not self.grid:
            if trace.electric.grid.shape != self.grid.shape or not np.allclose(
                trace.electric.grid.points, self.grid.points, atol=1e-14, rtol=0
            ):
                raise ValueError("Q inverse trace must lie on its surface chart")
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


@dataclass(frozen=True)
class FresnelEncounter:
    grid: SurfaceGrid
    incident: ElectromagneticField
    reflected: ElectromagneticField
    transmitted: ElectromagneticField
    incident_directions: np.ndarray
    transmitted_directions: np.ndarray
    transverse_projection_error: float


def fresnel_transmission(trace, interface, directions):
    """T: local Fresnel E/H traces using a declared incident eikonal direction.

    The small non-WKB longitudinal part of a diffracted incident field is removed
    for the local plane-wave boundary model, and its relative norm is reported.
    """
    grid = trace.electric.grid
    normal = grid.normals
    ni, nt = interface.minus.material.n, interface.plus.material.n
    if ni == nt:
        # The transparent-interface limit is the identity for any Maxwell field,
        # including diffracted fields with no single local plane-wave polarization.
        reflected = ElectromagneticField(
            ElectricField(
                grid, trace.electric.wavelength0, np.zeros_like(trace.electric.values)
            ),
            np.zeros_like(trace.magnetic),
        )
        return FresnelEncounter(
            grid, trace, reflected, trace, directions, directions, 0.0
        )
    dt = transmitted_direction(directions, normal, ni, nt)
    if np.any(np.abs(dt.imag) > 1e-12):
        raise ValueError(
            "TIR requires an evanescent interface model; the forward eikonal path does not support it"
        )
    dt = dt.real
    electric = trace.electric.values
    longitudinal = np.sum(electric * directions, axis=-1)[..., None] * directions
    error = np.linalg.norm(longitudinal) / max(np.linalg.norm(electric), 1e-300)
    electric = electric - longitudinal
    transmitted = apply_fresnel(
        electric, fresnel_transmission_matrix(directions, dt, normal, ni, nt)
    )
    reflected = apply_fresnel(
        electric, fresnel_reflection_matrix(directions, dt, normal, ni, nt)
    )

    def field(values, direction, index):
        return ElectromagneticField(
            ElectricField(grid, trace.electric.wavelength0, values),
            index / Z0 * np.cross(direction, values),
        )

    return FresnelEncounter(
        grid,
        field(electric, directions, ni),
        field(reflected, reflected_direction(directions, normal), ni),
        field(transmitted, dt, nt),
        directions,
        dt,
        float(error),
    )


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
