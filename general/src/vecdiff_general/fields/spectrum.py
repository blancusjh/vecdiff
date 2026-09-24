"""Modal field representations with explicit spatial validity and material context."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._arrays import frozen_array, positive
from ..fourier.cartesian import frequency_grid
from ..media.medium import Z0, HomogeneousRegion
from ..sampling.grids import PlaneGrid
from .electromagnetic_field import ElectricField, ElectromagneticField


@dataclass(frozen=True, eq=False)
class PlaneWaveExpansion:
    """Discrete modal amplitudes, including any quadrature weights.

    E(r) = sum amplitudes[j] exp(i k[j] dot (r-origin)). The region owns
    the material context. Complex k obeys a bilinear dispersion relation.
    """

    wavevectors: np.ndarray
    amplitudes: np.ndarray
    origin: np.ndarray
    wavelength0: float
    region: HomogeneousRegion

    def __post_init__(self):
        wavelength = positive(self.wavelength0, "vacuum wavelength")

        if not isinstance(self.region, HomogeneousRegion):
            raise TypeError("spectrum requires a HomogeneousRegion")

        k = frozen_array(self.wavevectors, dtype=complex)
        amplitudes = frozen_array(self.amplitudes, dtype=complex)
        origin = frozen_array(self.origin)

        if k.ndim != 2 or k.shape[-1] != 3 or amplitudes.shape != k.shape:
            raise ValueError("wavevectors and amplitudes must have shape (modes, 3)")

        if origin.shape != (3,):
            raise ValueError("phase origin must have shape (3,)")

        wavenumber = 2 * np.pi * self.region.material.n / wavelength
        dispersion = np.sum(k * k, axis=-1) - wavenumber**2
        scale = np.maximum(np.sum(np.abs(k) ** 2, axis=-1), wavenumber**2)

        if np.any(np.abs(dispersion) > 1e-10 * scale):
            raise ValueError("wavevectors violate the region's dispersion relation")

        transverse_error = np.abs(np.sum(k * amplitudes, axis=-1))
        transverse_scale = np.linalg.norm(k, axis=-1) * np.linalg.norm(
            amplitudes, axis=-1
        )

        if np.any(transverse_error > 1e-10 * transverse_scale):
            raise ValueError("electric modal amplitudes are not transverse")

        for name, value in (
            ("wavevectors", k),
            ("amplitudes", amplitudes),
            ("origin", origin),
        ):
            object.__setattr__(self, name, value)

        object.__setattr__(self, "wavelength0", wavelength)

    @property
    def magnetic_amplitudes(self):
        k0 = 2 * np.pi / self.wavelength0

        return np.cross(self.wavevectors, self.amplitudes) / (k0 * Z0)

    def evaluate(self, grid, *, chunk_size=256, backend="direct", tolerance=1e-12):
        """Evaluate SI E/H through the selected nonuniform Fourier backend."""
        from ..fourier.nufft import NUFFT

        coefficients = np.concatenate(
            (self.amplitudes, self.magnetic_amplitudes), axis=-1
        )
        values = NUFFT(
            coefficients,
            self.wavevectors,
            grid.points - self.origin,
            sign=1,
            backend=backend,
            tolerance=tolerance,
            chunk_size=chunk_size,
        )
        electric = ElectricField(grid, self.wavelength0, values[..., :3])

        return ElectromagneticField(electric, values[..., 3:])

    def electric(self, grid, **kwargs):
        return self.evaluate(grid, **kwargs).electric


@dataclass(frozen=True, eq=False)
class PlaneSpectrum:
    """FFT lattice plus its homogeneous modal solution.

    Created by angular_spectrum(). The FFT origin is the central physical grid
    point at index (ny // 2, nx // 2). Transverse wavevectors must follow the
    centered FFT lattice in (v, u) array order. Arbitrary modes belong in
    PlaneWaveExpansion; they cannot be reinterpreted as FFT coefficients.
    """

    grid: PlaneGrid
    expansion: PlaneWaveExpansion
    roundoff_removed_rms: float = 0.0
    evanescent_removed_rms: float = 0.0

    def __post_init__(self):
        if not isinstance(self.grid, PlaneGrid) or not isinstance(
            self.expansion, PlaneWaveExpansion
        ):
            raise TypeError("PlaneSpectrum requires PlaneGrid and PlaneWaveExpansion")

        if len(self.expansion.amplitudes) != np.prod(self.grid.shape):
            raise ValueError("FFT spectrum must contain one mode per grid cell")

        # IFFT2 references the central physical point, not an arbitrary origin.
        ny, nx = self.grid.shape
        center = self.grid.points[ny // 2, nx // 2]

        if not np.array_equal(self.expansion.origin, center):
            raise ValueError(
                "FFT phase origin must equal the central physical grid point"
            )

        # Check every bin, including empty bins, before permitting FFT synthesis.
        fx, fy = frequency_grid(self.grid.coordinates)
        expected = 2 * np.pi * np.stack((fx, fy), axis=-1).reshape(-1, 2)
        wavevectors = self.expansion.wavevectors
        transverse = wavevectors @ self.grid.frame.basis[:, :2]

        # Respect the frame's accepted orthogonality error as well as roundoff.
        basis = self.grid.frame.basis
        frame_error = np.linalg.norm(basis.T @ basis - np.eye(3), ord=2)
        scale = np.linalg.norm(wavevectors, axis=-1)
        tolerance = (64 * np.finfo(float).eps + frame_error) * scale[:, None]

        if np.any(np.abs(transverse - expected) > tolerance):
            raise ValueError(
                "transverse wavevectors must match the centered FFT lattice and ordering"
            )

    def evaluate(self, grid, **kwargs):
        return self.expansion.evaluate(grid, **kwargs)

    def propagate(self, distance, *, model="asm"):
        """Propagate through the homogeneous region using an explicit model."""
        from ..propagation.propagation import propagate_spectrum

        return propagate_spectrum(self, distance, model=model)


@dataclass(frozen=True)
class LocalSpectrum:
    """Bounds compare with the same discrete surface-current Green integral.

    They exclude source quadrature, boundary-model error and roundoff. Electric
    and magnetic absolute vector-error bounds are V/m and A/m respectively.
    """

    expansion: PlaneWaveExpansion
    radius: float
    electric_error_bound: float
    magnetic_error_bound: float

    def __post_init__(self):
        for value in (
            self.radius,
            self.electric_error_bound,
            self.magnetic_error_bound,
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    "radius and absolute error bounds must be finite and nonnegative"
                )

    @property
    def center(self):
        return self.expansion.origin

    def evaluate(self, grid, **kwargs):
        distance = np.linalg.norm(grid.points - self.center, axis=-1)
        tolerance = 32 * np.finfo(float).eps * max(1.0, np.max(np.abs(self.center)))

        if np.any(distance > self.radius + tolerance):
            raise ValueError("observation leaves the local spectrum's validity ball")

        return self.expansion.evaluate(grid, **kwargs)

    def electric(self, grid, **kwargs):
        return self.evaluate(grid, **kwargs).electric


@dataclass(frozen=True)
class HalfSpaceSpectrum:
    """Propagating Weyl quadrature valid beyond the source z envelope."""

    expansion: PlaneWaveExpansion
    boundary_z: float
    direction: int

    def evaluate(self, grid, **kwargs):
        if np.any(self.direction * (grid.points[..., 2] - self.boundary_z) < 0):
            raise ValueError(
                "Weyl representation is valid only beyond the source z envelope"
            )

        return self.expansion.evaluate(grid, **kwargs)
