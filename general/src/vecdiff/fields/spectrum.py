"""Plane-wave representations of the field of one homogeneous region.

``Spectrum`` is what every one of them is: the field of a region, held as
plane waves referred to a plane. ``PlaneSpectrum`` holds them on an FFT
lattice; ``fields.LatticeSpectrum``, ``fields.HankelSpectrum`` and
``fields.BeamSpectrum`` are the other forms the propagation produces.
``PlaneWaveExpansion`` is a set of modes with a phase origin and no plane.
"""

from __future__ import annotations

import importlib.util
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np

from vecdiff._arrays import frozen_array, positive
from vecdiff._backend import bd
from vecdiff.fields.electromagnetic_field import (
    ElectricField,
    ElectromagneticField,
)
from vecdiff.fourier.cartesian import frequency_grid
from vecdiff.media.medium import Z0, HomogeneousRegion
from vecdiff.sampling.grids import PlaneGrid


def summation(method="auto", tolerance=None, chunk_size=None, threads=None):
    """Options of a plane-wave sum, with "auto" resolved and defaults filled.

    method "nufft" runs FINUFFT type-3 transforms to ``tolerance``; "direct"
    forms the exponentials in blocks of ``chunk_size`` modes; "auto" is
    "nufft" when FINUFFT is installed. ``threads`` are FINUFFT's (None:
    VECDIFF_NUFFT_THREADS, else all cores).
    """
    if method == "auto":
        method = "nufft" if importlib.util.find_spec("finufft") else "direct"
    if method not in ("nufft", "direct"):
        raise ValueError("method must be 'auto', 'nufft' or 'direct'")
    if threads is not None and (isinstance(threads, bool) or int(threads) < 0):
        raise ValueError("threads must be None or a nonnegative integer")
    return dict(
        method=method,
        tolerance=1e-12 if tolerance is None else float(tolerance),
        chunk_size=256 if chunk_size is None else chunk_size,
        threads=threads,
    )


class Spectrum(ABC):
    """The field of one homogeneous region, held as plane waves referred to a plane.

    The field fills ``region`` and solves its Maxwell equations; the plane is
    where the representation was formed, ``grid`` its chart there (an FFT
    lattice, a lattice of carried samples or a polar quadrature), and
    ``plane`` that chart's Frame. E and H are given on any grid of points.
    Every Spectrum sums with the same options: ``method`` ("auto", "nufft"
    or "direct"), ``tolerance`` and ``chunk_size`` (None: the precision and
    block size the representation's sums default to) and ``threads``
    (FINUFFT threads; None: VECDIFF_NUFFT_THREADS, else all cores).

    Its attributes ``region`` (a HomogeneousRegion), ``wavelength0`` (the
    vacuum wavelength) and ``grid`` are set by each kind.
    """

    @property
    def plane(self):
        """The Frame of the plane the spectrum is referred to."""
        return self.grid.frame

    @abstractmethod
    def evaluate(
        self, grid, *, method="auto", tolerance=None, chunk_size=None, threads=None
    ):
        """E and H (an ElectromagneticField, SI units) at the points of ``grid``."""

    def electric(
        self, grid, *, method="auto", tolerance=None, chunk_size=None, threads=None
    ):
        """E (an ElectricField) at the points of ``grid``."""
        return self.evaluate(
            grid,
            method=method,
            tolerance=tolerance,
            chunk_size=chunk_size,
            threads=threads,
        ).electric

    def propagate(self, distance, *, magnetic=True):
        """The field on the chart moved ``distance`` along the plane's normal.

        An ElectromagneticField, or the ElectricField alone when ``magnetic``
        is False.
        """
        distance = float(distance)
        if not math.isfinite(distance):
            raise ValueError("distance must be finite")
        grid = self.grid.translated(distance * self.plane.normal)
        return self.evaluate(grid) if magnetic else self.electric(grid)

    @abstractmethod
    def relabeled(self, region):
        """The same field, attributed to another region of the same material."""


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

        wavenumber = 2 * bd.pi * self.region.material.n / wavelength
        dispersion = bd.sum(k * k, axis=-1) - wavenumber**2
        scale = bd.maximum(bd.sum(bd.abs(k) ** 2, axis=-1), wavenumber**2)

        if bd.any(bd.abs(dispersion) > 1e-10 * scale):
            raise ValueError("wavevectors violate the region's dispersion relation")

        # |k| |a| by hypot: the sum of squares underflows for |a| < 1e-154,
        # where a valid but faint mode would otherwise be rejected.
        def norm(vectors):
            size = bd.abs(vectors[..., 0])
            for axis in (1, 2):
                size = bd.hypot(size, bd.abs(vectors[..., axis]))
            return size

        transverse_error = bd.abs(bd.sum(k * amplitudes, axis=-1))
        transverse_scale = norm(k) * norm(amplitudes)

        if bd.any(transverse_error > 1e-10 * transverse_scale):
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
        k0 = 2 * bd.pi / self.wavelength0

        return bd.cross(self.wavevectors, self.amplitudes) / (k0 * Z0)

    def evaluate(
        self, grid, *, chunk_size=256, method="direct", tolerance=1e-12, threads=None
    ):
        """Evaluate SI E/H through the selected nonuniform Fourier summation method.

        method "nufft" runs FINUFFT on ``threads`` threads (None:
        VECDIFF_NUFFT_THREADS, else all cores).
        """
        from vecdiff.fourier.nufft import NUFFT

        coefficients = bd.concatenate(
            (bd.asarray(self.amplitudes), bd.asarray(self.magnetic_amplitudes)), axis=-1
        )
        values = NUFFT(
            coefficients,
            self.wavevectors,
            bd.asarray(grid.points) - bd.asarray(self.origin),
            sign=1,
            method=method,
            tolerance=tolerance,
            chunk_size=chunk_size,
            threads=threads,
        )
        electric = ElectricField(grid, self.wavelength0, values[..., :3])

        return ElectromagneticField(electric, values[..., 3:])

    def electric(self, grid, **kwargs):
        """Evaluate E without constructing or evaluating magnetic amplitudes."""
        from vecdiff.fourier.nufft import NUFFT

        values = NUFFT(
            self.amplitudes,
            self.wavevectors,
            grid.points - self.origin,
            sign=1,
            **kwargs,
        )
        return ElectricField(grid, self.wavelength0, values)


@dataclass(frozen=True, eq=False)
class PlaneSpectrum(Spectrum):
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

        if len(self.expansion.amplitudes) != math.prod(self.grid.shape):
            raise ValueError("FFT spectrum must contain one mode per grid cell")

        # IFFT2 references the central physical point, not an arbitrary origin.
        ny, nx = self.grid.shape
        center = self.grid.points[ny // 2, nx // 2]

        if not bd.array_equal(self.expansion.origin, center):
            raise ValueError(
                "FFT phase origin must equal the central physical grid point"
            )

        # Check every bin, including empty bins, before permitting FFT synthesis.
        fx, fy = frequency_grid(self.grid.coordinates)
        expected = 2 * bd.pi * bd.stack((fx, fy), axis=-1).reshape(-1, 2)
        wavevectors = self.expansion.wavevectors
        transverse = wavevectors @ self.grid.frame.basis[:, :2]

        # Respect the frame's accepted orthogonality error as well as roundoff.
        basis = self.grid.frame.basis
        frame_error = bd.linalg.norm(basis.T @ basis - bd.eye(3), ord=2)
        scale = bd.linalg.norm(wavevectors, axis=-1)
        tolerance = (64 * np.finfo(float).eps + frame_error) * scale[:, None]

        if bd.any(bd.abs(transverse - expected) > tolerance):
            raise ValueError(
                "transverse wavevectors must match the centered FFT lattice and ordering"
            )

    @property
    def region(self):
        return self.expansion.region

    @property
    def wavelength0(self):
        return self.expansion.wavelength0

    def evaluate(
        self, grid, *, method="auto", tolerance=None, chunk_size=None, threads=None
    ):
        options = summation(method, tolerance, chunk_size, threads)
        return self.expansion.evaluate(grid, **options)

    def electric(
        self, grid, *, method="auto", tolerance=None, chunk_size=None, threads=None
    ):
        options = summation(method, tolerance, chunk_size, threads)
        return self.expansion.electric(grid, **options)

    def propagate(self, distance, *, magnetic=True, model="asm"):
        """The field on the lattice moved ``distance`` along its normal, by FFT.

        model "asm" is exact; "fresnel" takes the paraxial phase.
        """
        from vecdiff.propagation.homogeneous import propagate_spectrum

        return propagate_spectrum(self, distance, model=model, magnetic=magnetic)

    def relabeled(self, region):
        return PlaneSpectrum(
            self.grid,
            replace(self.expansion, region=region),
            self.roundoff_removed_rms,
            self.evanescent_removed_rms,
        )
