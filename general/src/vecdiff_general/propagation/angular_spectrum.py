"""Homogeneous vector angular spectra and branch-dependent Ez completion.

The material dispersion and longitudinal completion belong to propagation.
Fourier functions only transform arrays and construct frequency coordinates.
"""

import numpy as np

from ..fields.electromagnetic_field import ElectricField, TransverseElectricField
from ..fields.spectrum import PlaneSpectrum, PlaneWaveExpansion
from ..fourier.cartesian import FFT2, frequency_grid
from ..media.medium import HomogeneousRegion
from ..sampling.grids import PlaneGrid


def angular_spectrum(
    field,
    region,
    *,
    direction=1,
    evanescent="retain",
    coefficient_rtol=64 * np.finfo(float).eps,
):
    """Represent a plane field using homogeneous vector propagation modes.

    Two-component inputs prescribe local u/v components. Complete electric
    inputs are validated without projection. The selected material and branch
    determine k_normal and, when needed, the longitudinal electric component.

    'discard' removes modes outside the propagating disk; 'retain' keeps the
    decaying branch. Coefficients below coefficient_rtol times the largest mode
    are zeroed before completion. Set this threshold to zero for strict
    retention. Both removed RMS amplitudes are reported in V/m.
    """

    if not isinstance(field, (ElectricField, TransverseElectricField)):
        raise TypeError("expected complete or transverse electric field")

    if not isinstance(field.grid, PlaneGrid):
        raise TypeError("FFT angular spectrum requires a PlaneGrid")

    if not isinstance(region, HomogeneousRegion):
        raise TypeError("region must be a HomogeneousRegion")

    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")

    if evanescent not in ("retain", "discard"):
        raise ValueError("evanescent must be 'retain' or 'discard'")

    if not np.isfinite(coefficient_rtol) or not 0 <= coefficient_rtol < 1:
        raise ValueError("coefficient_rtol must lie in [0, 1)")

    # The mathematical transform takes arrays and lattice parameters only.
    grid = field.grid
    fx, fy = frequency_grid(grid.coordinates)
    frequency_u = 2 * np.pi * fx
    frequency_v = 2 * np.pi * fy

    # The modal sum uses Fourier-series amplitudes, so divide the DFT by N.
    sample_count = np.prod(grid.shape)
    coefficients = FFT2(field.values) / sample_count

    # The optical dispersion and chosen branch determine the normal frequency.
    wavenumber = 2 * np.pi * region.material.n / field.wavelength0
    transverse_squared = frequency_u**2 + frequency_v**2
    normal_wavenumber = direction * np.sqrt(
        (wavenumber**2 - transverse_squared).astype(complex)
    )

    magnitude = np.linalg.norm(coefficients, axis=-1)
    roundoff = magnitude < coefficient_rtol * np.max(magnitude)
    roundoff_removed = float(np.linalg.norm(coefficients[roundoff]))
    coefficients[roundoff] = 0

    evanescent_removed = 0.0

    if evanescent == "discard":
        discarded = transverse_squared > wavenumber**2
        evanescent_removed = float(np.linalg.norm(coefficients[discarded]))
        coefficients[discarded] = 0

    # Complete only explicitly transverse inputs, in the plane's local basis.
    if isinstance(field, TransverseElectricField):
        occupied = np.any(coefficients != 0, axis=-1)
        grazing = np.abs(normal_wavenumber) < 1e-12 * wavenumber

        if np.any(grazing & occupied):
            raise ValueError(
                "populated grazing modes require a full vector polarization basis"
            )

        transverse_dot = (
            frequency_u * coefficients[..., 0] + frequency_v * coefficients[..., 1]
        )
        longitudinal = np.divide(
            -transverse_dot,
            normal_wavenumber,
            out=np.zeros_like(normal_wavenumber),
            where=~grazing,
        )

        local_coefficients = np.concatenate(
            (coefficients, longitudinal[..., None]), axis=-1
        )
        coefficients = local_coefficients @ grid.frame.basis.T

    # Cartesian modes retain the original physical phase origin.
    wavevectors = (
        frequency_u[..., None] * grid.frame.u
        + frequency_v[..., None] * grid.frame.v
        + normal_wavenumber[..., None] * grid.frame.normal
    )

    expansion = PlaneWaveExpansion(
        wavevectors.reshape(-1, 3),
        coefficients.reshape(-1, 3),
        grid.points[grid.shape[0] // 2, grid.shape[1] // 2],
        field.wavelength0,
        region,
    )

    return PlaneSpectrum(grid, expansion, roundoff_removed, evanescent_removed)
