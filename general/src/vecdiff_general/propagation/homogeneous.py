"""Homogeneous propagation on physically embedded parallel planes."""

import numpy as np

from vecdiff_general.fields import (
    ElectricField,
    ElectromagneticField,
    TransverseElectricField,
)
from vecdiff_general.fields.angular_spectrum import angular_spectrum
from vecdiff_general.fields.spectrum import PlaneSpectrum, PlaneWaveExpansion
from vecdiff_general.fourier.cartesian import IFFT2
from vecdiff_general.fourier.nufft import phase_factor


def propagate_spectrum(
    spectrum: PlaneSpectrum, distance: float, *, model="asm"
) -> ElectromagneticField:
    """Propagate an FFT spectrum by a signed normal distance in metres.

    ASM uses the exact homogeneous dispersion relation. Fresnel uses the
    paraxial phase k - k_perp**2 / (2*k), retaining the input vector amplitudes.
    The latter is an approximate spatial solution; it is not labelled exact
    Maxwell propagation merely because the electric polarization is transverse.
    """
    distance = float(distance)

    if not np.isfinite(distance):
        raise ValueError("distance must be finite")

    if model not in {"asm", "fresnel"}:
        raise ValueError("model must be 'asm' or 'fresnel'")

    expansion = spectrum.expansion
    occupied = np.any(expansion.amplitudes != 0, axis=-1)
    phase = np.zeros(len(occupied), dtype=complex)
    displacement = distance * spectrum.grid.frame.normal

    if model == "asm":
        phase[occupied] = phase_factor(
            expansion.wavevectors[occupied], displacement[None, :]
        )[0]
    else:
        wavevectors = expansion.wavevectors[occupied]

        if np.any(np.abs(wavevectors.imag) > 0):
            raise ValueError("paraxial Fresnel propagation excludes evanescent modes")

        wavenumber = 2 * np.pi * expansion.region.material.n / expansion.wavelength0
        normal_component = wavevectors.real @ spectrum.grid.frame.normal
        transverse_squared = np.sum(wavevectors.real**2, axis=-1) - normal_component**2
        paraxial_component = np.sign(normal_component) * (
            wavenumber - transverse_squared / (2 * wavenumber)
        )
        phase[occupied] = np.exp(1j * paraxial_component * distance)

    grid = spectrum.grid.translated(displacement)
    shape = grid.shape + (3,)

    electric_modes = expansion.amplitudes * phase[:, None]
    magnetic_modes = expansion.magnetic_amplitudes * phase[:, None]

    # IFFT2 divides by N; modal amplitudes already carry that normalization.
    sample_count = np.prod(grid.shape)
    electric = sample_count * IFFT2(electric_modes.reshape(shape))
    magnetic = sample_count * IFFT2(magnetic_modes.reshape(shape))

    return ElectromagneticField(
        ElectricField(grid, expansion.wavelength0, electric), magnetic
    )


def propagate(
    field: ElectricField | TransverseElectricField,
    distance: float,
    region,
    *,
    model="asm",
    direction=1,
    evanescent="retain",
):
    """Complete/decompose a plane field, propagate, and return physical E/H."""
    spectrum = angular_spectrum(
        field, region, direction=direction, evanescent=evanescent
    )

    return propagate_spectrum(spectrum, distance, model=model)


class PlaneTransport:
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
        from vecdiff_general.fourier.nufft import phase_factor

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
