"""Homogeneous propagation on physically embedded parallel planes."""

import math

from vecdiff_general._backend import bd
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
    spectrum: PlaneSpectrum, distance: float, *, model="asm", magnetic=True
) -> ElectromagneticField:
    """Propagate an FFT spectrum by a signed normal distance in metres.

    magnetic=False returns only the ElectricField and never forms the
    magnetic modes or their transform (half the memory and work for E alone).

    ASM uses the exact homogeneous dispersion relation. Fresnel uses the
    paraxial phase k - k_perp**2 / (2*k), retaining the input vector amplitudes.
    The latter is an approximate spatial solution; it is not labelled exact
    Maxwell propagation merely because the electric polarization is transverse.
    """
    distance = float(distance)

    if not bd.isfinite(distance):
        raise ValueError("distance must be finite")

    if model not in {"asm", "fresnel"}:
        raise ValueError("model must be 'asm' or 'fresnel'")

    expansion = spectrum.expansion
    occupied = bd.any(expansion.amplitudes != 0, axis=-1)
    phase = bd.zeros(len(occupied), dtype=complex)
    displacement = distance * spectrum.grid.frame.normal

    if model == "asm":
        phase[occupied] = phase_factor(
            expansion.wavevectors[occupied], displacement[None, :]
        )[0]
    else:
        wavevectors = expansion.wavevectors[occupied]

        if bd.any(bd.abs(wavevectors.imag) > 0):
            raise ValueError("paraxial Fresnel propagation excludes evanescent modes")

        wavenumber = 2 * bd.pi * expansion.region.material.n / expansion.wavelength0
        normal_component = wavevectors.real @ spectrum.grid.frame.normal
        transverse_squared = bd.sum(wavevectors.real**2, axis=-1) - normal_component**2
        paraxial_component = bd.sign(normal_component) * (
            wavenumber - transverse_squared / (2 * wavenumber)
        )
        phase[occupied] = bd.exp(1j * paraxial_component * distance)

    grid = spectrum.grid.translated(displacement)
    shape = grid.shape + (3,)

    electric_modes = expansion.amplitudes * phase[:, None]

    # IFFT2 divides by N; modal amplitudes already carry that normalization.
    sample_count = math.prod(grid.shape)
    electric = sample_count * IFFT2(electric_modes.reshape(shape))
    electric = ElectricField(grid, expansion.wavelength0, electric)

    if not magnetic:
        return electric

    magnetic_modes = expansion.magnetic_amplitudes * phase[:, None]

    return ElectromagneticField(
        electric, sample_count * IFFT2(magnetic_modes.reshape(shape))
    )


def propagate(
    field: ElectricField | TransverseElectricField,
    distance: float,
    region,
    *,
    model="asm",
    direction=1,
    evanescent="retain",
    magnetic=True,
):
    """Complete/decompose a plane field, propagate, and return physical E/H.

    magnetic=False returns E alone (an ElectricField) without forming H.
    """
    spectrum = angular_spectrum(
        field, region, direction=direction, evanescent=evanescent
    )

    return propagate_spectrum(spectrum, distance, model=model, magnetic=magnetic)


class PlaneTransport:
    """R_xi: exact homogeneous dispersion, on a parallel FFT lattice."""

    def __call__(self, spectrum, distance):
        return propagate_spectrum(spectrum, distance, model="asm")

    def spectrum(self, spectrum, distance):
        """Keep the modal representation between consecutive R/Q operators."""
        if not bd.isfinite(distance):
            raise ValueError("distance must be finite")
        displacement = distance * spectrum.grid.frame.normal
        grid = spectrum.grid.translated(displacement)
        old = spectrum.expansion
        from vecdiff_general.fourier.nufft import phase_factor

        amplitudes = bd.zeros_like(old.amplitudes)
        occupied = bd.any(old.amplitudes != 0, axis=-1)
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
