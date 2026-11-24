"""Homogeneous propagation on physically embedded parallel planes."""

import math

from vecdiff._backend import bd
from vecdiff.fields import (
    ElectricField,
    ElectromagneticField,
    TransverseElectricField,
)
from vecdiff.fields.angular_spectrum import angular_spectrum
from vecdiff.fields.spectrum import PlaneSpectrum, PlaneWaveExpansion, _occupied
from vecdiff.fourier.cartesian import _centered
from vecdiff.fourier.nufft import phase_factor
from vecdiff.media.medium import Z0


def propagate_spectrum(
    spectrum: PlaneSpectrum, distance: float, *, model="asm", magnetic=True
) -> ElectromagneticField:
    """Propagate an FFT spectrum by a signed normal distance.

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
    occupied = _occupied(expansion)
    phase = bd.zeros(len(occupied), dtype=complex)
    displacement = distance * spectrum.grid.frame.normal

    if model == "asm":
        phase[occupied] = phase_factor(
            expansion.wavevectors[occupied], displacement[None, :]
        )[0]
    else:
        from vecdiff.media.absorption import require_transparent

        require_transparent(
            expansion.region,
            "the paraxial (Fresnel) model",
            "its phase k - k_perp^2 / (2 k) is written for the propagating waves "
            "of a real wavenumber; model='asm' is exact there",
        )
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

    # IFFT2 divides by N; modal amplitudes already carry that normalization.
    # The modes are finite (validated) and so is each phase factor; the
    # fields are checked once, as computed samples.
    sample_count = math.prod(grid.shape)
    electric_modes = expansion.amplitudes * phase[:, None]
    electric = _centered(
        electric_modes.reshape(shape), inverse=True, scale=sample_count, overwrite=True
    )
    del electric_modes
    electric = ElectricField._computed(grid, expansion.wavelength0, electric)

    if not magnetic:
        return electric

    def form_magnetic():
        # cross(k, a) phase, one component at a time in place; 1 / (k0 Z0)
        # rides on the transform's scale.
        k, a = expansion.wavevectors, expansion.amplitudes
        modes = bd.empty_like(a)
        scratch = bd.empty(len(a), complex)
        for i, (p, q) in enumerate(((1, 2), (2, 0), (0, 1))):
            component = bd.multiply(k[:, p], a[:, q], out=modes[:, i])
            component -= bd.multiply(k[:, q], a[:, p], out=scratch)
            component *= phase
        del scratch
        k0 = 2 * bd.pi / expansion.wavelength0
        return _centered(
            modes.reshape(shape),
            inverse=True,
            scale=sample_count / (k0 * Z0),
            overwrite=True,
        )

    return ElectromagneticField._computed(electric, form_magnetic)


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

    def __repr__(self):
        return "PlaneTransport(exact homogeneous dispersion on an FFT lattice)"

    def __call__(self, spectrum, distance):
        return propagate_spectrum(spectrum, distance, model="asm")

    def spectrum(self, spectrum, distance):
        """Keep the modal representation between consecutive R/Q operators."""
        if not bd.isfinite(distance):
            raise ValueError("distance must be finite")
        displacement = distance * spectrum.grid.frame.normal
        grid = spectrum.grid.translated(displacement)
        old = spectrum.expansion
        from vecdiff.fourier.nufft import phase_factor

        amplitudes = bd.zeros_like(old.amplitudes)
        occupied = bd.any(old.amplitudes != 0, axis=-1)
        amplitudes[occupied] = (
            old.amplitudes[occupied]
            * phase_factor(old.wavevectors[occupied], displacement[None, :])[0, :, None]
        )
        # The lattice and its modes are the spectrum's, translated with its
        # grid; a phase keeps each amplitude transverse, so only finiteness
        # needs a check.
        if not bd.isfinite(amplitudes).all():
            raise ValueError("arrays must contain finite values")
        return PlaneSpectrum._trusted(
            grid,
            PlaneWaveExpansion._trusted(
                old.wavevectors, amplitudes, grid.center, old.wavelength0, old.region
            ),
            0.0,
            0.0,
        )
