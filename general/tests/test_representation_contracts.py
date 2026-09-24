"""Reject inconsistent FFT representations before they can corrupt propagation."""

from dataclasses import replace

import numpy as np
import pytest
import vecdiff_general as optics
from numpy.testing import assert_allclose


def plane_spectrum(direction=1, spacing=532e-9):
    wavelength = 532e-9
    basis = np.array([[0.8, 0, 0.6], [0, 1, 0], [-0.6, 0, 0.8]])
    frame = optics.Frame([2e-6, -3e-6, 5e-6], basis)
    grid = optics.CartesianGrid(
        (np.arange(7) - 1.3) * spacing,
        (np.arange(6) + 0.7) * spacing,
        frame,
    )
    phase = np.exp(2j * np.pi * np.arange(7) / 7)
    values = np.zeros(grid.shape + (2,), dtype=complex)
    values[..., 1] = phase
    field = optics.TransverseElectricField(grid, wavelength, values)
    air = optics.HomogeneousRegion("air", optics.Dielectric(1))

    return optics.angular_spectrum(field, air, direction=direction)


def test_repeated_plane_waves_are_not_an_fft_lattice():
    spectrum = plane_spectrum()
    count = np.prod(spectrum.grid.shape)
    wave = spectrum.expansion
    center = (spectrum.grid.shape[0] // 2) * spectrum.grid.shape[1]
    center += spectrum.grid.shape[1] // 2
    repeated = replace(
        wave,
        wavevectors=np.repeat(wave.wavevectors[center : center + 1], count, axis=0),
        amplitudes=np.zeros((count, 3)),
    )

    with pytest.raises(ValueError, match="lattice and ordering"):
        optics.PlaneSpectrum(spectrum.grid, repeated)


def test_fft_constructor_rejects_permuted_modes_without_restricting_general_expansion():
    spectrum = plane_spectrum()
    wave = spectrum.expansion
    permuted = replace(
        wave,
        wavevectors=wave.wavevectors[::-1],
        amplitudes=wave.amplitudes[::-1],
    )

    # Reordering a general modal sum does not change its physical field.
    assert_allclose(
        permuted.evaluate(spectrum.grid).electric.values,
        wave.evaluate(spectrum.grid).electric.values,
        atol=2e-14,
    )

    with pytest.raises(ValueError, match="lattice and ordering"):
        optics.PlaneSpectrum(spectrum.grid, permuted)


def test_fft_constructor_rejects_reanchored_origin_even_if_modal_field_is_preserved():
    spectrum = plane_spectrum()
    wave = spectrum.expansion
    displacement = np.array([0.13, -0.07, 0.11]) * wave.wavelength0
    phase = np.exp(1j * (wave.wavevectors @ displacement))
    reanchored = replace(
        wave,
        origin=wave.origin + displacement,
        amplitudes=wave.amplitudes * phase[:, None],
    )

    assert_allclose(
        reanchored.evaluate(spectrum.grid).electric.values,
        wave.evaluate(spectrum.grid).electric.values,
        atol=2e-14,
    )

    with pytest.raises(ValueError, match="phase origin"):
        optics.PlaneSpectrum(spectrum.grid, reanchored)


def test_fft_validation_respects_existing_frame_orthogonality_tolerance():
    basis = np.eye(3)
    basis[2, 0] = 1e-13
    grid = optics.CartesianGrid(
        np.arange(4) * 1e-6, np.arange(3) * 1e-6, optics.Frame(basis=basis)
    )
    values = np.zeros(grid.shape + (2,), dtype=complex)
    values[..., 1] = 1
    field = optics.TransverseElectricField(grid, 532e-9, values)
    air = optics.HomogeneousRegion("air", optics.Dielectric(1))

    spectrum = optics.angular_spectrum(field, air)
    propagated = spectrum.propagate(0)
    direct = spectrum.evaluate(grid)

    assert_allclose(propagated.electric.values, direct.electric.values, atol=3e-12)


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("spacing", [532e-9, 532e-10])
def test_valid_fft_representation_preserves_direct_electric_and_magnetic_fields(
    direction, spacing
):
    spectrum = plane_spectrum(direction, spacing)

    for distance in (0, direction * 0.2 * spectrum.expansion.wavelength0):
        propagated = spectrum.propagate(distance)
        direct = spectrum.evaluate(propagated.grid)

        assert_allclose(propagated.electric.values, direct.electric.values, atol=3e-14)
        assert_allclose(propagated.magnetic, direct.magnetic, atol=3e-16)
