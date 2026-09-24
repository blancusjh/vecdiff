"""Power, image geometry and Maxwell checks for the ideal pupil model."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from vecdiff_general import (
    CartesianGrid,
    Dielectric,
    Frame,
    HomogeneousRegion,
    TransverseElectricField,
    angular_spectrum,
)
from vecdiff_general.propagation.imaging import image_spectrum, scalar_image


def source(count=65, spacing=0.15e-6, state=(1, 0.4j)):
    axis = (np.arange(count) - count // 2) * spacing
    grid = CartesianGrid(axis, axis)
    x, y = np.meshgrid(axis, axis)
    g = np.exp(-((x - 0.9e-6) ** 2 + y**2) / (1.2e-6) ** 2)
    field = TransverseElectricField(grid, 532e-9, g[..., None] * state)
    air = HomogeneousRegion("air", Dielectric(1))

    return angular_spectrum(field, air, evanescent="discard"), g


@pytest.mark.parametrize("magnification", [-0.5, 1, 2])
def test_imaging_preserves_passed_power_and_maxwell_transversality(magnification):
    initial, _ = source()
    image_medium = HomogeneousRegion("image", Dielectric(1.33))
    na = min(0.55, 0.8 / abs(magnification))
    result = image_spectrum(
        initial,
        image_medium,
        magnification=magnification,
        numerical_aperture=na,
        image_frame=Frame([0, 0, 40e-6]),
    )
    image = result.propagate(0)
    k = initial.expansion.wavevectors
    keep = (
        np.linalg.norm(k[:, :2].real, axis=1)
        <= 2 * np.pi / 532e-9 * abs(magnification) * na
    )
    incoming = initial.expansion.amplitudes
    magnetic = initial.expansion.magnetic_amplitudes
    flux = 0.5 * np.cross(incoming.conj(), magnetic).real[:, 2]
    expected = flux[keep].sum() * initial.grid.weights.sum()
    assert_allclose(image.power(), expected, rtol=2e-12)
    assert_allclose(
        result.grid.spacing, np.array(initial.grid.spacing) * abs(magnification)
    )
    assert_allclose(result.grid.points[..., 2], 40e-6)
    residual = np.sum(
        result.expansion.wavevectors * result.expansion.amplitudes, axis=1
    )
    assert np.max(abs(residual)) < 1e-8
    assert_allclose(result.propagate(3e-6).power(), image.power(), rtol=2e-12)


@pytest.mark.parametrize("count", [64, 65])
def test_negative_magnification_inverts_the_image(count):
    initial, _ = source(count=count)
    image = image_spectrum(
        initial, initial.expansion.region, magnification=-1, numerical_aperture=0.7
    ).propagate(0)
    intensity = image.electric.norm2()
    x = image.grid.points[..., 0]
    centroid = np.sum(x * intensity) / np.sum(intensity)
    assert abs(centroid + 0.9e-6) < 0.015e-6


def test_identity_and_scalar_low_na_limit():
    initial, g = source(spacing=1e-6)
    region = initial.expansion.region
    # Use a broad, slowly varying object to exercise the paraxial limit.
    x, y = np.meshgrid(initial.grid.u, initial.grid.v)
    g = np.exp(-(x**2 + y**2) / (8e-6) ** 2)
    initial = angular_spectrum(
        TransverseElectricField(initial.grid, 532e-9, np.stack((g, 0 * g), axis=-1)),
        region,
    )
    result = image_spectrum(initial, region, magnification=1, numerical_aperture=0.2)
    assert_allclose(
        result.propagate(0).electric.values,
        initial.propagate(0).electric.values,
        atol=2e-9,
    )
    scalar = scalar_image(
        g, initial.grid, 532e-9, region, region, magnification=1, numerical_aperture=0.2
    )
    assert (
        np.linalg.norm(abs(scalar) ** 2 - result.propagate(0).electric.norm2())
        / np.linalg.norm(abs(scalar) ** 2)
        < 0.001
    )


def test_negative_magnification_maps_an_offset_coordinate_window():
    initial, _ = source()
    offset = 3e-6
    shifted_grid = CartesianGrid(initial.grid.u + offset, initial.grid.v)
    shifted_field = TransverseElectricField(
        shifted_grid, 532e-9, initial.propagate(0).electric.values[..., :2]
    )
    shifted = angular_spectrum(shifted_field, initial.expansion.region)
    image = image_spectrum(
        shifted, initial.expansion.region, magnification=-0.5, numerical_aperture=0.7
    ).propagate(0)
    intensity = image.electric.norm2()
    centroid = np.sum(image.grid.points[..., 0] * intensity) / intensity.sum()

    assert abs(centroid + 0.5 * (offset + 0.9e-6)) < 0.015e-6


def test_jones_pupil_attenuation_has_expected_power_cost():
    initial, _ = source()
    kwargs = dict(magnification=-0.5, numerical_aperture=0.6)
    reference = image_spectrum(initial, initial.expansion.region, **kwargs).propagate(0)
    attenuated = image_spectrum(
        initial, initial.expansion.region, pupil=0.3 * np.eye(2), **kwargs
    ).propagate(0)
    assert_allclose(attenuated.power(), 0.09 * reference.power(), rtol=1e-12)


def test_unresolved_pupil_is_rejected():
    initial, _ = source(spacing=1e-6)
    with pytest.raises(ValueError, match="resolve"):
        image_spectrum(
            initial, initial.expansion.region, magnification=1, numerical_aperture=0.8
        )


def test_aplanatic_focus_power_and_transversality():
    from vecdiff_general.propagation.imaging import focus_pupil

    coordinates = (np.arange(129) - 64) * 0.18e-6
    grid = CartesianGrid(coordinates, coordinates)
    x, y = np.meshgrid(coordinates, coordinates)
    amplitude = np.exp(-(x * x + y * y) / (4e-6) ** 2)
    field = TransverseElectricField(
        grid, 532e-9, amplitude[..., None] * np.array([1, 1j])
    )
    air = HomogeneousRegion("air", Dielectric(1))
    medium = HomogeneousRegion("image", Dielectric(1.3))
    focal_length = 12e-6
    na = 0.85
    focused = focus_pupil(
        field, air, medium, focal_length=focal_length, numerical_aperture=na
    )
    support = np.hypot(x, y) <= focal_length * na / 1.3
    from vecdiff_general import Z0

    expected = np.sum(abs(field.values[support]) ** 2) * grid.spacing[0] ** 2 / (2 * Z0)
    assert_allclose(focused.propagate(0).power(), expected, rtol=2e-12)
    residual = np.sum(
        focused.expansion.wavevectors * focused.expansion.amplitudes, axis=-1
    )
    assert np.max(abs(residual)) < 1e-8
    assert_allclose(focused.grid.frame.origin, [0, 0, focal_length])


def test_inverse_pupil_recovers_complex_vector_and_accounts_for_input_power():
    from vecdiff_general.propagation.imaging import apply_pupil

    initial, _ = source()
    initial = image_spectrum(
        initial, initial.expansion.region, magnification=1, numerical_aperture=0.65
    )
    pupil = np.array([[0.7, 0.12j], [0.05, 0.4]])
    prepared = apply_pupil(initial, np.linalg.inv(pupil))
    recovered = apply_pupil(prepared, pupil)
    assert_allclose(
        recovered.expansion.amplitudes, initial.expansion.amplitudes, atol=2e-16
    )
    assert prepared.propagate(0).power() > initial.propagate(0).power()
