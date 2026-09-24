"""Independent quadratures and physical normalization of focal propagators."""

import numpy as np
import pytest
import vecdiff_stigmatic as optics
from numpy.testing import assert_allclose


@pytest.mark.parametrize("backend", ["hankel", "nufft"])
def test_focus_backends_preserve_all_complex_components(backend):
    if backend == "nufft":
        pytest.importorskip("finufft")
    system = optics.StigmaticSystem(
        -0.01, 0.006, optics.Dielectric(1), optics.Dielectric(1.5)
    )
    rays = system.rays(0.0015, radial_order=20, azimuthal_order=64)
    source = optics.cartesian_source(rays, 532e-9, polarization=(1, 0.3j))
    focus = optics.DebyeFocus(optics.transfer(rays, source))
    grid = optics.PointGrid(
        focus.origin
        + 532e-9 * np.array([[0, 0, 0], [0.4, -0.3, 0.2], [-0.8, 0.2, -0.5]])
    )
    reference = focus.evaluate(grid)
    alternative = focus.evaluate(grid, backend=backend)
    assert_allclose(
        alternative.electric.values, reference.electric.values, rtol=2e-10, atol=1e-9
    )
    assert_allclose(alternative.magnetic, reference.magnetic, rtol=2e-10, atol=1e-11)


def test_fresnel_focus_has_gaussian_width_physical_coordinates_and_parseval_power():
    coordinate = np.linspace(-150e-6, 150e-6, 128, endpoint=False)
    grid = optics.PlaneGrid(coordinate, coordinate)
    waist = 30e-6
    radius_squared = np.sum(grid.points[..., :2] ** 2, axis=-1)
    amplitude = np.exp(-radius_squared / waist**2)
    components = np.stack((amplitude, 0.3j * amplitude), axis=-1)
    pupil = optics.TransverseElectricField(grid, 532e-9, components)
    medium = optics.Dielectric(1.5)
    focal_distance = 3e-3
    result = optics.fresnel_focus(pupil, focal_distance, medium)
    expected_spacing = (
        pupil.wavelength0 / medium.n * focal_distance / (128 * grid.spacing[0])
    )
    assert_allclose(result.grid.spacing, expected_spacing)
    assert_allclose(result.grid.frame.origin, [0, 0, focal_distance])
    incident_power = (
        medium.n
        / (2 * optics.Z0)
        * np.sum(np.abs(components) ** 2 * grid.weights[..., None])
    )
    assert_allclose(result.power(), incident_power, rtol=2e-13)
    expected_waist = pupil.wavelength0 * focal_distance / (np.pi * medium.n * waist)
    radial = np.sum(
        (result.grid.points - result.grid.frame.origin)[..., :2] ** 2, axis=-1
    )
    expected_intensity = np.exp(-2 * radial / expected_waist**2)
    actual_intensity = np.abs(result.electric.Ex) ** 2
    assert_allclose(
        actual_intensity / actual_intensity.max(), expected_intensity, atol=2e-11
    )
