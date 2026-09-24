import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import jv
from vecdiff_stigmatic import (
    DebyeFocus,
    Dielectric,
    PointGrid,
    StigmaticSystem,
    cartesian_source,
    transfer,
)

WAVELENGTH = 532e-9


def system(n0=1.0, ni=1.5):
    return StigmaticSystem(-0.010, 0.006, Dielectric(n0), Dielectric(ni))


@pytest.mark.parametrize("n0,ni", [(1.0, 1.5), (1.5, 1.0)])
def test_constant_optical_path_snell_and_geometric_power(n0, ni):
    optical_system = system(n0, ni)
    rays = optical_system.rays(0.001, radial_order=20, azimuthal_order=40)
    assert_allclose(rays.optical_path, optical_system.surface.optical_path, rtol=2e-14)
    normal = rays.surface_grid.normals
    k_tangent_jump = np.cross(
        normal, n0 * rays.incident_direction - ni * rays.transmitted_direction
    )
    assert_allclose(k_tangent_jump, 0, atol=2e-15)
    result = transfer(
        rays, cartesian_source(rays, WAVELENGTH, polarization=(1.0, 0.3j))
    )
    budget = result.power_budget()
    assert abs(budget["balance_W"]) / budget["incident_W"] < 3e-14
    assert_allclose(
        result.entrance.power(), result.incident_surface.power(), rtol=3e-14
    )
    assert_allclose(result.exit.power(), result.transmitted_surface.power(), rtol=3e-14)
    assert_allclose(
        np.sum(result.exit.electric.values * rays.transmitted_direction, axis=-1),
        0,
        atol=1e-14,
    )


def test_local_fresnel_traces_satisfy_all_four_conditions():
    rays = system().rays(0.002, radial_order=24, azimuthal_order=48)
    result = transfer(
        rays, cartesian_source(rays, WAVELENGTH, polarization=(1.0, 0.4j))
    )
    e1 = (
        result.incident_surface.electric.values
        + result.reflected_surface.electric.values
    )
    h1 = result.incident_surface.magnetic + result.reflected_surface.magnetic
    e2, h2 = (
        result.transmitted_surface.electric.values,
        result.transmitted_surface.magnetic,
    )
    normal = rays.surface_grid.normals
    assert_allclose(np.cross(normal, e2 - e1), 0, atol=1e-14)
    assert_allclose(np.cross(normal, h2 - h1), 0, atol=1e-16)
    assert_allclose(np.sum(normal * (1.5**2 * e2 - e1), axis=-1), 0, atol=1e-14)
    assert_allclose(np.sum(normal * (h2 - h1), axis=-1), 0, atol=1e-16)


def test_debye_against_independent_bessel_reduction_including_longitudinal_sign():
    rays = system().rays(0.002, radial_order=48, azimuthal_order=96)
    result = transfer(rays, cartesian_source(rays, WAVELENGTH))
    focus = DebyeFocus(result)
    x = np.linspace(-1.5, 1.5, 13) * WAVELENGTH
    points = np.column_stack((x, np.zeros_like(x), np.full_like(x, system().image_z)))
    actual = focus.electric(PointGrid(points)).values
    # Analytic phi integration: cos(phi) exp(-i a cos(phi)) -> -2 pi i J1(a).
    k = 2 * np.pi * 1.5 / WAVELENGTH
    sin_image = -rays.transmitted_direction[:, 0, 0]
    cos_image = rays.transmitted_direction[:, 0, 2]
    cos_object = rays.incident_direction[:, 0, 2]
    geometric = rays.geometric_amplitude[:, 0]
    p = geometric * result.tp[:, 0] / cos_object
    s = geometric * result.ts[:, 0]
    weights = rays.image_solid_angle_weights[:, 0] * 96 / (2 * np.pi)
    argument = k * x[:, None] * sin_image[None, :]
    prefactor = (
        -1j * k * system().image_z * np.exp(1j * k * system().image_z) / (2 * np.pi)
    )
    ex = prefactor * np.sum(
        np.pi
        * (
            (p * cos_image + s) * jv(0, argument)
            - (p * cos_image - s) * jv(2, argument)
        )
        * weights,
        axis=1,
    )
    ez = prefactor * np.sum(
        -2j * np.pi * p * sin_image * jv(1, argument) * weights, axis=1
    )
    assert_allclose(actual[:, 0], ex, rtol=3e-12, atol=1e-10)
    assert_allclose(actual[:, 2], ez, rtol=3e-12, atol=1e-10)
    assert np.max(abs(actual[:, 1])) < 1e-10


@pytest.mark.parametrize("polarization", ["radial", "azimuthal", "circular"])
def test_polarization_symmetry_at_focus(polarization):
    rays = system().rays(0.002, radial_order=24, azimuthal_order=64)
    if polarization == "radial":
        components = np.stack((np.cos(rays.azimuth), np.sin(rays.azimuth)), axis=-1)
    elif polarization == "azimuthal":
        components = np.stack((-np.sin(rays.azimuth), np.cos(rays.azimuth)), axis=-1)
    else:
        components = (1.0, 1j)
    result = transfer(rays, cartesian_source(rays, WAVELENGTH, polarization=components))
    field = DebyeFocus(result).electric(PointGrid([[0, 0, system().image_z]])).values[0]
    if polarization == "radial":
        assert abs(field[2]) > 100
        assert np.linalg.norm(field[:2]) < 1e-10
    elif polarization == "azimuthal":
        assert np.linalg.norm(field) < 1e-10
    else:
        assert_allclose(field[1], 1j * field[0], rtol=1e-13)
        assert abs(field[2]) < 1e-10


def test_azimuthal_source_has_no_longitudinal_electric_component_off_axis():
    rays = system().rays(0.002, radial_order=24, azimuthal_order=64)
    components = np.stack((-np.sin(rays.azimuth), np.cos(rays.azimuth)), axis=-1)
    source = cartesian_source(rays, WAVELENGTH, polarization=components)
    result = transfer(rays, source)
    points = PointGrid(
        [
            [0.5 * WAVELENGTH, 0.3 * WAVELENGTH, 0.006],
            [WAVELENGTH, 0, 0.006 + WAVELENGTH],
        ]
    )
    assert np.max(abs(DebyeFocus(result).electric(points).Ez)) < 1e-10


def test_quadrature_convergence_of_full_complex_field():
    points = PointGrid(
        [
            [0.4 * WAVELENGTH, 0.2 * WAVELENGTH, 0.006],
            [WAVELENGTH, 0, 0.006 + 0.3 * WAVELENGTH],
        ]
    )
    evaluations = []
    for order in (16, 32):
        rays = system().rays(0.002, radial_order=order, azimuthal_order=4 * order)
        result = transfer(
            rays, cartesian_source(rays, WAVELENGTH, polarization=(1, 0.4j))
        )
        evaluations.append(DebyeFocus(result).evaluate(points))
    for a, b in (
        (evaluations[0].electric.values, evaluations[1].electric.values),
        (evaluations[0].magnetic, evaluations[1].magnetic),
    ):
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < 1e-7


def test_unsupported_conjugates_and_apertures_fail_explicitly():
    with pytest.raises(ValueError, match="object_z"):
        StigmaticSystem(0.010, 0.006, Dielectric(1), Dielectric(1.5))
    with pytest.raises(ValueError, match="branch"):
        system().rays(0.020)
