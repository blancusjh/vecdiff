import numpy as np
from numpy.testing import assert_allclose
from vecdiff_general import (
    Dielectric,
    HomogeneousRegion,
    PlaneGrid,
    PointGrid,
    TransverseElectricField,
    angular_spectrum,
)
from vecdiff_general.comparisons.curved_interface import scatter_surface
from vecdiff_general.interfaces.dielectric_interface import DielectricInterface
from vecdiff_general.sampling.surface_sampling import sample_surface
from vecdiff_general.surfaces.asphere import EvenAsphere
from vecdiff_general.surfaces.axisymmetric import SphericalCap
from vecdiff_general.surfaces.surface import Plane


def test_spherical_cap_normal_and_area_jacobian():
    surface = SphericalCap(0.01)
    xy = np.array([[0.001, 0.002], [-0.002, 0.001]])
    points = surface.points(xy)
    center = np.array([0, 0, 0.01])
    assert_allclose(np.linalg.norm(points - center, axis=-1), 0.01, rtol=1e-14)
    assert_allclose(surface.normals(xy), (center - points) / 0.01, atol=1e-14)
    asphere = EvenAsphere(100.0)
    assert_allclose(asphere.points(xy), points, atol=1e-16)
    assert_allclose(asphere.normals(xy), surface.normals(xy), atol=1e-14)


def make_radiation(order):
    wavelength = 532e-9
    first = HomogeneousRegion("before", Dielectric(1))
    second = HomogeneousRegion("after", Dielectric(1))
    axis = (np.arange(48) - 24) * 0.31 * wavelength
    grid = PlaneGrid(axis, axis)
    x, y = np.meshgrid(axis, axis)
    envelope = np.exp(-(x * x + y * y) / (1.2 * wavelength) ** 2)
    values = np.stack((envelope, 0.3j * envelope), axis=-1)
    spectrum = angular_spectrum(
        TransverseElectricField(grid, wavelength, values), first, evanescent="discard"
    )
    surface = Plane()
    quadrature = sample_surface(
        surface, 5 * wavelength, radial_order=order, azimuthal_order=2 * order
    )
    interaction = scatter_surface(
        spectrum, DielectricInterface(surface, first, second), quadrature
    )
    return spectrum, interaction


def test_green_radiation_convergence_and_equal_index_control():
    wavelength = 532e-9
    points = PointGrid(np.array([[0.2, 0.1, 2], [-0.5, 0.3, 3]]) * wavelength)
    source, coarse = make_radiation(32)
    _, fine = make_radiation(48)
    coarse_field = coarse.transmitted.evaluate(points)
    fine_field = fine.transmitted.evaluate(points)
    reference = source.evaluate(points)
    for low, high in [
        (coarse_field.electric.values, fine_field.electric.values),
        (coarse_field.magnetic, fine_field.magnetic),
    ]:
        assert np.linalg.norm(low - high) / np.linalg.norm(high) < 1e-6
    assert (
        np.linalg.norm(fine_field.electric.values - reference.electric.values)
        / np.linalg.norm(reference.electric.values)
        < 5e-4
    )
    assert np.linalg.norm(fine.reflected_trace.electric.values) < 1e-12


def test_local_spectrum_bound_against_full_green_integral():
    wavelength = 532e-9
    _, interaction = make_radiation(32)
    center = np.array([0, 0, 100 * wavelength])
    radius = 0.1 * wavelength
    local = interaction.transmitted.local_spectrum(center, radius)
    points = PointGrid(
        center + radius * np.array([[0, 0, 0], [0.7, 0.2, 0.1], [0, 0, -1]])
    )
    exact = interaction.transmitted.evaluate(points)
    approximate = local.evaluate(points)
    electric_error = np.max(
        np.linalg.norm(exact.electric.values - approximate.electric.values, axis=-1)
    )
    magnetic_error = np.max(
        np.linalg.norm(exact.magnetic - approximate.magnetic, axis=-1)
    )
    assert electric_error <= local.electric_error_bound
    assert magnetic_error <= local.magnetic_error_bound


def test_reconstructed_boundary_residual_distinguishes_local_identity():
    """Converged radiation need not solve the curved dielectric boundary problem."""
    from vecdiff_general import PlaneWaveExpansion
    from vecdiff_general.observables import coherent_sum
    from vecdiff_general.observables.diagnostics import interface_residuals
    from vecdiff_general.sampling.near_surface import near_surface_grid

    wavelength = 532e-9
    first = HomogeneousRegion("before", Dielectric(1))
    second = HomogeneousRegion("after", Dielectric(1.5))
    source = PlaneWaveExpansion(
        [[0, 0, 2 * np.pi / wavelength]], [[1, 0, 0]], [0, 0, 0], wavelength, first
    )
    surface = SphericalCap(10 * wavelength)
    interface = DielectricInterface(surface, first, second)
    center = np.array([2 * wavelength, 0])
    point = surface.points(center)
    normal = surface.normals(center)
    offset = 0.002 * wavelength
    traces = []
    for order in (112, 144):
        quadrature = near_surface_grid(
            surface,
            center,
            5 * wavelength,
            offset,
            radial_order=order,
            azimuthal_order=128,
        )
        interaction = scatter_surface(source, interface, quadrature)
        local = interface_residuals(
            coherent_sum([interaction.incident_trace, interaction.reflected_trace]),
            interaction.transmitted_trace,
            quadrature.normals,
            first.material,
            second.material,
            reference_electric=1,
            reference_index=1,
        )
        assert max(local.values()) < 1e-12
        minus_grid = PointGrid([point - offset * normal])
        plus_grid = PointGrid([point + offset * normal])
        traces.append(
            (
                coherent_sum(
                    [
                        source.evaluate(minus_grid),
                        interaction.reflected.evaluate(minus_grid),
                    ]
                ),
                interaction.transmitted.evaluate(plus_grid),
            )
        )
    for side in (0, 1):
        assert_allclose(
            traces[0][side].electric.values, traces[1][side].electric.values, atol=1e-9
        )
        assert_allclose(traces[0][side].magnetic, traces[1][side].magnetic, atol=1e-11)
    residuals = interface_residuals(
        *traces[-1],
        normal,
        first.material,
        second.material,
        reference_electric=1,
        reference_index=1,
    )
    # This is deliberately a nonzero-offset diagnostic, not a claimed boundary limit.
    assert 1e-3 < max(residuals.values()) < 0.2


def test_propagating_weyl_spectrum_converges_to_far_green_control():
    """The omitted evanescent contribution is small only in this selected case."""
    import pytest

    pytest.importorskip("finufft")
    wavelength = 532e-9
    _, interaction = make_radiation(32)
    points = PointGrid(wavelength * np.array([[0.2, 0.1, 8], [-0.3, 0.1, 12]]))
    reference = interaction.transmitted.evaluate(points)
    values = []
    for order in (48, 64):
        spectrum = interaction.transmitted.angular_spectrum(
            polar_order=order, azimuthal_order=2 * order, backend="nufft"
        )
        values.append(spectrum.evaluate(points))
    for component in ("electric", "magnetic"):
        coarse = (
            values[0].electric.values if component == "electric" else values[0].magnetic
        )
        fine = (
            values[1].electric.values if component == "electric" else values[1].magnetic
        )
        exact = (
            reference.electric.values if component == "electric" else reference.magnetic
        )
        assert np.linalg.norm(coarse - fine) / np.linalg.norm(fine) < 1e-9
        assert np.linalg.norm(fine - exact) / np.linalg.norm(exact) < 1e-6
    with pytest.raises(ValueError):
        spectrum.evaluate(PointGrid([[0, 0, -wavelength]]))
