"""Independent Maxwell checks; prescribed Fresnel data is never the reference."""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose
from vecdiff_general.propagation.references import (
    MieSphere,
    boundary_meridian,
)
from vecdiff_general.propagation.references.diagnostics import (
    boundary_residuals,
    closed_flux_residual,
    maxwell_residuals,
)
from vecdiff_general.propagation.references.mfs import AxisymmetricMFS


def problem(n=1.5, wavelength=1.0, radius=0.5, polarization=(1, 0, 0)):
    air = o.HomogeneousRegion("air", o.Dielectric(1))
    glass = o.HomogeneousRegion("glass", o.Dielectric(n))
    source = o.PlaneWaveExpansion(
        [[0, 0, 2 * np.pi / wavelength]], [polarization], [0, 0, 0], wavelength, air
    )
    return source, o.SphereVolume([0, 0, 0], radius), glass


@pytest.mark.parametrize("polarization", [(1, 0, 0), (0, 1, 0), (1, 1j, 0)])
def test_mie_boundary_flux_and_multipole_convergence(polarization):
    source, sphere, glass = problem(polarization=polarization)
    solution = MieSphere(source, sphere, glass)
    boundary = boundary_meridian(sphere, 19).surface(7, phase=0.17)
    assert max(boundary_residuals(solution, boundary).values()) < 2e-9
    outer = boundary_meridian(o.SphereVolume([0, 0, 0], 1.0), 48).surface(4)
    assert abs(closed_flux_residual(solution, outer)) < 1e-10
    q = solution.efficiencies()
    assert_allclose(q["scattering"], q["extinction"], rtol=1e-13)
    grid = o.PointGrid([[0.1, 0.2, 0.15], [0.1, 0.2, 0.8]])
    finer = MieSphere(source, sphere, glass, order=solution.order + 4)
    assert_allclose(
        solution.evaluate(grid).electric.values,
        finer.evaluate(grid).electric.values,
        atol=1e-10,
    )


def test_mie_equal_index_rotated_translated_and_si_scaling():
    air = o.HomogeneousRegion("air", o.Dielectric(1.2))
    wave = np.array([0.6, 0, 0.8]) * 2 * np.pi * 1.2 / 532e-9
    source = o.PlaneWaveExpansion(
        [wave], [[0.8, 0.3j, -0.6]], [1e-6, 0, -1e-6], 532e-9, air
    )
    sphere = o.SphereVolume([2e-6, 0, 1e-6], 600e-9)
    grid = o.PointGrid(sphere.center + [[0, 0, 0], [1e-7, 2e-7, -1e-7], [0, 0, 8e-7]])
    actual = MieSphere(source, sphere, air).evaluate(grid)
    expected = source.evaluate(grid)
    assert_allclose(actual.electric.values, expected.electric.values, atol=1e-11)
    assert_allclose(actual.magnetic, expected.magnetic, atol=1e-13)
    normalized = problem(radius=0.5)
    physical = problem(radius=0.5 * 532e-9, wavelength=532e-9)
    p = np.array([[0, 0, 0.7], [0.1, 0, 0.2]])
    assert_allclose(
        MieSphere(*normalized).evaluate(o.PointGrid(p)).electric.values,
        MieSphere(*physical).evaluate(o.PointGrid(p * 532e-9)).electric.values,
        atol=1e-11,
    )


def test_mfs_converges_to_independent_mie_on_both_sides():
    source, sphere, glass = problem(polarization=(0.7, 0.2j, 0))
    exact = MieSphere(source, sphere, glass)
    boundary = boundary_meridian(sphere, 45).surface(3, phase=0.27)
    probes = o.PointGrid([[0, 0, 0], [0.1, 0.2, 0.1], [0, 0, 0.8], [0.2, -0.3, 0.7]])
    errors, jumps = [], []
    for count in (16, 32):
        solution = AxisymmetricMFS(
            source, sphere, glass, source_count=count, source_offset=0.25, azimuths=128
        )
        errors.append(
            np.linalg.norm(
                solution.evaluate(probes).electric.values
                - exact.evaluate(probes).electric.values
            )
        )
        jumps.append(max(boundary_residuals(solution, boundary).values()))
    assert errors[1] < errors[0] / 50 and errors[1] < 2e-6
    assert jumps[1] < jumps[0] / 50 and jumps[1] < 2e-5
    outer = boundary_meridian(o.SphereVolume([0, 0, 0], 1), 40).surface(4)
    assert abs(closed_flux_residual(solution, outer)) < 2e-6


@pytest.mark.parametrize("method", [MieSphere, AxisymmetricMFS])
def test_maxwell_equations_and_fourth_order_differencing(method):
    source, sphere, glass = problem()
    options = (
        {}
        if method is MieSphere
        else dict(source_count=32, source_offset=0.25, azimuths=128)
    )
    solution = method(source, sphere, glass, **options)
    for points, index in (([[0.1, 0.15, 0.12]], 1.5), ([[0.2, 0.1, 0.8]], 1)):
        coarse = maxwell_residuals(
            solution.evaluate, points, wavelength0=1, index=index, step=0.015 / index
        )
        fine = maxwell_residuals(
            solution.evaluate, points, wavelength0=1, index=index, step=0.0075 / index
        )
        assert max(fine.values()) < 2e-6
        assert fine["faraday"] < coarse["faraday"] / 10
        assert fine["vector_wave_E"] < coarse["vector_wave_E"] / 10


def test_lens_closed_geometry_and_no_contrast_control():
    source, _, _ = problem()
    volume = o.VolumeBetweenSurfaces(
        o.Plane(), o.SphericalCap(-4, o.Frame([0, 0, 0.6])), 1
    )
    boundary = boundary_meridian(volume, 30).surface(8)
    # Front, back and rim must all participate; closed normals integrate to zero.
    assert_allclose(
        np.sum(boundary.normals * boundary.weights[..., None], axis=(0, 1)),
        0,
        atol=1e-12,
    )
    solution = AxisymmetricMFS(
        source, volume, source.region, source_count=16, source_offset=0.06, azimuths=128
    )
    probes = o.PointGrid([[0.1, 0.2, 0.3], [0, 0, 1.5]])
    assert_allclose(
        solution.evaluate(probes).electric.values,
        source.evaluate(probes).electric.values,
        atol=2e-4,
    )


def test_reference_guards():
    source, sphere, glass = problem()
    with pytest.raises(ValueError, match="budget"):
        AxisymmetricMFS(source, sphere, glass, source_count=32, max_unknowns=10)
    with pytest.raises(ValueError, match="half-size"):
        AxisymmetricMFS(source, sphere, glass, source_offset=0.5)
    with pytest.raises(ValueError, match="outside"):
        MieSphere(source, sphere, glass).evaluate(
            o.PointGrid([[0, 0, 1]]), side="interior"
        )


def test_reference_package_is_not_imported_by_primary_evaluation():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vecdiff_general; assert not any(m.startswith('vecdiff_general.propagation.references') for m in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_comparison_window_is_explicit_and_fast_evaluation_matches_direct():
    from vecdiff_general.propagation.references.benchmarks import compare_case

    case = compare_case(
        "plano-convex",
        1,
        grids=(24,),
        window_apertures=6,
        run_bem=False,
        run_fem=False,
    )
    solution = case["solutions"]["Ours"]
    # "Ours" must be the actual operational solver, without an incident/control
    # field addition from the historical bypass experiment.
    assert isinstance(solution, o.HighFrequencySolution)
    grid = solution.transmitted[0].grid
    radius = case["system"].apertures[0]
    assert_allclose(len(grid.u) * grid.spacing[0], 6 * radius, rtol=1e-13)
    assert case["rows"][0]["window_apertures"] == 6
    actual = solution.evaluate(case["probes"])
    expected = solution.evaluate(case["probes"], backend="direct")
    assert_allclose(actual.electric.values, expected.electric.values, atol=2e-10)
    assert_allclose(actual.magnetic, expected.magnetic, atol=2e-12)
    entrance = o.CartesianGrid(
        grid.u, grid.v, case["system"].interfaces[0].surface.frame
    )
    operational = o.propagate_interfaces(case["source"], case["system"], entrance)
    assert_allclose(
        actual.electric.values,
        operational.evaluate(case["probes"]).electric.values,
        atol=2e-10,
    )
    with pytest.raises(ValueError, match="window_apertures"):
        compare_case("plano-convex", 1, window_apertures=2)
