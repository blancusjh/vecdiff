"""Native solvers against analytic Maxwell fields, mesh/PML and SI checks."""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose
from vecdiff_general.propagation.references import (
    DielectricBEM,
    DielectricFEM,
    MieSphere,
    SurfaceMesh,
    boundary_meridian,
    surface_mesh,
    tetrahedral_mesh,
)
from vecdiff_general.propagation.references.benchmarks import lens_case
from vecdiff_general.propagation.references.diagnostics import (
    boundary_residuals,
    closed_flux_residual,
    maxwell_residuals,
)


def problem(index=1.5, scale=1.0):
    air = o.HomogeneousRegion("air", o.Dielectric(1))
    glass = o.HomogeneousRegion("glass", o.Dielectric(index))
    source = o.PlaneWaveExpansion(
        [[0, 0, 2 * np.pi / scale]], [[1, 0, 0]], [0, 0, 0], scale, air
    )
    sphere = o.SphereVolume([0, 0, 0], 0.3 * scale)
    return source, sphere, glass


def probes(scale=1.0):
    return o.PointGrid(
        scale
        * np.array([[0, 0, 0], [0.06, 0.09, 0.12], [0.2, -0.15, 0.6], [0.1, 0.6, 0.15]])
    )


def test_bem_mesh_convergence_both_fields_and_boundary_traces():
    s, v, g = problem()
    exact = MieSphere(s, v, g).evaluate(probes())
    errors = []
    jumps = []
    for level in (0, 1, 2):
        mesh = SurfaceMesh.sphere(v, level)
        sol = DielectricBEM(s, mesh, g)
        actual = sol.evaluate(probes())
        errors.append(
            [
                np.linalg.norm(actual.electric.values - exact.electric.values)
                / np.linalg.norm(exact.electric.values),
                np.linalg.norm(actual.magnetic - exact.magnetic)
                / np.linalg.norm(exact.magnetic),
            ]
        )
        jumps.append(max(boundary_residuals(sol, mesh.boundary_grid()).values()))
        assert sol.algebraic_residual < 1e-11
    assert np.all(np.array(errors)[1:] < 0.4 * np.array(errors)[:-1])
    assert max(errors[-1]) < 0.04
    assert jumps[2] < jumps[1] < jumps[0]
    assert jumps[-1] < 0.17  # Low-order trace accuracy is deliberately not overstated.


def test_bem_quadrature_maxwell_flux_and_resource_guard():
    s, v, g = problem()
    mesh = SurfaceMesh.sphere(v, 1)
    coarse = DielectricBEM(s, mesh, g, quadrature=3, near_quadrature=5)
    fine = DielectricBEM(s, mesh, g, quadrature=5, near_quadrature=9)
    a = coarse.evaluate(probes()).electric.values
    b = fine.evaluate(probes()).electric.values
    assert np.linalg.norm(a - b) / np.linalg.norm(b) < 0.02
    exterior = np.array([[0.2, -0.15, 0.6], [0.1, 0.6, 0.15]])
    checks = maxwell_residuals(fine.evaluate, exterior, wavelength0=1, index=1)
    assert max(checks.values()) < 2e-5
    outer = boundary_meridian(o.SphereVolume([0, 0, 0], 0.8), 24).surface(12)
    assert abs(closed_flux_residual(fine, outer)) < 0.025
    with pytest.raises(ValueError, match="unknowns"):
        DielectricBEM(s, mesh, g, max_unknowns=10)
    with pytest.raises(ValueError, match="side"):
        fine.evaluate(mesh.boundary_grid())
    with pytest.raises(ValueError, match="edges/vertices"):
        fine.evaluate(o.PointGrid(mesh.vertices[:1]), side="exterior")


def test_bem_equal_index_and_si_scaling():
    results = []
    for scale in (1, 532e-9):
        s, v, g = problem(index=1, scale=scale)
        sol = DielectricBEM(s, SurfaceMesh.sphere(v, 1), g)
        result = sol.evaluate(probes(scale))
        expected = s.evaluate(probes(scale))
        assert (
            np.linalg.norm(result.electric.values - expected.electric.values)
            / np.linalg.norm(expected.electric.values)
            < 0.025
        )
        assert (
            np.linalg.norm(result.magnetic - expected.magnetic)
            / np.linalg.norm(expected.magnetic)
            < 0.025
        )
        results.append(result.electric.values)
    assert_allclose(*results, atol=1e-12)


def test_surface_mesh_rejects_open_or_inverted_boundary():
    _, v, _ = problem()
    mesh = SurfaceMesh.sphere(v, 0)
    with pytest.raises(ValueError, match="two oppositely"):
        SurfaceMesh(mesh.vertices, mesh.triangles[:-1])
    with pytest.raises(ValueError, match="outward"):
        SurfaceMesh(mesh.vertices, mesh.triangles[:, [0, 2, 1]])


@pytest.mark.parametrize("shape", ["plano-convex", "biconvex"])
def test_lens_meshes_and_finite_bem_fields(shape):
    pytest.importorskip("gmsh")
    s, g, v, _ = lens_case(1, shape=shape, wavelength0=1)
    mesh = surface_mesh(v, 0.25)
    assert mesh.locate([[0, 0, 0.15], [0, 0, 0.8]])[0].tolist() == [True, False]
    sol = DielectricBEM(s, mesh, g)
    assert np.isfinite(sol.evaluate(o.PointGrid([[0, 0, 0.7]])).electric.values).all()


def test_fem_vector_solution_converges_to_mie():
    pytest.importorskip("gmsh")
    s, v, g = problem()
    exact = MieSphere(s, v, g).evaluate(probes())
    errors = []
    for h in (0.3, 0.16):
        mesh = tetrahedral_mesh(v, edge_size=h, physical_padding=0.4, pml_width=0.4)
        sol = DielectricFEM(s, mesh, g)
        field = sol.evaluate(probes())
        errors.append(
            [
                np.linalg.norm(field.electric.values - exact.electric.values)
                / np.linalg.norm(exact.electric.values),
                np.linalg.norm(field.magnetic - exact.magnetic)
                / np.linalg.norm(exact.magnetic),
            ]
        )
        assert sol.algebraic_residual < 1e-10
        residuals = boundary_residuals(sol, mesh.boundary_grid())
        assert residuals["tangential_E"] < 1e-11
        assert residuals["normal_B"] < 1e-11
    assert np.all(np.array(errors[1]) < 0.7 * np.array(errors[0]))
    assert max(errors[-1]) < 0.4
    with pytest.raises(ValueError, match="physical box"):
        sol.evaluate(o.PointGrid([[0, 0, 0.9]]))
    with pytest.raises(ValueError, match="unknowns"):
        DielectricFEM(s, mesh, g, max_unknowns=10)


def test_fem_equal_index_and_scale_invariance():
    pytest.importorskip("gmsh")
    from vecdiff_general.propagation.references import TetrahedralMesh

    s, v, g = problem(index=1)
    mesh = tetrahedral_mesh(v, edge_size=0.3, physical_padding=0.4, pml_width=0.4)
    sol = DielectricFEM(s, mesh, g)
    field = sol.evaluate(probes())
    assert_allclose(
        field.electric.values, s.evaluate(probes()).electric.values, atol=1e-13
    )
    assert_allclose(field.magnetic, s.evaluate(probes()).magnetic, atol=1e-13)
    s, v, g = problem()
    base = DielectricFEM(s, mesh, g).evaluate(probes())
    scale = 532e-9
    scaled = TetrahedralMesh(
        mesh.vertices * scale,
        mesh.tetrahedra,
        mesh.interior,
        mesh.center * scale,
        mesh.halfwidth * scale,
        mesh.pml_width * scale,
    )
    s, v, g = problem(scale=scale)
    physical = DielectricFEM(s, scaled, g).evaluate(probes(scale))
    assert_allclose(base.electric.values, physical.electric.values, atol=2e-11)
    assert_allclose(base.magnetic, physical.magnetic, atol=2e-13)


def test_bem_oblique_coherent_illumination_in_nonunit_background():
    air = o.HomogeneousRegion("background", o.Dielectric(1.2))
    glass = o.HomogeneousRegion("inclusion", o.Dielectric(1.65))
    directions = np.array([[0.6, 0, 0.8], [0, -0.8, 0.6]])
    amplitudes = np.array([[0.8, 0.2j, -0.6], [0.3j, 0.6, 0.8]])
    source = o.PlaneWaveExpansion(
        2 * np.pi * 1.2 * directions, amplitudes, [0.1, -0.2, 0.05], 1.0, air
    )
    sphere = o.SphereVolume([0.05, -0.03, 0.1], 0.25)
    grid = o.PointGrid(
        sphere.center
        + np.array([[0, 0, 0], [0.02, 0.04, 0.06], [0.2, -0.1, 0.5], [0.4, 0.3, -0.2]])
    )
    exact = []
    for wave, amplitude in zip(source.wavevectors, amplitudes):
        mode = o.PlaneWaveExpansion([wave], [amplitude], source.origin, 1.0, air)
        exact.append(MieSphere(mode, sphere, glass).evaluate(grid))
    numerical = DielectricBEM(source, SurfaceMesh.sphere(sphere, 2), glass).evaluate(
        grid
    )
    expected_e = sum(f.electric.values for f in exact)
    expected_h = sum(f.magnetic for f in exact)
    assert (
        np.linalg.norm(numerical.electric.values - expected_e)
        / np.linalg.norm(expected_e)
        < 0.035
    )
    assert (
        np.linalg.norm(numerical.magnetic - expected_h) / np.linalg.norm(expected_h)
        < 0.04
    )


def test_bem_cache_reloads_identical_currents(tmp_path):
    from vecdiff_general.propagation.references import DielectricBEM, SurfaceMesh
    from vecdiff_general.volumes import SphereVolume

    air = o.HomogeneousRegion("air", o.Dielectric(1.0))
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    source = o.PlaneWaveExpansion([[0, 0, 2 * np.pi]], [[1, 0, 0]], [0, 0, 0], 1, air)
    mesh = SurfaceMesh.sphere(SphereVolume([0, 0, 0], 0.3), 1)
    first = DielectricBEM(source, mesh, glass, cache=tmp_path)
    second = DielectricBEM(source, mesh, glass, cache=tmp_path)
    assert first.loaded_from is None and second.loaded_from is not None
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    other = DielectricBEM(source, mesh, glass, cache=tmp_path, quadrature=5)
    assert other.loaded_from is None


def test_finite_rim_coupling_improves_a_lens_field_against_bem():
    """Compare complex vector fields over a common detector patch, not peaks."""
    wavelength = 532e-9
    source, glass, volume, system = lens_case(4, wavelength0=wavelength)
    radius = 2 * wavelength
    count = 192
    axis = (np.arange(count) - count // 2) * (24 * radius / count)
    grid = o.CartesianGrid(axis, axis, system.interfaces[0].surface.frame)
    old = o.propagate_interfaces(source, system, grid, rim_coupling="off")
    improved = o.propagate_interfaces(source, system, grid)
    assert old.coupled_elements == ()
    assert improved.coupled_elements == ((0, 1),)

    coordinate = np.linspace(-wavelength, wavelength, 21)
    x, y = np.meshgrid(coordinate, coordinate)
    detector = o.PointGrid(np.stack((x, y, np.full_like(x, 2.192e-6)), axis=-1))
    reference = (
        DielectricBEM(source, surface_mesh(volume, 2 * radius / 12), glass)
        .evaluate(detector)
        .electric.values
    )
    denominator = np.linalg.norm(reference)
    old_error = np.linalg.norm(old.evaluate(detector).electric.values - reference)
    new_error = np.linalg.norm(improved.evaluate(detector).electric.values - reference)
    assert new_error / denominator < 0.20
    assert new_error < old_error / 2

    # Smaller bodies were found to overshoot with the slice approximation.
    source, _, _, system = lens_case(2, wavelength0=wavelength)
    small = o.propagate_interfaces(source, system, grid)
    assert small.coupled_elements == ()
