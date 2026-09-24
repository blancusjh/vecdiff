"""Independent limits and numerical controls for R Q^-1 T Q R."""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose
from vecdiff_general.observables import coherent_sum
from vecdiff_general.observables.diagnostics import interface_residuals


def setup(count=24, length=24.0, n=1.0):
    region = o.HomogeneousRegion("medium", o.Dielectric(n))
    axis = (np.arange(count) - count // 2) * length / count
    return region, o.CartesianGrid(axis, axis)


def mode(region, grid, ix=1, iy=0):
    kx, ky = (
        2 * np.pi * ix / (len(grid.u) * grid.spacing[0]),
        2 * np.pi * iy / (len(grid.v) * grid.spacing[1]),
    )
    k = 2 * np.pi * region.material.n
    wavevector = np.array([kx, ky, np.sqrt(k * k - kx * kx - ky * ky)])
    amplitude = np.cross(wavevector / k, [0, 1, 0])
    return o.PlaneWaveExpansion([wavevector], [amplitude], [0, 0, 0], 1.0, region)


def spectrum(source, grid):
    return o.angular_spectrum(source.evaluate(grid).electric, source.region)


def test_Q_matches_direct_modal_sum_and_inverse_on_curved_chart():
    region, grid = setup()
    surface = o.FreeformSurface(
        lambda xy: 0.5 * np.cos(2 * np.pi * xy[..., 0] / 24),
        lambda xy: np.stack(
            (
                -np.pi / 24 * np.sin(2 * np.pi * xy[..., 0] / 24),
                np.zeros(xy.shape[:-1]),
            ),
            -1,
        ),
    )
    source = mode(region, grid, 3, 2)
    Q = o.SurfaceTransport(grid, surface, region, 1.0)
    trace = Q.forward(spectrum(source, grid))
    assert_allclose(
        trace.electric.values, source.evaluate(Q.grid).electric.values, atol=2e-10
    )
    assert_allclose(trace.magnetic, source.evaluate(Q.grid).magnetic, atol=2e-12)
    recovered, report = Q.inverse(trace)
    assert report.relative_residual < 1e-8
    assert_allclose(
        o.PlanarASM()(recovered, 0).electric.values,
        source.evaluate(grid).electric.values,
        atol=1e-7,
    )


def test_negative_height_is_not_the_curved_inverse():
    region, grid = setup()

    def h(xy):
        return 2 * np.cos(2 * np.pi * xy[..., 0] / 24)

    def grad(xy):
        return np.stack(
            (-np.pi / 6 * np.sin(2 * np.pi * xy[..., 0] / 24), np.zeros(xy.shape[:-1])),
            -1,
        )

    Q = o.SurfaceTransport(grid, o.FreeformSurface(h, grad), region, 1.0)
    minus = o.SurfaceTransport(
        grid, o.FreeformSurface(lambda xy: -h(xy), lambda xy: -grad(xy)), region, 1.0
    )
    source = mode(region, grid, 5)
    s = spectrum(source, grid)
    coefficients = s.expansion.amplitudes.reshape(grid.shape + (3,)) * Q.size
    curved = Q._synthesize(coefficients)
    from vecdiff_general.fourier.cartesian import FFT2

    wrong = minus._synthesize(FFT2(curved))
    expected = source.evaluate(grid).electric.values
    assert np.linalg.norm(wrong - expected) / np.linalg.norm(expected) > 0.02
    recovered, report = Q.inverse(Q.forward(s))
    assert report.relative_residual < 1e-8


@pytest.mark.parametrize("polarization", [(1, 0, 0), (0, 1, 0), (1, 1j, 0)])
def test_plane_slab_boundary_flux_and_evaluation(polarization):
    air, grid = setup()
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    end = o.HomogeneousRegion("exit", o.Dielectric(1))
    system = o.OpticalSystem(
        (
            o.DielectricInterface(o.Plane(), air, glass),
            o.DielectricInterface(o.Plane(o.Frame([0, 0, 5])), glass, end),
        )
    )
    source = o.PlaneWaveExpansion(
        [[0, 0, 2 * np.pi]], [polarization], [0, 0, 0], 1.0, air
    )
    solution = o.propagate_interfaces(source, system, grid)
    observation = grid.translated([0, 0, 8])
    expected = 0.96 * np.asarray(polarization) * np.exp(2j * np.pi * (7.5 + 3))
    assert_allclose(
        solution.evaluate(observation).electric.values,
        np.broadcast_to(expected, grid.shape + (3,)),
        atol=1e-10,
    )
    for interface, encounter in zip(system.interfaces, solution.encounters[0]):
        residuals = interface_residuals(
            coherent_sum([encounter.incident, encounter.reflected]),
            encounter.transmitted,
            encounter.grid.normals,
            interface.minus.material,
            interface.plus.material,
            reference_electric=1,
            reference_index=1,
        )
        assert max(residuals.values()) < 1e-12
        assert_allclose(
            encounter.incident.power() + encounter.reflected.power(),
            encounter.transmitted.power(),
            rtol=1e-12,
        )
    assert_allclose(
        solution.evaluate(observation).power(),
        solution.encounters[0][-1].transmitted.power(),
        rtol=1e-10,
    )


def test_oblique_plane_matches_exact_fresnel_and_no_green(monkeypatch):
    region, grid = setup()
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    interface = o.DielectricInterface(o.Plane(), region, glass)
    source = mode(region, grid, 4, 2)
    reference = o.scatter_plane(source, o.PlanarInterface(o.Frame(), region, glass))
    from vecdiff_general.comparisons.surface_radiation import SurfaceRadiation

    monkeypatch.setattr(
        SurfaceRadiation,
        "evaluate",
        lambda *a, **k: pytest.fail("Green entered primary path"),
    )
    solution = o.propagate_high_frequency(source, o.OpticalSystem((interface,)), grid)
    observation = grid.translated([0, 0, 7])
    assert_allclose(
        solution.evaluate(observation).electric.values,
        reference.transmitted.evaluate(observation).electric.values,
        atol=2e-9,
    )
    assert_allclose(
        solution.evaluate(
            o.PointGrid([[0.2, 0.3, 7]]), backend="direct"
        ).electric.values,
        reference.transmitted.evaluate(o.PointGrid([[0.2, 0.3, 7]])).electric.values,
        atol=2e-9,
    )


def test_curved_equal_index_control_and_flux():
    region, grid = setup()
    surface = o.FreeformSurface(
        lambda xy: 0.3 * np.cos(2 * np.pi * xy[..., 0] / 24),
        lambda xy: np.stack(
            (
                -0.3 * 2 * np.pi / 24 * np.sin(2 * np.pi * xy[..., 0] / 24),
                np.zeros(xy.shape[:-1]),
            ),
            -1,
        ),
    )
    other = o.HomogeneousRegion("other", o.Dielectric(1))
    source = mode(region, grid, 3)
    solution = o.propagate_interfaces(
        source, o.OpticalSystem((o.DielectricInterface(surface, region, other),)), grid
    )
    observation = grid.translated([0, 0, 4])
    assert_allclose(
        solution.evaluate(observation).electric.values,
        source.evaluate(observation).electric.values,
        atol=1e-7,
    )
    assert_allclose(
        solution.evaluate(observation).power(),
        source.evaluate(observation).power(),
        rtol=1e-9,
    )


def lens_solution(count, length=80.0, **options):
    air, grid = setup(count, length)
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    output = o.HomogeneousRegion("output", o.Dielectric(1))
    lens = o.OpticalSystem(
        (
            o.DielectricInterface(o.Plane(), air, glass),
            o.DielectricInterface(
                o.SphericalCap(-80, o.Frame([0, 0, 8])), glass, output
            ),
        ),
        (24.0, 24.0),
    )

    # Smooth pupil permits separating sampling convergence from rim staircasing.
    def envelope(p):
        r = np.linalg.norm(p[..., :2], axis=-1)
        taper = np.cos(np.pi / 2 * np.clip((r - 16) / 8, 0, 1)) ** 4
        return (
            np.exp(-((r / 16) ** 2))[..., None] * taper[..., None] * np.array([1, 0, 0])
        )

    source = o.EikonalField(
        lambda p: p[..., 2], lambda p: np.array([0, 0, 1]), envelope, 1.0, air
    )
    return o.propagate_interfaces(source, lens, grid, **options)


def test_lens_grid_convergence_boundary_and_flux():
    probes = o.PointGrid([[0, 0, 168], [1, 0, 168], [0, 1, 168], [0, 0, 140]])
    solutions = [lens_solution(n) for n in (64, 80, 96)]
    values = [s.evaluate(probes, backend="direct").electric.values for s in solutions]
    coarse = np.linalg.norm(values[0] - values[2]) / np.linalg.norm(values[2])
    fine = np.linalg.norm(values[1] - values[2]) / np.linalg.norm(values[2])
    assert fine < coarse and fine < 0.002
    final = solutions[-1]
    assert max(r.relative_residual for r in final.inverse_reports[0]) < 0.002
    for encounter in final.encounters[0]:
        assert_allclose(
            encounter.incident.power() + encounter.reflected.power(),
            encounter.transmitted.power(),
            rtol=1e-12,
        )
    # Reconstructed Maxwell fields and local Fresnel traces are different checks.
    Q = final.transports[-1][1]
    reconstructed = Q.forward(final.transmitted[0])
    target = final.encounters[0][-1].transmitted
    assert (
        np.linalg.norm(reconstructed.magnetic - target.magnetic)
        / np.linalg.norm(target.magnetic)
        < 0.02
    )  # leading-order WKB magnetic trace
    assert abs(reconstructed.power() / target.power() - 1) < 0.005
    assert_allclose(
        final.evaluate(Q.plane.translated([0, 0, 160])).power(),
        o.PlanarASM()(final.transmitted[0], 0).power(),
        rtol=1e-12,
    )


def test_sampled_input_uses_Q_and_matches_analytic_input():
    region, grid = setup()
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    source = mode(region, grid, 2)
    system = o.OpticalSystem((o.DielectricInterface(o.Plane(), region, glass),))
    a = o.propagate_interfaces(source, system, grid)
    b = o.propagate_interfaces(spectrum(source, grid), system, grid)
    observation = grid.translated([0, 0, 3])
    assert_allclose(
        a.evaluate(observation).electric.values,
        b.evaluate(observation).electric.values,
        atol=1e-9,
    )


@pytest.mark.parametrize("curved", [False, True])
def test_placed_chart_aperture_masks_trace_and_preserves_local_flux(curved):
    basis = np.array([[0.8, 0, 0.6], [0, 1, 0], [-0.6, 0, 0.8]])
    frame = o.Frame([10, -20, 30], basis)
    region, base = setup()
    grid = o.CartesianGrid(base.u, base.v, frame)
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    surface = o.SphericalCap(80, frame) if curved else o.Plane(frame)
    source = o.PlaneWaveExpansion(
        [2 * np.pi * frame.normal], [frame.u], frame.origin, 1.0, region
    )
    s = o.propagate_interfaces(
        source,
        o.OpticalSystem((o.DielectricInterface(surface, region, glass),), (8,), (8,)),
        grid,
    )
    encounter = s.encounters[0][0]
    radius = np.linalg.norm((grid.points - frame.origin) @ basis[:, :2], axis=-1)
    assert_allclose(
        encounter.transmitted.electric.values[radius > 8 + 1e-12], 0, atol=1e-12
    )
    assert_allclose(
        encounter.incident.power() + encounter.reflected.power(),
        encounter.transmitted.power(),
        rtol=1e-12,
    )


@pytest.mark.parametrize("height", [-2.0, 2.0])
def test_constant_height_Q_is_planar_ASM(height):
    region, grid = setup()
    surface = o.FreeformSurface(
        lambda xy: np.full(xy.shape[:-1], height), lambda xy: np.zeros_like(xy)
    )
    Q = o.SurfaceTransport(grid, surface, region, 1.0)
    source = mode(region, grid, 3, 2)
    initial = spectrum(source, grid)
    trace = Q.forward(initial)
    assert_allclose(
        trace.electric.values,
        o.PlanarASM()(initial, height).electric.values,
        atol=1e-12,
    )
    recovered, report = Q.inverse(trace)
    assert report.relative_residual < 1e-10
    assert report.magnetic_relative_residual < 1e-10
    assert report.solver == "fft" and report.iterations == 0
    assert_allclose(
        recovered.expansion.amplitudes, initial.expansion.amplitudes, atol=1e-10
    )


def test_two_curved_equal_index_interfaces_repeat_the_operator_chain():
    region, grid = setup()
    other = o.HomogeneousRegion("middle", o.Dielectric(1))
    end = o.HomogeneousRegion("end", o.Dielectric(1))

    def height(xy):
        return 0.2 * np.cos(2 * np.pi * xy[..., 0] / 24)

    def gradient(xy):
        return np.stack(
            (
                -0.2 * 2 * np.pi / 24 * np.sin(2 * np.pi * xy[..., 0] / 24),
                np.zeros(xy.shape[:-1]),
            ),
            -1,
        )

    first = o.FreeformSurface(height, gradient)
    second = o.FreeformSurface(height, gradient, o.Frame([0, 0, 4]))
    system = o.OpticalSystem(
        (
            o.DielectricInterface(first, region, other),
            o.DielectricInterface(second, other, end),
        )
    )
    source = mode(region, grid, 3, 1)
    solution = o.propagate_interfaces(source, system, grid)
    observation = grid.translated([0, 0, 8])
    assert_allclose(
        solution.evaluate(observation).electric.values,
        source.evaluate(observation).electric.values,
        atol=2e-7,
    )
    with pytest.raises(ValueError, match="surface envelope"):
        solution.evaluate(grid.translated([0, 0, 4]))


def test_inverse_nonconvergence_is_reported_as_failure():
    region, grid = setup()
    Q = o.SurfaceTransport(grid, o.SphericalCap(30), region, 1.0)
    source = mode(region, grid, 4, 2)
    with pytest.raises(ValueError, match="did not converge"):
        Q.inverse(Q.forward(spectrum(source, grid)), max_iterations=1)


def test_Q_rejects_a_backward_spectrum():
    region, grid = setup()
    field = o.TransverseElectricField(
        grid, 1.0, np.broadcast_to([1, 0], grid.shape + (2,))
    )
    backwards = o.angular_spectrum(field, region, direction=-1)
    Q = o.SurfaceTransport(grid, o.Plane(), region, 1.0)
    with pytest.raises(ValueError, match="forward propagating"):
        Q.forward(backwards)


def test_physical_trace_metric_and_complex_adjoint_on_steep_chart():
    region, grid = setup(count=20, length=12)
    surface = o.FreeformSurface(
        lambda xy: (
            np.cos(2 * np.pi * xy[..., 0] / 12)
            + 0.8 * np.sin(2 * np.pi * xy[..., 1] / 12)
        ),
        lambda xy: np.stack(
            (
                -2 * np.pi / 12 * np.sin(2 * np.pi * xy[..., 0] / 12),
                0.8 * 2 * np.pi / 12 * np.cos(2 * np.pi * xy[..., 1] / 12),
            ),
            -1,
        ),
    )
    Q = o.SurfaceTransport(grid, surface, region, 1.0)
    rng = np.random.default_rng(420)
    E = rng.normal(size=grid.shape + (3,)) + 1j * rng.normal(size=grid.shape + (3,))
    projected = E - Q.grid.normals * np.sum(E * Q.grid.normals, -1)[..., None]
    weighted = Q._trace_weight(Q._tangent(E))
    assert_allclose(
        np.sum(abs(weighted) ** 2 * grid.weights[..., None]),
        np.sum(abs(projected) ** 2 * Q.grid.weights[..., None]),
        rtol=2e-14,
    )
    A = Q._inverse_operator()
    x = rng.normal(size=A.shape[1]) + 1j * rng.normal(size=A.shape[1])
    y = rng.normal(size=A.shape[0]) + 1j * rng.normal(size=A.shape[0])
    assert_allclose(np.vdot(A @ x, y), np.vdot(x, A.H @ y), rtol=2e-13)


def test_transparent_curved_interface_preserves_diffracted_E_and_H():
    region, grid = setup(count=48)
    waves = [mode(region, grid, ix, iy) for ix, iy in ((0, 0), (9, 2), (-5, 4))]
    source = o.PlaneWaveExpansion(
        np.concatenate([wave.wavevectors for wave in waves]),
        np.concatenate([wave.amplitudes * a for wave, a in zip(waves, (1, 0.4j, 0.3))]),
        [0, 0, 0],
        1.0,
        region,
    )
    other = o.HomogeneousRegion("same material", o.Dielectric(1))
    surface = o.FreeformSurface(
        lambda xy: 0.4 * np.cos(2 * np.pi * xy[..., 0] / 24),
        lambda xy: np.stack(
            (
                -0.4 * 2 * np.pi / 24 * np.sin(2 * np.pi * xy[..., 0] / 24),
                np.zeros(xy.shape[:-1]),
            ),
            -1,
        ),
    )
    solution = o.propagate_interfaces(
        spectrum(source, grid),
        o.OpticalSystem((o.DielectricInterface(surface, region, other),)),
        grid,
    )
    encounter = solution.encounters[0][0]
    exact = source.evaluate(encounter.grid)
    assert_allclose(
        encounter.transmitted.electric.values, exact.electric.values, atol=1e-10
    )
    assert_allclose(encounter.transmitted.magnetic, exact.magnetic, atol=1e-12)
    assert_allclose(encounter.reflected.electric.values, 0, atol=1e-15)
    assert encounter.transverse_projection_error == 0
    # This is a diffracted field: projecting onto one local ray would change it.
    removed = np.sum(exact.electric.values * encounter.incident_directions, -1)
    assert np.linalg.norm(removed) / np.linalg.norm(exact.electric.values) > 0.05
    actual = solution.evaluate(grid.translated([0, 0, 8]))
    expected = source.evaluate(actual.electric.grid)
    assert_allclose(actual.electric.values, expected.electric.values, atol=2e-8)
    assert_allclose(actual.magnetic, expected.magnetic, atol=2e-10)
    assert_allclose(actual.power(), expected.power(), rtol=2e-9)


def test_inverse_reports_incompatible_magnetic_trace_without_fitting_it():
    region, grid = setup()
    Q = o.SurfaceTransport(grid, o.Plane(), region, 1.0)
    trace = Q.forward(spectrum(mode(region, grid), grid))
    inconsistent = o.ElectromagneticField(trace.electric, 2 * trace.magnetic)
    _, report = Q.inverse(inconsistent)
    assert report.relative_residual < 1e-12
    assert_allclose(report.magnetic_relative_residual, 0.5, atol=1e-12)


@pytest.mark.parametrize("curved", [False, True])
def test_joint_inverse_adjoint_and_compatible_multimode_field(curved):
    region, grid = setup(count=32, length=16)
    surface = (
        o.FreeformSurface(
            lambda xy: 0.7 * np.cos(2 * np.pi * xy[..., 0] / 16),
            lambda xy: np.stack(
                (
                    -0.7 * 2 * np.pi / 16 * np.sin(2 * np.pi * xy[..., 0] / 16),
                    np.zeros(xy.shape[:-1]),
                ),
                -1,
            ),
        )
        if curved
        else o.Plane()
    )
    Q = o.SurfaceTransport(grid, surface, region, 1)
    A = Q._inverse_operator()
    rng = np.random.default_rng(442)
    x = rng.normal(size=A.shape[1]) + 1j * rng.normal(size=A.shape[1])
    y = rng.normal(size=A.shape[0]) + 1j * rng.normal(size=A.shape[0])
    assert_allclose(np.vdot(A @ x, y), np.vdot(x, A.H @ y), rtol=3e-13)
    waves = [mode(region, grid, *pair) for pair in ((1, 3), (-4, 2), (5, -1))]
    source = o.PlaneWaveExpansion(
        np.concatenate([s.wavevectors for s in waves]),
        np.concatenate([s.amplitudes * a for s, a in zip(waves, (1, 0.4j, 0.3))]),
        [0, 0, 0],
        1,
        region,
    )
    original = spectrum(source, grid)
    trace = Q.forward(original)
    result, report = Q.inverse(trace)
    assert report.relative_residual < 2e-8
    assert report.magnetic_relative_residual < 2e-8
    assert_allclose(
        result.expansion.amplitudes, original.expansion.amplitudes, atol=2e-8
    )
    if not curved:
        assert report.solver == "fft" and report.iterations == 0
