"""Independent limits and numerical controls for R Q^-1 T Q R."""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose


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


@pytest.mark.parametrize("cache_compact", [False, True])
def test_streamed_curved_Q_preserves_complex_fields_and_adjoint(
    monkeypatch, cache_compact
):
    from vecdiff_general.propagation import surface_transfer as transfer

    region, grid = setup()
    surface = o.SphericalCap(30)
    dense = o.SurfaceTransport(grid, surface, region, 1.0)
    budget = dense.rank * dense.size * 16 if cache_compact else 0
    monkeypatch.setattr(transfer, "_PRECOMPUTE_BYTES", budget)
    monkeypatch.setattr(transfer, "_COMPACT_FACTOR_BYTES", budget)
    factor_orders = []
    original_jv = transfer.jv

    def counted_jv(order, argument):
        factor_orders.append(order)
        return original_jv(order, argument)

    monkeypatch.setattr(transfer, "jv", counted_jv)
    streamed = o.SurfaceTransport(grid, surface, region, 1.0)
    assert streamed._terms is None
    assert (streamed._compact_factors is not None) == cache_compact
    assert not factor_orders  # Unused incident-Q plans do not compute factors.
    source = mode(region, grid, 3, 2)
    field = streamed.forward(spectrum(source, grid))
    exact = source.evaluate(streamed.grid)
    assert_allclose(field.electric.values, exact.electric.values, atol=2e-10)
    assert_allclose(field.magnetic, exact.magnetic, atol=2e-12)
    assert len(factor_orders) == streamed.rank * (1 if cache_compact else 2)
    rng = np.random.default_rng(51)
    values = rng.normal(size=grid.shape + (3,)) + 1j * rng.normal(
        size=grid.shape + (3,)
    )
    assert_allclose(streamed._adjoint(values), dense._adjoint(values), atol=2e-13)
    ids = np.flatnonzero(dense.active)
    # Band plans use the same streamed recurrence for weighted TQ applications.
    monkeypatch.setattr(transfer, "_PRECOMPUTE_BYTES", 1 << 30)
    dense_band = transfer.BandSynthesis(dense, ids)
    monkeypatch.setattr(transfer, "_PRECOMPUTE_BYTES", budget)
    streamed_band = transfer.BandSynthesis(dense, ids)
    assert streamed_band.terms is None
    values[~dense.active] = 0
    factor_orders.clear()
    assert_allclose(streamed_band(values), dense_band(values), atol=2e-13)
    assert_allclose(
        streamed_band(0.7j * values), 0.7j * dense_band(values), atol=2e-13
    )
    assert len(factor_orders) == (0 if cache_compact else 2 * streamed_band.rank)


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
        o.PlaneTransport()(recovered, 0).electric.values,
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
        o.PlaneTransport()(initial, height).electric.values,
        atol=1e-12,
    )
    recovered, report = Q.inverse(trace)
    assert report.relative_residual < 1e-10
    assert report.magnetic_relative_residual < 1e-10
    assert report.solver == "fft" and report.iterations == 0
    assert_allclose(
        recovered.expansion.amplitudes, initial.expansion.amplitudes, atol=1e-10
    )


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
