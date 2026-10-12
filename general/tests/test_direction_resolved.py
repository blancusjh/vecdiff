"""Independent modal phase and Fresnel checks for compact direction-resolved transfer."""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose
from vecdiff_general.propagation._directions import transmitted_direction
from vecdiff_general.propagation._fresnel import (
    apply_fresnel,
    apply_transmission,
    fresnel_transmission_matrix,
    polarization_basis,
)
from vecdiff_general.propagation.surface_transfer import BandSynthesis


@pytest.mark.parametrize("n1,n2", [(1, 1.5), (1.5, 1), (1.3, 1.3)])
def test_fused_fresnel_matches_boundary_matrix_including_TIR(n1, n2):
    rng = np.random.default_rng(724)
    d = rng.normal(size=(120, 3))
    d[:, 2] = abs(d[:, 2]) + 0.3
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    n = rng.normal(size=d.shape)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    n *= np.where(np.sum(d * n, axis=-1) < 0, -1.0, 1.0)[:, None]
    n[:4] = d[:4]
    wave = rng.normal(size=(120, 2)) + 1j * rng.normal(size=(120, 2))
    electric = np.column_stack((wave, -np.sum(d[:, :2] * wave, axis=-1) / d[:, 2]))
    dt = transmitted_direction(d, n, n1, n2)
    expected = apply_fresnel(electric, fresnel_transmission_matrix(d, dt, n, n1, n2))
    if n1 == n2:
        # The analytic matched-index boundary is exactly the identity, even
        # near grazing where subtractive Snell formulas lose precision.
        expected, dt = electric, d
    actual = apply_transmission(d, n, n1, n2, wave)
    assert_allclose(actual[:, :3], expected, atol=4e-14, rtol=4e-14)
    assert_allclose(actual[:, 3:], n2 * np.cross(dt, expected), atol=4e-14, rtol=4e-14)


@pytest.mark.parametrize("count", [31, 32])
@pytest.mark.parametrize("workers", [1, 2])
def test_band_Q_matches_direct_on_nonspherical_graph(count, workers):
    axis = (np.arange(count) - count // 2) * 0.4
    plane = o.CartesianGrid(axis, axis, o.Frame([0, 0, 2]))
    surface = o.FreeformSurface(
        lambda xy: 1.3 * np.cos(xy[..., 0] / 3) + 0.2 * xy[..., 1],
        lambda xy: np.stack(
            (-1.3 / 3 * np.sin(xy[..., 0] / 3), np.full(xy.shape[:-1], 0.2)), axis=-1
        ),
        frame=plane.frame,
    )
    region = o.HomogeneousRegion("air", o.Dielectric(1))
    q = o.SurfaceTransport(plane, surface, region, 1)
    support = q.active & (q.ku > 0) & (q.ku < 1.8) & (q.kv > -1) & (q.kv < 0.8)
    ids = np.flatnonzero(support)
    plan = BandSynthesis(q, ids[::-1], workers=workers)
    rng = np.random.default_rng(849)
    coefficients = np.zeros(plane.shape + (2,), complex)
    coefficients[support] = rng.normal(size=(len(ids), 2)) + 1j * rng.normal(
        size=(len(ids), 2)
    )
    k = np.stack((q.ku[support], q.kv[support], q.kz[support]), axis=-1)
    direct = (
        np.exp(1j * (q.grid.points.reshape(-1, 3) - plane.frame.origin) @ k.T)
        @ coefficients[support]
        / q.size
    )
    assert_allclose(plan(coefficients).reshape(-1, 2), direct, atol=2e-13, rtol=2e-11)
    assert_allclose(
        plan(coefficients), q._synthesize(coefficients), atol=2e-13, rtol=2e-11
    )
    assert plan.rank < q.rank
    assert plan.tail_bound < 1e-12


def test_single_mode_band_has_rank_one_without_losing_sag_phase():
    axis = np.arange(-12, 12) * 0.4
    plane = o.CartesianGrid(axis, axis)
    surface = o.SphericalCap(12, plane.frame)
    q = o.SurfaceTransport(
        plane, surface, o.HomogeneousRegion("air", o.Dielectric(1)), 1
    )
    ids = np.array([q.size // 2 + len(axis) // 2 + 1])
    plan = BandSynthesis(q, ids)
    a = np.zeros(plane.shape + (2,), complex)
    a.reshape(-1, 2)[ids] = [1 + 2j, 0.7 - 0.3j]
    assert plan.rank == 1
    assert_allclose(plan(a), q._synthesize(a), atol=2e-13, rtol=2e-11)


def test_direction_resolved_curved_trace_matches_direct_modal_fresnel():
    from vecdiff_general.comparisons import scatter_surface

    axis = np.arange(-12, 12) * 0.5
    plane = o.CartesianGrid(axis, axis)
    air = o.HomogeneousRegion("air", o.Dielectric(1))
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    surface = o.FreeformSurface(
        lambda xy: 0.2 * np.cos(xy[..., 0] / 3) + 0.03 * xy[..., 1],
        lambda xy: np.stack(
            (-0.2 / 3 * np.sin(xy[..., 0] / 3), np.full(xy.shape[:-1], 0.03)), axis=-1
        ),
    )
    boundary = o.DielectricInterface(surface, air, glass)
    q = o.SurfaceTransport(plane, surface, air, 1)
    x, y = plane.points[..., 0], plane.points[..., 1]
    values = np.stack(
        (
            np.exp(2j * np.pi * x / 12) + 0.4 * np.exp(-2j * np.pi * y / 12),
            0.3j * np.exp(2j * np.pi * (x + y) / 12),
        ),
        axis=-1,
    )
    source = o.angular_spectrum(
        o.TransverseElectricField(plane, 1, values), air, evanescent="discard"
    )
    actual = o.DirectionResolvedFresnel().transmit(source, boundary, q).field
    expected = scatter_surface(source, boundary, q.grid).transmitted_trace
    assert_allclose(
        actual.electric.values, expected.electric.values, atol=2e-12, rtol=2e-12
    )
    assert_allclose(actual.magnetic, expected.magnetic, atol=2e-14, rtol=2e-12)


@pytest.mark.parametrize("n1,n2", [(1, 1.5), (1.5, 1), (1.3, 1.3)])
def test_unity_coefficients_refract_and_preserve_polarization_amplitudes(n1, n2):
    # Normal, oblique and (on glass-to-air transmission) TIR directions.
    sine = np.array([0.0, 0.3, 0.8])
    d = np.stack((sine, 0 * sine, np.sqrt(1 - sine**2)), axis=-1)
    normal = np.array([0, 0, 1.0])
    dt = transmitted_direction(d, normal, n1, n2)
    s, pi, pt = polarization_basis(d, dt, normal)
    incident = (0.7 + 0.2j) * s + (0.3 - 0.4j) * pi
    expected = (0.7 + 0.2j) * s + (0.3 - 0.4j) * pt
    actual = apply_transmission(
        d, normal, n1, n2, incident[:, :2], coefficients="unity"
    )
    assert_allclose(actual[:, :3], expected, atol=1e-14)
    assert_allclose(np.sum(dt * actual[:, :3], axis=-1), 0, atol=1e-14)
    assert_allclose(actual[:, 3:], n2 * np.cross(dt, expected), atol=1e-14)


def test_direction_resolved_unity_on_curved_surface_matches_modal_sum():
    axis = np.arange(-8, 8) * 0.5
    plane = o.CartesianGrid(axis, axis)
    air = o.HomogeneousRegion("air", o.Dielectric(1))
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    surface = o.EvenAsphere(0.02, conic=-1)
    boundary = o.DielectricInterface(surface, air, glass)
    q = o.SurfaceTransport(plane, surface, air, 1)
    x, y = plane.points[..., 0], plane.points[..., 1]
    values = np.stack(
        (np.exp(2j * np.pi * x / 8), 0.4j * np.exp(-2j * np.pi * y / 8)), axis=-1
    )
    source = o.angular_spectrum(
        o.TransverseElectricField(plane, 1, values), air, evanescent="discard"
    )
    actual = (
        o.DirectionResolvedFresnel(coefficients="unity")
        .transmit(source, boundary, q)
        .field
    )
    expected_e = np.zeros(q.grid.shape + (3,), complex)
    expected_h = np.zeros_like(expected_e)
    from vecdiff_general.media import Z0

    for k, amplitude in zip(source.expansion.wavevectors, source.expansion.amplitudes):
        if not np.any(amplitude):
            continue
        d = k.real / (2 * np.pi)
        dt = transmitted_direction(d, q.grid.normals, 1, 1.5)
        s, pi, pt = polarization_basis(d, dt, q.grid.normals)
        electric = (
            np.sum(s * amplitude, axis=-1)[..., None] * s
            + np.sum(pi * amplitude, axis=-1)[..., None] * pt
        )
        phase = np.exp(1j * (q.grid.points - source.expansion.origin) @ k)[..., None]
        expected_e += phase * electric
        expected_h += phase * 1.5 * np.cross(dt, electric) / Z0
    assert_allclose(actual.electric.values, expected_e, atol=2e-12)
    assert_allclose(actual.magnetic, expected_h, atol=2e-14)


def test_direction_resolved_rejects_unknown_coefficient_model():
    with pytest.raises(ValueError, match="coefficients"):
        o.DirectionResolvedFresnel(coefficients="scalar")
