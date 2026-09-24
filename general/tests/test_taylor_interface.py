"""Physical and algebraic controls for the experimental material recurrence."""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose
from vecdiff_general.media.medium import Z0
from vecdiff_general.observables import coherent_sum
from vecdiff_general.observables.maxwell import interface_l2_residuals
from vecdiff_general.propagation.experiments import _taylor
from vecdiff_general.propagation.experiments.curvature import TaylorInterfaceOperator
from vecdiff_general.propagation.experiments.taylor_interface import interface_expansion
from vecdiff_general.propagation.operators import fresnel_transmission


def test_fortran_taylor_product_matches_direct_convolution():
    assert _taylor.taylor_product is not None
    space = _taylor.TaylorSpace(8, 3)
    rng = np.random.default_rng(152)
    a = rng.normal(size=(len(space.indices), 3)) + 1j * rng.normal(
        size=(len(space.indices), 3)
    )
    b = rng.normal(size=a.shape) + 1j * rng.normal(size=a.shape)
    expected = np.add.reduceat(a[space.left] * b[space.right], space.starts, axis=0)
    actual = (_taylor.Taylor(space, a) * _taylor.Taylor(space, b)).c
    assert_allclose(actual, expected, atol=3e-14, rtol=3e-14)


@pytest.mark.parametrize("slope", [0, 0.2])
def test_tilted_plane_is_exact_fresnel_at_every_order(slope):
    e = interface_expansion(
        lambda x, y: slope * x,
        np.array([[0.1, 0.2], [-0.2, 0.3]]),
        n_incident=1,
        n_transmitted=1.5,
        order=3,
    )
    grid = e.surface_grid(np.ones(2))
    incident = e.trace("incident", 0.1, grid=grid)
    a, b = [o.HomogeneousRegion(str(n), o.Dielectric(n)) for n in (1, 1.5)]
    interface = o.DielectricInterface(o.Plane(), a, b)
    expected = fresnel_transmission(
        incident, interface, np.broadcast_to([0.0, 0.0, 1.0], (2, 3))
    )
    for n in range(4):
        for branch in ("reflected", "transmitted"):
            field = e.trace(branch, 0.1, order=n, grid=grid)
            assert_allclose(
                field.electric.values,
                getattr(expected, branch).electric.values,
                atol=1e-13,
            )
            assert_allclose(
                field.magnetic, getattr(expected, branch).magnetic, atol=1e-15
            )


def test_index_matched_curved_surface_is_transparent_through_order_three():
    e = interface_expansion(
        lambda x, y: 1 - (1 - x * x - y * y) ** 0.5,
        np.array([[0.1, 0.2], [0.3, 0.2]]),
        n_incident=1.2,
        n_transmitted=1.2,
        order=3,
    )
    i = e.trace("incident", 0.2)
    t = e.trace("transmitted", 0.2)
    r = e.trace("reflected", 0.2)
    assert_allclose(t.electric.values, i.electric.values, atol=1e-13)
    assert_allclose(t.magnetic, i.magnetic, atol=1e-15)
    assert np.linalg.norm(r.electric.values) < 1e-13


def test_extra_derivative_depth_preserves_retained_coefficients():
    args = dict(n_incident=1.5, n_transmitted=1, order=3)
    xy = np.array([[0.21, 0.13]])

    def height(x, y):
        return -(1 - (1 - x * x - y * y) ** 0.5)

    a = interface_expansion(height, xy, **args)
    b = interface_expansion(height, xy, derivative_depth=3, **args)
    for branch in range(3):
        for m in range(4):
            for aj, bj in zip(a.amplitudes[branch][m], b.amplitudes[branch][m]):
                assert_allclose(aj.c[0], bj.c[0], atol=3e-12, rtol=3e-12)
    for branch in ("incident", "reflected", "transmitted"):
        trace = a.trace(branch, 0.1)
        jet = a.jet(branch, 0.1)
        assert_allclose(trace.electric.values, jet.field.electric.values, atol=1e-13)
        assert_allclose(trace.magnetic, jet.field.magnetic, atol=1e-15)


def test_rotation_covariance_on_mixed_curvature_graph():
    angle = 0.37
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def height(x, y):
        return 0.5 * x * x + 0.35 * y * y + 0.12 * x * y + 0.08 * x * x * y

    xy = np.array([[0.21, -0.12]])
    rotated = xy @ rotation[:2, :2].T
    a = interface_expansion(height, xy, n_incident=1, n_transmitted=1.5)
    b = interface_expansion(
        lambda x, y: height(c * x + s * y, -s * x + c * y),
        rotated,
        n_incident=1,
        n_transmitted=1.5,
        amplitude=rotation @ np.array([1, 0, 0]),
    )
    for branch in ("reflected", "transmitted"):
        fa, fb = a.trace(branch, 0.1), b.trace(branch, 0.1)
        assert_allclose(fa.electric.values @ rotation.T, fb.electric.values, atol=2e-13)
        assert_allclose(fa.magnetic @ rotation.T, fb.magnetic, atol=2e-15)


def test_complete_curl_improves_with_order_and_expected_scale():
    e = interface_expansion(
        lambda x, y: 1 - (1 - x * x - y * y) ** 0.5,
        np.array([[0.2, 0.1], [0.3, 0.2]]),
        n_incident=1,
        n_transmitted=1.5,
    )
    g = e.surface_grid(np.ones(2))
    residuals = []
    for n in range(4):
        errors = []
        for wavelength in (0.1, 0.05):
            i, r, t = [
                e.trace(b, wavelength, order=n, grid=g)
                for b in ("incident", "reflected", "transmitted")
            ]
            jump = interface_l2_residuals(
                coherent_sum([i, r]), t, i, g, n_minus=1, n_plus=1.5
            )
            errors.append(jump["tangential_H"])
        assert_allclose(errors[0] / errors[1], 2 ** (n + 1), rtol=1e-7)
        residuals.append(errors[0])
    assert np.all(np.diff(residuals) < 0)
    assert residuals[-1] < residuals[0] / 100


def test_full_curl_from_cartesian_finite_difference_of_local_continuation():
    e = interface_expansion(
        lambda x, y: 0.5 * x * x + 0.35 * y * y + 0.12 * x * y,
        np.array([[0.2, 0.1]]),
        n_incident=1,
        n_transmitted=1.5,
    )
    powers = e.phases[2].space.powers

    def value(offset):
        u, v, z = offset
        duv = np.array([u, v, 0.0])
        hp = np.prod(duv[None, :] ** powers, axis=1) @ e.calculus.h.c[:, 0]
        ss = z - (hp - e.calculus.h.c[0, 0])
        monomials = np.prod(np.array([u, v, ss])[None, :] ** powers, axis=1)
        phase = monomials @ e.phases[2].c[:, 0]
        ik = 20j * np.pi
        amplitude = sum(
            np.array([monomials @ a.c[:, 0] for a in coeff]) / ik**m
            for m, coeff in enumerate(e.amplitudes[2])
        )
        return np.exp(ik * phase) * amplitude

    step = 1e-5
    gradient = []
    for d in np.eye(3) * step:
        gradient.append(
            (value(-2 * d) - 8 * value(-d) + 8 * value(d) - value(2 * d)) / (12 * step)
        )
    g = np.array(gradient)
    curl = np.array([g[1, 2] - g[2, 1], g[2, 0] - g[0, 2], g[0, 1] - g[1, 0]])
    assert_allclose(
        curl / (20j * np.pi * Z0),
        e.trace("transmitted", 0.1).magnetic[0],
        atol=5e-14,
        rtol=1e-10,
    )


def test_order_zero_Q_composition_matches_production_field():
    air, glass = [o.HomogeneousRegion(str(n), o.Dielectric(n)) for n in (1, 1.5)]
    surface = o.SphericalCap(10)
    interface = o.DielectricInterface(surface, air, glass)
    axis = (np.arange(32) - 16) * 8 / 32
    grid = o.CartesianGrid(axis, axis)
    source = o.PlaneWaveExpansion([[0, 0, 2 * np.pi]], [[1, 0, 0]], [0, 0, 0], 1, air)
    operator = TaylorInterfaceOperator(
        source,
        interface,
        grid,
        height=lambda x, y: 1 - (1 - x * x - y * y) ** 0.5,
        length_scale=10,
        order=0,
        aperture=3,
        max_sine=0.85,
    )
    spectrum, _ = operator.spectrum(0)
    production = o.propagate_interfaces(
        source, o.OpticalSystem((interface,), (3,), (3,)), grid
    )
    points = o.PointGrid([[0.2, 0.1, 5], [0, 0, 7]])
    assert_allclose(
        spectrum.evaluate(points).electric.values,
        production.evaluate(points).electric.values,
        atol=2e-8,
    )


def test_rejects_nonpropagating_or_grazing_local_expansion():
    with pytest.raises(ValueError, match="propagating"):
        interface_expansion(
            lambda x, y: 1 - (1 - x * x - y * y) ** 0.5,
            np.array([[0.8, 0]]),
            n_incident=1.5,
            n_transmitted=1,
        )
