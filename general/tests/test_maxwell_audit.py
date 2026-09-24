"""Independent derivative and boundary diagnostics, including negative controls."""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose
from vecdiff_general.media.medium import Z0
from vecdiff_general.observables import coherent_sum
from vecdiff_general.observables.maxwell import (
    ElectricFieldJet,
    interface_l2_residuals,
    plane_wave_jet,
)
from vecdiff_general.propagation.experiments.asymptotic import wkb_electric_jet
from vecdiff_general.propagation.operators import fresnel_transmission


def test_modal_jet_independent_curl_and_bulk_with_counterpropagating_modes():
    region = o.HomogeneousRegion("test", o.Dielectric(1.5))
    k = 3 * np.pi
    directions = np.array([[0.6, 0, 0.8], [0, 0, -1], [0, 0.8, 0.6]])
    a = np.cross(directions, [1, 2j, 3])
    source = o.PlaneWaveExpansion(k * directions, a, [0, 0, 0], 1.0, region)
    grid = o.PointGrid(np.random.default_rng(43).normal(size=(13, 3)))
    jet = plane_wave_jet(source, grid, chunk_size=5)
    exact = source.evaluate(grid)
    assert_allclose(jet.field.electric.values, exact.electric.values, atol=1e-13)
    assert_allclose(jet.field.magnetic, exact.magnetic, atol=1e-15)
    assert max(jet.bulk_residuals(1.5, np.ones(13)).values()) < 2e-15
    step = 1e-4
    for axis in range(3):

        def shifted(multiple):
            return source.evaluate(
                o.PointGrid(grid.points + multiple * step * np.eye(3)[axis])
            ).electric.values

        fd = (shifted(-2) - 8 * shifted(-1) + 8 * shifted(1) - shifted(2)) / (12 * step)
        assert_allclose(jet.gradient[:, axis], fd, atol=5e-11)


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_wkb_full_curl_retains_last_amplitude_derivative(order):
    # E_x=(1 + x/(ik)^N) exp(ik z). Its extra curl is zero, but
    # E_y=x/(ik)^N supplies a NONZERO final curl in H_z at order N+1.
    grid = o.PointGrid([[0.2, 0.1, 0.4], [-0.3, 0.4, 0.9]])
    shape = (order + 1, 2)
    a = np.zeros(shape + (3,), complex)
    da = np.zeros(shape + (3, 3), complex)
    dda = np.zeros(shape + (3, 3, 3), complex)
    a[0, :, 0] = 1
    a[order, :, 1] = grid.points[:, 0]
    da[order, :, 0, 1] = 1
    jet = wkb_electric_jet(
        grid,
        1,
        grid.points[:, 2],
        np.tile([0, 0, 1], (2, 1)),
        np.zeros((2, 3, 3)),
        a,
        da,
        dda,
        order=order,
    )
    ik = 2j * np.pi
    missing = np.exp(ik * grid.points[:, 2]) / ik ** (order + 1)
    assert_allclose(Z0 * jet.field.magnetic[:, 2], missing, atol=1e-15)
    # Complete curl-curl includes grad(div E); a polynomial longitudinal test
    # below separately distinguishes it from the vector Helmholtz operator.
    assert max(jet.bulk_residuals(1, np.ones(2)).values()) < 1e-15


def test_nonconstant_phase_and_amplitude_hessian_against_polynomial_derivatives():
    grid = o.PointGrid([[0.1, -0.2, 0.3], [0.4, 0.2, -0.1]])
    x, y, z = grid.points.T
    phase = z + 0.3 * x * y
    w = np.column_stack((0.3 * y, 0.3 * x, np.ones(2)))
    ds = np.zeros((2, 3, 3))
    ds[:, 0, 1] = ds[:, 1, 0] = 0.3
    a = np.zeros((1, 2, 3), complex)
    a[0, :, 0] = 1 + x**2 + y * z
    da = np.zeros((1, 2, 3, 3), complex)
    da[0, :, :, 0] = np.column_stack((2 * x, z, y))
    dda = np.zeros((1, 2, 3, 3, 3), complex)
    dda[0, :, 0, 0, 0] = 2
    dda[0, :, 1, 2, 0] = dda[0, :, 2, 1, 0] = 1
    jet = wkb_electric_jet(grid, 1, phase, w, ds, a, da, dda, order=0)

    def field(p):
        xx, yy, zz = p.T
        return (1 + xx**2 + yy * zz) * np.exp(2j * np.pi * (zz + 0.3 * xx * yy))

    step = 2e-4
    # Off-diagonal derivatives exercise tensor-axis ordering.
    for i in range(3):
        for j in range(3):
            di, dj = step * np.eye(3)[i], step * np.eye(3)[j]
            fd = (
                field(grid.points + di + dj)
                - field(grid.points + di - dj)
                - field(grid.points - di + dj)
                + field(grid.points - di - dj)
            ) / (4 * step**2)
            assert_allclose(jet.hessian[:, i, j, 0], fd, atol=3e-5)


def test_curl_curl_residual_distinguishes_longitudinal_helmholtz_solution():
    grid = o.PointGrid([[0, 0, 0]])
    e = np.array([[0, 0, 1]], complex)
    grad = np.zeros((1, 3, 3), complex)
    hess = np.zeros((1, 3, 3, 3), complex)
    grad[0, 2, 2] = 2j * np.pi
    hess[0, 2, 2, 2] = -((2 * np.pi) ** 2)
    residuals = ElectricFieldJet(
        o.ElectricField(grid, 1, e), grad, hess
    ).bulk_residuals(1, np.ones(1))
    assert residuals["helmholtz_E"] == 0
    assert_allclose(residuals["curl_curl_E"], 1)
    assert_allclose(residuals["divergence_E"], 1)


def test_area_boundary_audit_and_bad_magnetic_negative_control():
    air, glass = [o.HomogeneousRegion(str(n), o.Dielectric(n)) for n in (1, 1.5)]
    direction = np.array([0.6, 0, 0.8])
    plane = o.CartesianGrid(np.arange(4.0), np.arange(4.0))
    q = o.SurfaceTransport(plane, o.Plane(), air, 1)
    source = o.PlaneWaveExpansion(
        [2 * np.pi * direction], [[0, 1, 0]], [0, 0, 0], 1, air
    )
    incident = source.evaluate(q.grid)
    encounter = fresnel_transmission(
        incident, o.DielectricInterface(o.Plane(), air, glass), direction
    )
    minus = coherent_sum([incident, encounter.reflected])
    residuals = interface_l2_residuals(
        minus, encounter.transmitted, incident, q.grid, n_minus=1, n_plus=1.5
    )
    assert max(residuals.values()) < 1e-14
    wrong = o.ElectromagneticField(
        encounter.transmitted.electric, np.zeros_like(encounter.transmitted.magnetic)
    )
    failed = interface_l2_residuals(
        minus, wrong, incident, q.grid, n_minus=1, n_plus=1.5
    )
    assert failed["tangential_E"] < 1e-14
    assert failed["tangential_H"] > 0.5
    assert failed["normal_flux_L2"] > 0.5


def test_area_audit_rejects_different_physical_points():
    grid = o.SurfaceGrid([[0, 0, 0]], [[0, 0, 1]], [1])
    shifted = o.PointGrid([[0, 0, 1]])
    a = o.ElectromagneticField(o.ElectricField(grid, 1, [[1, 0, 0]]), [[0, 1 / Z0, 0]])
    b = o.ElectromagneticField(
        o.ElectricField(shifted, 1, [[1, 0, 0]]), [[0, 1 / Z0, 0]]
    )
    with pytest.raises(ValueError, match="identical physical points"):
        interface_l2_residuals(a, b, a, grid, n_minus=1, n_plus=1)
