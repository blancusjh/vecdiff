"""Coverage masks, region phase, one-sided traces and interface-cache reuse."""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose, assert_array_equal


def curved_problem(transmission="spectral", index=1.15, basis=None):
    basis = np.eye(3) if basis is None else basis
    frame = o.Frame([0, 0, 0], basis)
    axis = np.arange(-16, 16) / 8
    grid = o.CartesianGrid(axis, axis, o.Frame(-basis[:, 2], basis))
    air = o.HomogeneousRegion("before", o.Dielectric(1))
    glass = o.HomogeneousRegion("inside", o.Dielectric(index))
    after = o.HomogeneousRegion("after", o.Dielectric(1))
    front = o.FreeformSurface(
        lambda p: 0.2 * np.cos(p[..., 0]),
        lambda p: np.stack(
            (-0.2 * np.sin(p[..., 0]), np.zeros_like(p[..., 0])), axis=-1
        ),
        frame=frame,
    )
    back = o.Plane(o.Frame(0.8 * basis[:, 2], basis))
    system = o.OpticalSystem(
        (
            o.DielectricInterface(front, air, glass),
            o.DielectricInterface(back, glass, after),
        )
    )
    values = np.zeros(grid.shape + (2,), complex)
    values[..., 0] = 1
    source = o.TransverseElectricField(grid, 1, values)
    path = o.PropagationPipeline(system, transmission=transmission)
    return source, path


@pytest.mark.parametrize("transmission", ["local", "spectral"])
def test_coverage_fills_curved_band_without_repeating_transmission(
    transmission, monkeypatch
):
    source, path = curved_problem(transmission)
    calls = []
    transmit = path.transmission.transmit

    def counted(*args):
        calls.append(1)
        return transmit(*args)

    monkeypatch.setattr(path.transmission, "transmit", counted)
    kw = dict(y=0, window=(-1.5, 1.5, -0.3, 1.3), shape=(33, 25), side="plus")
    reduced = path(source, **kw)
    assert not reduced.valid.all()
    assert np.isnan(reduced.values[~reduced.valid]).all()
    assert len(calls) == 2
    path.coverage = "complete"
    complete = path(source, **kw)
    assert len(calls) == 2
    assert complete.valid.all()
    assert np.isfinite(complete.values).all()
    assert_allclose(
        complete.values[reduced.valid], reduced.values[reduced.valid], atol=1e-10
    )
    assert path.last_coverage.extended.any()
    assert_array_equal(path.last_coverage.computed, complete.valid)
    # In the curved band, upstream points must still use the incident field.
    points = complete.grid.points
    upstream = path.last_coverage.extended & (path.last_coverage.regions == 0)
    assert upstream.any()
    assert_allclose(
        complete.Ex[upstream],
        np.exp(2j * np.pi * (points[..., 2][upstream] + 1)),
        atol=2e-12,
    )
    path.coverage = "reduced"
    again = path(source, **kw)
    assert_array_equal(again.valid, reduced.valid)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "basis", [np.eye(3), np.array([[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]])]
)
def test_complete_curved_regions_match_independent_homogeneous_plane_wave(basis):
    source, path = curved_problem(index=1, basis=basis)
    path.coverage = "complete"
    local = np.array(
        [
            [0, 0, -0.1],
            [0, 0, 0.19999],
            [0, 0, 0.20001],
            [1, 0, 0.79],
            [1, 0, 0.80001],
            [0, 0, 1.2],
        ]
    )
    field = path(source, o.PointGrid(local @ basis.T))
    expected = np.exp(2j * np.pi * (local[:, 2] + 1))[:, None] * basis[:, 0]
    assert_allclose(field.values, expected, atol=3e-12)
    assert_array_equal(path.last_coverage.regions, [0, 0, 1, 1, 2, 2])
    assert not path.last_coverage.extended.any()  # Equal-index faces need no exclusion.


@pytest.mark.parametrize("coverage", ["reduced", "complete"])
def test_flat_interface_sided_limits_match_analytic_fresnel(coverage):
    source, path = curved_problem()
    air, glass = path.interfaces[0].minus, path.interfaces[0].plus
    path = o.PropagationPipeline(
        o.OpticalSystem((o.DielectricInterface(o.Plane(), air, glass),)),
        coverage=coverage,
    )
    with pytest.raises(ValueError, match="side"):
        path(source, z=0, shape=(3, 3))
    minus = path(source, z=0, side="minus", shape=(3, 3))
    plus = path(source, z=0, side="plus", shape=(3, 3))
    assert minus.valid.all() and plus.valid.all()
    assert_allclose(minus.Ex, 1, atol=1e-12)
    assert_allclose(plus.Ex, 2 / 2.15, atol=1e-12)


def test_all_masked_target_does_not_solve_interfaces(monkeypatch):
    source, path = curved_problem()

    def forbidden(*args):
        raise AssertionError("No valid observation requires transmission")

    monkeypatch.setattr(path.transmission, "transmit", forbidden)
    field = path(source, z=0.1, window=(-0.1, 0.1, -0.1, 0.1), shape=(3, 3))
    assert not field.valid.any()
    assert not path.last_solution.interactions


def test_invalid_coverage_and_masked_incident_fields_are_explicit():
    with pytest.raises(ValueError, match="coverage"):
        o.PropagationPipeline(coverage="regional")
    source, path = curved_problem()
    grid = source.grid
    mask = np.ones(grid.shape, bool)
    mask[0, 0] = False
    field = o.ElectricField(grid, 1, np.ones(grid.shape + (3,)), valid=mask)
    assert not field.valid.flags.writeable and not field.values.flags.writeable
    with pytest.raises(ValueError, match="uncomputed"):
        path(field, z=2)
    with pytest.raises(ValueError, match="finite"):
        o.ElectricField(grid, 1, np.full(grid.shape + (3,), np.nan))
