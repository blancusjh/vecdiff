"""Coverage masks, region phase, one-sided traces and interface-cache reuse."""

import numpy as np
import pytest
import vecdiff as o
from numpy.testing import assert_allclose, assert_array_equal

bd, host = o.backend.bd, o.backend.to_numpy


def curved_problem(transmission="spectral", index=1.15, basis=None):
    basis = np.eye(3) if basis is None else basis
    frame = o.geometry.Frame([0, 0, 0], basis)
    axis = np.arange(-16, 16) / 8
    grid = o.sampling.CartesianGrid(axis, axis, o.geometry.Frame(-basis[:, 2], basis))
    air = o.media.HomogeneousRegion("before", o.media.Dielectric(1))
    glass = o.media.HomogeneousRegion("inside", o.media.Dielectric(index))
    after = o.media.HomogeneousRegion("after", o.media.Dielectric(1))
    front = o.surfaces.FreeformSurface(
        lambda p: 0.2 * np.cos(p[..., 0]),
        lambda p: np.stack(
            (-0.2 * np.sin(p[..., 0]), np.zeros_like(p[..., 0])), axis=-1
        ),
        frame=frame,
    )
    back = o.surfaces.Plane(o.geometry.Frame(0.8 * basis[:, 2], basis))
    system = o.system.OpticalSystem(
        (
            o.system.DielectricInterface(front, air, glass),
            o.system.DielectricInterface(back, glass, after),
        )
    )
    values = np.zeros(grid.shape + (2,), complex)
    values[..., 0] = 1
    source = o.fields.TransverseElectricField(grid, 1, values)
    return source, o.alternatives.WavePropagator(system, transmission=transmission)


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
    solution = path(source)
    kw = dict(y=0, window=(-1.5, 1.5, -0.3, 1.3), shape=(33, 25), side="plus")
    reduced = solution.electric(**kw)
    assert not reduced.valid.all()
    assert np.isnan(reduced.values[~reduced.valid]).all()
    assert len(calls) == 2
    complete = solution.electric(**kw, coverage="complete")
    assert len(calls) == 2
    assert complete.valid.all()
    assert np.isfinite(complete.values).all()
    assert_allclose(
        complete.values[reduced.valid], reduced.values[reduced.valid], atol=1e-10
    )
    coverage = solution.coverage(**kw, coverage="complete")
    assert coverage.extended.any()
    assert_array_equal(coverage.computed, complete.valid)
    # In the curved band, upstream points must still use the incident field.
    points = complete.grid.points
    upstream = coverage.extended & (coverage.regions == 0)
    assert upstream.any()
    assert_allclose(
        complete.Ex[upstream],
        np.exp(2j * np.pi * (points[..., 2][upstream] + 1)),
        atol=2e-12,
    )
    again = solution.electric(**kw)
    assert_array_equal(again.valid, reduced.valid)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "basis", [np.eye(3), np.array([[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]])]
)
def test_complete_curved_regions_match_independent_homogeneous_plane_wave(basis):
    source, path = curved_problem(index=1, basis=basis)
    solution = path(source)
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
    target = o.sampling.PointGrid(local @ basis.T)
    field = solution.electric(target, coverage="complete")
    expected = np.exp(2j * np.pi * (local[:, 2] + 1))[:, None] * basis[:, 0]
    assert_allclose(field.values, expected, atol=3e-12)
    coverage = solution.coverage(target, coverage="complete")
    assert_array_equal(coverage.regions, [0, 0, 1, 1, 2, 2])
    assert not coverage.extended.any()  # Equal-index faces need no exclusion.


@pytest.mark.parametrize("coverage", ["reduced", "complete"])
def test_flat_interface_sided_limits_match_analytic_fresnel(coverage):
    source, path = curved_problem()
    air, glass = path.interfaces[0].minus, path.interfaces[0].plus
    solution = o.alternatives.WavePropagator(
        o.system.OpticalSystem(
            (o.system.DielectricInterface(o.surfaces.Plane(), air, glass),)
        )
    )(source)
    with pytest.raises(ValueError, match="side"):
        solution.electric(z=0, shape=(3, 3), coverage=coverage)
    minus = solution.electric(z=0, side="minus", shape=(3, 3), coverage=coverage)
    plus = solution.electric(z=0, side="plus", shape=(3, 3), coverage=coverage)
    assert minus.valid.all() and plus.valid.all()
    assert_allclose(minus.Ex, 1, atol=1e-12)
    assert_allclose(plus.Ex, 2 / 2.15, atol=1e-12)


def test_a_target_inside_the_curved_band_is_all_masked():
    source, path = curved_problem()
    solution = path(source)
    field = solution.field(z=0.1, window=(-0.1, 0.1, -0.1, 0.1), shape=(3, 3))
    assert not field.electric.valid.any()
    assert np.isnan(field.magnetic).all()


def test_invalid_coverage_and_masked_incident_fields_are_explicit():
    source, path = curved_problem()
    with pytest.raises(ValueError, match="coverage"):
        path(source).electric(z=2, coverage="regional")
    grid = source.grid
    mask = np.ones(grid.shape, bool)
    mask[0, 0] = False
    field = o.fields.ElectricField(grid, 1, np.ones(grid.shape + (3,)), valid=mask)
    if o.backend.get_backend() == "CPU":  # CuPy arrays cannot be made read-only
        assert not field.valid.flags.writeable and not field.values.flags.writeable
    with pytest.raises(ValueError, match="uncomputed"):
        path(field)
    with pytest.raises(ValueError, match="finite"):
        o.fields.ElectricField(grid, 1, np.full(grid.shape + (3,), np.nan))
