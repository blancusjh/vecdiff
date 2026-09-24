"""Surface edges never cut the field; declared stops (pupils) do."""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose
from vecdiff_general.io.prescription import load_system, save_system
from vecdiff_general.propagation.high_frequency import propagate_high_frequency
from vecdiff_general.propagation.operators import PlanarASM

WAVELENGTH = 1.0
K = 2 * np.pi / WAVELENGTH


def _setup(count=64, width=16.0):
    axis = (np.arange(count) - count // 2) * width / count
    air = o.HomogeneousRegion("air", o.Dielectric(1.0))
    source = o.PlaneWaveExpansion(
        [[0, 0, K]], [[1.0, 0, 0]], [0, 0, 0], WAVELENGTH, air
    )
    return o.CartesianGrid(axis, axis), air, source


def _matched_element(air, thickness, apertures=(4.0, 4.0), stops=None):
    # Same index as air, different material identity: a genuine element.
    inside = o.HomogeneousRegion("matched glass", o.Dielectric(1.0, "matched"))
    return o.OpticalSystem(
        (
            o.DielectricInterface(o.Plane(), air, inside),
            o.DielectricInterface(o.Plane(o.Frame([0, 0, thickness])), inside, air),
        ),
        apertures=apertures,
        stops=stops,
    )


def _plane_wave_error(solution, z):
    probe = o.PointGrid(
        np.array([[x, y, z] for x in (-3.0, 0.0, 2.5) for y in (-2.0, 0.0, 3.0)])
    )
    field = solution.evaluate(probe, backend="direct").electric.values
    exact = np.zeros_like(field)
    exact[:, 0] = np.exp(1j * K * z)
    return np.linalg.norm(field - exact) / np.linalg.norm(exact)


@pytest.mark.parametrize("thickness", [0.05, 2.0])
def test_edges_do_not_cut_an_index_matched_element_is_invisible(thickness):
    grid, air, source = _setup()
    solution = propagate_high_frequency(source, _matched_element(air, thickness), grid)
    assert len(solution.exterior_spectra[0]) == 1
    assert _plane_wave_error(solution, thickness + 6.0) < 1e-12


def test_inside_and_outside_light_recombine_before_the_exit():
    grid, air, source = _setup()
    thickness = 0.7
    solution = propagate_high_frequency(source, _matched_element(air, thickness), grid)
    inner = PlanarASM().spectrum(solution.interface_spectra[0][0], thickness)
    total = (
        inner.expansion.amplitudes
        + solution.exterior_spectra[0][0].expansion.amplitudes
    )
    centre = np.argmax(abs(total[:, 0]))
    assert_allclose(total[centre, 0], np.exp(1j * K * thickness), atol=1e-12)
    assert abs(np.delete(total, centre, axis=0)).max() < 1e-12


def test_a_stop_cuts_and_matches_a_stop_on_an_unbounded_surface():
    grid, air, source = _setup()
    probe = o.PointGrid(np.array([[0.0, 0.0, 6.0], [2.0, 1.0, 7.0], [5.0, -1.0, 6.5]]))
    at_edge = propagate_high_frequency(
        source, _matched_element(air, 0.7, stops=(4.0, None)), grid
    )
    unbounded = propagate_high_frequency(
        source,
        _matched_element(air, 0.7, apertures=(None, None), stops=(4.0, None)),
        grid,
    )
    assert at_edge.exterior_spectra[0] == ()
    assert_allclose(
        at_edge.evaluate(probe, backend="direct").electric.values,
        unbounded.evaluate(probe, backend="direct").electric.values,
        atol=1e-12,
    )
    assert _plane_wave_error(at_edge, 6.7) > 0.1


def test_a_stop_at_the_exit_also_blocks_light_passing_outside():
    grid, air, source = _setup()
    exit_stop = propagate_high_frequency(
        source, _matched_element(air, 0.7, stops=(None, 4.0)), grid
    )
    # Index matched: equivalent to one pupil of radius 4 at the exit plane.
    single = propagate_high_frequency(
        source,
        _matched_element(air, 0.7, apertures=(None, None), stops=(None, 4.0)),
        grid,
    )
    probe = o.PointGrid(np.array([[0.0, 0.0, 6.0], [2.0, 1.0, 7.0], [5.0, -1.0, 6.5]]))
    assert_allclose(
        exit_stop.evaluate(probe, backend="direct").electric.values,
        single.evaluate(probe, backend="direct").electric.values,
        atol=1e-10,
    )


def test_open_systems_and_growing_edges():
    grid, air, source = _setup()
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    face = o.DielectricInterface(o.Plane(), air, glass)
    with pytest.raises(ValueError, match="open system"):
        propagate_high_frequency(source, o.OpticalSystem((face,), (4.0,)), grid)
    propagate_high_frequency(source, o.OpticalSystem((face,), (4.0,), (4.0,)), grid)
    growing = o.OpticalSystem(
        (face, o.DielectricInterface(o.Plane(o.Frame([0, 0, 1.0])), glass, air)),
        apertures=(3.0, 4.0),
    )
    with pytest.raises(ValueError, match="exceeds the entrance edge"):
        propagate_high_frequency(source, growing, grid)
    with pytest.raises(ValueError, match="stop"):
        o.OpticalSystem((face,), (4.0,), (-1.0,))


def test_prescription_round_trip_keeps_stops(tmp_path):
    _, air, _ = _setup()
    system = _matched_element(air, 0.7, stops=(3.5, None))
    path = tmp_path / "lens.json"
    save_system(system, path)
    assert load_system(path).stops == (3.5, None)
