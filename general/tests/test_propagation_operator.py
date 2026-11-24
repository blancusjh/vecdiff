"""Physical and API contracts for field/system/target propagation."""

import numpy as np
import pytest
import vecdiff as o
from numpy.testing import assert_allclose


def problem(transmission="spectral"):
    air = o.media.HomogeneousRegion("air", o.media.Dielectric(1))
    glass = o.media.HomogeneousRegion("glass", o.media.Dielectric(1.5))
    axis = np.arange(-8, 8) * 0.25
    grid = o.sampling.CartesianGrid(axis, axis, o.geometry.Frame([0, 0, -1]))
    values = np.zeros(grid.shape + (2,), complex)
    values[..., 0] = 1
    source = o.fields.TransverseElectricField(grid, 1, values)
    interfaces = [
        o.system.Refraction(o.surfaces.Plane.z(0), air, glass),
        o.system.Refraction(o.surfaces.Plane.z(2), glass, air),
    ]
    return source, o.alternatives.WavePropagator(
        o.system.OpticalSystem(interfaces), transmission=transmission
    )


@pytest.mark.parametrize("mode", ["local", "spectral"])
def test_piecewise_planar_fields_and_interface_sides(mode):
    source, propagate = problem(mode)
    z = np.array([-0.5, 0.5, 3.0])
    grid = o.sampling.PointGrid(np.stack((np.zeros(3), np.zeros(3), z), axis=-1))
    solution = propagate(source)
    field = solution.electric(grid)
    expected = np.array(
        [
            np.exp(2j * np.pi * 0.5),
            0.8 * np.exp(2j * np.pi * (1 + 1.5 * 0.5)),
            0.96 * np.exp(2j * np.pi * (1 + 1.5 * 2 + 1)),
        ]
    )
    assert isinstance(field, o.fields.ElectricField)
    assert_allclose(field.Ex, expected, atol=2e-12)
    assert_allclose(field.values[..., 1:], 0, atol=1e-13)
    with pytest.raises(ValueError, match="side"):
        solution.electric(z=0, shape=(3, 3))
    minus = solution.electric(z=0, shape=(3, 3), side="minus")
    plus = solution.electric(z=0, shape=(3, 3), side="plus")
    assert_allclose(plus.Ex, 0.8 * minus.Ex, atol=1e-12)


def test_a_solution_serves_every_target_and_its_transports_are_reused(monkeypatch):
    source, propagate = problem()
    calls = []
    original = propagate.transmission.transmit

    def counted(*args):
        calls.append(args[1])
        return original(*args)

    monkeypatch.setattr(propagate.transmission, "transmit", counted)
    solution = propagate(source)
    assert len(calls) == len(solution.legs) == 2
    for z in (0.5, 1, 3, -0.5):
        solution.electric(z=z, shape=(4, 4))
    assert len(calls) == 2
    again = propagate(source)
    assert len(calls) == 4 and again.legs is solution.legs


def test_a_target_named_by_coordinate_surface_or_grid_is_one_target():
    source, original = problem()
    solution = original(source)
    kw = dict(window=(-1, 1, -1, 1), shape=(7, 9))
    a = solution.electric(z=3, **kw)
    b = solution.electric(o.surfaces.Plane.z(3), **kw)
    c = solution.electric(o.surfaces.Plane.z(3).sample(u=a.grid.u, v=a.grid.v))
    assert_allclose(a.values, b.values, atol=1e-13)
    assert_allclose(a.values, c.values, atol=1e-13)
    assert a.grid.shape == (7, 9)


@pytest.mark.parametrize("coordinate", ["x", "y", "z"])
def test_coordinate_planes_and_auto_sampling(coordinate):
    source, propagate = problem()
    solution = propagate(source)
    where = dict({coordinate: 3}, window=(-0.4, 0.4, 3, 4))
    field = solution.electric(**where)
    assert_allclose(field.grid.points[..., "xyz".index(coordinate)], 3)
    assert max(field.grid.spacing) <= 0.4 / 1.5 * (1 + 1e-12)
    assert solution.coverage(**where).sampling["mode"] == "automatic"


def test_curved_and_rotated_targets_match_analytic_homogeneous_wave():
    source, old = problem()
    region = old.elements[0].before
    solution = o.propagation.Propagator(o.system.OpticalSystem(medium=region))(source)
    tilted = o.surfaces.Plane(
        o.geometry.Frame([0, 0, 4], [[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]])
    )
    curved = o.surfaces.SphericalCap(20, o.geometry.Frame([0, 0, 4]))
    for surface in [tilted, curved]:
        result = solution.electric(surface, window=(-1, 1, -1, 1), shape=(9, 9))
        assert_allclose(
            result.Ex, np.exp(2j * np.pi * (result.grid.points[..., 2] + 1)), atol=1e-11
        )


def test_electric_output_does_not_evaluate_magnetic_amplitudes(monkeypatch):
    source, propagate = problem()
    solution = propagate(source)

    def forbidden(self):
        raise AssertionError("electric-only observation requested H")

    monkeypatch.setattr(
        o.fields.PlaneWaveExpansion, "magnetic_amplitudes", property(forbidden)
    )
    solution.electric(z=3, shape=(4, 4))


@pytest.mark.parametrize("angle_mode", [0, 1])
def test_local_oblique_plane_wave_matches_exact_fresnel(angle_mode):
    source, propagate = problem("local")
    source = o.fields.TransverseElectricField(
        source.grid,
        1,
        np.stack(
            (
                np.exp(2j * np.pi * angle_mode * source.grid.points[..., 0] / 4),
                0.3j * np.exp(2j * np.pi * angle_mode * source.grid.points[..., 0] / 4),
            ),
            axis=-1,
        ),
    )
    spectrum = o.fields.angular_spectrum(source, propagate.elements[0].before)
    interface = propagate.elements[0]
    transport = o.alternatives.SurfaceTransport(
        o.sampling.CartesianGrid(source.grid.u, source.grid.v),
        interface.surface,
        interface.before,
        1,
    )
    shifted = o.propagation.PlaneTransport().spectrum(spectrum, 1)
    actual = propagate.transmission.transmit(shifted, interface, transport)
    exact = o.alternatives.scatter_plane(
        spectrum,
        o.system.PlanarInterface(
            interface.surface.frame, interface.before, interface.after
        ),
    )
    expected = exact.evaluate(transport.grid, side="plus")
    assert_allclose(actual.field.electric.values, expected.electric.values, atol=2e-12)
    assert_allclose(actual.field.magnetic, expected.magnetic, atol=2e-14)
    assert actual.report.eikonal_relative_residual < 1e-12


def test_local_detects_counterpropagating_phase_cancellation():
    source, propagate = problem("local")
    # Two forward beams at +-38.7 degrees have a common summed phase whose
    # gradient magnitude does not satisfy the medium's eikonal equation.
    axis = np.arange(-16, 16) * 0.25
    grid = o.sampling.CartesianGrid(axis, axis, source.grid.frame)
    x = grid.points[..., 0]
    values = np.stack((np.zeros_like(x), np.cos(2 * np.pi * 5 * x / 8)), axis=-1)
    with pytest.raises(ValueError, match="single-direction approximation"):
        propagate(o.fields.TransverseElectricField(grid, 1, values))


def test_zero_local_field_is_transmitted_and_an_unknown_treatment_refused():
    source, propagate = problem("local")
    zero = o.fields.TransverseElectricField(
        source.grid, 1, np.zeros_like(source.values)
    )
    assert_allclose(propagate(zero).electric(z=3, shape=(3, 3)).values, 0)
    with pytest.raises(ValueError, match="transmission"):
        o.alternatives.WavePropagator(transmission="unknown")


@pytest.mark.parametrize("ka", [100, 1000, 10000])
def test_surface_electric_jet_against_analytic_modal_derivatives(ka):
    """Large-size two-mode transfer check, not a full macroscopic benchmark."""
    radius = ka / (2 * np.pi)
    axis = (np.arange(64) - 32) * 0.2 * radius / 64
    grid = o.sampling.CartesianGrid(axis, axis)
    x = grid.points[..., 0]
    values = np.stack(
        (1 + 0.2 * np.exp(2j * np.pi * x / (0.2 * radius)), 0.3j * np.ones_like(x)),
        axis=-1,
    )
    air = o.media.HomogeneousRegion("air", o.media.Dielectric(1))
    spectrum = o.fields.angular_spectrum(
        o.fields.TransverseElectricField(grid, 1, values), air
    )
    transport = o.alternatives.SurfaceTransport(
        grid, o.surfaces.SphericalCap(radius), air, 1
    )
    electric, derivative = transport.electric_jet(spectrum)
    modes = spectrum.expansion
    keep = np.any(modes.amplitudes != 0, axis=-1)
    k, a = modes.wavevectors[keep], modes.amplitudes[keep]
    phase = np.exp(
        1j * np.einsum("...j,mj->...m", transport.grid.points - modes.origin, k)
    )
    expected = np.einsum("...m,mc->...c", phase, a)
    gradient = np.einsum("...m,mc,mj->...cj", phase, a, 1j * k)
    assert_allclose(electric, expected, atol=1e-9)
    assert_allclose(derivative, gradient, atol=1e-8)
