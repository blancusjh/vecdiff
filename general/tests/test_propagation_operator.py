"""Physical and API contracts for field/system/target propagation."""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose


def problem(transmission="spectral"):
    air = o.HomogeneousRegion("air", o.Dielectric(1))
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    axis = np.arange(-8, 8) * 0.25
    grid = o.CartesianGrid(axis, axis, o.Frame([0, 0, -1]))
    values = np.zeros(grid.shape + (2,), complex)
    values[..., 0] = 1
    source = o.TransverseElectricField(grid, 1, values)
    interfaces = [
        o.DielectricInterface(o.Plane.z(0), air, glass),
        o.DielectricInterface(o.Plane.z(2), glass, air),
    ]
    return source, o.PropagationPipeline(
        o.OpticalSystem(interfaces), transmission=transmission
    )


@pytest.mark.parametrize("mode", ["local", "spectral"])
def test_piecewise_planar_fields_and_interface_sides(mode):
    source, propagate = problem(mode)
    z = np.array([-0.5, 0.5, 3.0])
    grid = o.PointGrid(np.stack((np.zeros(3), np.zeros(3), z), axis=-1))
    field = propagate(source, grid)
    expected = np.array(
        [
            np.exp(2j * np.pi * 0.5),
            0.8 * np.exp(2j * np.pi * (1 + 1.5 * 0.5)),
            0.96 * np.exp(2j * np.pi * (1 + 1.5 * 2 + 1)),
        ]
    )
    assert isinstance(field, o.ElectricField)
    assert_allclose(field.Ex, expected, atol=2e-12)
    assert_allclose(field.values[..., 1:], 0, atol=1e-13)
    with pytest.raises(ValueError, match="side"):
        propagate(source, z=0, shape=(3, 3))
    minus = propagate(source, z=0, shape=(3, 3), side="minus")
    plus = propagate(source, z=0, shape=(3, 3), side="plus")
    assert_allclose(plus.Ex, 0.8 * minus.Ex, atol=1e-12)


def test_target_stops_before_unneeded_interface_and_reuses_source(monkeypatch):
    source, propagate = problem()
    calls = []
    original = propagate.transmission.transmit

    def counted(*args):
        calls.append(args[1])
        return original(*args)

    monkeypatch.setattr(propagate.transmission, "transmit", counted)
    propagate(source, z=0.5, shape=(4, 4))
    assert len(calls) == len(propagate._operators) == 1
    propagate(source, z=1, shape=(4, 4))
    assert len(calls) == 1
    propagate(source, z=3, shape=(4, 4))
    assert len(calls) == 2
    propagate(source, z=-0.5, shape=(4, 4))
    assert len(calls) == 2
    propagate.clear_cache()
    propagate(source, z=3, shape=(4, 4))
    assert len(calls) == 4


def test_target_equivalence_and_source_owned_by_system():
    source, original = problem()
    system = o.OpticalSystem(original.interfaces, incident_field=source)
    propagate = o.PropagationPipeline(system)
    kw = dict(window=(-1, 1, -1, 1), shape=(7, 9))
    a = propagate(z=3, **kw)
    b = propagate(o.Plane.z(3), **kw)
    c = propagate(target_surface=o.Plane.z(3).sample(u=a.grid.u, v=a.grid.v))
    assert_allclose(a.values, b.values, atol=1e-13)
    assert_allclose(a.values, c.values, atol=1e-13)
    assert a.grid.shape == (7, 9)


@pytest.mark.parametrize("coordinate", ["x", "y", "z"])
def test_coordinate_planes_and_auto_sampling(coordinate):
    source, propagate = problem()
    field = propagate(source, **{coordinate: 3}, window=(-0.4, 0.4, 3, 4))
    assert_allclose(field.grid.points[..., "xyz".index(coordinate)], 3)
    assert max(field.grid.spacing) <= 0.4 / 1.5 * (1 + 1e-12)
    assert propagate.last_sampling["mode"] == "automatic"


def test_curved_and_rotated_targets_match_analytic_homogeneous_wave():
    source, old = problem()
    region = old.interfaces[0].minus
    propagate = o.PropagationPipeline(o.OpticalSystem(incident_region=region))
    tilted = o.Plane(o.Frame([0, 0, 4], [[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]]))
    curved = o.SphericalCap(20, o.Frame([0, 0, 4]))
    for surface in [tilted, curved]:
        result = propagate(source, surface, window=(-1, 1, -1, 1), shape=(9, 9))
        assert_allclose(
            result.Ex, np.exp(2j * np.pi * (result.grid.points[..., 2] + 1)), atol=1e-11
        )


def test_electric_output_does_not_evaluate_magnetic_amplitudes(monkeypatch):
    source, propagate = problem()
    propagate.solve(source)

    def forbidden(self):
        raise AssertionError("electric-only observation requested H")

    monkeypatch.setattr(
        o.PlaneWaveExpansion, "magnetic_amplitudes", property(forbidden)
    )
    propagate(source, z=3, shape=(4, 4))


@pytest.mark.parametrize("angle_mode", [0, 1])
def test_local_oblique_plane_wave_matches_exact_fresnel(angle_mode):
    source, propagate = problem("local")
    source = o.TransverseElectricField(
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
    spectrum = o.angular_spectrum(source, propagate.interfaces[0].minus)
    interface = propagate.interfaces[0]
    transport = o.SurfaceTransport(
        o.CartesianGrid(source.grid.u, source.grid.v),
        interface.surface,
        interface.minus,
        1,
    )
    shifted = o.PlaneTransport().spectrum(spectrum, 1)
    actual = propagate.transmission.transmit(shifted, interface, transport)
    exact = o.scatter_plane(
        spectrum,
        o.PlanarInterface(interface.surface.frame, interface.minus, interface.plus),
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
    grid = o.CartesianGrid(axis, axis, source.grid.frame)
    x = grid.points[..., 0]
    values = np.stack((np.zeros_like(x), np.cos(2 * np.pi * 5 * x / 8)), axis=-1)
    with pytest.raises(ValueError, match="single-direction approximation"):
        propagate(o.TransverseElectricField(grid, 1, values), z=3, shape=(4, 4))


def test_zero_local_field_and_compatibility_alias():
    source, propagate = problem("local")
    zero = o.TransverseElectricField(source.grid, 1, np.zeros_like(source.values))
    assert_allclose(propagate(zero, z=3, shape=(3, 3)).values, 0)
    assert o.DirectionResolvedFresnel is o.SpectralFresnel
    with pytest.raises(ValueError, match="transmission"):
        o.PropagationPipeline(transmission="unknown")


@pytest.mark.parametrize("ka", [100, 1000, 10000])
def test_surface_electric_jet_against_analytic_modal_derivatives(ka):
    """Large-size two-mode transfer check, not a full macroscopic benchmark."""
    radius = ka / (2 * np.pi)
    axis = (np.arange(64) - 32) * 0.2 * radius / 64
    grid = o.CartesianGrid(axis, axis)
    x = grid.points[..., 0]
    values = np.stack(
        (1 + 0.2 * np.exp(2j * np.pi * x / (0.2 * radius)), 0.3j * np.ones_like(x)),
        axis=-1,
    )
    air = o.HomogeneousRegion("air", o.Dielectric(1))
    spectrum = o.angular_spectrum(o.TransverseElectricField(grid, 1, values), air)
    transport = o.SurfaceTransport(grid, o.SphericalCap(radius), air, 1)
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
