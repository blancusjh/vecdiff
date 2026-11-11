"""The method's entry point: Propagator(system)(light) and its Solution."""

import numpy as np
import pytest
import vecdiff as o
from numpy.testing import assert_allclose

host = o.backend.to_numpy
F, S, P = o.fields, o.surfaces, o.propagation
Frame = o.geometry.Frame
AIR = o.media.HomogeneousRegion("air", o.media.Dielectric(1.0))
GLASS = o.media.HomogeneousRegion("glass", o.media.Dielectric(1.5))
GLASS2 = o.media.HomogeneousRegion("glass, behind", o.media.Dielectric(1.5))


def converging(radius=10.0):
    return F.FocusedBeam(
        F.profiles.Gaussian(0.3),
        F.polarization.x,
        focal_length=1.0,
        max_sine=0.4,
        wavelength0=1.0,
        region=AIR,
        frame=Frame([0, 0, 1.5 * radius]),
    )


def system(radius=10.0):
    cap = S.SphericalCap(radius, Frame([0, 0, -radius]))
    return (
        o.system.DielectricInterface(cap, AIR, GLASS),
        o.system.DielectricInterface(S.Plane(Frame([0, 0, 2.0])), GLASS, GLASS2),
    )


def points(z, count=5):
    x = np.linspace(-1.0, 1.0, count)
    return o.sampling.PointGrid(np.c_[x, 0.4 * x, np.full(count, z)])


def test_solution_holds_one_spectrum_per_region_and_one_report_per_refraction():
    solution = P.Propagator(system())(converging())
    assert [s.region for s in solution.spectra] == [AIR, GLASS, GLASS2]
    assert all(isinstance(s, F.Spectrum) for s in solution.spectra)
    assert solution.spectrum is solution.spectra[-1]
    assert len(solution.reports) == len(solution.legs) == 1
    assert isinstance(solution.reports[0], o.transport.TransportReport)
    summary = str(solution)
    assert "surface 1 (SphericalCap, air -> glass)" in summary
    assert "unsupported" in summary and "orders (-1, 1)" in summary


def test_field_gives_e_and_h_together_and_electric_is_its_electric_part():
    solution = P.Propagator(system())(converging())
    target = points(6.0)
    field = solution.field(target)
    assert isinstance(field, F.ElectromagneticField) and field.electric.valid.all()
    expected = solution.spectrum.evaluate(target)
    scale = abs(host(expected.electric.values)).max()
    assert_allclose(
        host(field.electric.values), host(expected.electric.values), atol=1e-13 * scale
    )
    assert_allclose(host(field.magnetic), host(expected.magnetic), atol=1e-13 * scale)
    assert_allclose(
        host(solution.electric(target).values),
        host(field.electric.values),
        atol=1e-12 * scale,
    )


def test_field_takes_every_point_from_the_spectrum_of_its_region():
    beam = converging()
    solution = P.Propagator(system())(beam)
    before, inside, behind = points(-12.0, 3), points(1.0, 3), points(4.0, 3)
    together = o.sampling.PointGrid(
        np.concatenate([host(g.points) for g in (before, inside, behind)])
    )
    field = host(solution.electric(together).values)
    expected = np.concatenate(
        [
            host(beam.evaluate(before).electric.values),
            host(solution.spectra[1].electric(inside).values),
            host(solution.spectra[2].electric(behind).values),
        ]
    )
    assert_allclose(field, expected, atol=1e-12 * abs(expected).max())
    report = solution.coverage(together)
    assert host(report.regions).tolist() == [0] * 3 + [1] * 3 + [2] * 3
    assert report.sampling["mode"] == "explicit"


def test_propagator_takes_only_the_options_of_the_method():
    for retired in ("transport", "transmission", "max_sine", "axisymmetric"):
        with pytest.raises(TypeError, match=retired):
            P.Propagator(system(), **{retired: None})
    with pytest.raises(ValueError, match="realization must be"):
        P.Propagator(system(), realization="wave")
    with pytest.raises(ValueError, match="one reference-plane offset per surface"):
        P.Propagator(system(), planes=(0.0,))
    with pytest.raises(TypeError, match="light must be"):
        P.Propagator(system())(converging().plane_waves(5.0))
    axis = (np.arange(16) - 8) * 0.5
    grid = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, -12.0]))
    values = np.zeros(grid.shape + (2,), complex)
    values[..., 0] = 1
    sampled = F.TransverseElectricField(grid, 1.0, values)
    with pytest.raises(ValueError, match="carrier=.*WavePropagator"):
        P.Propagator(system())(sampled)
    with pytest.raises(ValueError, match="incident region"):
        P.Propagator(system())(F.angular_spectrum(sampled, GLASS), carrier=None)
