"""Every plane-wave representation of a region's field is one Spectrum."""

import numpy as np
import pytest
import vecdiff as o
from numpy.testing import assert_allclose
from vecdiff.fields.lattice_spectrum import lattice_spectrum

host = o.backend.to_numpy
F, S = o.fields, o.surfaces
Frame = o.geometry.Frame
AIR = o.media.HomogeneousRegion("air", o.media.Dielectric(1.0))
GLASS = o.media.HomogeneousRegion("glass", o.media.Dielectric(1.5))
GLASS2 = o.media.HomogeneousRegion("glass, next region", o.media.Dielectric(1.5))


def gaussian_lattice():
    axis = (np.arange(32) - 16) * 0.4
    grid = o.sampling.CartesianGrid(axis, axis, Frame([0.2, -0.1, 1.0]))
    x, y = np.meshgrid(axis, axis)
    envelope = np.exp(-(x**2 + y**2) / 4**2)
    return grid, np.stack((envelope, 0.3j * envelope), axis=-1)


def plane_spectrum():
    grid, values = gaussian_lattice()
    field = F.TransverseElectricField(grid, 1.0, values)
    return F.angular_spectrum(field, GLASS, evanescent="discard")


def lattice():
    grid, values = gaussian_lattice()
    return lattice_spectrum(grid, GLASS, 1.0, values, cutoff=0.995)


def beam():
    gaussian = F.PlaneBeam(
        F.profiles.Gaussian(3.0),
        F.polarization.x,
        wavelength0=1.0,
        region=GLASS,
        frame=Frame([0, 0, -2.0]),
    )
    chart = o.sampling.PolarGrid(
        np.linspace(0, 9, 9), 2 * np.pi * np.arange(8) / 8, Frame([0, 0, -1.0])
    )
    return F.BeamSpectrum(gaussian, chart)


def hankel():
    incident = F.PlaneBeam(
        F.profiles.Gaussian(3.0),
        F.polarization.x,
        wavelength0=1.0,
        region=AIR,
        frame=Frame([0, 0, -2.0]),
    )
    leg = o.transport.MeridianTransport(
        o.system.DielectricInterface(S.SphericalCap(30.0), AIR, GLASS),
        incident.carrier,
        incident.frame,
        1.0,
        scale=6.0,
        extent=12.0,
    )
    chart = o.sampling.PolarGrid(
        np.linspace(0, 9, 9), 2 * np.pi * np.arange(8) / 8, Frame([0, 0, -2.0])
    )
    transmitted, _ = leg.transmit(F.BeamSpectrum(incident, chart))
    return transmitted


KINDS = {"plane": plane_spectrum, "lattice": lattice, "beam": beam, "hankel": hankel}
POINTS = o.sampling.PointGrid(
    np.array([[0.3, -0.2, 2.0], [1.1, 0.4, 3.5], [-0.7, 0.9, 6.0]])
)


@pytest.mark.parametrize("kind", list(KINDS))
def test_each_spectrum_knows_its_region_wavelength_and_plane(kind):
    spectrum = KINDS[kind]()
    assert isinstance(spectrum, F.Spectrum)
    assert spectrum.region == GLASS and spectrum.wavelength0 == 1.0
    assert isinstance(spectrum.plane, Frame) and spectrum.plane is spectrum.grid.frame


@pytest.mark.parametrize("kind", list(KINDS))
def test_electric_is_the_electric_part_of_evaluate(kind):
    spectrum = KINDS[kind]()
    both = spectrum.evaluate(POINTS, method="direct")
    alone = spectrum.electric(POINTS, method="direct")
    assert isinstance(both, F.ElectromagneticField)
    assert isinstance(alone, F.ElectricField)
    scale = abs(host(both.electric.values)).max()
    assert scale > 0
    assert_allclose(host(alone.values), host(both.electric.values), atol=1e-12 * scale)


@pytest.mark.parametrize("kind", list(KINDS))
def test_propagate_gives_the_field_on_the_chart_moved_along_the_normal(kind):
    spectrum = KINDS[kind]()
    moved = spectrum.grid.translated(1.5 * spectrum.plane.normal)
    expected = spectrum.evaluate(moved, method="direct")
    actual = spectrum.propagate(1.5)
    scale = abs(host(expected.electric.values)).max()
    assert actual.grid.shape == moved.shape
    assert_allclose(
        host(actual.electric.values),
        host(expected.electric.values),
        atol=1e-10 * scale,
    )
    assert_allclose(host(actual.magnetic), host(expected.magnetic), atol=1e-10 * scale)
    assert isinstance(spectrum.propagate(1.5, magnetic=False), F.ElectricField)


@pytest.mark.parametrize("kind", list(KINDS))
def test_relabeled_spectrum_is_the_same_field_in_the_named_region(kind):
    spectrum = KINDS[kind]()
    relabeled = spectrum.relabeled(GLASS2)
    assert isinstance(relabeled, type(spectrum)) and relabeled.region == GLASS2
    assert_allclose(
        host(relabeled.electric(POINTS, method="direct").values),
        host(spectrum.electric(POINTS, method="direct").values),
        atol=0,
    )


@pytest.mark.parametrize("kind", list(KINDS))
def test_every_spectrum_takes_the_same_summation_options_and_no_others(kind):
    spectrum = KINDS[kind]()
    spectrum.evaluate(POINTS, method="direct", tolerance=1e-10, chunk_size=64)
    spectrum.electric(POINTS, method="auto", threads=1)
    with pytest.raises(TypeError, match="backend"):
        spectrum.evaluate(POINTS, backend="direct")
    with pytest.raises(TypeError, match="backend"):
        spectrum.electric(POINTS, backend="direct")
    with pytest.raises(ValueError, match="method must be"):
        spectrum.evaluate(POINTS, method="fast")


@pytest.mark.parametrize("kind", list(KINDS))
def test_the_spectra_of_forward_light_travel_along_their_normal(kind):
    assert KINDS[kind]().direction == 1


def test_a_backward_plane_spectrum_says_so_and_propagates_along_the_normal():
    grid, values = gaussian_lattice()
    field = F.TransverseElectricField(grid, 1.0, values)
    back = F.angular_spectrum(field, GLASS, direction=-1, evanescent="discard")
    assert back.direction == -1
    moved = grid.translated(-1.5 * grid.frame.normal)
    expected = back.evaluate(moved, method="direct")
    actual = back.propagate(-1.5)
    scale = abs(host(expected.electric.values)).max()
    assert_allclose(
        host(actual.electric.values), host(expected.electric.values), atol=1e-10 * scale
    )
