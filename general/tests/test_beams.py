"""Beams, their profiles, polarization states and carriers, against closed forms."""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose
from scipy.special import jv

bd, host = o.backend.bd, o.backend.to_numpy

AIR = o.media.HomogeneousRegion("air", o.media.Dielectric(1.0))
profiles, polarization = o.fields.profiles, o.fields.polarization
rng = np.random.default_rng(3)


def test_profiles_and_polarization_states_stay_separate():
    assert_allclose(profiles.Uniform(2.0)([0, 2.0, 2.1], 0), [1, 1, 0])
    assert_allclose(profiles.Uniform()(5.0, 7.0), 1)
    assert_allclose(profiles.Gaussian(2.0)(1.0, 1.0), np.exp(-0.5))
    assert profiles.Gaussian(2.0)(0, profiles.Gaussian(2.0).support) < 1.2e-16
    azimuth = np.array([0.0, np.pi / 2, 2.0])
    assert_allclose(polarization.radial.vectors(azimuth)[2], [np.cos(2), np.sin(2)])
    assert_allclose(polarization.azimuthal.vectors(azimuth)[1], [-1, 0], atol=1e-16)
    assert_allclose(polarization.cartesian(1, 1j).vectors(azimuth)[0], [1, 1j])
    with pytest.raises(ValueError):
        polarization.Jones((1, 0), "helical")


@pytest.mark.parametrize(
    "carrier",
    [
        o.fields.Carrier.converging([0.1, -0.2, 3.0]),
        o.fields.Carrier.diverging([0.3, 0.0, -2.0]),
        o.fields.Carrier.collimated([0.2, 0.1, 1.0], [0, 1, 0]),
    ],
)
def test_carriers_obey_the_eikonal_equation_and_their_derivatives(carrier):
    points = rng.normal(size=(20, 3))
    s = carrier.directions(points)
    assert_allclose(np.linalg.norm(s, axis=-1), 1, atol=1e-15)
    step = 1e-6
    for axis in range(3):
        d = np.eye(3)[axis] * step
        slope = (carrier.path(points + d) - carrier.path(points - d)) / (2 * step)
        assert_allclose(slope, s[:, axis], atol=1e-8)
        derivative = (carrier.gradient(points + d) - carrier.gradient(points - d)) / (
            2 * step
        )
        assert_allclose(derivative, carrier.jacobians(points)[..., axis], atol=1e-7)


def test_converging_carrier_is_singular_only_at_its_focus():
    carrier = o.fields.Carrier.converging([0, 0, 1.0])
    with pytest.raises(ValueError, match="unit vector"):
        carrier.directions([[0, 0, 1.0]])


def test_plane_beam_reproduces_its_plane_and_the_lattice_spectrum_away_from_it():
    profile, state = profiles.Gaussian(2.5), polarization.cartesian(1, 0.4j)
    beam = o.fields.PlaneBeam(profile, state, wavelength0=1.0, region=AIR)
    xy = rng.uniform(-6, 6, size=(200, 2))
    on_plane = beam.evaluate(o.sampling.PointGrid(np.c_[xy, np.zeros(200)]))
    expected = host(profile(xy[:, 0], xy[:, 1]))[:, None] * np.array([1, 0.4j])
    assert_allclose(on_plane.electric.values[:, :2], expected, atol=1e-13)
    # An FFT lattice wide enough that its images vanish is the same beam.
    axis = np.arange(-128, 128) * 0.25
    lattice = o.sampling.CartesianGrid(axis, axis)
    x, y = np.meshgrid(axis, axis)
    transverse = o.fields.TransverseElectricField(
        lattice, 1.0, host(profile(x, y))[..., None] * np.array([1, 0.4j])
    )
    spectrum = o.fields.angular_spectrum(transverse, AIR, evanescent="discard")
    probes = o.sampling.PointGrid(np.c_[xy[:20], np.full(20, 3.0)])
    assert_allclose(
        beam.evaluate(probes).electric.values,
        spectrum.evaluate(probes).electric.values,
        atol=1e-12,
    )


def test_unbounded_uniform_plane_beam_is_one_plane_wave():
    beam = o.fields.PlaneBeam(
        profiles.Uniform(), polarization.y, wavelength0=1.0, region=AIR
    )
    waves = beam.plane_waves(100.0)
    assert len(waves.amplitudes) == 1
    assert_allclose(waves.amplitudes[0], [0, 1, 0])
    with pytest.raises(ValueError, match="support"):
        o.fields.PlaneBeam(
            lambda x, y: x * 0 + 1, polarization.x, wavelength0=1, region=AIR
        )


def test_focused_beam_matches_the_appendix_c_closed_form():
    """Circular input, P = 1, apodization 'none': Eqs. (C.2)-(C.4) at azimuth zero."""
    umax = 0.8
    beam = o.fields.FocusedBeam(
        profiles.Uniform(),
        polarization.cartesian(1, 1j),
        focal_length=500.0,
        max_sine=umax,
        wavelength0=1.0,
        region=AIR,
        apodization="none",
    )
    k = 2 * np.pi
    t, w = np.polynomial.legendre.leggauss(400)
    u, w = umax / 2 * (t + 1), umax / 2 * w
    c = np.sqrt(1 - u**2)
    norm = np.sum(w * u * (1 + c) / 2)
    r, z = np.linspace(0, 3, 13), 1.7
    J0, J1, J2 = (jv(n, k * np.outer(r, u)) for n in (0, 1, 2))
    phase = w * u * np.exp(1j * k * z * c) / norm
    closed = np.stack(
        (
            (J0 * (1 + c) / 2 + J2 * (1 - c) / 2) @ phase,
            1j * ((J0 * (1 + c) / 2 - J2 * (1 - c) / 2) @ phase),
            -1j * ((J1 * u) @ phase),
        ),
        -1,
    )
    focus = beam.evaluate(o.sampling.PointGrid([[0.0, 0, 0]])).electric.values[0]
    assert_allclose(focus / focus[0], [1, 1j, 0], atol=1e-13)
    field = beam.evaluate(o.sampling.PointGrid(np.c_[r, 0 * r, z + 0 * r]))
    assert_allclose(field.electric.values / focus[0], closed, atol=1e-12)


def test_aplanatic_focused_beam_carries_the_pupil_power():
    beam = o.fields.FocusedBeam(
        profiles.Uniform(),
        polarization.x,
        focal_length=200.0,
        max_sine=0.6,
        wavelength0=1.0,
        region=AIR,
    )
    _, _, weight, density = beam.quadrature(0.0)
    k = beam.wavenumber
    # Through any transverse plane, P = (n / 2 Z0) (2 pi / k)² integral |F|² dOmega.
    power = (2 * np.pi / k) ** 2 * np.sum(weight * np.sum(abs(density) ** 2, -1))
    assert_allclose(power, np.pi * (200.0 * 0.6) ** 2, rtol=1e-12)


def test_radial_and_azimuthal_foci_have_their_symmetries():
    kwargs = dict(focal_length=300.0, max_sine=0.9, wavelength0=1.0, region=AIR)
    axis = o.sampling.PointGrid(np.c_[np.zeros(5), np.zeros(5), np.linspace(-2, 2, 5)])
    radial = o.fields.FocusedBeam(profiles.Uniform(), polarization.radial, **kwargs)
    E = radial.evaluate(axis).electric.values
    assert np.max(abs(E[:, :2])) < 1e-12 * np.max(abs(E[:, 2]))
    azimuthal = o.fields.FocusedBeam(
        profiles.Uniform(), polarization.azimuthal, **kwargs
    )
    ring = o.sampling.PointGrid(rng.normal(size=(30, 3)))
    E = azimuthal.evaluate(ring).electric.values
    assert np.max(abs(E[:, 2])) < 1e-12 * np.max(abs(E))


def test_plane_waves_accept_faint_transverse_modes_and_reject_faint_longitudinal_ones():
    air = o.media.HomogeneousRegion("air", o.media.Dielectric(1.0))
    k = 2 * np.pi * np.array([[0, 0, 1.0], [0.6, 0, 0.8]])
    faint = 1e-200 * np.array([[1, 0.3j, 0], [0.8, 0.1, -0.6]])
    o.fields.PlaneWaveExpansion(k, faint, [0, 0, 0], 1.0, air)
    with pytest.raises(ValueError, match="transverse"):
        o.fields.PlaneWaveExpansion(
            k, faint + [[0, 0, 1e-201], [0, 0, 0]], [0, 0, 0], 1.0, air
        )

    def gaussian(x, y):  # underflows to 1e-188 before the rim of the pupil
        return np.exp(-(x**2 + y**2) / 0.05**2)

    beam = o.fields.FocusedBeam(
        gaussian,
        o.fields.polarization.x,
        focal_length=1.0,
        max_sine=1.0,
        wavelength0=1e-3,
        region=air,
    )
    assert len(beam.plane_waves(0.0).amplitudes)
