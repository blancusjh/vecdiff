import numpy as np
import pytest
from vecdiff import *
from vecdiff.propagation.surface_radiation import SurfaceRadiation
from vecdiff.observables.electromagnetism import boundary_residuals


def test_sphere_area_and_freeform_geometry():
    sphere = Sphere(2.); samples = sample_surface(sphere, (-1, 1), (0, 2*np.pi), 20, 32)
    np.testing.assert_allclose(sum(samples.weights), 16*np.pi, rtol=1e-14)
    free = FreeformSurface(lambda x, y: .2*x*x+.1*x*y, lambda x, y: (.4*x+.1*y, .1*x))
    u, v = np.array([.3, -.2]), np.array([.5, .1])
    n, jac = free.normal_and_jacobian(u, v); a, b = free.tangents(u, v)
    np.testing.assert_allclose(np.sum(n*a, axis=-1), 0, atol=1e-15)
    np.testing.assert_allclose(np.sum(n*b, axis=-1), 0, atol=1e-15)


def test_curved_transform_linearity_and_boundary_data():
    cap = SphericalCap(4.); sampling = sample_surface(cap, (0, 2), (0, 2*np.pi), 32, 48)
    interface = DielectricInterface(cap, Medium(), Medium(1.5))
    a = plane_wave(); b = plane_wave((.1, 0, np.sqrt(.99)), (0, .3j, 0))
    both = ElectricSpectrum(np.concatenate((a.wavevectors, b.wavevectors)), np.concatenate((a.amplitudes, b.amplitudes)))
    results = [interface_transform(x, interface, sampling) for x in (a, b, both)]
    for branch in ("reflected", "transmitted"):
        for component in ("J", "M"):
            one, two, total = [getattr(getattr(r, branch), component) for r in results]
            np.testing.assert_allclose(total, one+two, atol=2e-15)
    d = results[-1].boundary
    errors = boundary_residuals(d.incident_E+d.reflected_E, d.incident_H+d.reflected_H,
                                d.transmitted_E, d.transmitted_H, sampling.normals, Medium(), Medium(1.5))
    assert max(errors.values()) < 1e-14


@pytest.mark.parametrize("direction", [-1, 1])
def test_polar_spectral_transform_matches_direct(direction):
    cap = SphericalCap(3.); sampling = sample_surface(cap, (0, 1.5), (0, 2*np.pi), 36, 64)
    result = interface_transform(plane_wave(), DielectricInterface(cap, Medium(), Medium(1.5)), sampling)
    radiation = result.transmitted if direction == 1 else result.reflected
    grid = CartesianGrid.from_spacing(.29, 24)
    a = radiation.spectrum(grid, direction=direction)
    b = radiation.spectrum(grid, direction=direction, backend="polar", radial_count=1800)
    np.testing.assert_allclose(a.amplitudes, b.amplitudes, atol=4e-9, rtol=1e-7)


def test_native_green_integral_matches_independent_reference():
    from references.stratton_chu import franz_integral
    cap = SphericalCap(3.); sampling = sample_surface(cap, (0, 1), (0, 2*np.pi), 20, 32)
    result = interface_transform(plane_wave(), DielectricInterface(cap, Medium(), Medium(1.5)), sampling)
    rad = result.transmitted
    points = np.array([[.1, .2, 1.], [.7, -.2, 2.]])
    e, h = rad.evaluate(points)
    ref = franz_integral(points, sampling.points, rad.J, rad.M, sampling.weights, 3*np.pi, 1.5)
    np.testing.assert_allclose(e, ref, atol=3e-15)
    # Maxwell curl identities are independent of this E-only reference.
    step = 1e-4
    de, dh = [], []
    for axis in np.eye(3):
        ep, hp = rad.evaluate(points+step*axis); em, hm = rad.evaluate(points-step*axis)
        de.append((ep-em)/(2*step)); dh.append((hp-hm)/(2*step))
    def curl(d): return np.stack((d[1][:, 2]-d[2][:, 1], d[2][:, 0]-d[0][:, 2], d[0][:, 1]-d[1][:, 0]), axis=-1)
    np.testing.assert_allclose(curl(de), 2j*np.pi*h, atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(curl(dh), -2j*np.pi*1.5**2*e, atol=3e-6, rtol=3e-6)


def test_continuous_angular_spectrum_polar_matches_direct_transform():
    cap = SphericalCap(1.5); sampling = sample_surface(cap, (0, .6), (0, 2*np.pi), 24, 40)
    result = interface_transform(plane_wave(), DielectricInterface(cap, Medium(), Medium(1.5)), sampling)
    rad = result.transmitted
    polar = rad.angular_spectrum(n_theta=60, n_phi=48, backend="polar", radial_count=800, max_order=2)
    direct = rad.angular_spectrum(n_theta=60, n_phi=48, backend="direct")
    points = np.array([[.1, .15, 10.], [-.3, .2, 12.]])
    e_polar, _ = polar.evaluate(points)
    e_direct, _ = direct.evaluate(points)
    np.testing.assert_allclose(e_polar, e_direct, rtol=1e-10, atol=1e-11)


def test_mie_boundary_reference_and_normalization():
    pytest.importorskip("miepython")
    from references.mie import fields
    theta = np.linspace(.1, 3., 41); normal = np.stack((np.sin(theta), np.zeros_like(theta), np.cos(theta)), axis=-1)
    for environment in (1., 1.2):
        eo, ho = fields(normal*(1+1e-8), 1., environment_index=environment)
        ei, hi = fields(normal*(1-1e-8), 1., environment_index=environment)
        residual = boundary_residuals(eo, ho, ei, hi, normal, Medium(environment), Medium(1.5))
        assert max(residual.values()) < 3e-7


def test_richards_wolf_reference_uses_maxwell_spectrum():
    from references.richards_wolf import spectrum
    rw = spectrum(.85, n_theta=12, n_phi=20)
    np.testing.assert_allclose(np.sum(rw.wavevectors*rw.amplitudes, axis=-1), 0, atol=2e-15)
