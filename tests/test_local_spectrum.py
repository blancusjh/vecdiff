import numpy as np
import pytest
from vecdiff import (EvenAsphere, Medium, plane_wave, DielectricInterface,
                     sample_surface, interface_transform, Frame)


def radiation(radius, origin=(0, 0, 0), nr=20, nphi=40):
    surface = EvenAsphere(-1/radius, -2.25, frame=Frame(origin=origin))
    sampling = sample_surface(surface, (0, 1.2*radius), (0, 2*np.pi), nr, nphi)
    return interface_transform(plane_wave(medium=Medium(1.5)),
                              DielectricInterface(surface, Medium(1.5), Medium()), sampling).transmitted


@pytest.mark.parametrize("radius", [20., 200., 20000.])
def test_bound_covers_full_kernel_and_converges_with_scale(radius):
    rad = radiation(radius)
    center = np.array([0, 0, 2*radius])
    ball = 5.
    field = rad.local_spectrum(center, ball)
    rng = np.random.default_rng(73)
    d = rng.normal(size=(30, 3)); d *= ball/np.linalg.norm(d, axis=1)[:, None]
    points = center+d
    e, h = field.evaluate(points); er, hr = rad.evaluate(points)
    assert np.max(np.linalg.norm(e-er, axis=-1)) <= field.electric_error_bound
    assert np.max(np.linalg.norm(h-hr, axis=-1)) <= field.magnetic_error_bound
    err = np.linalg.norm(np.c_[e-er,h-hr])/np.linalg.norm(np.c_[er,hr])
    if radius >= 20000: assert err < 1e-3  # sampled E/H error target at this scale
    k, a = field.spectrum.wavevectors, field.spectrum.amplitudes
    assert np.max(abs(np.sum(k*a, axis=-1))) < 1e-12*np.linalg.norm(a)
    with pytest.raises(ValueError, match="outside"): field.evaluate(center+[6,0,0])


def test_nufft_and_translation_preserve_complex_phase():
    pytest.importorskip("finufft")
    rad = radiation(20000.)
    f = rad.local_spectrum([0,0,40000.], 3.)
    p = np.array([[x,0,40000.+z] for x in (-1.,0.,1.) for z in (-2.,0.,2.)])
    e,h = f.evaluate(p)
    en,hn = f.evaluate(p, backend="nufft")
    np.testing.assert_allclose(en, e, atol=1e-8*np.max(abs(e)), rtol=1e-8)
    np.testing.assert_allclose(hn, h, atol=1e-8*np.max(abs(h)), rtol=1e-8)
    shift = np.array([17.,-3.,100.])
    moved = radiation(20000., shift).local_spectrum(np.array([0,0,40000.])+shift,3.)
    em,hm = moved.evaluate(p+shift)
    phase = np.exp(1j*3*np.pi*shift[2])
    np.testing.assert_allclose(em,e*phase,atol=2e-9*np.max(abs(e)),rtol=2e-9)
    np.testing.assert_allclose(hm,h*phase,atol=2e-9*np.max(abs(h)),rtol=2e-9)


@pytest.mark.parametrize("radius", [-1, np.nan, np.inf, 100])
def test_invalid_observation_balls(radius):
    with pytest.raises(ValueError): radiation(20).local_spectrum([0,0,40], radius)


def test_bound_with_arbitrary_currents_in_nonunit_medium():
    from vecdiff.propagation.surface_radiation import SurfaceRadiation
    original = radiation(20)
    rng = np.random.default_rng(123)
    shape = original.J.shape
    j = rng.normal(size=shape)+1j*rng.normal(size=shape)
    m = rng.normal(size=shape)+1j*rng.normal(size=shape)
    rad = SurfaceRadiation(original.sampling,j,m,1.,Medium(1.8))
    field = rad.local_spectrum([0,0,40],3.)
    points = np.array([[0,0,40],[3,0,40],[0,0,37],[0,0,43]])
    e,h=field.evaluate(points);er,hr=rad.evaluate(points)
    assert np.max(np.linalg.norm(e-er,axis=1)) <= field.electric_error_bound
    assert np.max(np.linalg.norm(h-hr,axis=1)) <= field.magnetic_error_bound
