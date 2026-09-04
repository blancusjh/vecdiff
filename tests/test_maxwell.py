import numpy as np
import pytest
from vecdiff import (Medium, Frame, Plane, DielectricInterface, plane_wave,
                     interface_transform, ElectricSpectrum)
from vecdiff.interfaces.fresnel import solve
from vecdiff.observables.electromagnetism import boundary_residuals, poynting


@pytest.mark.parametrize("n1,n2", [(1., 1.5), (1.5, 1.), (1., 1.)])
@pytest.mark.parametrize("angle", [0., 1e-9, 20., 41.810314895778596, 56.309932474020215, 75., 89.9])
@pytest.mark.parametrize("polarization", ["s", "p", "circular"])
def test_fresnel_all_boundary_conditions_and_flux(n1, n2, angle, polarization):
    a = np.deg2rad(angle); u = np.array([np.sin(a), 0., np.cos(a)])
    s, p = np.array([0., 1., 0.]), np.array([np.cos(a), 0., -np.sin(a)])
    e = s if polarization == "s" else p if polarization == "p" else (s+1j*p)/np.sqrt(2)
    m1, m2 = Medium(n1), Medium(n2); k0 = 2*np.pi
    f = solve(k0*n1*u, e, [0, 0, 1], m1, m2)
    hi = n1*np.cross(u, e); hr = np.cross(f.reflected_k, f.reflected_E)/k0
    ht = np.cross(f.transmitted_k, f.transmitted_E)/k0
    residual = boundary_residuals((e+f.reflected_E)[None], (hi+hr)[None], f.transmitted_E[None], ht[None],
                                 np.array([[0, 0, 1]]), m1, m2, electric_scale=1, magnetic_scale=n1)
    assert max(residual.values()) < 2e-12
    incoming, reflected, transmitted = (poynting(x, y)[2] for x, y in [(e, hi), (f.reflected_E, hr), (f.transmitted_E, ht)])
    assert abs((incoming+reflected-transmitted)/incoming) < 1e-10
    for k, a in [(f.reflected_k, f.reflected_E), (f.transmitted_k, f.transmitted_E)]:
        assert abs(np.sum(k*a)) < 1e-12


def test_tilted_normal_incidence_degeneracy():
    n = np.array([.6, 0, .8]); e = np.array([.8, 0, -.6])
    f = solve(2*np.pi*n, e, n, Medium(), Medium(1.5))
    np.testing.assert_allclose(f.transmitted_E, .8*e, atol=1e-14)


def test_reconstructed_fields_on_tilted_displaced_plane():
    a = .43; rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    frame = Frame([.3, -.2, .7], rotation); surface = Plane(frame)
    local = np.array([[0., 0., 1.], [.2, .1, np.sqrt(.95)], [-.4, .2, np.sqrt(.8)]])
    e = np.cross(local, [0, 1, 0]).astype(complex)*np.array([1, .2j, -.3])[:, None]
    wave = ElectricSpectrum(2*np.pi*frame.vectors(local), frame.vectors(e))
    out = interface_transform(wave, DielectricInterface(surface, Medium(), Medium(1.5)))
    q = surface.position(np.linspace(-3, 3, 31), np.linspace(.3, 2, 31))
    ei, hi = wave.evaluate(q); er, hr = out.reflected.evaluate(q); et, ht = out.transmitted.evaluate(q)
    errors = boundary_residuals(ei+er, hi+hr, et, ht, frame.rotation[:, 2], Medium(), Medium(1.5))
    assert max(errors.values()) < 2e-14


def test_tir_reconstructed_evanescent_field_decays():
    m = Medium(1.5); angle = np.deg2rad(60)
    wave = plane_wave((np.sin(angle), 0, np.cos(angle)), (0, 1, 0), medium=m)
    out = interface_transform(wave, DielectricInterface(Plane(), m, Medium()))
    q = np.array([[.2, 0, 0], [.2, 0, .5]])
    e, _ = out.transmitted.evaluate(q)
    decay = np.exp(-out.transmitted.wavevectors[0, 2].imag*.5)
    np.testing.assert_allclose(np.linalg.norm(e[1])/np.linalg.norm(e[0]), decay)


def test_brewster_p_reflection_zero():
    a = np.arctan(1.5); u = np.array([np.sin(a), 0, np.cos(a)])
    f = solve(2*np.pi*u, [np.cos(a), 0, -np.sin(a)], [0, 0, 1], Medium(), Medium(1.5))
    assert np.linalg.norm(f.reflected_E) < 2e-16
