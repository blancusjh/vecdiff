import numpy as np
import pytest
from vecdiff import Medium, Frame, plane_wave
from vecdiff.media.layers import LayerStack
from vecdiff.propagation.layered_propagation import propagate_layers
from vecdiff.propagation.multiple_scattering import coherent_feedback, ConvergenceError
from vecdiff.observables import boundary_residuals, poynting


@pytest.mark.parametrize("method", ["successive", "gmres"])
def test_coherent_round_trips(method):
    reflection = .98*np.exp(.001j)
    result = coherent_feedback(np.array([.3, .2j]), lambda x: reflection*x,
                               method=method, max_iterations=2000)
    np.testing.assert_allclose(result.state, np.array([.3, .2j])/(1-reflection), rtol=1e-9)
    assert result.relative_residual < 1e-10


def test_feedback_never_returns_unconverged_field():
    with pytest.raises(ConvergenceError):
        coherent_feedback([1.], lambda x: .999*x, max_iterations=3)


def test_coupled_complex_feedback_and_strict_gmres_budget():
    matrix = np.array([[.9+.03j, .12j], [.07, -.2j]])
    b = np.array([1., .3j])
    result = coherent_feedback(b, lambda x: matrix@x, method="gmres", max_iterations=3)
    np.testing.assert_allclose(result.state, np.linalg.solve(np.eye(2)-matrix, b), rtol=1e-12)
    with pytest.raises(ConvergenceError):
        coherent_feedback(b, lambda x: matrix@x, method="gmres", max_iterations=1)


@pytest.mark.parametrize("angle", [0., 30., 60.])
@pytest.mark.parametrize("indices", [(1., 1.5, 1.), (1.5, 1., 1.5), (1., 2., 1.3, 1.7)])
@pytest.mark.parametrize("polarization", ["s", "p"])
def test_layer_fields_all_boundaries_and_flux(angle, indices, polarization):
    theta = np.deg2rad(angle)
    u = [np.sin(theta), 0, np.cos(theta)]
    e = [0, 1, 0] if polarization == "s" else [np.cos(theta), 0, -np.sin(theta)]
    stack = LayerStack(tuple(Medium(n) for n in indices), (.23,)*(len(indices)-2))
    wave = plane_wave(u, e, medium=stack.media[0])
    field = propagate_layers(wave, stack)
    for j, z in enumerate(stack.boundaries):
        q = np.array([[.2, -.3, z], [.4, .1, z]])
        e1, h1 = field.evaluate(q, region=j); e2, h2 = field.evaluate(q, region=j+1)
        jumps = boundary_residuals(e1, h1, e2, h2, [0, 0, 1], stack.media[j], stack.media[j+1], electric_scale=1, magnetic_scale=indices[0])
        assert max(jumps.values()) < 1e-12
    e1, h1 = field.evaluate([[0, 0, 0]], region=0)
    e2, h2 = field.evaluate([[0, 0, stack.boundaries[-1]]], region=len(indices)-1)
    np.testing.assert_allclose(poynting(e1, h1)[..., 2], poynting(e2, h2)[..., 2], atol=1e-15)


def test_fabry_perot_phase_and_airy_formula():
    for wavelength in np.linspace(.8, 1.2, 51):
        n = 4.; thickness = 1.; r = (1-n)/(1+n)
        phase = 2*np.pi*n*thickness/wavelength
        airy_t = (1-r*r)*np.exp(1j*phase)/(1-r*r*np.exp(2j*phase))
        out = propagate_layers(plane_wave(wavelength=wavelength), LayerStack((Medium(), Medium(n), Medium()), (thickness,)))
        np.testing.assert_allclose(out.transmission[0], airy_t, atol=3e-15)


def test_thick_evanescent_layer_is_finite():
    theta = np.deg2rad(60)
    wave = plane_wave([np.sin(theta), 0, np.cos(theta)], [0, 1, 0], medium=Medium(1.5))
    field = propagate_layers(wave, LayerStack((Medium(1.5), Medium(), Medium(1.5)), (1000.,)))
    e, h = field.evaluate([[0, 0, 500]], region=1)
    assert np.isfinite(e).all() and np.isfinite(h).all()
    np.testing.assert_allclose(abs(field.reflection), 1, atol=1e-14)


def test_rotated_displaced_stack_boundary():
    angle = .4; rot = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    frame = Frame([.2, .3, -.4], rot)
    stack = LayerStack((Medium(), Medium(2.), Medium()), (.37,), frame)
    field = propagate_layers(plane_wave(rot[:, 2], rot[:, 0]), stack)
    for j, z in enumerate(stack.boundaries):
        q = frame.points([[.1, -.2, z]])
        e1, h1 = field.evaluate(q, region=j); e2, h2 = field.evaluate(q, region=j+1)
        assert max(boundary_residuals(e1, h1, e2, h2, rot[:, 2], stack.media[j], stack.media[j+1]).values()) < 1e-13
