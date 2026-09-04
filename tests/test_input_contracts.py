import numpy as np
import pytest
from vecdiff import (ElectricSpectrum, Medium, LayerStack, plane_wave,
                     propagate_layers, coherent_feedback)
from vecdiff.fourier.nufft import synthesize
from vecdiff.observables.electromagnetism import boundary_residuals


@pytest.mark.parametrize("backend", ["direct", "nufft"])
def test_empty_spectrum_and_target_sets(backend):
    spectrum = ElectricSpectrum(np.empty((0, 3)), np.empty((0, 3)))
    e, h = spectrum.evaluate([[0, 0, 0]], backend=backend)
    assert np.count_nonzero(e) == np.count_nonzero(h) == 0
    assert plane_wave().evaluate(np.empty((0, 3)), backend=backend)[0].shape == (0, 3)


@pytest.mark.parametrize("kwargs", [{"method": "unknown"}, {"max_iterations": 1.5}, {"restart": 0}])
def test_feedback_rejects_invalid_controls_even_for_zero_input(kwargs):
    with pytest.raises(ValueError):
        coherent_feedback([0.], lambda x: x, **kwargs)


def test_single_interface_layer_with_evanescent_exit_is_stable_far_away():
    incident = plane_wave((np.sin(1.), 0, np.cos(1.)), (0, 1, 0), medium=Medium(1.5))
    field = propagate_layers(incident, LayerStack((Medium(1.5), Medium()), ()))
    with np.errstate(over="raise", invalid="raise"):
        e, h = field.evaluate([[0, 0, 1000]], region=1)
    assert np.isfinite(e).all() and np.isfinite(h).all()
    assert np.max(abs(e)) < 1e-100


@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf])
def test_boundary_diagnostic_rejects_invalid_weights(value):
    e, h = plane_wave().evaluate([[0, 0, 0]])
    with pytest.raises(ValueError):
        boundary_residuals(e, h, e, h, [0, 0, 1], Medium(), Medium(), weights=[value])


def test_fourier_rejects_invalid_chunk():
    with pytest.raises(ValueError):
        synthesize([[0, 0, 0]], [1], [[0, 0, 0]], chunk=0)
