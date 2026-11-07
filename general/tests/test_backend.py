"""The array backend switch, and GPU results against the CPU where a GPU exists."""

import importlib.util

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose
from scipy.special import jv

backend = o.backend
AIR = o.media.HomogeneousRegion("air", o.media.Dielectric(1.0))


def cuda_available():
    if importlib.util.find_spec("cupy") is None:
        return False
    import cupy

    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except cupy.cuda.runtime.CUDARuntimeError:
        return False


gpu = pytest.mark.skipif(not cuda_available(), reason="needs CuPy and a CUDA device")


@pytest.fixture
def on_gpu():
    """Run the test body on CUDA and restore the starting backend afterwards."""
    start = backend.get_backend()
    backend.set_backend("CUDA")
    yield
    backend.set_backend(start)


def test_backend_names_and_host_conversion():
    assert backend.get_backend() in ("CPU", "CUDA")
    with pytest.raises(ValueError, match="CPU"):
        backend.set_backend("TPU")
    assert isinstance(backend.to_numpy([1.0, 2.0]), np.ndarray)
    if backend.get_backend() == "CPU":
        assert backend.bd.asarray([1, 2]).__class__ is np.ndarray


def focused():
    return o.fields.FocusedBeam(
        o.fields.profiles.Gaussian(30.0),
        o.fields.polarization.radial,
        focal_length=50.0,
        max_sine=0.9,
        wavelength0=1.0,
        region=AIR,
    )


def test_cpu_results_are_reference_for_the_gpu_tests():
    """The CPU path is unchanged by the backend module: identical to its own rerun."""
    if backend.get_backend() != "CPU":
        pytest.skip("CPU reference")
    points = o.sampling.PointGrid(np.random.default_rng(0).normal(size=(50, 3)))
    first = focused().evaluate(points).electric.values
    assert isinstance(first, np.ndarray) and not first.flags.writeable
    assert_allclose(focused().evaluate(points).electric.values, first, rtol=0, atol=0)


@gpu
def test_gpu_beam_spectrum_and_fft_match_the_cpu(on_gpu):
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(400, 3))
    lattice = rng.normal(size=(64, 48, 3)) + 1j * rng.normal(size=(64, 48, 3))
    backend.set_backend("CPU")
    cpu_beam = focused().evaluate(o.sampling.PointGrid(raw))
    cpu_fft = o.fourier.FFT2(lattice)
    backend.set_backend("CUDA")
    gpu_beam = focused().evaluate(o.sampling.PointGrid(raw))
    assert backend.bd.__repr__().endswith("cupy>")
    assert not isinstance(gpu_beam.electric.values, np.ndarray)
    assert_allclose(
        backend.to_numpy(gpu_beam.electric.values),
        cpu_beam.electric.values,
        rtol=1e-12,
        atol=1e-12 * abs(cpu_beam.electric.values).max(),
    )
    assert_allclose(
        backend.to_numpy(gpu_beam.magnetic),
        cpu_beam.magnetic,
        atol=1e-12 * abs(cpu_beam.magnetic).max(),
    )
    gpu_fft = o.fourier.FFT2(lattice)
    assert_allclose(backend.to_numpy(gpu_fft), cpu_fft, atol=1e-12)
    assert_allclose(backend.to_numpy(o.fourier.IFFT2(gpu_fft)), lattice, atol=1e-14)


@gpu
def test_gpu_bessel_functions_reach_scipy_where_scipy_is_exact(on_gpu):
    from vecdiff_general._backend import jv as device_jv

    rng = np.random.default_rng(2)
    order = rng.integers(-40, 41, 20000)
    x = rng.uniform(0, 60, 20000)
    expected = jv(order, x)
    envelope = np.maximum(abs(expected), 1 / np.sqrt(np.maximum(x, 1)))
    error = abs(backend.to_numpy(device_jv(order, x)) - expected) / envelope
    assert error.max() < 2e-14
    with pytest.raises(ValueError, match="integer"):
        device_jv(np.array([0.5]), np.array([1.0]))
