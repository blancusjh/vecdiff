import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import ive, jv
from vecdiff_general._backend import to_numpy
from vecdiff_general.fourier.cartesian import FFT2, IFFT2
from vecdiff_general.fourier.nufft import NUFFT
from vecdiff_general.fourier.polar import HT, azimuthal_coefficients, hankel_sum


@pytest.mark.parametrize("shape", [(9, 12, 3), (8, 7, 2, 3), (6, 8), (7, 9)])
def test_centered_fft_parseval_and_direct_coefficients(shape):
    generator = np.random.default_rng(7)
    g = generator.normal(size=shape) + 1j * generator.normal(size=shape)
    ny, nx = shape[:2]

    G = FFT2(g)
    reconstructed = IFFT2(G)

    assert_allclose(reconstructed, g, atol=2e-15)
    assert_allclose(np.sum(abs(G) ** 2) / (ny * nx), np.sum(abs(g) ** 2), rtol=1e-14)

    # Independent Fourier sum at frequency indices (1, 2) about the center.
    y = np.arange(ny) - ny // 2
    x = np.arange(nx) - nx // 2
    phase = np.exp(-2j * np.pi * (y[:, None] / ny + 2 * x[None, :] / nx))
    phase = phase.reshape((ny, nx) + (1,) * (g.ndim - 2))
    expected = np.sum(g * phase, axis=(0, 1))

    assert_allclose(G[ny // 2 + 1, nx // 2 + 2], expected, atol=3e-14)


def test_gaussian_hankel_pair_and_nonzero_order():
    radius = np.linspace(0, 8, 8001)
    frequency = np.linspace(0, 4, 15)
    actual = HT(np.exp(-(radius**2)), radius, frequency)
    assert_allclose(actual, 0.5 * np.exp(-(frequency**2) / 4), atol=9e-8)
    actual = HT(radius * np.exp(-(radius**2)), radius, frequency, order=1)
    assert_allclose(actual, frequency / 4 * np.exp(-(frequency**2) / 4), atol=1e-12)


@pytest.mark.parametrize("sources, targets", [(3000, 1500), (1200, 4000)])
def test_fast_hankel_sum_matches_the_direct_sum(sources, targets):
    """Asymptotic NUFFT far field and Chebyshev near field, cut along either axis."""
    pytest.importorskip("finufft")
    generator = np.random.default_rng(11)
    x = np.r_[0.0, generator.uniform(0, 400.0, sources - 1)]
    t = np.r_[0.0, generator.uniform(0, 9.0, targets - 1)]
    orders = [0, 1, -1, 2, -3, 6]
    c = generator.normal(size=(sources, 6)) + 1j * generator.normal(size=(sources, 6))
    c *= np.exp(5j * x)[:, None]
    fast = to_numpy(hankel_sum(c, x, t, order=orders, method="fast"))
    probe = np.r_[0, generator.choice(targets, 60, replace=False)]
    direct = hankel_sum(c, x, t[probe], order=orders, method="direct")
    expected = np.stack(
        [jv(n, np.outer(t[probe], x)) @ c[:, i] for i, n in enumerate(orders)], -1
    )
    assert_allclose(direct, expected, atol=1e-13 * abs(expected).max())
    error = np.linalg.norm(fast[probe] - expected, axis=0) / np.linalg.norm(
        expected, axis=0
    )
    assert error.max() < 1e-12
    assert hankel_sum(c[:, 0], x, t[:3], order=2).shape == (3,)
    with pytest.raises(ValueError, match="integer"):
        hankel_sum(c[:, 0], x, t, order=0.5)


@pytest.mark.parametrize("order", [0, 1, 3])
def test_fast_hankel_transform_reproduces_dlmf_pairs(order):
    """DLMF 10.22.51 (Gaussian) and 10.22.67 (Weber's second integral)."""
    pytest.importorskip("finufft")
    x, w = np.polynomial.legendre.leggauss(64)
    edges = np.linspace(0, 120, 121)
    r = ((edges[:-1, None] + edges[1:, None]) / 2 + x / 2).ravel()
    dr = np.tile(w / 2, 120)
    q = np.linspace(0, 40, 4001)
    p, b = 0.06, 25.0
    gaussian = HT(
        np.exp(-((p * r) ** 2)) * r**order, r, q, order=order, weights=dr, method="fast"
    )
    closed = q**order / (2 * p**2) ** (order + 1) * np.exp(-(q**2) / (4 * p**2))
    assert_allclose(gaussian, closed, atol=1e-12 * abs(closed).max())
    weber = HT(
        np.exp(-((p * r) ** 2)) * jv(order, b * r),
        r,
        q,
        order=order,
        weights=dr,
        method="fast",
    )
    closed = (
        np.exp(-((q - b) ** 2) / (4 * p**2))
        * ive(order, q * b / (2 * p**2))
        / (2 * p**2)
    )
    assert_allclose(weber, closed, atol=1e-12 * abs(closed).max())


def test_harmonics_reject_unresolved_tail():
    phi = 2 * np.pi * np.arange(32) / 32
    values = np.exp(3j * phi)
    orders, coefficients = azimuthal_coefficients(values, max_order=3)
    assert_allclose(coefficients[orders == 3], 1, atol=1e-14)
    with pytest.raises(ValueError, match="tail"):
        azimuthal_coefficients(values, max_order=2)


def test_nufft_matches_direct_complex_vector_synthesis():
    pytest.importorskip("finufft")
    generator = np.random.default_rng(8)
    wavevectors = generator.normal(size=(80, 3)) * 1e7
    amplitudes = generator.normal(size=(80, 6)) + 1j * generator.normal(size=(80, 6))
    points = generator.normal(size=(31, 3)) * 1e-6
    direct = NUFFT(amplitudes, wavevectors, points, sign=1, method="direct")
    accelerated = NUFFT(amplitudes, wavevectors, points, sign=1, method="nufft")
    assert np.linalg.norm(direct - accelerated) / np.linalg.norm(direct) < 5e-12
