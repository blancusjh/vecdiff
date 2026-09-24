"""Fourier sums and quadrature checked against definitions and analytic pairs."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose
from vecdiff_general import fourier
from vecdiff_general.fourier import NUFFT


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("sign", [-1, 1])
def test_nufft_matches_exponential_sum_with_independent_batches(dimension, sign):
    pytest.importorskip("finufft")
    random = np.random.default_rng(14)
    x = random.normal(size=(19, dimension))
    q = random.normal(size=(4, 5, dimension))
    g = random.normal(size=(19, 2, 3)) + 1j * random.normal(size=(19, 2, 3))

    expected = np.zeros((4, 5, 2, 3), dtype=complex)

    for position, value in zip(x, g):
        phase = np.exp(sign * 1j * np.sum(q * position, axis=-1))
        expected += phase[..., None, None] * value

    direct = NUFFT(g, x, q, sign=sign, backend="direct", chunk_size=3)
    accelerated = NUFFT(g, x, q, sign=sign)

    assert_allclose(direct, expected, atol=2e-14)
    assert_allclose(accelerated, expected, atol=2e-10)


@pytest.mark.parametrize("backend", ["direct", "nufft"])
def test_nonuniform_continuous_pair_has_correct_measure_sign_and_normalization(backend):
    if backend == "nufft":
        pytest.importorskip("finufft")

    nodes, weights = leggauss(96)
    x = 6 * nodes[:, None]
    q = 9 * nodes[:, None]
    dx = 6 * weights
    dq = 9 * weights
    gaussian = np.exp(-(x[:, 0] ** 2))
    g = np.stack((gaussian, x[:, 0] * gaussian), axis=-1)

    G = fourier.transform(
        g, mode="nonuniform", coordinates=x, frequencies=q, weights=dx, backend=backend
    )
    expected_gaussian = np.sqrt(np.pi) * np.exp(-(q[:, 0] ** 2) / 4)
    expected = np.stack(
        (expected_gaussian, -0.5j * q[:, 0] * expected_gaussian), axis=-1
    )

    assert_allclose(G, expected, atol=2e-11)

    recovered = fourier.inverse(
        G, mode="nonuniform", coordinates=x, frequencies=q, weights=dq, backend=backend
    )

    assert_allclose(recovered, g, atol=2e-9)


def test_nonuniform_integral_agrees_with_centered_cartesian_transform():
    x = 0.13 + np.arange(9) * 0.2
    y = -0.47 + np.arange(6) * 0.3
    X, Y = np.meshgrid(x, y)
    coordinates = np.stack((X, Y), axis=-1).reshape(-1, 2)
    fx, fy = fourier.frequency_grid((x, y))
    frequencies = 2 * np.pi * np.stack((fx, fy), axis=-1)
    g = np.exp(-(X**2) - Y**2) * (1 + 0.4j * Y)

    cartesian = fourier.transform(g, mode="cartesian", coordinates=(x, y))
    nonuniform = fourier.transform(
        g.ravel(),
        mode="nonuniform",
        coordinates=coordinates,
        frequencies=frequencies,
        weights=np.full(g.size, 0.2 * 0.3),
        backend="direct",
    )

    assert_allclose(nonuniform, cartesian, atol=3e-14)

    frequency_area = (2 * np.pi) ** 2 / (g.size * 0.2 * 0.3)
    recovered = fourier.inverse(
        nonuniform.ravel(),
        mode="nonuniform",
        coordinates=coordinates,
        frequencies=frequencies.reshape(-1, 2),
        weights=np.full(g.size, frequency_area),
        backend="direct",
    )

    assert_allclose(recovered.reshape(g.shape), g, atol=3e-14)


def test_complex_fourier_sum_preserves_decay_and_ignores_zero_growing_terms():
    q = np.array([[1.0], [2.0]])
    x = np.array([[2j], [-1000j]])
    g = np.array([1, 0])

    actual = NUFFT(g, x, q, sign=1, backend="direct")

    assert_allclose(actual, np.exp(-2 * q[:, 0]))

    with pytest.raises(ValueError, match="overflows"):
        NUFFT([1, 1], x, q, sign=1, backend="direct")

    with pytest.raises(ValueError, match="real coordinates"):
        NUFFT(g, x, q, sign=1)


def test_nonuniform_quadrature_requires_an_explicit_valid_measure():
    x = np.array([[0.0], [1.0]])
    q = np.array([[0.0]])

    with pytest.raises(ValueError, match="quadrature weights"):
        fourier.transform([1, 1], mode="nonuniform", coordinates=x, frequencies=q)

    with pytest.raises(ValueError, match="nonnegative"):
        fourier.transform(
            [1, 1], mode="nonuniform", coordinates=x, frequencies=q, weights=[1, -1]
        )
