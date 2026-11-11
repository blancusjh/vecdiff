"""Fourier sums and quadrature checked against definitions and analytic pairs."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose
from vecdiff import fourier
from vecdiff.fourier import NUFFT


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

    direct = NUFFT(g, x, q, sign=sign, method="direct", chunk_size=3)
    accelerated = NUFFT(g, x, q, sign=sign)

    assert_allclose(direct, expected, atol=2e-14)
    assert_allclose(accelerated, expected, atol=2e-10)


@pytest.mark.parametrize("method", ["direct", "nufft"])
def test_nonuniform_continuous_pair_has_correct_measure_sign_and_normalization(method):
    if method == "nufft":
        pytest.importorskip("finufft")

    nodes, weights = leggauss(96)
    x = 6 * nodes[:, None]
    q = 9 * nodes[:, None]
    dx = 6 * weights
    dq = 9 * weights
    gaussian = np.exp(-(x[:, 0] ** 2))
    g = np.stack((gaussian, x[:, 0] * gaussian), axis=-1)

    G = fourier.transform(
        g, mode="nonuniform", coordinates=x, frequencies=q, weights=dx, method=method
    )
    expected_gaussian = np.sqrt(np.pi) * np.exp(-(q[:, 0] ** 2) / 4)
    expected = np.stack(
        (expected_gaussian, -0.5j * q[:, 0] * expected_gaussian), axis=-1
    )

    assert_allclose(G, expected, atol=2e-11)

    recovered = fourier.inverse(
        G, mode="nonuniform", coordinates=x, frequencies=q, weights=dq, method=method
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
        method="direct",
    )

    assert_allclose(nonuniform, cartesian, atol=3e-14)

    frequency_area = (2 * np.pi) ** 2 / (g.size * 0.2 * 0.3)
    recovered = fourier.inverse(
        nonuniform.ravel(),
        mode="nonuniform",
        coordinates=coordinates,
        frequencies=frequencies.reshape(-1, 2),
        weights=np.full(g.size, frequency_area),
        method="direct",
    )

    assert_allclose(recovered.reshape(g.shape), g, atol=3e-14)


def test_complex_fourier_sum_preserves_decay_and_ignores_zero_growing_terms():
    q = np.array([[1.0], [2.0]])
    x = np.array([[2j], [-1000j]])
    g = np.array([1, 0])

    actual = NUFFT(g, x, q, sign=1, method="direct")

    assert_allclose(actual, np.exp(-2 * q[:, 0]))

    with pytest.raises(ValueError, match="overflows"):
        NUFFT([1, 1], x, q, sign=1, method="direct")

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


def test_nufft_honours_the_thread_setting(monkeypatch):
    pytest.importorskip("finufft")
    random = np.random.default_rng(3)
    x, q = random.normal(size=(300, 3)), random.normal(size=(40, 3))
    g = random.normal(size=(300, 2)) + 1j * random.normal(size=(300, 2))
    expected = NUFFT(g, x, q, method="direct")
    for threads in (1, 0, 3):
        assert_allclose(NUFFT(g, x, q, threads=threads), expected, atol=1e-10)
    monkeypatch.setenv("VECDIFF_NUFFT_THREADS", "2")
    assert_allclose(NUFFT(g, x, q), expected, atol=1e-10)
    with pytest.raises(ValueError, match="threads"):
        NUFFT(g, x, q, threads=-1)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("tilt", [0.0, 0.7])
def test_coplanar_targets_sum_in_two_dimensions_as_the_direct_sum(
    monkeypatch, sign, tilt
):
    """Targets on one plane, axis-aligned or tilted and far from the origin."""
    finufft = pytest.importorskip("finufft")
    from vecdiff.fourier.nufft import _plane

    random = np.random.default_rng(21)
    x = random.normal(size=(300, 3)) * 6
    g = random.normal(size=(300, 4)) + 1j * random.normal(size=(300, 4))
    c, s = np.cos(tilt), np.sin(tilt)
    axes = np.array([[c, 0.0, -s], [0.3 * s, np.sqrt(0.91), 0.3 * c]]).T
    a, b = np.meshgrid(np.linspace(-3, 3, 40), np.linspace(-2, 4, 30))
    q = [5.0, -2.0, 30.0] + np.stack((a, b), -1) @ axes.T
    bound = 1e-11 * np.abs(g).sum()
    three = []
    original = finufft.nufft3d3

    def counted(*args, **kwargs):
        three.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(finufft, "nufft3d3", counted)
    expected = NUFFT(g, x, q, sign=sign, method="direct")
    assert_allclose(NUFFT(g, x, q, sign=sign), expected, rtol=0, atol=bound)
    assert not three and _plane(q.reshape(-1, 3), x, 1e-12) is not None
    # One target off the plane by more than the tolerance allows: 3D.
    q[0, 0] += 1e-11 * np.cross(*axes.T)
    expected = NUFFT(g, x, q, sign=sign, method="direct")
    assert_allclose(NUFFT(g, x, q, sign=sign), expected, rtol=0, atol=bound)
    assert three and _plane(q.reshape(-1, 3), x, 1e-12) is None


def test_plane_wave_fields_on_a_tilted_plane_match_the_direct_sum():
    pytest.importorskip("finufft")
    import vecdiff as o

    region = o.media.HomogeneousRegion("glass", o.media.Dielectric(1.5))
    random = np.random.default_rng(4)
    s = random.normal(size=(200, 3))
    s /= np.linalg.norm(s, axis=1)[:, None]
    e = random.normal(size=(200, 3)) + 1j * random.normal(size=(200, 3))
    e -= s * np.sum(s * e, axis=1)[:, None]
    waves = o.fields.PlaneWaveExpansion(
        2 * np.pi * 1.5 * s, e, [0.3, 0.1, -0.2], 1.0, region
    )
    basis = np.linalg.qr(random.normal(size=(3, 3)))[0]
    basis[:, 2] *= np.sign(np.linalg.det(basis))
    axis = np.linspace(-4, 4, 33)
    grid = o.sampling.CartesianGrid(axis, axis, o.geometry.Frame([1, 2, 9], basis))
    direct = waves.evaluate(grid, method="direct")
    fast = waves.evaluate(grid, method="nufft")
    for a, b in (
        (fast.electric.values, direct.electric.values),
        (fast.magnetic, direct.magnetic),
    ):
        assert_allclose(a, b, rtol=0, atol=1e-11 * np.abs(b).max())
