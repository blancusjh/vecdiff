import numpy as np
import pytest
from numpy.testing import assert_allclose
from vecdiff_general import (
    Dielectric,
    Frame,
    HomogeneousRegion,
    PlaneGrid,
    PlaneWaveExpansion,
    PointGrid,
)
from vecdiff_general.fourier.cartesian import FFT2, IFFT2
from vecdiff_general.fourier.nufft import NUFFT
from vecdiff_general.fourier.polar import HT, azimuthal_coefficients
from vecdiff_general.media.layers import LayerStack
from vecdiff_general.propagation.layered_propagation import propagate_layers


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
    direct = NUFFT(amplitudes, wavevectors, points, sign=1, backend="direct")
    accelerated = NUFFT(amplitudes, wavevectors, points, sign=1, backend="nufft")
    assert np.linalg.norm(direct - accelerated) / np.linalg.norm(direct) < 5e-12


@pytest.mark.parametrize("thickness_wavelengths", [0.1, 0.5, 1.1, 3.2])
def test_slab_against_airy_and_each_boundary(thickness_wavelengths):
    wavelength = 532e-9
    regions = tuple(
        HomogeneousRegion(name, Dielectric(index))
        for name, index in [("left", 1), ("glass", 1.5), ("right", 1)]
    )
    thickness = thickness_wavelengths * wavelength
    stack = LayerStack(regions, (thickness,))
    incident = PlaneWaveExpansion(
        [[0, 0, 2 * np.pi / wavelength]],
        [[1, 0.3j, 0]],
        [0, 0, 0],
        wavelength,
        regions[0],
    )
    solution = propagate_layers(incident, stack)
    phase = np.exp(2j * np.pi * 1.5 * thickness / wavelength)
    expected_reflection = (-0.2 + 0.2 * phase**2) / (1 - 0.04 * phase**2)
    expected_transmission = 0.96 * phase / (1 - 0.04 * phase**2)
    assert_allclose(solution.reflection[0], expected_reflection, atol=1e-14)
    assert_allclose(solution.transmission[0], expected_transmission, atol=1e-14)
    assert_allclose(
        abs(expected_reflection) ** 2 + abs(expected_transmission) ** 2, 1, atol=1e-14
    )
    for boundary, position in enumerate(stack.boundaries):
        grid = PointGrid([[0, 0, position], [wavelength / 3, 0, position]])
        before = solution.evaluate(grid, region=boundary)
        after = solution.evaluate(grid, region=boundary + 1)
        assert_allclose(
            before.electric.values[..., :2], after.electric.values[..., :2], atol=3e-14
        )
        assert_allclose(before.magnetic[..., :2], after.magnetic[..., :2], atol=3e-16)


def test_frustrated_tir_flux_and_thick_gap_stability():
    wavelength = 532e-9
    regions = tuple(
        HomogeneousRegion(name, Dielectric(index))
        for name, index in [("left", 1.5), ("gap", 1), ("right", 1.5)]
    )
    direction = np.array([np.sin(1.0), 0, np.cos(1.0)])
    incident = PlaneWaveExpansion(
        [2 * np.pi * 1.5 / wavelength * direction],
        [[0, 1, 0]],
        [0, 0, 0],
        wavelength,
        regions[0],
    )
    powers = []
    for width in (0.15 * wavelength, 1000 * wavelength):
        solution = propagate_layers(incident, LayerStack(regions, (width,)))
        grid = PlaneGrid(
            np.arange(4) * wavelength, np.arange(4) * wavelength, Frame([0, 0, width])
        )
        before = solution.evaluate(grid.translated([0, 0, -width]), region=0)
        after = solution.evaluate(grid, region=2)
        powers.append(after.power())
        assert_allclose(before.power(), after.power(), atol=1e-25, rtol=1e-12)
        middle = solution.evaluate(PointGrid([[0, 0, width / 2]]), region=1)
        assert np.isfinite(middle.electric.values).all()
    assert powers[0] > 0
    assert powers[1] < 1e-100 * powers[0]
