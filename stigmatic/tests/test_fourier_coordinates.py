"""Typed physical grids and the common Cartesian/polar Fourier convention."""

import numpy as np
import pytest
import vecdiff_stigmatic as optics
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose
from vecdiff_stigmatic import fourier


def gauss_axis(limit, count):
    nodes, weights = leggauss(count)

    return limit * (nodes + 1) / 2, limit * weights / 2


def test_grid_types_fix_array_order_and_physical_coordinates():
    basis = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    frame = optics.Frame([1e-3, 2e-3, 3e-3], basis)
    cartesian = optics.CartesianGrid([0, 2e-6, 4e-6], [-1e-6, 1e-6], frame)

    assert cartesian.array_order == ("v", "u")
    assert cartesian.shape == (2, 3)
    assert cartesian.coordinate_system == "cartesian"
    assert optics.PlaneGrid is optics.CartesianGrid
    assert_allclose(
        cartesian.points[1, 2], frame.origin + 4e-6 * frame.u + 1e-6 * frame.v
    )

    polar = optics.PolarGrid([0, 1e-6, 2e-6], np.arange(4) * np.pi / 2, frame)

    assert polar.array_order == ("radius", "azimuth")
    assert polar.shape == (3, 4)
    assert polar.coordinate_system == "polar"
    assert_allclose(polar.points[1, 1], frame.origin + 1e-6 * frame.v)
    assert_allclose(polar.weights.sum(), np.pi * (2e-6) ** 2)
    assert_allclose(polar.translated([0, 0, 5e-6]).points, polar.points + [0, 0, 5e-6])

    values = np.zeros(polar.shape + (2,), dtype=complex)
    transverse = optics.TransverseElectricField(polar, 532e-9, values)

    assert transverse.grid is polar


def test_polar_grid_rejects_duplicate_period_and_invalid_radial_order():
    with pytest.raises(ValueError, match="duplicate endpoint"):
        optics.PolarGrid([0, 1], np.linspace(0, 2 * np.pi, 8))

    with pytest.raises(ValueError, match="increasing"):
        optics.PolarGrid([1, 0], np.arange(8) * np.pi / 4)


def test_cartesian_transform_preserves_coordinate_phase_and_continuous_measure():
    u = 0.23 + np.arange(12) * 0.13
    v = -0.17 + np.arange(9) * 0.21
    grid = optics.CartesianGrid(u, v)
    frequency_u, frequency_v = fourier.frequency_grid(grid.shape, grid.spacing)
    q_u, q_v = frequency_u[2, 3], frequency_v[2, 3]
    values = np.exp(1j * (q_u * grid.points[..., 0] + q_v * grid.points[..., 1]))

    transformed = fourier.transform(
        values, mode=grid.coordinate_system, coordinates=grid.coordinates
    )
    expected = np.zeros(grid.shape, dtype=complex)
    expected[2, 3] = grid.weights.sum()

    assert_allclose(transformed, expected, atol=2e-14)

    recovered = fourier.inverse(
        transformed, mode=grid.coordinate_system, coordinates=grid.coordinates
    )

    assert_allclose(recovered, values, atol=2e-14)


def test_polar_transform_and_inverse_against_nonaxisymmetric_gaussians():
    radius, radial_weights = gauss_axis(6, 96)
    frequency, frequency_weights = gauss_axis(14, 112)
    azimuth = 0.17 + 2 * np.pi * np.arange(16) / 16
    frequency_azimuth = -0.23 + 2 * np.pi * np.arange(20) / 20
    grid = optics.PolarGrid(radius, azimuth, dr_weights=radial_weights)
    orders = (0, 1, -2)

    values = np.stack(
        [
            radius[:, None] ** abs(order)
            * np.exp(-(radius[:, None] ** 2))
            * np.exp(1j * order * azimuth)
            for order in orders
        ],
        axis=-1,
    )

    transformed = fourier.transform(
        values,
        mode=grid.coordinate_system,
        coordinates=grid.coordinates,
        frequencies=(frequency, frequency_azimuth),
        weights=grid.dr_weights,
    )
    expected = np.stack(
        [
            np.pi
            * (-1j) ** abs(order)
            * (frequency[:, None] / 2) ** abs(order)
            * np.exp(-(frequency[:, None] ** 2) / 4)
            * np.exp(1j * order * frequency_azimuth)
            for order in orders
        ],
        axis=-1,
    )

    assert_allclose(transformed, expected, atol=5e-13)

    recovered = fourier.inverse(
        transformed,
        mode="polar",
        coordinates=grid.coordinates,
        frequencies=(frequency, frequency_azimuth),
        weights=frequency_weights,
    )

    assert_allclose(recovered, values, atol=5e-12)


def test_cartesian_and_polar_modes_share_the_same_fourier_convention():
    axis = (np.arange(128) - 64) * 0.1
    cartesian = optics.CartesianGrid(axis, axis)
    values = np.exp(-np.sum(cartesian.points[..., :2] ** 2, axis=-1))
    fft = fourier.transform(values, mode="cartesian", coordinates=cartesian.coordinates)

    radius, weights = gauss_axis(6, 96)
    azimuth = 2 * np.pi * np.arange(16) / 16
    q = 2 * np.pi * np.fft.fftfreq(len(axis), 0.1)[:10]
    radial_values = np.broadcast_to(
        np.exp(-(radius[:, None] ** 2)), (len(radius), len(azimuth))
    )
    polar = fourier.transform(
        radial_values,
        mode="polar",
        coordinates=(radius, azimuth),
        frequencies=(q, azimuth),
        weights=weights,
    )

    assert_allclose(fft[0, :10], polar[:, 0], atol=5e-13)


def test_transform_rejects_ambiguous_shape_and_unsupported_mode():
    with pytest.raises(ValueError, match="leading shape"):
        fourier.transform(
            np.zeros((3, 5)), mode="cartesian", coordinates=(np.arange(3), np.arange(5))
        )

    with pytest.raises(ValueError, match="mode"):
        fourier.transform(
            np.zeros((3, 3)), mode="guess", coordinates=(np.arange(3), np.arange(3))
        )
