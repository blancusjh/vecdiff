import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vecdiff import FieldCartesian, Grid
from vecdiff.polarization import polarization_from_components, polarization_map_from_field
from vecdiff.polarization_visualization import plot_polarization_map


def _first_curve_point(ax):
    return np.asarray(ax.collections[0].get_segments()[0][0])


def _ellipse_half_width(ax, ellipse_index, ellipse_points=4):
    segments = ax.collections[0].get_segments()
    points = np.asarray([segment[0] for segment in segments[ellipse_index * ellipse_points : (ellipse_index + 1) * ellipse_points]])
    return 0.5 * (float(np.max(points[:, 0])) - float(np.min(points[:, 0])))


def test_polarization_map_defaults_to_local_polar_ellipses():
    x = np.array([[0.0]])
    y = np.array([[1.0]])
    pol = polarization_from_components(np.array([[1.0 + 0.0j]]), np.array([[0.0 + 0.0j]]))

    fig, ax = plt.subplots()
    plot_polarization_map(x, y, pol, scale=1.0, ellipse_points=4, ax=ax)

    assert np.allclose(_first_curve_point(ax), [0.0, 2.0])
    plt.close(fig)


def test_polarization_map_cartesian_mode_preserves_xy_ellipses():
    x = np.array([[0.0]])
    y = np.array([[1.0]])
    pol = polarization_from_components(np.array([[1.0 + 0.0j]]), np.array([[0.0 + 0.0j]]))

    fig, ax = plt.subplots()
    plot_polarization_map(x, y, pol, scale=1.0, ellipse_points=4, ellipse_mode="cartesian", ax=ax)

    assert np.allclose(_first_curve_point(ax), [1.0, 1.0])
    plt.close(fig)


def test_polarization_map_rejects_unknown_ellipse_mode():
    x = np.array([[0.0]])
    y = np.array([[1.0]])
    pol = polarization_from_components(np.array([[1.0 + 0.0j]]), np.array([[0.0 + 0.0j]]))

    with pytest.raises(ValueError, match="ellipse_mode"):
        plot_polarization_map(x, y, pol, scale=1.0, ellipse_mode="cylindrical")


def test_polarization_map_default_keeps_uniform_sample_without_amplitude_cap():
    axis = np.linspace(-1.0, 1.0, 40)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    amp = np.ones_like(x)
    amp[::2, ::2] = 0.1
    pol = polarization_from_components(amp + 0.0j, np.zeros_like(x, dtype=complex))

    fig, ax = plt.subplots()
    plot_polarization_map(x, y, pol, target_ellipses=20, ellipse_points=4, min_intensity_fraction=0.0, ax=ax)

    assert len(ax.collections[0].get_segments()) == 20 * 20 * 4
    plt.close(fig)


def test_polarization_map_power_intensity_scaling_uses_gamma():
    x = np.array([[0.0, 2.0]])
    y = np.array([[0.0, 0.0]])
    pol = polarization_from_components(np.array([[1.0 + 0.0j, 0.25 + 0.0j]]), np.zeros_like(x, dtype=complex))

    fig_linear, ax_linear = plt.subplots()
    plot_polarization_map(
        x,
        y,
        pol,
        scale=1.0,
        ellipse_points=4,
        min_intensity_fraction=0.0,
        min_ellipse_scale=0.0,
        scale_by_intensity=True,
        intensity_scale_mode="linear",
        ellipse_mode="cartesian",
        ax=ax_linear,
    )

    fig_power, ax_power = plt.subplots()
    plot_polarization_map(
        x,
        y,
        pol,
        scale=1.0,
        ellipse_points=4,
        min_intensity_fraction=0.0,
        min_ellipse_scale=0.0,
        scale_by_intensity=True,
        intensity_scale_mode="power",
        intensity_scale_gamma=0.5,
        ellipse_mode="cartesian",
        ax=ax_power,
    )

    assert np.isclose(_ellipse_half_width(ax_linear, 1), 0.25)
    assert np.isclose(_ellipse_half_width(ax_power, 1), 0.5)
    plt.close(fig_linear)
    plt.close(fig_power)


def test_polarization_map_rejects_unknown_intensity_scale_mode():
    x = np.array([[0.0]])
    y = np.array([[1.0]])
    pol = polarization_from_components(np.array([[1.0 + 0.0j]]), np.array([[0.0 + 0.0j]]))

    with pytest.raises(ValueError, match="intensity_scale_mode"):
        plot_polarization_map(x, y, pol, scale=1.0, intensity_scale_mode="sqrt")


def test_polarization_map_from_cartesian_field_respects_half_size():
    axis = np.linspace(-2.0, 2.0, 5)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    grid = Grid.from_cartesian(X, Y)
    field = FieldCartesian(X + 0.0j, Y + 0.0j, grid=grid, symmetric=False)

    xx, yy, pol = polarization_map_from_field(field, half_size=1.0, n_img=3)

    assert np.allclose(xx[0], [-1.0, 0.0, 1.0])
    assert np.allclose(yy[:, 0], [-1.0, 0.0, 1.0])
    assert np.allclose(pol.ex, xx)
    assert np.allclose(pol.ey, yy)
