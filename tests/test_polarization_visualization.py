import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vecdiff import FieldCartesian, Grid
from vecdiff.polarization import polarization_from_components, polarization_map_from_field
from vecdiff.polarization_visualization import (
    plot_field_polarization,
    plot_polarization_map,
)


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


def test_polarization_map_threshold_is_an_intensity_fraction():
    x = np.array([[0.0, 1.0]])
    y = np.array([[0.0, 0.0]])
    pol = polarization_from_components(
        np.array([[1.0 + 0.0j, 0.2 + 0.0j]]),
        np.zeros_like(x, dtype=complex),
    )

    fig, ax = plt.subplots()
    plot_polarization_map(
        x,
        y,
        pol,
        scale=1.0,
        ellipse_points=4,
        min_intensity_fraction=0.1,
        ellipse_mode="cartesian",
        ax=ax,
    )

    # The second sample has 4% of the peak intensity and must be excluded.
    assert len(ax.collections[0].get_segments()) == 4
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


def test_polarization_map_default_threshold_covers_faint_secondary_maxima():
    # A sample at 0.5% of the peak intensity models a faint secondary lobe.
    # The default threshold must keep it so the whole diffraction pattern is
    # shown, not just the bright core.
    x = np.array([[0.0, 1.0]])
    y = np.array([[0.0, 0.0]])
    pol = polarization_from_components(
        np.array([[1.0 + 0.0j, np.sqrt(0.005) + 0.0j]]),
        np.zeros_like(x, dtype=complex),
    )

    fig, ax = plt.subplots()
    plot_polarization_map(x, y, pol, scale=1.0, ellipse_points=4, ellipse_mode="cartesian", ax=ax)

    assert len(ax.collections[0].get_segments()) == 2 * 4
    plt.close(fig)


def test_polarization_map_max_radius_limits_glyphs():
    x = np.array([[0.0, 1.0]])
    y = np.array([[0.0, 0.0]])
    pol = polarization_from_components(
        np.array([[1.0 + 0.0j, 1.0 + 0.0j]]),
        np.zeros_like(x, dtype=complex),
    )

    fig, ax = plt.subplots()
    plot_polarization_map(
        x, y, pol, scale=1.0, ellipse_points=4, ellipse_mode="cartesian", max_radius=0.5, ax=ax
    )

    # Only the sample at the origin (r=0) is within max_radius; r=1 is dropped.
    assert len(ax.collections[0].get_segments()) == 1 * 4
    plt.close(fig)


def test_polarization_map_default_glyphs_have_halo():
    x = np.array([[0.0]])
    y = np.array([[1.0]])
    pol = polarization_from_components(np.array([[1.0 + 0.0j]]), np.array([[0.5j]]))

    fig, ax = plt.subplots()
    plot_polarization_map(x, y, pol, scale=1.0, ellipse_points=8, ax=ax)

    assert len(ax.collections[0].get_path_effects()) > 0
    plt.close(fig)


def test_polarization_map_color_override_disables_halo():
    x = np.array([[0.0]])
    y = np.array([[1.0]])
    pol = polarization_from_components(np.array([[1.0 + 0.0j]]), np.array([[0.5j]]))

    fig, ax = plt.subplots()
    plot_polarization_map(
        x, y, pol, scale=1.0, ellipse_points=8, curve_kwargs={"color": "red"}, ax=ax
    )

    assert not ax.collections[0].get_path_effects()
    plt.close(fig)


def test_polarization_map_draws_arrowhead_for_linear_light():
    # Linear light along x: the ellipse collapses to a line whose tangent
    # vanishes at the turning point the arrowhead is placed on.  The renderer
    # must fall back to the major axis so a head is still drawn.
    x = np.array([[0.0]])
    y = np.array([[0.0]])
    pol = polarization_from_components(np.array([[1.0 + 0.0j]]), np.array([[0.0 + 0.0j]]))

    fig, ax = plt.subplots()
    plot_polarization_map(x, y, pol, scale=1.0, ellipse_points=8, ellipse_mode="cartesian", ax=ax)

    # collections[0] is the curve, collections[1] the arrowhead (two segments).
    assert len(ax.collections) == 2
    assert len(ax.collections[1].get_segments()) == 2
    plt.close(fig)


def test_field_polarization_ellipse_mode_defaults_to_cartesian():
    # A y-polarized sample off the x-axis: the Cartesian basis draws the ellipse
    # extent along y, while the (rejected) polar default would rotate it.
    axis = np.linspace(-1.0, 1.0, 5)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    grid = Grid.from_cartesian(X, Y)
    field = FieldCartesian(np.zeros_like(X, dtype=complex), np.ones_like(Y, dtype=complex), grid=grid, symmetric=False)

    ax, _ = plot_field_polarization(field, background=None, ellipse_points=4, min_intensity_fraction=0.0)
    segments = ax.collections[0].get_segments()
    # Each ellipse for purely y-polarized light is a vertical line: constant x,
    # varying y about its centre.  In the rejected polar basis it would tilt.
    first = np.asarray(segments[0])
    assert np.allclose(first[:, 0], first[0, 0])
    plt.close(ax.figure)
