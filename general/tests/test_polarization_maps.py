"""Glyph layout and sizing of polarization maps."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import vecdiff_general as optics
from vecdiff_general.visualization import plot_polarization
from vecdiff_general.visualization.polarization import (
    _layout_nodes,
    _size_fraction,
)

WAVELENGTH = 532e-9


def glyphs(axis):
    curves = next(
        item.get_segments()
        for item in axis.collections
        if item.get_gid() == "polarization-ellipses"
    )
    curves = np.asarray(curves)

    return (curves[:, 0] + curves[:, 32]) / 2, curves


def uniform_disk(radius=18e-6, state=(1, 1j)):
    grid = optics.PolarGrid(
        np.linspace(0, radius, 65), np.arange(240) * 2 * np.pi / 240
    )
    values = np.broadcast_to(
        np.asarray(state) / np.linalg.norm(state), grid.shape + (2,)
    )

    return optics.TransverseElectricField(grid, WAVELENGTH, values)


def test_polar_rings_have_even_arc_pitch_and_reach_the_rim():
    grid = uniform_disk().grid
    limits = np.array([-18, 18, -18, 18])
    centers, pitch = _layout_nodes(grid, 2.0, limits, "polar")
    radius = np.round(np.linalg.norm(centers, axis=1), 9)
    rings = np.unique(radius)

    assert rings[0] == 0
    assert_step = np.diff(rings)
    assert np.allclose(assert_step, pitch[0])
    # The last ring sits half a pitch inside the sampled disk: no rim band.
    assert np.isclose(18 - rings[-1], pitch[0] / 2)

    for ring in rings[1:]:
        angles = np.sort(np.arctan2(*centers[radius == ring][:, ::-1].T) % (2 * np.pi))
        steps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
        chord = 2 * ring * np.sin(steps[0] / 2)
        assert np.allclose(steps, steps[0])
        assert 0.85 * pitch[0] <= chord <= 1.25 * pitch[0]


def test_cartesian_lattice_is_centered_with_half_cell_margins():
    grid = optics.CartesianGrid(
        np.linspace(-5e-6, 5e-6, 41), np.linspace(-3e-6, 3e-6, 25)
    )
    centers, pitch = _layout_nodes(grid, 0.7, np.array([-5, 5, -3, 3]), "cartesian")
    x, y = np.unique(centers[:, 0]), np.unique(centers[:, 1])

    assert len(x) % 2 == 1 and len(y) % 2 == 1
    assert np.isclose(x[len(x) // 2], 0) and np.isclose(y[len(y) // 2], 0)
    assert np.isclose(x[0] + 5, (x[1] - x[0]) / 2)
    assert np.isclose(y[0] + 3, (y[1] - y[0]) / 2)


@pytest.mark.parametrize("distribution", ["polar", "cartesian"])
def test_uniform_pupil_is_filled_to_its_rim(distribution):
    figure, axis = plt.subplots(figsize=(5, 5))
    plot_polarization(uniform_disk(), axis=axis, distribution=distribution)
    centers, curves = glyphs(axis)
    plt.close(figure)
    extent = np.linalg.norm(curves - centers[:, None], axis=-1).max(axis=1)
    reach = np.linalg.norm(centers, axis=1) + extent

    assert reach.max() <= 18 * (1 + 1e-9)
    # The pupil center carries a glyph; a disk chart has no inner edge.
    assert np.linalg.norm(centers, axis=1).min() < 1e-9 * 18
    # No empty rim: the outermost glyphs come within one glyph size of the edge.
    assert 18 - reach.max() < extent.max()
    # Glyphs are centered on their nodes; on native rings a constant field
    # has constant glyphs (lattice nodes near the rim may shrink to fit).
    assert np.allclose(curves[:, :-1].mean(axis=1), centers, atol=1e-9 * 18)
    if distribution == "polar":
        assert np.allclose(extent, extent.max(), rtol=0.08)


def test_linear_states_are_centered_double_headed_arrows():
    grid = optics.CartesianGrid(
        np.linspace(-4e-6, 4e-6, 33), np.linspace(-4e-6, 4e-6, 33)
    )
    angle = np.deg2rad(30)
    values = np.broadcast_to([np.cos(angle), np.sin(angle)], grid.shape + (2,))
    field = optics.TransverseElectricField(grid, WAVELENGTH, values)
    figure, axis = plt.subplots(figsize=(5, 5))
    plot_polarization(field, axis=axis)
    centers, _ = glyphs(axis)
    heads = next(
        item for item in axis.collections if item.get_gid() == "polarization-rotation"
    )
    arrows = [np.asarray(path.vertices) for path in heads.get_paths()]
    plt.close(figure)

    assert len(arrows) == len(centers)
    for arrow, center in zip(arrows, centers):
        tips = arrow[[0, 5]]
        axis_direction = tips[0] - tips[1]
        assert np.allclose(tips.mean(axis=0), center)
        assert np.isclose(
            np.arctan2(axis_direction[1], axis_direction[0]) % np.pi, angle
        )


def test_size_scales_are_monotone_and_span_floor_to_one():
    relative = np.logspace(-4, 0, 50)

    for scale in ("log", "sqrt", "linear", 0.25):
        size = _size_fraction(relative, scale, 0.4, 1e-4)
        assert np.all(np.diff(size) >= 0)
        assert np.isclose(size[-1], 1)
        assert size[0] >= 0.4

    log = _size_fraction(np.array([1e-4, 1e-2, 1]), "log", 0.4, 1e-4)
    assert np.allclose(log, [0.4, 0.7, 1])


def test_weak_rings_keep_glyphs_with_intensity_ordered_sizes():
    axis_values = np.linspace(-3e-6, 3e-6, 121)
    grid = optics.CartesianGrid(axis_values, axis_values)
    x, y = np.meshgrid(axis_values, axis_values)
    radius = np.hypot(x, y)
    # Airy-like rings: three side lobes of decreasing strength.
    amplitude = np.sinc(radius / 0.5e-6) * np.exp(-radius / 3e-6)
    field = optics.TransverseElectricField(
        grid, WAVELENGTH, amplitude[..., None] * np.array([1, 0])
    )
    figure, axis = plt.subplots(figsize=(5, 5))
    plot_polarization(field, axis=axis, distribution="polar", crop=False)
    centers, curves = glyphs(axis)
    plt.close(figure)
    size = np.linalg.norm(curves[:, 0] - centers, axis=1)
    distance = np.linalg.norm(centers, axis=1)

    # Glyphs remain out to the third side lobe (|r| up to 1.75 µm).
    assert distance.max() > 1.75
    assert size[np.argmin(distance)] == size.max()
    # A fixed radial pitch near the ring period used to drop whole rings into
    # the zeros; every bright ring (between zeros at k * 0.5 µm) keeps glyphs.
    for lobe in range(1, 4):
        assert np.any((distance > 0.5 * lobe + 0.05) & (distance < 0.5 * lobe + 0.45))


def test_line_glyphs_are_equal_centered_direction_segments():
    grid = optics.CartesianGrid(
        np.linspace(-4e-6, 4e-6, 33), np.linspace(-4e-6, 4e-6, 33)
    )
    x, y = np.meshgrid(grid.u, grid.v)
    angle = np.deg2rad(20)
    amplitude = np.exp(-(x**2 + y**2) / 3e-6**2)
    values = amplitude[..., None] * np.array([np.cos(angle), np.sin(angle)])
    field = optics.TransverseElectricField(grid, WAVELENGTH, values)
    figure, axis = plt.subplots(figsize=(5, 5))
    plot_polarization(
        field, axis=axis, glyph="line", glyph_color="black", size_scale=None
    )
    lines = next(
        item for item in axis.collections if item.get_gid() == "polarization-lines"
    )
    segments = np.asarray(lines.get_segments())
    plt.close(figure)
    direction = segments[:, 1] - segments[:, 0]
    length = np.linalg.norm(direction, axis=1)

    assert np.allclose(length, length[0], rtol=0.08)
    assert np.allclose(np.arctan2(direction[:, 1], direction[:, 0]) % np.pi, angle)
    assert np.allclose(lines.get_colors()[:, :3], 0)
