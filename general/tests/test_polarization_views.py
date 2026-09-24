"""Physical projection, polarization degeneracies and readable glyph placement."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.distance import pdist, squareform
from scipy.special import j1
from vecdiff_general import CartesianGrid, ElectricField, Frame, PolarGrid
from vecdiff_general.visualization import (
    animate_polarization,
    plot_polarization,
    plot_polarization_scalar,
)
from vecdiff_general.visualization.polarization import (
    _ellipse_curves,
    _evaluate_glyphs,
    _phase_geometry,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def make_field(grid=None, state=(1, 1j, 0), envelope=None):
    if grid is None:
        axis = np.linspace(-4, 4, 81) * 1e-6
        grid = CartesianGrid(axis, axis)

    amplitude = np.ones(grid.shape) if envelope is None else envelope
    values = amplitude[..., None] * np.asarray(state) @ grid.frame.basis.T

    return ElectricField(grid, 532e-9, values)


def glyphs(axis):
    collection = next(
        item for item in axis.collections if item.get_gid() == "polarization-ellipses"
    )
    curves = np.asarray(collection.get_segments())
    centers = (curves[:, 0] + curves[:, 32]) / 2
    radii = np.linalg.norm(curves[:, 0] - centers, axis=-1)

    return curves, centers, radii


def test_component_intensity_can_skip_polarization_without_changing_values(monkeypatch):
    from vecdiff_general.visualization import fields

    field = make_field(state=(0, 0, 2))
    longitudinal = abs(field.values[..., 2]) ** 2

    def unexpected_overlay(*args, **kwargs):
        raise AssertionError("A component map must not build polarization glyphs")

    monkeypatch.setattr(fields, "plot_polarization", unexpected_overlay)
    axis = fields.plot_intensity(field, longitudinal, polarization=False)

    assert_allclose(axis.images[0].get_array(), longitudinal)
    assert_allclose(axis.get_xlim(), [-4, 4])
    assert not axis.collections


def electric_arrows(collection):
    polygons = np.array([path.vertices[:7] for path in collection.get_paths()])
    tails = (polygons[:, 3] + polygons[:, 4]) / 2

    return polygons, tails, polygons[:, 0]


def test_static_heads_are_off_axis_and_linear_has_two_outward_heads():
    chi = np.array([np.pi / 4, -np.pi / 4, 0])
    curves, heads, owners = _ellipse_curves(
        np.zeros(3), chi, np.zeros((3, 2)), np.ones(3)
    )
    assert_allclose(curves[:2, 0], [[1, 0], [1, 0]])
    assert_allclose(owners, [0, 1, 2])
    triangles = np.array(heads[:2])
    assert_allclose(triangles[:, 0], [[2**-0.5, 2**-0.5], [2**-0.5, -(2**-0.5)]])
    directions = triangles[:, 0] - triangles[:, 1:].mean(axis=1)
    assert directions[0, 1] > 0
    assert directions[1, 1] < 0
    assert_allclose(heads[2][[0, 5]], [[1, 0], [-1, 0]])
    assert len(heads[2]) == 10
    assert_allclose(curves[2, :, 1], 0)


def test_geometry_is_unchanged_by_global_phase_or_rotating_frame():
    field = make_field(state=(1, 0.4j, 0.7))
    expected = glyphs(plot_polarization(field))[0]
    phased = ElectricField(field.grid, field.wavelength0, field.values * np.exp(1.2j))
    assert_allclose(glyphs(plot_polarization(phased))[0], expected, atol=1e-12)

    basis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    grid = CartesianGrid(field.grid.u, field.grid.v, Frame([1, 2, 3], basis))
    rotated = make_field(grid, state=(1, 0.4j, 0.7))
    assert_allclose(glyphs(plot_polarization(rotated))[0], expected, atol=1e-12)


def test_glyphs_are_separated_and_count_is_bounded():
    axis = plot_polarization(make_field(), max_glyphs=60)
    _, centers, radii = glyphs(axis)
    assert 10 < len(centers) <= 60
    distance = squareform(pdist(centers))
    np.fill_diagonal(distance, np.inf)
    assert np.all(distance > radii[:, None] + radii[None, :])


def test_single_glyph_limit_keeps_a_readable_nonzero_state():
    axis = plot_polarization(make_field(), max_glyphs=1)
    _, centers, radii = glyphs(axis)
    assert len(centers) == 1
    assert radii[0] > 0
    assert not axis.texts


def test_faint_disconnected_lobe_remains_visible_and_crop_follows_displacement():
    field = make_field()
    x, y = np.meshgrid(field.grid.u * 1e6, field.grid.v * 1e6)
    amplitude = np.exp(-((x - 1.6) ** 2 + y**2) / 0.3)
    amplitude += 0.08 * np.exp(-((x + 1.8) ** 2 + y**2) / 0.3)
    field = make_field(field.grid, envelope=amplitude)
    axis = plot_polarization(field, max_glyphs=50)
    _, centers, _ = glyphs(axis)
    assert np.any(centers[:, 0] < -1)
    assert np.any(centers[:, 0] > 1)
    assert np.ptp(axis.get_ylim()) < 4


def test_secondary_diffraction_maxima_keep_polarization_glyphs():
    field = make_field()
    x, y = np.meshgrid(field.grid.u * 1e6, field.grid.v * 1e6)
    argument = 3 * np.hypot(x, y)
    amplitude = np.divide(
        2 * j1(argument), argument, out=np.ones_like(argument), where=argument > 0
    )
    axis = plot_polarization(make_field(field.grid, envelope=amplitude))
    _, centers, _ = glyphs(axis)
    radius = np.linalg.norm(centers, axis=1)

    # The first two Airy secondary maxima lie near 5.14/3 and 8.42/3 µm.
    assert np.count_nonzero((radius > 1.5) & (radius < 1.9)) >= 3
    assert np.count_nonzero((radius > 2.6) & (radius < 3.0)) >= 3


def test_polar_diffraction_layout_resolves_weak_rings_with_readable_sizes():
    field = make_field()
    x, y = np.meshgrid(field.grid.u * 1e6, field.grid.v * 1e6)
    argument = 3 * np.hypot(x, y)
    amplitude = np.divide(
        2 * j1(argument), argument, out=np.ones_like(argument), where=argument > 0
    )
    axis = plot_polarization(
        make_field(field.grid, envelope=amplitude), distribution="polar", crop=False
    )
    _, centers, radii = glyphs(axis)
    distance = np.linalg.norm(centers, axis=1)

    assert np.count_nonzero((distance > 1.5) & (distance < 1.9)) >= 12
    assert np.count_nonzero((distance > 2.6) & (distance < 3.0)) >= 20
    assert radii.min() > 0.5 * radii.max()


def test_circular_orientation_and_dark_field_are_masked():
    axis = plot_polarization_scalar(make_field(), "orientation")
    assert np.ma.getmaskarray(axis.images[0].get_array()).all()

    dark = make_field(state=(0, 0, 1))
    axis = plot_polarization(dark)
    assert not any(
        item.get_gid() == "polarization-ellipses" for item in axis.collections
    )
    assert "No projected field" in axis.texts[0].get_text()


def test_polar_origin_is_not_repeated_and_vectors_are_not_radialized():
    grid = PolarGrid(np.linspace(0, 4e-6, 41), np.arange(96) * 2 * np.pi / 96)
    axis = plot_polarization(make_field(grid, state=(0, 1, 0)))
    curves, centers, _ = glyphs(axis)
    assert np.count_nonzero(np.linalg.norm(centers, axis=1) < 1e-10) == 1
    assert_allclose(
        curves[..., 0], np.broadcast_to(centers[:, :1], curves.shape[:2]), atol=1e-14
    )


def test_small_subplot_uses_fewer_glyphs_and_resolution_does_not_set_density():
    field = make_field()
    large = plot_polarization(field)
    _, small = plt.subplots(figsize=(3, 3), layout="constrained")
    plot_polarization(field, axis=small, legend=False)
    assert len(glyphs(small)[1]) < len(glyphs(large)[1])

    grid = CartesianGrid(np.linspace(-4, 4, 161) * 1e-6, np.linspace(-4, 4, 161) * 1e-6)
    fine = plot_polarization(make_field(grid))
    ratio = len(glyphs(fine)[1]) / len(glyphs(large)[1])
    assert 0.75 < ratio < 1.25


def test_animation_preserves_phase_and_ellipse_radius():
    animation = animate_polarization(make_field(), frames=16, density=0.3)
    arrows, _ = animation._func(0)
    _, centers, radii = glyphs(arrows.axes)
    _, tails, tips = electric_arrows(arrows)
    first = tips - centers
    assert_allclose(tails, centers, atol=1e-14)
    animation._func(4)
    _, tails, tips = electric_arrows(arrows)
    assert_allclose(tails, centers, atol=1e-14)
    quarter = tips - centers

    assert np.all(first[:, 0] > 0)
    assert_allclose(first[:, 1], 0, atol=1e-14)
    assert_allclose(quarter[:, 0], 0, atol=1e-14)
    assert_allclose(quarter[:, 1], first[:, 0])
    assert_allclose(np.linalg.norm(quarter, axis=1), radii)


def test_phase_arrows_start_at_centers_and_heads_align_with_electric_vectors():
    vectors = np.array([[1, 0.4j], [1j, -0.4], [1, 0]], dtype=complex)
    centers = np.array([[2, -1], [-3, 4], [5, 7]])
    radii = np.ones(3)
    phase = 0.7
    segments, heads, owners = _phase_geometry(vectors, centers, radii, phase)
    expected = np.real(vectors * np.exp(-1j * phase))
    assert_allclose(segments[:, 0], centers[owners])
    assert_allclose(segments[:, 1], centers[owners] + expected[owners])
    assert_allclose(heads[:, 0], segments[:, 1])

    direction = heads[:, 0] - heads[:, 1:].mean(axis=1)
    assert np.all(np.sum(direction * expected[owners], axis=1) > 0)
    cross = (
        direction[:, 0] * expected[owners, 1] - direction[:, 1] * expected[owners, 0]
    )
    assert_allclose(cross, 0, atol=1e-14)

    segments, heads, owners = _phase_geometry(vectors, centers, radii, 0)
    assert 2 in owners
    assert_allclose(segments[-1, 1], centers[2] + [1, 0])

    segments, heads, owners = _phase_geometry(vectors, centers, radii, np.pi / 2)
    assert 2 not in owners


def test_rotated_ellipses_and_arrows_share_the_same_fixed_centers():
    from matplotlib.collections import PathCollection

    field = make_field(state=(1 + 0.3j, 0.7 + 0.9j, 0))
    animation = animate_polarization(field, frames=16, density=0.3)

    for frame in range(16):
        arrows, _ = animation._func(frame)
        curves, centers, _ = glyphs(arrows.axes)
        polygons, tails, tips = electric_arrows(arrows)
        assert_allclose(tails, centers, atol=1e-14)
        assert_allclose(
            (curves[:, :32] + curves[:, 32:64]) / 2,
            np.broadcast_to(centers[:, None], curves[:, :32].shape),
            atol=1e-14,
        )
        assert not any(isinstance(c, PathCollection) for c in arrows.axes.collections)
        assert not any(
            c.get_gid() == "polarization-vectors" for c in arrows.axes.collections
        )

        # The head's base is symmetric about the shaft and never behind its tail.
        direction = tips - tails
        base = (polygons[:, 1] + polygons[:, 6]) / 2
        assert np.all(np.sum((base - centers) * direction, axis=1) >= 0)


def test_static_phase_matches_animation_at_the_same_time():
    field = make_field(state=(np.exp(0.3j), 0.6j * np.exp(0.3j), 0))
    axis = plot_polarization(field, phase=1.2, density=0.3)
    expected = next(
        c for c in axis.collections if c.get_gid() == "polarization-rotation"
    )
    animation = animate_polarization(field, phase=1.2, density=0.3)
    arrows, _ = animation._func(0)
    assert_allclose(electric_arrows(arrows)[0], electric_arrows(expected)[0])


def test_dark_center_is_empty_and_neighbouring_glyphs_are_symmetric():
    field = make_field()
    x, y = np.meshgrid(field.grid.u * 1e6, field.grid.v * 1e6)
    values = np.stack((x, y, np.zeros_like(x)), axis=-1)
    values *= np.exp(-(x**2 + y**2) / 5)[..., None]
    field = ElectricField(field.grid, field.wavelength0, values)
    _, centers, radii = glyphs(plot_polarization(field))

    # No off-lattice glyph is inserted to fill the central polarization null.
    assert np.all(np.linalg.norm(centers, axis=1) > 2 * radii)
    for point in centers:
        assert np.min(np.linalg.norm(centers + point, axis=1)) < 1e-12

    # Every row uses the same columns: no intensity-ranked staggered placement.
    columns = np.unique(np.round(centers[:, 0], 10))
    rows = np.unique(np.round(centers[:, 1], 10))
    assert_allclose(columns, rows)
    steps = np.diff(columns)
    assert np.ptp(steps) <= 0.1000001  # at most one underlying grid interval


def test_linear_glyph_has_one_closed_outline_without_a_separate_shaft():
    from matplotlib.collections import LineCollection

    axis = plot_polarization(make_field(state=(1, 0, 0)))
    arrows = next(c for c in axis.collections if c.get_gid() == "polarization-rotation")
    curves, centers, _ = glyphs(axis)
    assert len(arrows.get_paths()) == len(curves)

    for path, center in zip(arrows.get_paths(), centers):
        assert len(path.vertices) == 11  # ten boundary corners plus closing vertex
        assert_allclose(path.vertices[0], path.vertices[-1])
        assert_allclose((path.vertices[0] + path.vertices[5]) / 2, center)

    for collection in axis.collections:
        if isinstance(collection, LineCollection):
            assert np.all(collection.get_colors()[:, 3] == 0)


def test_narrow_layout_has_clear_legend_and_matching_colorbar_height():
    field = make_field()
    x, y = np.meshgrid(field.grid.u * 1e6, field.grid.v * 1e6)
    field = make_field(field.grid, envelope=np.exp(-(x**2) / 8 - y**2 / 0.05))
    axis = plot_polarization(field, color_by="handedness")
    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend = axis.get_legend().get_window_extent(renderer)
    xlabel = axis.xaxis.label.get_window_extent(renderer)

    assert not legend.overlaps(xlabel)
    assert figure.get_figheight() < 4
    colorbar = axis.child_axes[0].get_window_extent(renderer)
    assert_allclose(colorbar.height / axis.get_window_extent(renderer).height, 0.82)


def test_intensity_threshold_does_not_use_amplitude_and_density_is_adjustable():
    field = make_field()
    values = field.values.copy()
    values[:, :30] *= 0.2
    field = ElectricField(field.grid, field.wavelength0, values)
    sparse = plot_polarization(field, min_intensity_fraction=0.1, density=0.4)
    dense = plot_polarization(field, min_intensity_fraction=0.1, density=1.2)
    sparse_centers = glyphs(sparse)[1]
    dense_centers = glyphs(dense)[1]

    assert np.all(dense_centers[:, 0] >= -1)
    assert len(dense_centers) > len(sparse_centers)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"density": 0},
        {"density": np.nan},
        {"max_glyphs": 0},
        {"stride": True},
        {"min_intensity_fraction": 1},
        {"intensity_gamma": -1},
    ],
)
def test_invalid_controls_fail_clearly(kwargs):
    with pytest.raises(ValueError):
        plot_polarization(make_field(), **kwargs)


def test_native_cartesian_layout_has_exact_equal_steps():
    field = make_field()
    _, centers, _ = glyphs(plot_polarization(field, density=0.73))

    for component, nodes in zip(centers.T, (field.grid.u, field.grid.v)):
        coordinates = np.unique(np.round(component, 12))
        steps = np.diff(coordinates)
        assert len(steps) > 3
        assert_allclose(steps, steps[0], atol=1e-12)
        distance = abs(component[:, None] - nodes[None, :] * 1e6)
        assert np.max(distance.min(axis=1)) < 1e-12


@pytest.mark.parametrize("native", [False, True])
def test_polar_layout_has_exact_concentric_rings_and_periodic_angles(native):
    grid = None

    if native:
        grid = PolarGrid(np.linspace(0, 4e-6, 81), np.arange(240) * 2 * np.pi / 240)

    field = make_field(grid)
    _, centers, _ = glyphs(plot_polarization(field, distribution="polar", crop=False))
    radius = np.linalg.norm(centers, axis=1)
    rings = np.unique(np.round(radius, 10))
    assert len(rings) >= 4
    assert np.count_nonzero(radius < 1e-10) == 1
    inner_rings = rings[rings < 3]
    assert_allclose(np.diff(inner_rings), np.diff(inner_rings)[0], atol=1e-10)

    for ring in rings[1:]:
        points = centers[abs(radius - ring) < 1e-9]
        angles = np.sort(np.arctan2(points[:, 1], points[:, 0]) % (2 * np.pi))
        steps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])

        # Full rings inside the square view must have exactly periodic angles.
        if ring < 3:
            assert len(points) >= 4
            assert_allclose(steps, 2 * np.pi / len(points), atol=1e-10)

        if native:
            assert np.min(abs(grid.radius * 1e6 - ring)) < 1e-9
            phase = angles * len(grid.azimuth) / (2 * np.pi)
            assert_allclose(phase, np.round(phase), atol=1e-10)


@pytest.mark.parametrize("distribution", ["cartesian", "polar"])
def test_whole_glyphs_stay_inside_polar_annulus(distribution):
    grid = PolarGrid(np.linspace(2e-6, 4e-6, 81), np.arange(240) * 2 * np.pi / 240)
    _, centers, radii = glyphs(
        plot_polarization(make_field(grid), distribution=distribution, crop=False)
    )
    distance = np.linalg.norm(centers, axis=1)
    assert len(centers) > 15
    assert np.all(distance - 1.25 * radii >= 2 - 1e-12)
    assert np.all(distance + 1.25 * radii <= 4 + 1e-12)


def test_alternate_layout_interpolates_complex_components_without_extrapolation():
    field = make_field()
    x, y = np.meshgrid(field.grid.u * 1e6, field.grid.v * 1e6)
    values = np.stack((1 + 0.3j * x, y + 0.5j, 2 * x - y), axis=-1)
    field = ElectricField(field.grid, field.wavelength0, values)
    centers = np.array([[0.137, -0.219], [-2.381, 1.627]])
    expected = np.column_stack(
        (
            1 + 0.3j * centers[:, 0],
            centers[:, 1] + 0.5j,
            2 * centers[:, 0] - centers[:, 1],
        )
    )
    assert_allclose(_evaluate_glyphs(field, centers), expected, atol=1e-14)
    assert np.isnan(_evaluate_glyphs(field, np.array([[5, 0]]))).all()


def test_polar_interpolation_closes_complex_phase_seam():
    angles = 0.17 + np.arange(80) * 2 * np.pi / 80
    grid = PolarGrid(np.linspace(0.5e-6, 4e-6, 51), angles)
    values = np.zeros(grid.shape + (3,), dtype=complex)
    values[..., 0] = np.exp(1j * angles)
    field = ElectricField(grid, 532e-9, values)
    angle = angles[0] - np.pi / 80
    center = 2 * np.array([[np.cos(angle), np.sin(angle)]])
    expected = (np.exp(1j * angles[-1]) + np.exp(1j * angles[0])) / 2
    assert_allclose(_evaluate_glyphs(field, center)[0, 0], expected, atol=1e-14)


def test_compressed_sizes_preserve_positions_and_readable_weak_states():
    field = make_field()
    x, y = np.meshgrid(field.grid.u * 1e6, field.grid.v * 1e6)
    field = make_field(field.grid, envelope=np.exp(-(x**2 + y**2) / 3))
    _, centers, radii = glyphs(plot_polarization(field))
    _, constant_centers, constant_radii = glyphs(
        plot_polarization(field, scale_by_intensity=False)
    )
    assert_allclose(centers, constant_centers, atol=1e-12)
    values = _evaluate_glyphs(field, centers)
    relative = np.sum(abs(values[:, :2]) ** 2, axis=1) / 2
    assert_allclose(radii / constant_radii, 0.5 + 0.5 * relative**0.25)
    assert np.all(radii >= 0.5 * constant_radii)
    assert np.all(np.diff(radii[np.argsort(relative)]) >= -1e-12)


@pytest.mark.parametrize("distribution", ["cartesian", "polar"])
def test_animation_uses_same_exact_layout_phasors_as_static_view(distribution):
    field = make_field()
    x, y = np.meshgrid(field.grid.u * 1e6, field.grid.v * 1e6)
    envelope = np.exp(0.7j * x - 0.3j * y)
    field = make_field(field.grid, state=(1, 0.6j, 0), envelope=envelope)
    options = {"distribution": distribution, "density": 0.6}
    phase = 0.41
    axis = plot_polarization(field, phase=phase, **options)
    static = next(
        item for item in axis.collections if item.get_gid() == "polarization-rotation"
    )
    animation = animate_polarization(field, phase=phase, **options)
    animated, _ = animation._func(0)

    for first, second in zip(static.get_paths(), animated.get_paths(), strict=True):
        assert_allclose(first.vertices, second.vertices, atol=1e-12)


def test_incoherent_orthogonal_fields_have_no_fictitious_ellipse():
    from vecdiff_general.observables.polarization import ensemble_stokes
    from vecdiff_general.visualization import plot_intensity

    first = make_field(state=(1, 0, 0))
    second = make_field(state=(0, 1j, 0))
    parameters = ensemble_stokes((first, second), (0.5, 0.5))
    assert_allclose(parameters[..., 0], 1)
    assert_allclose(parameters[..., 1:], 0)
    axis = plot_intensity((first, second), weights=(0.5, 0.5))
    assert not any(
        item.get_gid() == "polarization-ellipses" for item in axis.collections
    )
    assert "Unpolarized" in axis.texts[0].get_text()


def test_ensemble_polarized_component_and_intensity_overlay():
    from vecdiff_general.visualization import plot_intensity

    first = make_field(state=(1, 0, 0))
    second = make_field(state=(0, 1, 0))
    axis = plot_intensity((first, second), weights=(0.8, 0.2))
    _, _, _ = glyphs(axis)
    arrows = next(
        item for item in axis.collections if item.get_gid() == "polarization-rotation"
    )
    assert_allclose(arrows.get_facecolors()[:, 3], 0.6)
    assert_allclose(axis.images[0].get_array(), 1)
    with pytest.raises(ValueError, match="phase"):
        plot_polarization((first, second), phase=0)


def test_polarization_glyphs_are_grayscale_over_inferno():
    for color_by in ("uniform", "handedness"):
        axis = plot_polarization(make_field(), color_by=color_by)
        assert axis.images[0].cmap.name == "inferno"

        for collection in axis.collections:
            for colors in (collection.get_facecolors(), collection.get_edgecolors()):
                if len(colors):
                    assert_allclose(colors[:, 0], colors[:, 1])
                    assert_allclose(colors[:, 1], colors[:, 2])


def test_bright_core_receives_glyphs_even_with_a_large_weak_halo():
    grid = CartesianGrid(np.linspace(-2, 2, 401) * 1e-6, np.linspace(-2, 2, 401) * 1e-6)
    x, y = np.meshgrid(grid.u * 1e6, grid.v * 1e6)
    envelope = np.exp(-(x**2 + y**2) / 0.12) + 0.09
    field = make_field(grid, envelope=envelope)
    _, axis = plt.subplots(figsize=(5, 5))
    plot_polarization(field, axis=axis, distribution="polar", crop=False)
    _, centers, _ = glyphs(axis)
    radius = np.linalg.norm(centers, axis=1)
    core = np.count_nonzero(radius < 0.3)
    halo = np.count_nonzero((radius > 1) & (radius < 1.8))

    assert core >= 20
    assert core / (0.3**2) > 2.5 * halo / (1.8**2 - 1)


def test_changing_polarization_gets_more_detail_than_a_uniform_state():
    grid = CartesianGrid(np.linspace(-2, 2, 201) * 1e-6, np.linspace(-2, 2, 201) * 1e-6)
    x, y = np.meshgrid(grid.u * 1e6, grid.v * 1e6)
    angle = 2 * x + y
    changing = ElectricField(
        grid, 532e-9, np.stack((np.cos(angle), np.sin(angle), 0 * x), axis=-1)
    )
    uniform = make_field(grid, state=(1, 0, 0))
    varying_nodes = glyphs(plot_polarization(changing))[1]
    uniform_nodes = glyphs(plot_polarization(uniform))[1]

    assert len(varying_nodes) > 2 * len(uniform_nodes)


def test_bright_stripes_receive_resolved_glyphs_without_background_clutter():
    grid = CartesianGrid(np.linspace(-2, 2, 257) * 1e-6, np.linspace(-2, 2, 257) * 1e-6)
    x, y = np.meshgrid(grid.u * 1e6, grid.v * 1e6)
    envelope = sum(
        np.exp(-(((x - center) / 0.18) ** 2) - (y / 0.9) ** 4)
        for center in (-0.8, 0, 0.8)
    )
    field = make_field(grid, state=(1, 0, 0), envelope=envelope)
    axis = plot_polarization(field, density=1.25, crop=False)
    _, centers, _ = glyphs(axis)
    values = _evaluate_glyphs(field, centers)
    intensity = np.sum(abs(values[..., :2]) ** 2, axis=1)

    assert np.min(intensity) > 0.002 * np.max(envelope**2)
    for stripe in (-0.8, 0, 0.8):
        bright = (abs(centers[:, 0] - stripe) < 0.18) & (abs(centers[:, 1]) < 0.8)
        assert bright.sum() >= 10


def test_dense_linear_arrows_keep_a_visible_interior_inside_the_outline():
    axis = plot_polarization(make_field(state=(1, 0, 0)), density=2)
    arrows = next(
        item for item in axis.collections if item.get_gid() == "polarization-rotation"
    )
    axis.figure.canvas.draw()
    polygon = arrows.get_paths()[0].vertices
    screen = axis.transData.transform(polygon)
    shaft_width_points = np.linalg.norm(screen[2] - screen[8]) * 72 / axis.figure.dpi

    assert shaft_width_points > arrows.get_linewidths()[0] + 0.15


@pytest.mark.parametrize("figsize", [(5.5, 5), (3, 3)])
def test_narrow_annular_pupil_uses_interior_native_rings(figsize):
    grid = PolarGrid(np.linspace(18e-6, 20e-6, 65), np.arange(240) * 2 * np.pi / 240)
    _, axis = plt.subplots(figsize=figsize)
    plot_polarization(make_field(grid), axis=axis, crop=False)
    _, centers, radii = glyphs(axis)
    radius = np.linalg.norm(centers, axis=1)

    assert len(centers) >= 20
    assert np.all(radius - 1.25 * radii >= 18 - 1e-12)
    assert np.all(radius + 1.25 * radii <= 20 + 1e-12)
