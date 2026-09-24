"""Polarization views on physical planes, with intensity-aware glyph placement.

Ellipse construction and tangent markers recover ideas from the Claude renderer.
Components always follow the field contract: global Cartesian E projected onto
the grid frame, never inferred from array order. Matplotlib is optional.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

from .._arrays import positive, positive_integer
from ..fields import ElectricField, ElectromagneticField, TransverseElectricField
from ..observables.polarization import ellipse_parameters, stokes
from ..sampling import CartesianGrid, PointGrid, PolarGrid


def _plane_data(field, threshold):
    if not np.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError("min_intensity_fraction must lie in [0, 1)")

    if isinstance(field, ElectromagneticField):
        field = field.electric

    if not isinstance(field, (ElectricField, TransverseElectricField)):
        raise TypeError("expected an electric or electromagnetic field")

    grid = field.grid

    if not isinstance(grid, (CartesianGrid, PolarGrid)):
        raise TypeError("polarization maps require a CartesianGrid or PolarGrid")

    if isinstance(field, TransverseElectricField):
        values = field.values @ grid.frame.basis[:, :2].T
        field = ElectricField(grid, field.wavelength0, values)

    basis = grid.frame.basis[:, :2]
    parameters = stokes(field, basis)
    orientation, ellipticity, _ = ellipse_parameters(field, basis)
    intensity = parameters[..., 0]
    visible = intensity > threshold * intensity.max()

    # Chart coordinates avoid subtraction error when the frame is translated.
    if isinstance(grid, CartesianGrid):
        u, v = np.meshgrid(grid.u, grid.v)
    else:
        radius, azimuth = np.meshgrid(grid.radius, grid.azimuth, indexing="ij")
        u = radius * np.cos(azimuth)
        v = radius * np.sin(azimuth)

    return field, u, v, parameters, orientation, ellipticity, visible


def _view_limits(u, v, visible, crop):
    limits = []

    for coordinate in (u, v):
        lower, upper = coordinate.min(), coordinate.max()

        if crop and np.any(visible):
            active = coordinate[visible]
            span = max(np.ptp(active), 0.04 * (upper - lower))
            middle = 0.5 * (active.min() + active.max())
            lower = max(lower, middle - 0.58 * span)
            upper = min(upper, middle + 0.58 * span)

        limits.extend((lower * 1e6, upper * 1e6))

    return np.asarray(limits)


def _figure_size(limits):
    ratio = (limits[1] - limits[0]) / (limits[3] - limits[2])
    height = np.clip(5.0 / np.sqrt(ratio), 1.3, 7.5)
    width = height * ratio

    if width > 11.5:
        height *= 11.5 / width
        width = 11.5

    return max(4.5, width + 1.3), max(2.6, height + 1.3)


def _prepare_axis(axis, limits, figsize, title):
    import matplotlib.pyplot as plt

    if axis is None:
        _, axis = plt.subplots(
            figsize=figsize or _figure_size(limits),
            dpi=180,
            layout="constrained",
        )
    elif figsize is not None:
        raise ValueError("figsize applies only when creating a new figure")

    axis.set(
        xlim=limits[:2],
        ylim=limits[2:],
        xlabel="u (µm)",
        ylabel="v (µm)",
        aspect="equal",
        title=title,
    )
    axis.set_facecolor("#171717")
    axis.tick_params(labelsize=10)

    return axis


def _draw_values(axis, grid, u, v, values, *, cmap, norm, smooth):
    if isinstance(grid, CartesianGrid):
        du, dv = grid.spacing
        extent = (
            np.array(
                [u.min() - du / 2, u.max() + du / 2, v.min() - dv / 2, v.max() + dv / 2]
            )
            * 1e6
        )

        return axis.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="bilinear" if smooth else "nearest",
            interpolation_stage="data",
        )

    if smooth:
        # Close the periodic seam without extending across an annular hole.
        return axis.pcolormesh(
            np.concatenate((u, u[:, :1]), axis=1) * 1e6,
            np.concatenate((v, v[:, :1]), axis=1) * 1e6,
            np.ma.concatenate((values, values[:, :1]), axis=1),
            shading="gouraud",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )

    radius = grid.radius
    edges = np.r_[
        max(0, radius[0] - (radius[1] - radius[0]) / 2),
        (radius[:-1] + radius[1:]) / 2,
        radius[-1] + (radius[-1] - radius[-2]) / 2,
    ]
    angles = np.r_[grid.azimuth, grid.azimuth[0] + 2 * np.pi]
    angles -= np.pi / len(grid.azimuth)

    return axis.pcolormesh(
        edges[:, None] * np.cos(angles) * 1e6,
        edges[:, None] * np.sin(angles) * 1e6,
        values,
        shading="flat",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )


def _colorbar(axis, image, label):
    # An inset follows the actual equal-aspect axes, including very narrow beams.
    # A normal subplot colorbar follows the larger allocated layout slot instead.
    color_axis = axis.inset_axes([1.035, 0.09, 0.035, 0.82])
    bar = axis.figure.colorbar(image, cax=color_axis)
    bar.set_label(label, fontsize=10)
    bar.ax.tick_params(labelsize=9)


def _evaluate_glyphs(electric, centers):
    """Evaluate the display phasors; never interpolate angles or Stokes values.

    Coordinates are local micrometres. Native grid nodes reproduce their stored
    values. Alternate layouts use linear complex interpolation, with no
    extrapolation beyond the sampled plane or polar annulus.
    """
    grid = electric.grid
    local = centers * 1e-6

    if isinstance(grid, CartesianGrid):
        interpolator = RegularGridInterpolator(
            (grid.v, grid.u), electric.values, bounds_error=False, fill_value=np.nan
        )
        values = interpolator(local[:, ::-1])
    else:
        radius = np.linalg.norm(local, axis=1)
        azimuth = (np.arctan2(local[:, 1], local[:, 0]) - grid.azimuth[0]) % (2 * np.pi)
        azimuth += grid.azimuth[0]
        angles = np.append(grid.azimuth, grid.azimuth[0] + 2 * np.pi)
        periodic_values = np.concatenate(
            (electric.values, electric.values[:, :1]), axis=1
        )
        interpolator = RegularGridInterpolator(
            (grid.radius, angles),
            periodic_values,
            bounds_error=False,
            fill_value=np.nan,
        )
        # Roundoff on a native outer ring must not count as extrapolation.
        tolerance = 1e-12 * grid.radius[-1]
        inside = (radius >= grid.radius[0] - tolerance) & (
            radius <= grid.radius[-1] + tolerance
        )
        radius = np.where(
            inside, np.clip(radius, grid.radius[0], grid.radius[-1]), radius
        )
        values = interpolator(np.column_stack((radius, azimuth)))

    return values


def _layout_nodes(grid, center, spacing, limits, distribution, stride):
    """Exact rows or rings; no nearest-neighbour displacement of a glyph."""
    xmin, xmax, ymin, ymax = limits

    if distribution == "cartesian":
        if isinstance(grid, CartesianGrid):
            axes = []

            for coordinate, origin in zip((grid.u * 1e6, grid.v * 1e6), center):
                step = max(
                    stride, int(np.rint(spacing / (coordinate[1] - coordinate[0])))
                )
                step = int(np.ceil(step / stride)) * stride
                anchor = int(np.argmin(abs(coordinate[::stride] - origin))) * stride
                indices = np.arange(anchor % step, len(coordinate), step)
                axes.append(coordinate[indices])

            x, y = np.meshgrid(*axes)
        else:
            x_steps = np.arange(
                np.ceil((xmin - center[0]) / spacing),
                np.floor((xmax - center[0]) / spacing) + 1,
            )
            y_steps = np.arange(
                np.ceil((ymin - center[1]) / spacing),
                np.floor((ymax - center[1]) / spacing) + 1,
            )
            x, y = np.meshgrid(
                center[0] + spacing * x_steps, center[1] + spacing * y_steps
            )

        return np.column_stack((x.ravel(), y.ravel()))

    native = isinstance(grid, PolarGrid)

    if native:
        # The polar chart has a geometrical origin; intensity does not move it.
        center = np.zeros(2)
        available = grid.radius[::stride] * 1e6

        if grid.radius[0] > 0:
            # Start on an interior native ring. Starting on the inner boundary
            # can leave every subsequent ring too close to an edge to fit.
            margin = 1.25 * 0.46 * spacing
            available = available[
                (available >= grid.radius[0] * 1e6 + margin)
                & (available <= grid.radius[-1] * 1e6 - margin)
            ]

        rings = []
        last = -np.inf

        for radius in available:
            if radius - last >= spacing * (1 - 1e-12):
                rings.append(radius)
                last = radius
    else:
        corners = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]])
        maximum_radius = np.max(np.linalg.norm(corners - center, axis=1))
        rings = np.arange(0, maximum_radius + spacing / 2, spacing)

    nodes = []

    for radius in rings:
        if radius == 0:
            nodes.append(center[None, :])
            continue

        # Exact periodic angular steps; chord length stays at least spacing.
        count = max(1, int(np.floor(np.pi / np.arcsin(min(1, spacing / (2 * radius))))))

        if native:
            eligible = np.arange(0, len(grid.azimuth), stride)
            divisors = [
                n
                for n in range(1, count + 1)
                if len(grid.azimuth) % n == 0 and (len(grid.azimuth) // n) % stride == 0
            ]
            count = max(divisors, default=1)
            angles = grid.azimuth[
                eligible[0] + np.arange(count) * (len(grid.azimuth) // count)
            ]
        else:
            # Multiples of four retain the axes and quadrant symmetry.
            count = max(4, 4 * (count // 4))
            angles = np.arange(count) * 2 * np.pi / count

        nodes.append(
            center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
        )

    return np.concatenate(nodes) if nodes else np.empty((0, 2))


def _density_mask(centers, center, projected, peak, grid, distribution, states):
    """Resolve bright features and changing polarization on nested rows or rings.

    Normalized Stokes distances detect polarization changes without depending
    on global phase or the undefined orientation of a circular state. Polar
    rings are refined as whole rings, avoiding irregular missing angular nodes.
    """
    if not len(centers):
        return np.empty(0, dtype=bool)

    relative = np.nan_to_num(projected / peak)
    neighbors = cKDTree(centers).query(centers, k=min(5, len(centers)))[1]
    neighbors = np.atleast_2d(neighbors).reshape(len(centers), -1)
    differences = np.linalg.norm(states[:, None] - states[neighbors], axis=-1)
    reliable = np.minimum(relative[:, None], relative[neighbors]) > 0.002
    variation = np.max(np.where(reliable, differences, 0), axis=1)

    # A uniform plane wave needs an orientation key, not a dense carpet.
    structured = np.ptp(relative[np.isfinite(projected)]) > 0.1
    fine = ((relative >= 0.25) & structured) | (
        (variation >= 0.35) & (relative >= 0.02)
    )
    medium = (
        ((relative >= 0.20) & structured)
        | ((variation >= 0.12) & (relative >= 0.005))
        | ~structured
    )

    if distribution == "cartesian":
        indices_by_axis = []
        for coordinate, origin in zip(centers.T, center):
            nodes, indices = np.unique(coordinate, return_inverse=True)
            anchor = np.argmin(abs(nodes - origin))
            indices_by_axis.append((indices, anchor, len(nodes)))

        (ix, ax, nx), (iy, ay, ny) = indices_by_axis
        # Refine neighboring lattice cells together, not isolated glyphs.
        fine_grid = np.zeros((ny, nx), dtype=bool)
        medium_grid = np.zeros((ny, nx), dtype=bool)
        fine_grid[iy, ix], medium_grid[iy, ix] = fine, medium
        fine = maximum_filter(fine_grid, size=3)[iy, ix]
        medium = maximum_filter(medium_grid, size=3)[iy, ix]
        half = ((ix - ax) % 2 == 0) & ((iy - ay) % 2 == 0)
        quarter = ((ix - ax) % 4 == 0) & ((iy - ay) % 4 == 0)

        return fine | (medium & half) | quarter

    origin = np.zeros(2) if isinstance(grid, PolarGrid) else center
    radius = np.linalg.norm(centers - origin, axis=1)
    _, rings = np.unique(np.round(radius, 10), return_inverse=True)
    keep = np.zeros(len(centers), dtype=bool)
    ring_ids = np.unique(rings)

    for ring in ring_ids:
        indices = np.flatnonzero(rings == ring)
        if np.any(fine[indices]):
            level = 1
        elif np.any(medium[indices]):
            level = 2
        else:
            level = 4
        # Preserve every radial station. Weak or slowly varying rings reduce
        # angular density instead of opening unexplained radial gaps.
        maximum_step = min(level, max(1, len(indices) // 4))
        step = max(
            candidate
            for candidate in range(1, maximum_step + 1)
            if len(indices) % candidate == 0
        )
        keep[indices[::step]] = True

    return keep


def _select_glyphs(
    axis,
    electric,
    u,
    v,
    intensity,
    visible,
    density,
    maximum,
    stride,
    distribution,
    threshold,
    adaptive_density,
    ensemble=None,
):
    """Choose exact layout positions, mask dark states and keep full glyphs in view."""
    density = positive(density, "density")
    maximum = positive_integer(maximum, "max_glyphs")
    stride = 1 if stride is None else positive_integer(stride, "stride")
    grid = electric.grid

    if not np.any(visible):
        return np.empty((0, 2)), np.empty(0), np.empty((0, 3), dtype=complex)

    coordinates = np.column_stack((u.ravel(), v.ravel())) * 1e6
    mass = (intensity * grid.weights).ravel()
    center = np.sum(coordinates * mass[:, None], axis=0) / mass.sum()
    axis.figure.canvas.draw()
    bounds = axis.get_window_extent().transformed(
        axis.figure.dpi_scale_trans.inverted()
    )
    limits = (*axis.get_xlim(), *axis.get_ylim())
    xmin, xmax, ymin, ymax = limits
    data_per_inch = max((xmax - xmin) / bounds.width, (ymax - ymin) / bounds.height)
    # Resolve the bright core before allocating any budget to the halo.
    # Using all above-threshold area made a broad weak halo starve a tiny focus.
    core = intensity >= 0.25 * intensity.max()
    core_area = np.sum(grid.weights[core]) * 1e12
    screen_spacing = 5.5 / 72 * data_per_inch / np.sqrt(density)
    target = 65 if np.ptp(intensity) > 0.1 * intensity.max() else 260
    target *= np.clip(bounds.width * bounds.height / 16, 0.25, 1.5)
    spacing = max(screen_spacing, np.sqrt(core_area / (target * density)))

    if isinstance(grid, PolarGrid) and grid.radius[0] > 0:
        # A narrow annulus has a large area but little room across its width.
        # Fit the first glyph ring inside the band before budgeting azimuths.
        spacing = min(spacing, 0.6 * np.ptp(grid.radius) * 1e6)

    while True:
        centers = _layout_nodes(grid, center, spacing, limits, distribution, stride)
        from ..observables.polarization import stokes_components

        if ensemble is None:
            values = _evaluate_glyphs(electric, centers)
            components = values @ grid.frame.basis[:, :2]
            projected = np.sum(abs(components) ** 2, axis=1)
            parameters = stokes_components(components)
        else:
            members, weights = ensemble
            values = np.stack(
                [_evaluate_glyphs(member, centers) for member in members], axis=1
            )
            components = values @ grid.frame.basis[:, :2]
            projected = np.sum(
                abs(components) ** 2 * weights[None, :, None], axis=(1, 2)
            )
            parameters = np.sum(
                stokes_components(components) * weights[None, :, None], axis=1
            )
        radius = 0.46 * spacing
        x, y = centers.T
        margin = np.minimum.reduce((x - xmin, xmax - x, y - ymin, ymax - y))
        keep = np.isfinite(projected) & (projected > threshold * intensity.max())
        if adaptive_density:
            states = np.divide(
                parameters[:, 1:],
                projected[:, None],
                out=np.zeros_like(parameters[:, 1:]),
                where=projected[:, None] > threshold * intensity.max(),
            )
            keep &= _density_mask(
                centers, center, projected, intensity.max(), grid, distribution, states
            )

        keep &= margin >= 1.25 * radius

        if isinstance(grid, PolarGrid):
            distance = np.linalg.norm(centers, axis=1)
            keep &= distance + 1.25 * radius <= grid.radius[-1] * 1e6

            if grid.radius[0] > 0:
                keep &= distance - 1.25 * radius >= grid.radius[0] * 1e6

        centers, values = centers[keep], values[keep]

        if len(centers) <= maximum:
            break

        spacing *= max(1.12, np.sqrt(len(centers) / maximum))

    radii = np.full(len(centers), radius)

    if len(centers) > 1 and adaptive_density:
        distances = cKDTree(centers).query(centers, k=2)[0][:, 1]
        radii = np.minimum(0.46 * distances, radius)
        x, y = centers.T
        margin = np.minimum.reduce((x - xmin, xmax - x, y - ymin, ymax - y))

        if isinstance(grid, PolarGrid):
            distance = np.linalg.norm(centers, axis=1)
            margin = np.minimum(margin, grid.radius[-1] * 1e6 - distance)
            if grid.radius[0] > 0:
                margin = np.minimum(margin, distance - grid.radius[0] * 1e6)

        radii = np.minimum(radii, margin / 1.25)

    return centers, radii, values


def _glyph_size(relative_intensity, power, minimum):
    """Compressed intensity cue with a smooth readability floor."""
    return minimum + (1 - minimum) * np.clip(relative_intensity, 0, 1) ** power


def _electric_arrows(starts, ends, radii, *, double=False):
    """One closed silhouette per arrow: shaft and head share a single border."""
    displacement = ends - starts
    length = np.linalg.norm(displacement, axis=-1)
    direction = displacement / length[:, None]
    normal = np.column_stack((-direction[:, 1], direction[:, 0]))
    size = np.minimum(radii, length)
    neck = ends - 0.48 * size[:, None] * direction
    head = 0.34 * size[:, None] * normal
    # At dense display pitches the old shaft was thinner than its border.
    # Leave a visible white interior instead of an almost black silhouette.
    shaft = 0.085 * size[:, None] * normal

    if double:
        rear_neck = starts + 0.48 * size[:, None] * direction

        return np.stack(
            (
                ends,
                neck + head,
                neck + shaft,
                rear_neck + shaft,
                rear_neck + head,
                starts,
                rear_neck - head,
                rear_neck - shaft,
                neck - shaft,
                neck - head,
            ),
            axis=1,
        )

    return np.stack(
        (
            ends,
            neck + head,
            neck + shaft,
            starts + shaft,
            starts - shaft,
            neck - shaft,
            neck - head,
        ),
        axis=1,
    )


def _arrow_triangles(anchors, directions, radii):
    """Align rotation heads with the preceding arc chord, anchored by their tips."""
    length = np.linalg.norm(directions, axis=-1)
    direction = np.divide(
        directions,
        length[:, None],
        out=np.zeros_like(directions),
        where=length[:, None] > 0,
    )
    normal = np.column_stack((-direction[:, 1], direction[:, 0]))

    base = anchors - 0.48 * radii[:, None] * direction
    width = 0.22 * radii[:, None] * normal

    return np.stack((anchors, base + width, base - width), axis=1)


def _phase_vectors(electric, centers, radii):
    """Scale local phasors to the drawn semi-major axes without changing phase."""
    components = _evaluate_glyphs(electric, centers) @ electric.grid.frame.basis[:, :2]
    intensity = np.sum(np.abs(components) ** 2, axis=-1)
    linear = np.hypot(
        np.abs(components[:, 0]) ** 2 - np.abs(components[:, 1]) ** 2,
        2 * np.real(components[:, 0].conj() * components[:, 1]),
    )
    major_amplitude = np.sqrt((intensity + linear) / 2)

    return components * (radii / major_amplitude)[:, None]


def _phase_geometry(vectors, centers, radii, phase):
    """Electric arrows from fixed grid nodes to Re(E exp(-i phase)).

    The head follows the electric vector, not the velocity of its tip. Short
    vectors receive proportionally smaller heads so the base stays on the shaft.
    At a zero crossing the whole arrow vanishes, without leaving a dot.
    """
    displacement = np.real(vectors * np.exp(-1j * phase))
    length = np.linalg.norm(displacement, axis=-1)
    owners = np.flatnonzero(length > 1e-8 * radii)
    positions = centers + displacement
    segments = np.stack((centers[owners], positions[owners]), axis=1)
    head_radii = np.minimum(radii[owners], length[owners])
    heads = _electric_arrows(centers[owners], positions[owners], head_radii)

    return segments, heads, owners


def _ellipse_curves(orientation, ellipticity, centers, radii):
    """Phase-independent geometry; positive chi rotates u toward v."""
    angle = np.linspace(0, 2 * np.pi, 65)
    major = np.column_stack((np.cos(orientation), np.sin(orientation)))
    minor = np.column_stack((-np.sin(orientation), np.cos(orientation)))
    ratio = np.tan(ellipticity)
    curves = centers[:, None, :] + radii[:, None, None] * (
        major[:, None, :] * np.cos(angle)[None, :, None]
        + ratio[:, None, None] * minor[:, None, :] * np.sin(angle)[None, :, None]
    )

    # Keep rotation markers away from the major-axis vertices. Linear states
    # instead have two outward heads: oscillation, without a handedness claim.
    rotating = np.abs(ratio) > 0.04
    marker_angle = np.pi / 4
    displacement = major * np.cos(marker_angle) + ratio[:, None] * minor * np.sin(
        marker_angle
    )
    owners = np.concatenate((np.flatnonzero(rotating), np.flatnonzero(~rotating)))
    anchors = centers[rotating] + radii[rotating, None] * displacement[rotating]
    base_angle = marker_angle - 0.45
    base_offset = major * np.cos(base_angle) + ratio[:, None] * minor * np.sin(
        base_angle
    )
    bases = centers[rotating] + radii[rotating, None] * base_offset[rotating]
    direction = anchors - bases
    head_scale = np.linalg.norm(direction, axis=-1) / 0.48
    heads = list(_arrow_triangles(anchors, direction, head_scale))
    half_axis = radii[~rotating, None] * major[~rotating]
    heads.extend(
        _electric_arrows(
            centers[~rotating] - half_axis,
            centers[~rotating] + half_axis,
            radii[~rotating],
            double=True,
        )
    )

    return curves, heads, owners


def plot_polarization(
    field,
    *,
    axis=None,
    density=1.0,
    adaptive_density=True,
    min_intensity_fraction=0.002,
    max_glyphs=640,
    crop=True,
    background=True,
    intensity_gamma=0.3,
    scale_by_intensity=True,
    size_power=0.25,
    min_size_fraction=0.5,
    distribution="auto",
    color_by="uniform",
    colorbar=True,
    legend=True,
    figsize=None,
    title="Polarization in the observation plane",
    stride=None,
    phase=None,
    weights=None,
):
    """Draw polarization ellipses over projected |Eu|² + |Ev|².

    Accepts Cartesian or polar electric fields, transverse inputs and evaluated
    E/H. Axes are local u,v in µm. Projection omits normal E: this is not a 3D
    polarization view. Automatic cropping follows intensity support, including
    displaced and disconnected lobes. crop=False preserves the sampled window.

    density controls glyph count relative to physical figure space; max_glyphs
    bounds it. Dark nodes are excluded by a fraction of peak projected intensity.
    adaptive_density resolves the bright core and normalized-Stokes variation,
    with nested coarser rows or rings elsewhere. Set False for uniform density.
    distribution='auto' follows the grid chart: exact Cartesian rows or native
    polar rings. An explicit alternate distribution linearly interpolates the
    complex field at exact display positions, without modifying the input field.
    Dark positions are omitted, never shifted toward brighter neighbours.
    By default, relative glyph radius is 0.5 + 0.5 * (I / I_peak) ** 0.25.
    size_power and min_size_fraction control this compressed intensity cue;
    scale_by_intensity=False gives constant sizes. This is a display scale,
    not a physical length. Glyphs use grayscale over an inferno intensity background.
    color_by='uniform' uses white glyphs with dark outlines; 'handedness' uses
    different gray levels. Rotation remains encoded by the arrow direction.
    Static heads indicate rotation, or two-way linear oscillation. With phase
    in radians, electric arrows start at the fixed glyph centers and end at
    Re(E exp(-i phase)). Heads align with those vectors. No point markers are used.
    A list/tuple of fields with nonnegative weights denotes an incoherent
    ensemble. Glyphs show its polarized component; opacity is the projected
    degree of polarization. Unpolarized states have no ellipse. Such an ensemble
    has no single phase and cannot be animated as one electric phasor.
    Background gamma changes contrast only. Returns the Matplotlib axis.
    """
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import PowerNorm, to_rgba_array
    from matplotlib.lines import Line2D
    from matplotlib.transforms import ScaledTranslation

    if color_by not in ("handedness", "uniform"):
        raise ValueError("color_by must be handedness or uniform")

    intensity_gamma = positive(intensity_gamma, "intensity_gamma")
    size_power = positive(size_power, "size_power")

    if not np.isfinite(min_size_fraction) or not 0 <= min_size_fraction <= 1:
        raise ValueError("min_size_fraction must lie in [0, 1]")

    if distribution not in ("auto", "cartesian", "polar"):
        raise ValueError("distribution must be auto, cartesian, or polar")

    if phase is not None and not np.isfinite(phase):
        raise ValueError("phase must be finite")

    ensemble = None

    if isinstance(field, (tuple, list)):
        if phase is not None:
            raise ValueError("an incoherent ensemble has no single optical phase")

        from ..observables.polarization import ensemble_stokes, ensemble_weights

        members = tuple(_plane_data(member, 0)[0] for member in field)
        weights = ensemble_weights(members, weights)
        data = _plane_data(members[0], min_intensity_fraction)
        electric, u, v, _, orientation, ellipticity, _ = data
        parameters = ensemble_stokes(members, weights, electric.grid.frame.basis[:, :2])
        visible = parameters[..., 0] > min_intensity_fraction * parameters[..., 0].max()
        ensemble = members, weights
    else:
        if weights is not None:
            raise ValueError("weights require an ensemble of fields")

        data = _plane_data(field, min_intensity_fraction)
        electric, u, v, parameters, orientation, ellipticity, visible = data

    intensity = parameters[..., 0]
    limits = _view_limits(u, v, visible, crop)
    axis = _prepare_axis(axis, limits, figsize, title)

    if background:
        image = _draw_values(
            axis,
            electric.grid,
            u,
            v,
            intensity,
            cmap="inferno",
            norm=PowerNorm(intensity_gamma, 0, intensity.max() or 1),
            smooth=True,
        )

        if colorbar:
            _colorbar(axis, image, r"$|E_u|^2 + |E_v|^2$ (V²/m²)")

    if legend and color_by == "handedness":
        labels = ("Linear ↔", "u → v", "v → u")
        colors = ("#ffffff", "#cccccc", "#888888")
        handles = [
            Line2D([], [], color=color, label=label, lw=2)
            for color, label in zip(colors, labels)
        ]
        axis.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            bbox_transform=axis.transAxes
            + ScaledTranslation(0, -48 / 72, axis.figure.dpi_scale_trans),
            ncol=3,
            frameon=False,
            fontsize=9,
        )

    distribution = (
        electric.grid.coordinate_system if distribution == "auto" else distribution
    )
    centers, radii, values = _select_glyphs(
        axis,
        electric,
        u,
        v,
        intensity,
        visible,
        density,
        max_glyphs,
        stride,
        distribution,
        min_intensity_fraction,
        adaptive_density,
        ensemble,
    )

    if not len(centers):
        axis.text(
            0.5,
            0.5,
            "No glyphs fit the view"
            if np.any(visible)
            else "No projected field above threshold",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
        )
        return axis

    degree = np.ones(len(centers))

    if ensemble is None:
        points = (
            electric.grid.frame.origin
            + centers[:, :1] * 1e-6 * electric.grid.frame.u
            + centers[:, 1:] * 1e-6 * electric.grid.frame.v
        )
        glyph_field = ElectricField(PointGrid(points), electric.wavelength0, values)
        orientation, chi, amplitude = ellipse_parameters(
            glyph_field, electric.grid.frame.basis[:, :2]
        )
    else:
        from ..observables.polarization import stokes_components

        local = values @ electric.grid.frame.basis[:, :2]
        parameters = np.sum(stokes_components(local) * weights[None, :, None], axis=1)
        polarized = np.linalg.norm(parameters[:, 1:], axis=1)
        degree = np.clip(polarized / parameters[:, 0], 0, 1)
        keep = degree > 1e-3
        centers, radii, parameters = centers[keep], radii[keep], parameters[keep]
        polarized, degree = polarized[keep], degree[keep]
        orientation = 0.5 * np.arctan2(parameters[:, 2], parameters[:, 1])
        chi = 0.5 * np.arcsin(np.clip(parameters[:, 3] / polarized, -1, 1))
        amplitude = np.sqrt(parameters[:, 0])

        if not len(centers):
            axis.text(
                0.5,
                0.04,
                "Unpolarized in this projection",
                ha="center",
                transform=axis.transAxes,
                color="white",
            )
            return axis

    if scale_by_intensity:
        radii *= _glyph_size(
            amplitude**2 / intensity.max(), size_power, min_size_fraction
        )

    curves, heads, owners = _ellipse_curves(orientation, chi, centers, radii)
    rotating = np.abs(np.tan(chi)) > 0.04
    colors = np.full(len(centers), "#ffffff", dtype="<U7")

    if color_by == "handedness":
        colors[rotating & (chi > 0)] = "#cccccc"
        colors[rotating & (chi < 0)] = "#888888"

    arrow_colors = to_rgba_array(colors)
    arrow_colors[:, 3] = degree
    curve_colors = arrow_colors.copy()

    if phase is None:
        # Linear states are rendered by a single double-headed polygon. Keep
        # their exact trajectories in the collection without painting over them.
        curve_colors[~rotating, 3] = 0

    outline_colors = curve_colors.copy()
    outline_colors[:, :3] = to_rgba_array("#151515")[0, :3]
    radius_points = (
        radii
        * axis.get_window_extent().width
        / np.ptp(axis.get_xlim())
        * 72
        / axis.figure.dpi
    )
    line_widths = np.clip(0.24 * radius_points, 0.35, 0.85)
    outlines = LineCollection(
        curves, colors=outline_colors, linewidths=line_widths + 0.45, zorder=2
    )
    axis.add_collection(outlines)
    bodies = LineCollection(
        curves, colors=curve_colors, linewidths=line_widths, zorder=3
    )
    bodies.set_gid("polarization-ellipses")
    axis.add_collection(bodies)

    if phase is not None:
        # Keep the trajectory subordinate to the arrow so its tail is visibly
        # anchored at the center, especially when the ellipse collapses to a line.
        bodies.set_alpha(0.45)
        bodies.set_linewidth(0.85)
        outlines.set_alpha(0.3)
        outlines.set_linewidth(1.5)
        vectors = _phase_vectors(electric, centers, radii)
        segments, heads, owners = _phase_geometry(vectors, centers, radii, phase)

    arrows = PolyCollection(
        heads,
        facecolors=arrow_colors[owners],
        edgecolors="#151515",
        linewidths=0.12,
        zorder=6,
    )
    arrows.set_gid("polarization-rotation")
    arrows.set_joinstyle("miter")
    axis.add_collection(arrows)

    return axis


def plot_polarization_scalar(
    field,
    quantity="ellipticity",
    *,
    axis=None,
    min_intensity_fraction=0.002,
    crop=True,
    figsize=None,
    title=None,
):
    """Map ellipticity chi or axis orientation psi in degrees.

    Fixed scales permit comparison: chi in [-45,45], psi in [-90,90]. Orientation
    is masked at circular states, whose major axis is undefined. Color
    levels follow the signed angle; -90 and +90 degrees are the same axis. Angles are never interpolated across a wrap.
    """
    from matplotlib.colors import Normalize

    if quantity not in ("ellipticity", "orientation"):
        raise ValueError("quantity must be ellipticity or orientation")

    data = _plane_data(field, min_intensity_fraction)
    electric, u, v, parameters, orientation, ellipticity, visible = data
    limits = _view_limits(u, v, visible, crop)
    axis = _prepare_axis(axis, limits, figsize, title or quantity.capitalize())

    if quantity == "orientation":
        linear_fraction = np.hypot(parameters[..., 1], parameters[..., 2])
        visible &= linear_fraction > 1e-7 * parameters[..., 0]
        values, bound, cmap = orientation, 90, "inferno"
        label = "Major-axis angle ψ (°)"
    else:
        values, bound, cmap = ellipticity, 45, "inferno"
        label = "Ellipticity angle χ (°)"

    image = _draw_values(
        axis,
        electric.grid,
        u,
        v,
        np.ma.array(np.degrees(values), mask=~visible),
        cmap=cmap,
        norm=Normalize(-bound, bound),
        smooth=False,
    )
    _colorbar(axis, image, label)

    return axis


def plot_ellipses(field, *, stride=None, axis=None, relative_threshold=0.03, **kwargs):
    """Compatibility entry point; relative_threshold retains amplitude semantics.

    New code should use plot_polarization with min_intensity_fraction instead.
    """
    if not np.isfinite(relative_threshold) or not 0 <= relative_threshold < 1:
        raise ValueError("relative_threshold must lie in [0, 1)")

    return plot_polarization(
        field,
        stride=stride,
        axis=axis,
        min_intensity_fraction=relative_threshold**2,
        **kwargs,
    )
