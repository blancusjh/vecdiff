"""Polarization views on physical planes, with intensity-aware glyph placement.

Ellipse construction and tangent markers recover ideas from the Claude renderer.
Components always follow the field contract: global Cartesian E projected onto
the grid frame, never inferred from array order. Matplotlib is optional.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from .._arrays import positive, positive_integer
from ..fields import ElectricField, ElectromagneticField, TransverseElectricField
from ..observables.polarization import ellipse_parameters, stokes
from ..sampling import CartesianGrid, PolarGrid


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


def _structure_length(grid, parameters, threshold):
    """Intensity-weighted length over which amplitude or polarization changes.

    Uses sqrt(I) and the normalized Stokes vector, so optical phase (a tilted
    plane wave, a defocus) does not count as structure. Returns metres, or inf
    for a uniform field. An Airy pattern gives about lambda / (2 pi NA).
    """
    intensity = parameters[..., 0]
    peak = intensity.max()
    bright = intensity > max(threshold, 1e-3) * peak
    amplitude = np.sqrt(intensity)
    states = np.divide(
        parameters[..., 1:],
        intensity[..., None],
        out=np.zeros_like(parameters[..., 1:]),
        where=bright[..., None],
    )
    channels = np.concatenate((amplitude[..., None], states), axis=-1)

    if isinstance(grid, CartesianGrid):
        slopes = np.gradient(channels, grid.v, grid.u, axis=(0, 1))
    else:
        step = 2 * np.pi / len(grid.azimuth)
        radial = np.gradient(channels, grid.radius, axis=0)
        azimuthal = (np.roll(channels, -1, axis=1) - np.roll(channels, 1, axis=1)) / (
            2 * step
        )
        radius = np.maximum(grid.radius, grid.radius[-1] * 1e-9)[:, None, None]
        slopes = (radial, azimuthal / radius)

    squared = sum(slope**2 for slope in slopes)
    variation = squared[..., 0] + intensity * np.sum(squared[..., 1:], axis=-1) / 4
    numerator = np.sum(grid.weights * intensity)
    denominator = np.sum(grid.weights * variation)

    return np.sqrt(numerator / denominator) if denominator > 0 else np.inf


def _odd_lattice(lower, upper, spacing):
    """Odd node count filling [lower, upper] with half-cell margins.

    The middle node sits at the window center, so symmetric windows sample
    their axes and no edge receives a wider empty band than the other.
    """
    count = max(1, 2 * int(np.floor(((upper - lower) / spacing - 1) / 2 + 0.5)) + 1)
    step = (upper - lower) / count

    return lower + (np.arange(count) + 0.5) * step, step


def _ring_count(radius, spacing):
    """Glyphs on one ring: arc pitch near spacing, quadrant symmetric if possible."""
    if radius <= 0:
        return 1

    natural = max(1, int(np.rint(2 * np.pi * radius / spacing)))
    quadrant = 4 * max(1, int(np.rint(natural / 4)))
    chord = 2 * radius * np.sin(np.pi / quadrant)

    return quadrant if 0.85 * spacing <= chord <= 1.25 * spacing else natural


def _radial_stations(profile, start, end, step):
    """Ring radii that put a glyph ring on every resolved bright ring.

    A fixed radial pitch close to the ring period of a focus beats against it:
    whole glyph rings fall into dark zeros and leave empty bands. Stations are
    therefore taken first at prominent maxima of the azimuthal mean intensity
    (strongest first, at least 0.8 pitch apart), then gaps are filled evenly at
    about one pitch. start and end are always stations unless a maximum is near.
    """
    fine = np.linspace(start, end, max(3, int(np.ceil(8 * (end - start) / step)) + 1))
    level = np.nan_to_num(profile(fine))
    window = max(1, int(np.rint(1.5 * step / (fine[1] - fine[0]))))
    maxima = []

    for index in np.argsort(level)[::-1]:
        value = level[index]

        if value <= 0 or index in (0, len(fine) - 1):
            continue

        left = level[max(0, index - window) : index]
        right = level[index + 1 : index + 1 + window]
        near = level[max(0, index - 2) : index + 3]

        # A diffraction ring falls below half its maximum on both sides within
        # 1.5 pitches; aberration ripple and interpolation noise do not.
        if value < near.max() or max(left.min(), right.min()) > 0.5 * value:
            continue

        if all(abs(fine[index] - chosen) >= 0.8 * step for chosen in maxima):
            maxima.append(fine[index])

    stations = sorted(maxima)

    for anchor in (start, end):
        if all(abs(anchor - chosen) >= 0.8 * step for chosen in stations):
            stations.append(anchor)

    stations = np.sort(stations)
    filled = [stations[:1]]

    for lower, upper in zip(stations[:-1], stations[1:]):
        count = max(1, int(np.rint((upper - lower) / step)))
        filled.append(lower + (upper - lower) * np.arange(1, count + 1) / count)

    return np.concatenate(filled)


def _layout_nodes(grid, spacing, limits, distribution, profile=None):
    """Glyph centers filling the visible sampled region without empty bands.

    Cartesian rows fill the view with half-cell margins. Polar rings are
    concentric about the chart origin; their pitch is adjusted so the last ring
    sits half a pitch inside the sampled disk (or the nearest view edge), and
    each ring holds an even angular pitch. With profile (mean intensity versus
    radius), rings follow the bright rings of the field; see _radial_stations.
    Returns centers and nominal pitches.
    """
    xmin, xmax, ymin, ymax = limits

    if distribution == "cartesian":
        x, step_x = _odd_lattice(xmin, xmax, spacing)
        y, step_y = _odd_lattice(ymin, ymax, spacing)
        x, y = np.meshgrid(x, y)
        centers = np.column_stack((x.ravel(), y.ravel()))

        return centers, np.full(len(centers), min(step_x, step_y))

    inner = 0.0

    if isinstance(grid, PolarGrid):
        inner, outer = grid.radius[0] * 1e6, grid.radius[-1] * 1e6
        origin = np.zeros(2)
        reach = outer
    else:
        origin = np.zeros(2)

        if not (xmin < 0 < xmax and ymin < 0 < ymax):
            origin = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])

        outer = min(
            origin[0] - xmin, xmax - origin[0], origin[1] - ymin, ymax - origin[1]
        )
        corners = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]])
        reach = np.max(np.linalg.norm(corners - origin, axis=1))

    if inner > 0:
        count = max(1, int(np.rint((outer - inner) / spacing)))
        step = (outer - inner) / count
        rings = inner + (np.arange(int(np.ceil((reach - inner) / step))) + 0.5) * step
    else:
        count = max(1, int(np.rint(outer / spacing - 0.5)))
        step = outer / (count + 0.5)
        rings = np.arange(int(np.floor(reach / step + 0.5)) + 1) * step

    rings = rings[rings <= reach * (1 + 1e-12)]

    if profile is not None and len(rings) > 1:
        rings = _radial_stations(profile, rings[0], rings[-1], step)

    nodes = []

    for radius in rings:
        number = _ring_count(radius, step)
        angles = np.arange(number) * 2 * np.pi / number
        nodes.append(
            origin + radius * np.column_stack((np.cos(angles), np.sin(angles)))
        )

    centers = np.concatenate(nodes)

    return centers, np.full(len(centers), step)


def _glyph_spacing(axis, grid, parameters, limits, density, threshold):
    """Display pitch in micrometres: resolve structure, stay legible on screen."""
    bounds = axis.get_window_extent().transformed(
        axis.figure.dpi_scale_trans.inverted()
    )
    xmin, xmax, ymin, ymax = limits
    data_per_inch = max((xmax - xmin) / bounds.width, (ymax - ymin) / bounds.height)
    span = min(xmax - xmin, ymax - ymin)
    legible = 0.17 * data_per_inch
    sparse = span / 15
    structure = 1.6 * _structure_length(grid, parameters, threshold) * 1e6

    return np.clip(structure, legible, max(legible, sparse)) / np.sqrt(density)


def _glyph_states(electric, centers, ensemble):
    """Local projected phasors and Stokes values at the exact glyph centers."""
    from ..observables.polarization import stokes_components

    basis = electric.grid.frame.basis[:, :2]

    if ensemble is None:
        values = _evaluate_glyphs(electric, centers)
        parameters = stokes_components(values @ basis)
    else:
        members, weights = ensemble
        values = np.stack(
            [_evaluate_glyphs(member, centers) for member in members], axis=1
        )
        parameters = np.sum(
            stokes_components(values @ basis) * weights[None, :, None], axis=1
        )

    return values, parameters


def _select_glyphs(
    axis,
    electric,
    parameters,
    limits,
    density,
    maximum,
    distribution,
    threshold,
    ensemble,
):
    """Exact glyph centers, their local states and radii fitting view and support."""
    density = positive(density, "density")
    maximum = positive_integer(maximum, "max_glyphs")
    grid = electric.grid
    peak = parameters[..., 0].max()
    axis.figure.canvas.draw()
    spacing = _glyph_spacing(axis, grid, parameters, limits, density, threshold)
    xmin, xmax, ymin, ymax = limits

    profile = None

    if distribution == "polar":
        angles = np.arange(48) * 2 * np.pi / 48
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        origin = np.zeros(2)

        if not isinstance(grid, PolarGrid) and not (
            xmin < 0 < xmax and ymin < 0 < ymax
        ):
            origin = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])

        def profile(radii):
            points = origin + radii[:, None, None] * directions[None]
            level = _glyph_states(electric, points.reshape(-1, 2), ensemble)[1][:, 0]

            level = level.reshape(len(radii), -1)
            finite = np.isfinite(level)

            return np.sum(np.where(finite, level, 0), axis=1) / np.maximum(
                finite.sum(axis=1), 1
            )

    while True:
        centers, pitch = _layout_nodes(grid, spacing, limits, distribution, profile)
        radii = 0.44 * pitch

        if len(centers) > 1:
            neighbour = cKDTree(centers).query(centers, k=2)[0][:, 1]
            radii = np.minimum(radii, 0.44 * neighbour)

        x, y = centers.T
        margin = np.minimum.reduce((x - xmin, xmax - x, y - ymin, ymax - y))

        if isinstance(grid, PolarGrid):
            distance = np.linalg.norm(centers, axis=1)
            margin = np.minimum(margin, grid.radius[-1] * 1e6 - distance)
            if grid.radius[0] > 0:
                margin = np.minimum(margin, distance - grid.radius[0] * 1e6)

        # A node near a curved or clipped edge keeps its position and a slightly
        # smaller glyph; only glyphs that would shrink by over 40 % are omitted.
        margin = margin * (1 - 1e-9)
        inside = margin >= 0.6 * radii
        radii = np.minimum(radii, margin)
        centers, radii = centers[inside], radii[inside]

        if not len(centers):
            return centers, radii, np.empty((0, 3), dtype=complex), np.empty((0, 4))

        values, local = _glyph_states(electric, centers, ensemble)
        keep = np.isfinite(local[:, 0]) & (local[:, 0] > threshold * peak)

        if np.count_nonzero(keep) <= maximum:
            return centers[keep], radii[keep], values[keep], local[keep]

        spacing *= max(1.05, np.sqrt(np.count_nonzero(keep) / maximum))


def _size_fraction(relative, scale, floor, threshold):
    """Monotone display size in [floor, 1] for relative intensity in (threshold, 1].

    'log' spreads the decades between threshold and the peak evenly, so weak
    side lobes stay readable and their ordering is preserved. 'sqrt' follows
    field amplitude, 'linear' follows intensity, a number p follows I**p.
    """
    relative = np.clip(relative, threshold, 1)

    if scale == "log":
        level = 1 - np.log(relative) / np.log(threshold) if threshold > 0 else 1
    elif scale == "sqrt":
        level = np.sqrt(relative)
    elif scale == "linear":
        level = relative
    else:
        level = relative**scale

    return floor + (1 - floor) * np.clip(level, 0, 1)


def _ellipse_curves(orientation, ellipticity, centers, radii):
    """Glyph geometry centered on its node; positive chi rotates u toward v.

    Rotating states carry one head at the minor-axis vertex, tangent to the
    trajectory, so the head is parallel to the major axis for thin ellipses.
    Linear states are centered double-headed arrows along the oscillation,
    which reproduces a pivot='middle' quiver without a false sign.
    """
    angle = np.linspace(0, 2 * np.pi, 65)
    major = np.column_stack((np.cos(orientation), np.sin(orientation)))
    minor = np.column_stack((-np.sin(orientation), np.cos(orientation)))
    ratio = np.tan(ellipticity)
    curves = centers[:, None, :] + radii[:, None, None] * (
        major[:, None, :] * np.cos(angle)[None, :, None]
        + ratio[:, None, None] * minor[:, None, :] * np.sin(angle)[None, :, None]
    )

    rotating = np.abs(ratio) > 0.05
    owners = np.concatenate((np.flatnonzero(rotating), np.flatnonzero(~rotating)))
    anchors = centers[rotating] + (radii * ratio)[rotating, None] * minor[rotating]
    # d/dtheta at theta = pi/2 is -major; the sign of ratio already encodes handedness.
    direction = -major[rotating]
    # Center the head on the vertex instead of ending it there.
    size = 0.95 * radii[rotating]
    anchors += 0.24 * size[:, None] * direction
    heads = list(_arrow_triangles(anchors, direction, size))
    half_axis = radii[~rotating, None] * major[~rotating]
    heads.extend(
        _electric_arrows(
            centers[~rotating] - half_axis,
            centers[~rotating] + half_axis,
            radii[~rotating],
            double=True,
        )
    )

    return curves, heads, owners, rotating


def plot_polarization(
    field,
    *,
    axis=None,
    density=1.0,
    distribution="auto",
    min_intensity_fraction=1e-4,
    max_glyphs=4000,
    size_scale="log",
    size_floor=0.4,
    crop=True,
    background=True,
    intensity_gamma=0.5,
    color_by="uniform",
    colorbar=True,
    legend=True,
    figsize=None,
    title="Polarization in the observation plane",
    phase=None,
    weights=None,
    glyph="ellipse",
    glyph_color="white",
    cmap="inferno",
    norm=None,
    reference_intensity=None,
    colorbar_label=None,
):
    """Draw polarization ellipses over projected |Eu|² + |Ev|².

    Accepts Cartesian or polar electric fields, transverse inputs and evaluated
    E/H. Axes are local u,v in µm. Projection omits normal E: this is not a 3D
    polarization view. Automatic cropping follows intensity support, including
    displaced and disconnected lobes. crop=False preserves the sampled window.

    Every glyph is centered on its layout node. distribution='cartesian' fills
    the view with an odd, window-centered lattice; 'polar' uses concentric
    rings about the chart origin with an even arc pitch on every ring, the
    outermost ring half a pitch inside the sampled disk. 'auto' follows the
    grid chart. The complex field is linearly interpolated at the exact nodes;
    angles and Stokes values are never interpolated, glyphs never shifted.

    The pitch resolves the intensity-weighted amplitude/polarization length of
    the field (a quarter of an Airy ring period, roughly), bounded by screen
    legibility and at least 15 glyphs across the view. density multiplies the
    glyph count per area; max_glyphs bounds it. Glyphs are omitted only where
    projected intensity is below min_intensity_fraction of its peak.

    size_scale maps intensity to glyph size: 'log' (default) spreads the decades
    from min_intensity_fraction to the peak evenly, so weak rings keep readable
    glyphs; 'sqrt' is amplitude, 'linear' intensity, a number p gives I**p;
    None draws constant sizes. size_floor is the relative size at threshold.
    This is a display scale, not a physical length.

    Linear states are centered double-headed arrows, like a middle-pivot quiver.
    Rotating states have one tangent head at the minor-axis vertex. Glyphs are
    white with dark outlines; color_by='handedness' uses gray levels by sense.
    With phase in radians, electric arrows start at the glyph centers and end
    at Re(E exp(-i phase)). A list/tuple of fields with nonnegative weights is
    an incoherent ensemble: glyphs show its polarized component, opacity is the
    projected degree of polarization, and no single phase exists.
    glyph='line' draws only the transverse polarization direction: a centered,
    headless segment along the major axis (ellipticity and handedness are not
    shown). glyph_color sets the glyph color; dark glyphs carry no outline.

    The background is projected intensity, divided by reference_intensity when
    given (for example an on-axis value shared by several panels). norm is any
    Matplotlib normalization (LogNorm(1e-5, 1) shows weak lobes and minima);
    otherwise intensity_gamma is a PowerNorm exponent (0.5 shows |E|). cmap and
    colorbar_label set the color map and label. Returns the Matplotlib axis.
    """
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import PowerNorm, to_rgba_array
    from matplotlib.lines import Line2D
    from matplotlib.transforms import ScaledTranslation

    if color_by not in ("handedness", "uniform"):
        raise ValueError("color_by must be handedness or uniform")

    if glyph not in ("ellipse", "line"):
        raise ValueError("glyph must be ellipse or line")

    if glyph == "line" and phase is not None:
        raise ValueError("phase arrows require ellipse glyphs")

    if reference_intensity is not None:
        reference_intensity = positive(reference_intensity, "reference_intensity")

    intensity_gamma = positive(intensity_gamma, "intensity_gamma")

    if size_scale not in (None, "log", "sqrt", "linear"):
        size_scale = positive(size_scale, "size_scale")

    if not np.isfinite(size_floor) or not 0 <= size_floor <= 1:
        raise ValueError("size_floor must lie in [0, 1]")

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
        electric, u, v, *_ = data
        parameters = ensemble_stokes(members, weights, electric.grid.frame.basis[:, :2])
        visible = parameters[..., 0] > min_intensity_fraction * parameters[..., 0].max()
        ensemble = members, weights
    else:
        if weights is not None:
            raise ValueError("weights require an ensemble of fields")

        data = _plane_data(field, min_intensity_fraction)
        electric, u, v, parameters, _, _, visible = data

    intensity = parameters[..., 0]
    limits = _view_limits(u, v, visible, crop)
    axis = _prepare_axis(axis, limits, figsize, title)

    if background:
        from matplotlib.colors import LogNorm

        shown = (
            intensity
            if reference_intensity is None
            else intensity / reference_intensity
        )

        if norm is None:
            norm = PowerNorm(intensity_gamma, 0, shown.max() or 1)
        elif isinstance(norm, LogNorm) and norm.vmin is not None:
            # Zeros are the darkest color, not masked holes in the map.
            shown = np.maximum(shown, norm.vmin)

        image = _draw_values(
            axis, electric.grid, u, v, shown, cmap=cmap, norm=norm, smooth=True
        )

        if colorbar:
            if colorbar_label is None:
                colorbar_label = (
                    r"$|E_u|^2 + |E_v|^2$ (V²/m²)"
                    if reference_intensity is None
                    else r"$(|E_u|^2 + |E_v|^2)\,/\,I_{\mathrm{ref}}$"
                )
            _colorbar(axis, image, colorbar_label)

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
    peak = intensity.max()
    centers, radii, values, local = (
        _select_glyphs(
            axis,
            electric,
            parameters,
            (*axis.get_xlim(), *axis.get_ylim()),
            density,
            max_glyphs,
            distribution,
            min_intensity_fraction,
            ensemble,
        )
        if np.any(visible)
        else (np.empty((0, 2)),) * 4
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

    # Stokes values at the nodes: no angle is ever interpolated.
    polarized = np.linalg.norm(local[:, 1:], axis=1)
    degree = np.clip(polarized / local[:, 0], 0, 1)

    if ensemble is not None:
        keep = degree > 1e-3
        centers, radii, local = centers[keep], radii[keep], local[keep]
        values, polarized, degree = values[keep], polarized[keep], degree[keep]

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
    else:
        degree = np.ones(len(centers))

    orientation = 0.5 * np.arctan2(local[:, 2], local[:, 1])
    chi = 0.5 * np.arcsin(
        np.clip(
            np.divide(
                local[:, 3], polarized, out=np.zeros(len(local)), where=polarized > 0
            ),
            -1,
            1,
        )
    )

    if size_scale is not None:
        radii = radii * _size_fraction(
            local[:, 0] / peak, size_scale, size_floor, min_intensity_fraction
        )

    radius_points = (
        radii
        * axis.get_window_extent().width
        / np.ptp(axis.get_xlim())
        * 72
        / axis.figure.dpi
    )

    if glyph == "line":
        half = (
            0.65
            * radii[:, None]
            * np.column_stack((np.cos(orientation), np.sin(orientation)))
        )
        colors = to_rgba_array([glyph_color] * len(centers))
        colors[:, 3] *= degree
        lines = LineCollection(
            np.stack((centers - half, centers + half), axis=1),
            colors=colors,
            linewidths=np.clip(0.14 * radius_points, 0.8, 1.6),
            capstyle="butt",
            zorder=3,
        )
        lines.set_gid("polarization-lines")
        axis.add_collection(lines)

        return axis

    curves, heads, owners, rotating = _ellipse_curves(orientation, chi, centers, radii)
    colors = np.array([glyph_color] * len(centers), dtype=object)

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

    if np.mean(to_rgba_array(glyph_color)[0, :3]) < 0.5:
        # A dark outline around a dark glyph only thickens it.
        outline_colors[:, 3] = 0

    line_widths = np.clip(0.16 * radius_points, 0.3, 0.9)
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
        linewidths=0.12 if np.mean(to_rgba_array(glyph_color)[0, :3]) >= 0.5 else 0,
        zorder=6,
    )
    arrows.set_gid("polarization-rotation")
    arrows.set_joinstyle("miter")
    axis.add_collection(arrows)

    return axis


def plot_focal_polarization(
    field,
    *,
    axis=None,
    reference_intensity=None,
    floor=None,
    glyph="line",
    density=6.0,
    title="Focal-plane polarization",
    **options,
):
    """Focal-plane polarization map in the notebooks' common style.

    Background: transverse intensity divided by its on-axis value (by default
    the sample nearest the chart origin), on ``hot`` with a logarithmic scale
    from ``floor`` to 1, so weak side lobes and minima stay distinct. The
    default floor puts the plane's median intensity mid-scale (floor = median²)
    and is never below 1e-5 or half the deepest minimum: a plane lit
    throughout (light passing beside a small lens) would otherwise be
    overexposed, while a tight focus keeps 1e-5. Glyphs: dense black transverse
    polarization directions (glyph='ellipse' for rotating states) on a dense
    lattice; their size follows intensity on a log scale, from 0.15 of full
    size at 1e-5 of the peak, so weak side lobes keep readable glyphs.
    Any plot_polarization option overrides these defaults. Axes read x, y.
    """
    from matplotlib.colors import LogNorm

    electric = field.electric if isinstance(field, ElectromagneticField) else field
    grid = electric.grid
    values = electric.values
    if values.shape[-1] == 3:
        values = values @ grid.frame.basis[:, :2]
    transverse = np.sum(abs(values) ** 2, -1)
    if reference_intensity is None:
        if isinstance(grid, CartesianGrid):
            reference_intensity = float(
                transverse[np.argmin(abs(grid.v)), np.argmin(abs(grid.u))]
            )
        else:
            reference_intensity = float(transverse[np.argmin(grid.radius)].mean())
    if floor is None:
        relative = transverse / reference_intensity
        floor = max(1e-5, 0.5 * float(relative.min()), float(np.median(relative)) ** 2)
    settings = dict(
        crop=False,
        glyph_color="black",
        size_scale="log",
        size_floor=0.15,
        cmap="hot",
        norm=LogNorm(floor, 1),
        min_intensity_fraction=min(floor, 0.5),
        colorbar_label="Transverse intensity / on-axis value",
    )
    settings.update(options)
    axis = plot_polarization(
        field,
        axis=axis,
        glyph=glyph,
        density=density,
        title=title,
        reference_intensity=reference_intensity,
        **settings,
    )
    axis.set(xlabel="x (µm)", ylabel="y (µm)")
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


def plot_ellipses(field, *, axis=None, relative_threshold=0.03, **kwargs):
    """Compatibility entry point; relative_threshold retains amplitude semantics.

    New code should use plot_polarization with min_intensity_fraction instead.
    """
    if not np.isfinite(relative_threshold) or not 0 <= relative_threshold < 1:
        raise ValueError("relative_threshold must lie in [0, 1)")

    return plot_polarization(
        field,
        axis=axis,
        min_intensity_fraction=relative_threshold**2,
        **kwargs,
    )
