"""Visualization helpers for polarization maps."""

from __future__ import annotations

from typing import Any, Literal, Mapping

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize, PowerNorm

from .polarization import PolarizationData


def _ellipse_glyph(ex, ey, points):
    """Build a smooth polarization-ellipse glyph from complex components.

    The ellipse is sampled uniformly in geometric angle (never by physical
    phase), so its points are evenly spaced and never pile up at the vertices --
    circular light gives a clean circle, linear light a clean straight line.
    The semi-major axis is normalized to 1; callers scale it to the glyph size.

    Returns ``(pts, head_point, head_dir, ellipticity)`` in the local (ex, ey)
    frame: ``pts`` is an ``(points, 2)`` array tracing the ellipse,
    ``head_point`` is the tip of the major axis, ``head_dir`` the direction the
    arrowhead points, and ``ellipticity`` is ``|minor/major|`` in ``[0, 1]``.

    ``head_dir`` is the tangent to the ellipse curve at the major-axis tip:
    perpendicular to the major axis for elliptical/circular light, with sign
    given by handedness (right- vs left-handed). For linear light the tangent
    vanishes and the caller must decide how to render the head (or omit it,
    since a linear oscillation has no handedness to indicate).
    """

    ex = complex(ex)
    ey = complex(ey)
    ax2 = abs(ex) ** 2
    ay2 = abs(ey) ** 2
    s0 = ax2 + ay2
    s1 = ax2 - ay2
    s2 = 2.0 * np.real(ex * np.conj(ey))
    s3 = -2.0 * np.imag(ex * np.conj(ey))

    psi = 0.5 * np.arctan2(s2, s1)
    chi = 0.5 * np.arcsin(np.clip(s3 / max(s0, np.finfo(float).eps), -1.0, 1.0))
    ratio = np.tan(chi)  # signed minor/major axis ratio in [-1, 1]

    theta = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    xe = np.cos(theta)
    ye = ratio * np.sin(theta)
    cos_p, sin_p = np.cos(psi), np.sin(psi)
    pts = np.column_stack([cos_p * xe - sin_p * ye, sin_p * xe + cos_p * ye])

    head_point = np.array([cos_p, sin_p])  # major-axis tip: rotate (1, 0)
    # Tangent to the ellipse at the major-axis tip (theta=0):
    # d/dtheta (xe, ye) = (0, ratio), rotated by psi. Perpendicular to the
    # major axis for elliptical/circular light, sign encodes handedness.
    # Vanishes for linear light (no handedness to point along).
    head_dir = np.array([-sin_p * ratio, cos_p * ratio])
    return pts, head_point, head_dir, abs(float(ratio))


def _polar_to_cartesian_basis(vectors, cx, cy):
    phi = np.arctan2(cy, cx)
    radial = np.array([np.cos(phi), np.sin(phi)])
    azimuthal = np.array([-np.sin(phi), np.cos(phi)])
    return vectors[:, :1] * radial + vectors[:, 1:] * azimuthal


def _curve_segments(curve):
    return np.stack([curve, np.roll(curve, -1, axis=0)], axis=1)


def _arrowhead_triangle(location, direction, length, width):
    """Return a 3-vertex filled arrowhead whose axial midpoint sits on ``location``.

    ``location`` is the point on the ellipse curve where the head is anchored:
    the apex extends ``length / 2`` outward along ``direction`` and the base
    extends ``length / 2`` inward. A single clean filled triangle (drawn as a
    polygon) reads as a crisp head, unlike a pair of haloed line segments
    which blob at small glyph sizes.

    Centring the head on its anchor -- rather than seating the apex there --
    keeps the ellipse curve visibly threading through the head, so the glyph
    reads as an "ellipse-with-marker" rather than an appendage growing off the
    tip.
    """

    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm <= np.finfo(float).eps or length <= np.finfo(float).eps:
        return None
    t = direction / norm
    perp = np.array([-t[1], t[0]])
    half = 0.5 * length
    apex = location + half * t
    base = location - half * t
    return np.array([apex, base + 0.5 * width * perp, base - 0.5 * width * perp])


def _slim_colorbar(mappable, ax, label=None):
    """Attach a slim colorbar that is clearly shorter than the axis.

    ``aspect=35`` gives a thin bar (thinner than Matplotlib's ~20 default) and
    ``shrink=0.65`` keeps its height well under the axis so it never dominates
    the panel visually even under ``constrained_layout``.
    """

    kwargs = {"aspect": 35, "shrink": 0.65, "pad": 0.03, "fraction": 0.045}
    if label is not None:
        kwargs["label"] = label
    return plt.colorbar(mappable, ax=ax, **kwargs)


def _line_kwargs(kwargs: Mapping[str, Any] | None, linewidth=1.2, color=None, zorder=None):
    out = {"linewidths": linewidth}
    if color is not None:
        out["colors"] = color
    if zorder is not None:
        out["zorder"] = zorder
    if kwargs:
        out.update(dict(kwargs))
    if "color" in out:
        out["colors"] = out.pop("color")
    if "linewidth" in out:
        out["linewidths"] = out.pop("linewidth")
    return out


def _intensity_scale_factor(relative_amp, mode, gamma, min_scale):
    relative_amp = float(np.clip(relative_amp, 0.0, 1.0))
    min_scale = float(min_scale)

    if mode == "linear":
        factor = relative_amp
    elif mode == "power":
        if gamma <= 0.0:
            raise ValueError("intensity_scale_gamma must be positive for power scaling.")
        factor = relative_amp ** float(gamma)
    elif mode == "log":
        if gamma <= 0.0:
            raise ValueError("intensity_scale_gamma must be positive for log scaling.")
        factor = np.log1p(float(gamma) * relative_amp) / np.log1p(float(gamma))
    else:
        raise ValueError("intensity_scale_mode must be 'linear', 'log', or 'power'.")

    return max(float(factor), min_scale)


def _halo(linewidth, fg="black", grow=1.3):
    """Return a path-effect stroke that outlines a glyph for legibility.

    A thin dark outline keeps white glyphs readable over both the dark and the
    bright ends of a perceptually-uniform background colormap -- essential when
    the map covers a full diffraction pattern whose lobes span that whole range.
    """

    return [pe.withStroke(linewidth=float(linewidth) + float(grow), foreground=fg)]


def _user_set(kwargs: Mapping[str, Any] | None, *keys: str) -> bool:
    """True if the caller explicitly provided any of ``keys`` in ``kwargs``."""

    return bool(kwargs) and any(key in kwargs for key in keys)


def _radius_mask(xs: np.ndarray, ys: np.ndarray, max_radius: float | None) -> np.ndarray:
    """Boolean mask selecting samples within ``max_radius`` of the origin.

    Used to limit glyphs to a chosen radial extent, i.e. up to a given order of
    diffraction maxima.  ``None`` keeps every sample.
    """

    if max_radius is None:
        return np.ones(np.shape(xs), dtype=bool)
    return (np.asarray(xs) ** 2 + np.asarray(ys) ** 2) <= float(max_radius) ** 2


def plot_polarization_map(
    x: np.ndarray,
    y: np.ndarray,
    pol: PolarizationData,
    stride: int | None = None,
    target_ellipses: int = 18,
    max_ellipses: int | None = None,
    max_radius: float | None = None,
    scale: float | None = None,
    ellipse_points: int = 72,
    min_intensity_fraction: float = 0.002,
    color_by_phase: bool = False,
    phase_cmap: str = "twilight_shifted",
    phase_colorbar: bool = True,
    scale_by_intensity: bool = False,
    intensity_scale_mode: Literal["linear", "log", "power"] = "power",
    intensity_scale_gamma: float = 0.5,
    min_ellipse_scale: float = 0.30,
    arrow_opening_angle: float = np.deg2rad(42.0),
    arrow_length: float = 0.45,
    arrow_head_absolute_length: float | None = None,
    head_fade_by_ellipticity: bool = False,
    linear_head_marker: bool = True,
    linear_head_threshold: float = 0.02,
    curve_kwargs: Mapping[str, Any] | None = None,
    arrowhead_kwargs: Mapping[str, Any] | None = None,
    ellipse_mode: Literal["polar", "cartesian"] = "polar",
    ax=None,
):
    """Draw local polarization ellipses over a sampled plane.

    ``ellipse_mode="polar"`` interprets ``pol.ex`` and ``pol.ey`` as local
    radial and azimuthal components.  ``ellipse_mode="cartesian"`` preserves
    the previous behavior and interprets them as x and y components.

    Ellipses are drawn wherever ``S0 = |Ex|² + |Ey|²`` exceeds
    ``min_intensity_fraction`` of the peak.  The default is deliberately low so
    the polarization state is shown across the whole diffraction pattern --
    including the secondary maxima -- while the dark nulls between lobes stay
    empty.  ``max_radius`` further limits the glyphs to a chosen radial extent
    (i.e. up to a given order of maxima); ``None`` uses the full window.

    By default ``scale_by_intensity`` is ``False`` so every ellipse is drawn at
    the same readable size and the polarization state in a faint outer lobe is
    just as legible as in the bright core.  Enable it (with
    ``intensity_scale_mode`` / ``intensity_scale_gamma``) to instead encode
    intensity through glyph size.

    Glyphs default to white with a thin dark halo so they read over both the
    dark and bright ends of the background colormap; pass ``color`` (or
    ``path_effects``) in ``curve_kwargs`` / ``arrowhead_kwargs`` to override.

    The arrowhead axis is aligned with the tangent to the ellipse curve at
    its anchor point -- perpendicular to the major axis for
    elliptical/circular light, with sign given by handedness. Its length is
    ``arrow_length * glyph_extent`` (proportional to the ellipse). Pass
    ``arrow_head_absolute_length`` to fix the head length in graph units
    instead, which decouples it from the ellipse size.

    Linear light has a vanishing curve tangent at the major-axis tip because
    the E vector reverses direction there.  With ``linear_head_marker=True``
    (default) the renderer falls back to the outward major-axis direction,
    which is a genuine physical vector -- the instantaneous field direction
    at the peak of the oscillation -- so a single arrowhead marks the sense
    of oscillation.  Set ``linear_head_marker=False`` to drop the head for
    linear samples.  ``linear_head_threshold`` sets the ellipticity below
    which the fallback is used (``|chi|`` roughly equal to this ratio for
    small values); the default of 0.02 (~1.1°) uses it only for glyphs that
    are indistinguishable from a straight segment at plot resolution.

    For numerically non-linear light (ellipticity above the threshold),
    ``head_fade_by_ellipticity`` controls the head *length*: when ``False``
    (default) the head keeps its full size, when ``True`` it is scaled by
    ``sqrt(ellipticity)`` so the head magnitude encodes handedness strength.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D coordinate arrays.")
    if x.shape != y.shape or x.shape != pol.ex.shape:
        raise ValueError("x, y, and polarization components must have matching shapes.")
    if ellipse_mode not in {"polar", "cartesian"}:
        raise ValueError("ellipse_mode must be 'polar' or 'cartesian'.")
    if intensity_scale_mode not in {"linear", "log", "power"}:
        raise ValueError("intensity_scale_mode must be 'linear', 'log', or 'power'.")

    s0 = np.asarray(pol.s0, dtype=float)
    s0_max = float(np.nanmax(s0)) + np.finfo(float).eps
    visible = s0 > float(min_intensity_fraction) * s0_max

    if stride is None:
        target_ellipses = max(int(target_ellipses), 4)
        if np.any(visible):
            rows, cols = np.nonzero(visible)
            active_size = min(np.ptp(rows) + 1, np.ptp(cols) + 1)
        else:
            active_size = min(x.shape)
        # Base the density on the visible region instead of the full canvas.
        # This preserves detail in a diffraction-limited focus even when the
        # displayed window is much larger than the focal spot.
        stride = max(1, int(np.ceil(active_size / target_ellipses)))
    else:
        stride = max(1, int(stride))

    xs = x[::stride, ::stride]
    ys = y[::stride, ::stride]
    ex = pol.ex[::stride, ::stride]
    ey = pol.ey[::stride, ::stride]
    amp = pol.amplitude[::stride, ::stride]
    phase = pol.phase[::stride, ::stride]

    amp_max = np.sqrt(s0_max)
    valid = s0[::stride, ::stride] > float(min_intensity_fraction) * s0_max
    valid &= _radius_mask(xs, ys, max_radius)
    if max_ellipses is not None and np.count_nonzero(valid) > max_ellipses:
        valid_amp = amp[valid]
        keep_count = max(1, int(max_ellipses))
        threshold_index = max(0, valid_amp.size - keep_count)
        adaptive_threshold = np.partition(valid_amp, threshold_index)[threshold_index]
        valid &= amp >= adaptive_threshold

    if scale is None:
        # Use geometric neighbour distances so this works for both Cartesian
        # meshes and polar meshes (whose azimuthal spacing vanishes at r=0 and
        # would otherwise collapse the glyph size to zero).
        radial_steps = np.hypot(np.diff(xs, axis=1), np.diff(ys, axis=1)) if xs.shape[1] > 1 else np.array([])
        angular_steps = np.hypot(np.diff(xs, axis=0), np.diff(ys, axis=0)) if xs.shape[0] > 1 else np.array([])
        steps = np.concatenate((radial_steps.ravel(), angular_steps.ravel()))
        steps = steps[np.isfinite(steps) & (steps > np.finfo(float).eps)]
        base_spacing = float(np.nanmedian(steps)) if steps.size else float(min(np.ptp(x), np.ptp(y)))
        # 0.38 keeps ellipse diameters at ~0.76x the sample spacing, leaving a
        # clear gap between neighbouring glyphs so they never merge.
        scale = 0.38 * base_spacing

    figure_segments = []
    head_polys = []
    colors = []
    head_colors = []

    head_width_ratio = 2.0 * np.tan(0.5 * float(arrow_opening_angle))

    for cx, cy, ex_i, ey_i, amp_i, phase_i, keep in zip(
        xs.ravel(),
        ys.ravel(),
        ex.ravel(),
        ey.ravel(),
        amp.ravel(),
        phase.ravel(),
        valid.ravel(),
    ):
        if not keep:
            continue

        if scale_by_intensity:
            relative_amp = amp_i / amp_max
            size_factor = _intensity_scale_factor(
                relative_amp,
                intensity_scale_mode,
                intensity_scale_gamma,
                min_ellipse_scale,
            )
        else:
            size_factor = 1.0
        glyph_extent = scale * size_factor

        pts, head_point, head_dir, ellipticity = _ellipse_glyph(ex_i, ey_i, ellipse_points)

        # Head at the major-axis tip.  For elliptical/circular light the
        # direction is the curve tangent (encodes handedness).  For linear
        # light the tangent vanishes at the tip -- because the E vector
        # reverses direction there -- so we fall back to the outward
        # major-axis direction, marking the instantaneous field direction at
        # the peak of the oscillation.
        linear_light = ellipticity < float(linear_head_threshold)
        if linear_light and linear_head_marker:
            head_pairs = [(head_point, head_point)]
        else:
            head_pairs = [(head_point, head_dir)]

        if ellipse_mode == "polar":
            pts = _polar_to_cartesian_basis(pts, cx, cy)
            head_pairs = [
                (
                    _polar_to_cartesian_basis(hp[None, :], cx, cy)[0],
                    _polar_to_cartesian_basis(hd[None, :], cx, cy)[0],
                )
                for hp, hd in head_pairs
            ]

        center = np.array([cx, cy])
        curve = glyph_extent * pts + center
        segments = _curve_segments(curve)
        figure_segments.append(segments)

        # Head length: full ``head_base`` for linear-light markers so the
        # tick-marks stay visible, and either full length or the
        # sqrt-of-ellipticity fade for genuinely elliptical light -- the
        # sqrt keeps a *visible* head as soon as handedness is perceptible.
        head_base = (
            float(arrow_head_absolute_length)
            if arrow_head_absolute_length is not None
            else arrow_length * glyph_extent
        )
        if linear_light and linear_head_marker:
            head_length = head_base
        elif head_fade_by_ellipticity:
            head_length = 0.0 if ellipticity < 0.01 else head_base * np.sqrt(ellipticity)
        else:
            head_length = head_base

        if color_by_phase:
            c = (phase_i + np.pi) / (2.0 * np.pi)
            colors.append(np.full(segments.shape[0], c))

        for head_p, head_d in head_pairs:
            head = _arrowhead_triangle(
                glyph_extent * head_p + center,
                head_d,
                head_length,
                head_width_ratio * head_length,
            )
            if head is not None:
                head_polys.append(head)
                if color_by_phase:
                    head_colors.append(c)

    if not figure_segments:
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))
        ax.set_aspect("equal", adjustable="box")
        return ax

    figure_segments = np.concatenate(figure_segments, axis=0)
    head_polys = np.asarray(head_polys) if head_polys else None

    # Resolve the arrowhead fill: honour an explicit head or curve colour, else
    # match the white body.  A dark halo is added only when nothing was overridden.
    if _user_set(arrowhead_kwargs, "color", "colors"):
        head_face = arrowhead_kwargs.get("color", arrowhead_kwargs.get("colors"))
    elif _user_set(curve_kwargs, "color", "colors"):
        head_face = curve_kwargs.get("color", curve_kwargs.get("colors"))
    else:
        head_face = "white"
    head_alpha = arrowhead_kwargs.get("alpha") if arrowhead_kwargs else None

    if color_by_phase:
        lc_kwargs = _line_kwargs(curve_kwargs, linewidth=0.45, zorder=3.0)
        lc_kwargs.pop("colors", None)
        cmap = lc_kwargs.pop("cmap", phase_cmap)
        lc = LineCollection(figure_segments, array=np.concatenate(colors), cmap=cmap, **lc_kwargs)
        if not _user_set(curve_kwargs, "path_effects"):
            lc.set_path_effects(_halo(lc_kwargs["linewidths"], grow=0.6))
        ax.add_collection(lc)
        if phase_colorbar:
            _slim_colorbar(lc, ax, label="Fase normalizada")

        if head_polys is not None:
            pc = PolyCollection(head_polys, array=np.asarray(head_colors), cmap=cmap, zorder=4.0)
            if not _user_set(arrowhead_kwargs, "path_effects"):
                pc.set_path_effects(_halo(0.4, grow=0.6))
            ax.add_collection(pc)
    else:
        # White glyphs with a thin dark halo stay legible over both the dark and
        # bright ends of the background colormap, so a full-coverage map reads
        # everywhere from the bright core out to the faint secondary maxima.
        lc_kwargs = _line_kwargs(curve_kwargs, linewidth=0.6, color="white", zorder=3.0)
        lc = LineCollection(figure_segments, **lc_kwargs)
        if not _user_set(curve_kwargs, "color", "colors", "path_effects"):
            lc.set_path_effects(_halo(lc_kwargs["linewidths"], grow=0.6))
        ax.add_collection(lc)

        if head_polys is not None:
            pc = PolyCollection(head_polys, facecolors=head_face, edgecolors="none", zorder=4.0)
            if head_alpha is not None:
                pc.set_alpha(float(head_alpha))
            if not _user_set(arrowhead_kwargs, "color", "colors", "path_effects"):
                pc.set_path_effects(_halo(0.4, grow=0.6))
            ax.add_collection(pc)

    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def _autocrop_extent(xx, yy, valid, padding=1.15):
    """Return a square ``(xmin, xmax, ymin, ymax)`` box tightly bounding ``valid``."""

    if not np.any(valid):
        return float(np.min(xx)), float(np.max(xx)), float(np.min(yy)), float(np.max(yy))

    radius = float(np.max(np.hypot(xx[valid], yy[valid]))) * padding
    radius = min(radius, float(np.max(np.hypot(xx, yy))))
    if radius <= 0.0:
        radius = float(np.max(np.hypot(xx, yy))) or 1.0
    return -radius, radius, -radius, radius


def plot_polarization_scalar_map(
    field,
    quantity: Literal["ellipticity", "orientation"],
    half_size: float | None = None,
    n_img: int = 500,
    min_intensity_fraction: float = 0.002,
    vmin: float | None = None,
    vmax: float | None = None,
    autocrop: bool = True,
    crop_padding: float = 1.15,
    ax=None,
):
    """Plot a scalar map of the local ellipticity angle or major-axis orientation.

    ``quantity="ellipticity"`` shows the ellipticity angle ``chi`` (0 deg is
    linear, +-45 deg is circular). ``quantity="orientation"`` shows the
    major-axis angle ``psi`` of the polarization ellipse. Both are masked
    below ``min_intensity_fraction`` of the peak intensity, where the local
    polarization state is not meaningfully defined.

    The color range defaults (``vmin``/``vmax`` left as ``None``) to the
    actual spread of the data instead of the full physical range: a field
    that stays close to linear polarization would otherwise wash out to a
    single near-white color against a fixed +-45/+-90 deg scale. Pass explicit
    ``vmin``/``vmax`` to compare several plots on the same scale. When
    ``autocrop`` is enabled, the axes are zoomed to the region where the
    signal actually exceeds ``min_intensity_fraction`` instead of showing the
    full sampled window, most of which would otherwise be blank.
    """

    from .polarization import polarization_map_from_field

    if quantity == "ellipticity":
        vmax_physical = 45.0
        cmap = "RdBu_r"
        label = r"Ángulo de elipticidad $\chi$ (°)"
    elif quantity == "orientation":
        vmax_physical = 90.0
        cmap = "twilight_shifted"
        label = r"Ángulo del eje mayor $\psi$ (°)"
    else:
        raise ValueError("quantity must be 'ellipticity' or 'orientation'.")

    xx, yy, pol = polarization_map_from_field(field, half_size=half_size, n_img=n_img)
    values_rad = pol.chi if quantity == "ellipticity" else pol.psi
    values_deg = np.rad2deg(values_rad)

    s0_max = float(np.nanmax(pol.s0)) + np.finfo(float).eps
    valid = pol.s0 > float(min_intensity_fraction) * s0_max
    masked = np.where(valid, values_deg, np.nan)

    if vmin is None or vmax is None:
        data_scale = float(np.nanmax(np.abs(masked[valid]))) if np.any(valid) else vmax_physical
        data_scale = min(max(data_scale * 1.15, 1e-6), vmax_physical)
        vmin = -data_scale if vmin is None else vmin
        vmax = data_scale if vmax is None else vmax

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        masked,
        extent=[float(np.min(xx)), float(np.max(xx)), float(np.min(yy)), float(np.max(yy))],
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    _slim_colorbar(im, ax, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if autocrop:
        xmin, xmax, ymin, ymax = _autocrop_extent(xx, yy, valid, padding=crop_padding)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    return ax, pol


def plot_polarization_quiver(
    x: np.ndarray,
    y: np.ndarray,
    pol: PolarizationData,
    stride: int | None = None,
    azimuthal_stride: int | None = None,
    target_arrows: int = 24,
    max_radius: float | None = None,
    length: float | None = None,
    arrow_length_fraction: float = 1.0,
    min_intensity_fraction: float = 0.002,
    min_cross_fraction: float = 0.0,
    scale_by_intensity: bool = False,
    color_by_cross_fraction: bool = False,
    cross_fraction_cmap: str = "viridis",
    cross_fraction_colorbar: bool = True,
    quiver_kwargs: Mapping[str, Any] | None = None,
    ax=None,
):
    """Overlay polarization-orientation arrows using Matplotlib quiver.

    The defaults are chosen as a coherent visual configuration: arrow length
    is tied to the sampled grid spacing, the shaft is long enough to read the
    local orientation, and the arrowhead is kept modest relative to the body.
    Arrows are white with a thin dark halo so they read over both the dark and
    bright ends of the background colormap.  Like the ellipse map, the low
    ``min_intensity_fraction`` default extends the arrows across the secondary
    maxima, and ``max_radius`` limits them to a chosen order of maxima.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    x = np.asarray(x)
    y = np.asarray(y)
    s0 = np.asarray(pol.s0, dtype=float)
    s0_max = float(np.nanmax(s0)) + np.finfo(float).eps
    visible = s0 > float(min_intensity_fraction) * s0_max
    if stride is None:
        target_arrows = max(int(target_arrows), 4)
        if np.any(visible):
            rows, cols = np.nonzero(visible)
            active_size = min(np.ptp(rows) + 1, np.ptp(cols) + 1)
        else:
            active_size = min(x.shape)
        stride = max(1, int(np.ceil(active_size / target_arrows)))
    else:
        stride = max(1, int(stride))
    azimuthal_stride = stride if azimuthal_stride is None else max(1, int(azimuthal_stride))

    xs = x[::azimuthal_stride, ::stride]
    ys = y[::azimuthal_stride, ::stride]
    psi = pol.psi[::azimuthal_stride, ::stride]
    amp = pol.amplitude[::azimuthal_stride, ::stride]

    amp_max = np.sqrt(s0_max)
    s0s = s0[::azimuthal_stride, ::stride]
    cross_fraction = np.divide(
        np.abs(pol.ey[::azimuthal_stride, ::stride]) ** 2,
        s0s,
        out=np.zeros_like(s0s, dtype=float),
        where=s0s > np.finfo(float).eps,
    )
    valid = s0s > float(min_intensity_fraction) * s0_max
    valid &= cross_fraction >= float(min_cross_fraction)
    valid &= _radius_mask(xs, ys, max_radius)

    if length is None:
        # Geometric neighbour distances work for both Cartesian meshes and
        # polar meshes, whose azimuthal spacing vanishes at r=0.
        radial_steps = np.hypot(np.diff(xs, axis=1), np.diff(ys, axis=1)) if xs.shape[1] > 1 else np.array([])
        angular_steps = np.hypot(np.diff(xs, axis=0), np.diff(ys, axis=0)) if xs.shape[0] > 1 else np.array([])
        steps = np.concatenate((radial_steps.ravel(), angular_steps.ravel()))
        steps = steps[np.isfinite(steps) & (steps > np.finfo(float).eps)]
        base_spacing = float(np.nanmedian(steps)) if steps.size else float(min(np.ptp(x), np.ptp(y)))
        length = float(arrow_length_fraction) * base_spacing

    if scale_by_intensity:
        length_factor = amp / amp_max
    else:
        length_factor = np.ones_like(amp)

    u = length * length_factor * np.cos(psi)
    v = length * length_factor * np.sin(psi)

    kwargs: dict[str, Any] = {
        "angles": "xy",
        "scale_units": "xy",
        "scale": 1.0,
        "pivot": "mid",
        "color": "white",
        "width": 0.0038,
        "headwidth": 3.6,
        "headlength": 4.2,
        "headaxislength": 3.8,
        "zorder": 4.0,
    }
    if quiver_kwargs:
        kwargs.update(dict(quiver_kwargs))

    apply_halo = not _user_set(quiver_kwargs, "color", "path_effects")

    if color_by_cross_fraction:
        kwargs.pop("color", None)
        kwargs.setdefault("cmap", cross_fraction_cmap)
        kwargs.setdefault("norm", Normalize(vmin=0.0, vmax=1.0))
        quiver = ax.quiver(xs[valid], ys[valid], u[valid], v[valid], cross_fraction[valid], **kwargs)
        if cross_fraction_colorbar:
            _slim_colorbar(quiver, ax, label=r"$|E_y|^2 / (|E_x|^2 + |E_y|^2)$")
    else:
        quiver = ax.quiver(xs[valid], ys[valid], u[valid], v[valid], **kwargs)

    if apply_halo:
        quiver.set_path_effects(_halo(0.0, grow=0.8))
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_field_polarization(
    field,
    half_size=None,
    n_img=500,
    background="intensity",
    intensity_gamma=0.35,
    cross_fraction_min_intensity=0.0,
    glyph="ellipse",
    sampling="cartesian",
    ax=None,
    **kwargs,
):
    """Plot field polarization with a Cartesian or polar glyph layout.

    ``sampling="cartesian"`` (default) places the glyphs on a square grid.
    ``sampling="polar"`` places them on evenly spaced concentric rings (with the
    number of glyphs per ring growing with radius, so the spacing stays uniform),
    which suits radially structured fields such as focal diffraction patterns.
    The polar layout is controlled by ``n_rings`` (default 12) and
    ``angular_spacing`` (default 1.0): values above 1 thin out the glyphs along
    each ring (larger azimuthal gaps) without changing the radial sampling,
    which leaves room to size the glyphs larger via an explicit ``scale``.

    Pass ``rings`` as a list of ``(radius, n_azimuthal)`` tuples to specify ring
    positions and azimuthal counts explicitly instead of using the automatic
    uniform layout.  A center-point glyph at ``r=0`` is always included.  The
    glyph scale is derived from the smallest consecutive ring spacing unless
    overridden via ``scale``.
    """

    from .polarization import polarization_from_components, polarization_map_from_field

    # Always sample the field on a Cartesian mesh for the raster background; the
    # glyph positions come from this mesh (Cartesian) or an even polar layout.
    bg_x, bg_y, bg_pol = polarization_map_from_field(field, half_size=half_size, n_img=n_img)
    if sampling == "cartesian":
        xx, yy, pol = bg_x, bg_y, bg_pol
    elif sampling == "polar":
        from scipy.interpolate import RegularGridInterpolator

        rings_spec = kwargs.pop("rings", None)   # [(radius, n_az), ...] custom layout
        n_rings = max(1, int(kwargs.pop("n_rings", 12)))
        angular_spacing = max(float(kwargs.pop("angular_spacing", 1.0)), np.finfo(float).eps)
        r_max = float(half_size) if half_size is not None else float(np.max(np.hypot(bg_x, bg_y)))
        dr = r_max / (n_rings + 0.5)
        # Always include an on-axis (r=0) glyph, then the concentric rings.
        xg, yg = [np.array([0.0])], [np.array([0.0])]
        if rings_spec is not None:
            for rk, n_az in rings_spec:
                ang = np.linspace(0.0, 2.0 * np.pi, int(n_az), endpoint=False)
                xg.append(float(rk) * np.cos(ang))
                yg.append(float(rk) * np.sin(ang))
            # Derive the reference spacing from successive ring radii so the
            # glyph scale stays consistent with the densest part of the layout.
            radii = sorted(float(r) for r, _ in rings_spec)
            if len(radii) >= 2:
                dr = float(np.min(np.diff(radii)))
            elif radii:
                dr = float(radii[0])
        else:
            for rk in dr * (np.arange(n_rings) + 1.0):
                n_az = max(6, int(round(2.0 * np.pi * rk / (angular_spacing * dr))))
                ang = np.linspace(0.0, 2.0 * np.pi, n_az, endpoint=False)
                xg.append(rk * np.cos(ang))
                yg.append(rk * np.sin(ang))
        xg = np.concatenate(xg)
        yg = np.concatenate(yg)

        x_axis = np.asarray(bg_x[0, :], dtype=float)
        y_axis = np.asarray(bg_y[:, 0], dtype=float)
        pts = np.column_stack((yg, xg))
        ex_g = RegularGridInterpolator((y_axis, x_axis), bg_pol.ex, bounds_error=False, fill_value=0.0)(pts)
        ey_g = RegularGridInterpolator((y_axis, x_axis), bg_pol.ey, bounds_error=False, fill_value=0.0)(pts)
        xx = xg[:, None]
        yy = yg[:, None]
        pol = polarization_from_components(ex_g[:, None], ey_g[:, None])
        # Glyph positions are irregular here, so fix the stride and glyph size
        # explicitly instead of inferring them from a grid.
        kwargs.setdefault("stride", 1)
        if glyph == "quiver":
            kwargs.setdefault("length", 0.7 * dr)
        else:
            kwargs.setdefault("scale", 0.45 * dr)
    else:
        raise ValueError("sampling must be 'cartesian' or 'polar'.")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    if background == "intensity":
        vmax = float(np.nanmax(bg_pol.s0))
        norm = PowerNorm(gamma=float(intensity_gamma), vmin=0.0, vmax=vmax) if vmax > 0.0 else None
        im = ax.imshow(
            bg_pol.s0,
            extent=[float(np.min(bg_x)), float(np.max(bg_x)), float(np.min(bg_y)), float(np.max(bg_y))],
            origin="lower",
            cmap="magma",
            aspect="equal",
            norm=norm,
        )
        colorbar = _slim_colorbar(im, ax, label=r"$|E_x|^2 + |E_y|^2$")
        if vmax > 0.0:
            intensity_ticks = np.linspace(0.0, vmax, 5)
            colorbar.set_ticks(intensity_ticks)
            colorbar.set_ticklabels([f"{value:.4g}" for value in intensity_ticks])
    elif background == "cross_fraction":
        vmax_intensity = float(np.nanmax(bg_pol.s0))
        cross_fraction = np.divide(
            np.abs(bg_pol.ey) ** 2,
            bg_pol.s0,
            out=np.full_like(bg_pol.s0, np.nan, dtype=float),
            where=bg_pol.s0 > np.finfo(float).eps,
        )
        cross_fraction[bg_pol.s0 < float(cross_fraction_min_intensity) * vmax_intensity] = np.nan
        im = ax.imshow(
            cross_fraction,
            extent=[float(np.min(bg_x)), float(np.max(bg_x)), float(np.min(bg_y)), float(np.max(bg_y))],
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        _slim_colorbar(im, ax, label=r"$|E_y|^2 / (|E_x|^2 + |E_y|^2)$")
    elif background is not None:
        raise ValueError("background must be 'intensity', 'cross_fraction', or None.")

    if glyph == "ellipse":
        # ``pol`` is expressed in Cartesian field components, so the ellipse
        # renderer must interpret it in the Cartesian basis unless the caller
        # explicitly asked otherwise.
        kwargs.setdefault("ellipse_mode", "cartesian")
        plot_polarization_map(xx, yy, pol, ax=ax, **kwargs)
    elif glyph == "quiver":
        plot_polarization_quiver(xx, yy, pol, ax=ax, **kwargs)
    else:
        raise ValueError("glyph must be 'ellipse' or 'quiver'.")
    return ax, pol


def plot_field_polarization_summary(
    field,
    half_size: float | None = None,
    n_img: int = 500,
    title: str | None = None,
    component_view: str = "abs",
    cmap: str = "hot",
    min_intensity_fraction: float = 0.002,
    crop_padding: float = 1.15,
    show_cross_fraction: bool = False,
    polarization_kwargs: Mapping[str, Any] | None = None,
    cross_fraction_kwargs: Mapping[str, Any] | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Combine the component, intensity, polarization, and scalar-angle maps into one figure.

    Lays out (component 1, component 2, intensity) on the top row and
    (polarization ellipses, ellipticity angle, major-axis orientation) on the
    bottom row, all cropped to the same region where the signal exceeds
    ``min_intensity_fraction`` of its peak -- avoiding both a scattered set of
    separate figures and the mostly-blank margins a fixed large window leaves
    around a compact focal spot. With ``show_cross_fraction=True`` a fourth
    column adds the ``Ey``-fraction diagnostic.
    """

    from .polarization import polarization_map_from_field
    from .view import field_cartesian_maps

    polarization_kwargs = dict(polarization_kwargs or {})

    ncols = 4 if show_cross_fraction else 3
    if figsize is None:
        figsize = (4.6 * ncols, 8.2)
    fig, axes = plt.subplots(2, ncols, figsize=figsize, constrained_layout=True)

    rep = {"abs": np.abs, "real": np.real, "imag": np.imag}[component_view]
    c1, c2, extent, labels = field_cartesian_maps(field, half_size=half_size, n_img=n_img)
    i1, i2 = rep(c1), rep(c2)
    intensity = i1**2 + i2**2

    for ax, img, label in ((axes[0, 0], i1, labels[0]), (axes[0, 1], i2, labels[1])):
        vmax = float(np.max(np.abs(img))) or 1.0
        im = ax.imshow(np.abs(img), extent=extent, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax, aspect="equal")
        ax.set_title(rf"$E_{{{label}}}$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    vmax = float(np.max(intensity)) or 1.0
    im = axes[0, 2].imshow(intensity, extent=extent, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax, aspect="equal")
    axes[0, 2].set_title("Intensidad")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    polarization_kwargs.setdefault("half_size", half_size)
    polarization_kwargs.setdefault("n_img", n_img)
    plot_field_polarization(field, ax=axes[1, 0], **polarization_kwargs)
    axes[1, 0].set_title("Polarización")

    plot_polarization_scalar_map(
        field, "ellipticity", half_size=half_size, n_img=n_img,
        min_intensity_fraction=min_intensity_fraction, autocrop=False, ax=axes[1, 1],
    )
    axes[1, 1].set_title("Ángulo de elipticidad")

    plot_polarization_scalar_map(
        field, "orientation", half_size=half_size, n_img=n_img,
        min_intensity_fraction=min_intensity_fraction, autocrop=False, ax=axes[1, 2],
    )
    axes[1, 2].set_title("Orientación del eje mayor")

    # The cross-polarization diagnostic deliberately uses very low intensity
    # thresholds to reveal faint nodal-ring features across the whole sampled
    # window, so it is excluded from the shared crop below.
    uncropped_axes = set()
    if show_cross_fraction:
        cross_fraction_kwargs = dict(cross_fraction_kwargs or {})
        cross_fraction_kwargs.setdefault("half_size", half_size)
        cross_fraction_kwargs.setdefault("n_img", n_img)
        cross_fraction_kwargs.setdefault("background", "cross_fraction")
        cross_fraction_kwargs.setdefault("glyph", "quiver")
        plot_field_polarization(field, ax=axes[0, 3], **cross_fraction_kwargs)
        axes[0, 3].set_title("Fracción de polarización cruzada")
        axes[1, 3].axis("off")
        uncropped_axes = {axes[0, 3]}

    xx, yy, pol = polarization_map_from_field(field, half_size=half_size, n_img=n_img)
    valid = pol.s0 > float(min_intensity_fraction) * (float(np.nanmax(pol.s0)) + np.finfo(float).eps)
    xmin, xmax, ymin, ymax = _autocrop_extent(xx, yy, valid, padding=crop_padding)
    for ax in axes.flat:
        if ax.has_data() and ax not in uncropped_axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    if title:
        fig.suptitle(title)
    return fig, axes


def _components_in_basis(field, basis: Literal["cartesian", "circular"]):
    """Return ``(c1, c2, label1, label2)`` for a Field in the requested transverse basis."""

    from .coordinate_transformation import cartesian_to_circular

    if basis == "cartesian":
        return np.asarray(field.x), np.asarray(field.y), "x", "y"
    if basis == "circular":
        L = getattr(field, "L", None)
        R = getattr(field, "R", None)
        if L is None or R is None:
            L, R = cartesian_to_circular(np.asarray(field.x), np.asarray(field.y))
        return np.asarray(L), np.asarray(R), "L", "R"
    raise ValueError("basis must be 'cartesian' or 'circular'.")


def _row_label(ax, text):
    """Attach a bold row label to the leftmost axes of a row."""

    ax.annotate(
        text,
        xy=(-0.28, 0.5),
        xycoords="axes fraction",
        ha="center",
        va="center",
        rotation=90,
        fontsize=13,
        fontweight="bold",
    )


def plot_incident_and_focal_components(
    incident,
    focal,
    *,
    basis: Literal["cartesian", "circular"] = "cartesian",
    incident_half_size: float | None = None,
    focal_half_size: float | None = None,
    n_img: int = 500,
    cmap: str = "hot",
    component_view: str = "abs",
    incident_label: str = "Campo incidente",
    focal_label: str = "Plano focal",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Stack the component maps of two fields in one figure.

    The top row shows the incident field's components in ``basis`` (plus its
    intensity), and the bottom row does the same for the focal-plane field.
    ``basis="cartesian"`` picks ``(Ex, Ey)``; ``basis="circular"`` picks
    ``(EL, ER)`` — for a Cartesian field the circular components are computed
    from ``(Ex, Ey)``. The intensity column is basis-invariant.
    """

    from .view import sample_component_pair_on_cartesian_mesh

    if figsize is None:
        figsize = (14.0, 8.4)
    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)

    rep = {"abs": np.abs, "real": np.real, "imag": np.imag}[component_view]
    rows = (
        (incident, incident_half_size, incident_label, 0),
        (focal, focal_half_size, focal_label, 1),
    )
    for fld, half_size, row_label, row in rows:
        c1_raw, c2_raw, lab1, lab2 = _components_in_basis(fld, basis)
        c1, c2, extent = sample_component_pair_on_cartesian_mesh(
            c1_raw, c2_raw, fld.grid, half_size=half_size, n_img=n_img
        )
        i1, i2 = rep(c1), rep(c2)
        intensity = np.abs(c1) ** 2 + np.abs(c2) ** 2

        for col, (img, label) in enumerate(((i1, lab1), (i2, lab2))):
            ax = axes[row, col]
            vmax = float(np.max(np.abs(img))) or 1.0
            im = ax.imshow(np.abs(img), extent=extent, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax, aspect="equal")
            ax.set_title(rf"$E_{{{label}}}$")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        vmax_int = float(np.max(intensity)) or 1.0
        ax = axes[row, 2]
        im = ax.imshow(intensity, extent=extent, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax_int, aspect="equal")
        ax.set_title("Intensidad")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        _row_label(axes[row, 0], row_label)

    if title:
        fig.suptitle(title)
    return fig, axes


def _autocrop_row(axes_row, field, half_size, n_img, min_intensity_fraction, crop_padding):
    from .polarization import polarization_map_from_field

    xx, yy, pol = polarization_map_from_field(field, half_size=half_size, n_img=n_img)
    valid = pol.s0 > float(min_intensity_fraction) * (float(np.nanmax(pol.s0)) + np.finfo(float).eps)
    xmin, xmax, ymin, ymax = _autocrop_extent(xx, yy, valid, padding=crop_padding)
    for ax in axes_row:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)


def plot_incident_and_focal_polarization_map(
    incident,
    focal,
    *,
    incident_half_size: float | None = None,
    focal_half_size: float | None = None,
    n_img: int = 500,
    min_intensity_fraction: float = 0.002,
    crop_padding: float = 1.15,
    autocrop: bool = True,
    incident_polarization_kwargs: Mapping[str, Any] | None = None,
    focal_polarization_kwargs: Mapping[str, Any] | None = None,
    incident_label: str = "Mapa de polarización del campo incidente",
    focal_label: str = "Mapa de polarización en el plano focal",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Place the local polarization-ellipse maps of two fields side by side.

    Left panel: incident field. Right panel: focal-plane field. Each panel
    is autocropped to its own intensity support so a diffuse pupil and a
    compact focal spot can share the figure without one of them becoming a
    dot. The per-panel titles (``incident_label``, ``focal_label``) describe
    which field is shown.
    """

    if figsize is None:
        # Side by side: each panel gets a ~8" square, wide enough for
        # individual ellipses to read at a glance.
        figsize = (16.0, 8.5)
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    panels = (
        (incident, incident_half_size, incident_label, incident_polarization_kwargs, 0),
        (focal, focal_half_size, focal_label, focal_polarization_kwargs, 1),
    )
    for fld, half_size, panel_title, pkwargs, col in panels:
        pkwargs = dict(pkwargs or {})
        pkwargs.setdefault("half_size", half_size)
        pkwargs.setdefault("n_img", n_img)
        plot_field_polarization(fld, ax=axes[col], **pkwargs)
        axes[col].set_title(panel_title)

        if autocrop:
            _autocrop_row([axes[col]], fld, half_size, n_img, min_intensity_fraction, crop_padding)

    if title:
        fig.suptitle(title)
    return fig, axes


def plot_incident_and_focal_polarization_angles(
    incident,
    focal,
    *,
    incident_half_size: float | None = None,
    focal_half_size: float | None = None,
    n_img: int = 500,
    min_intensity_fraction: float = 0.002,
    crop_padding: float = 1.15,
    autocrop: bool = True,
    incident_label: str = "Campo incidente",
    focal_label: str = "Plano focal",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Stack the ellipticity and major-axis orientation maps of two fields.

    Top row: incident field. Bottom row: focal-plane field. Columns are the
    ellipticity angle ``chi`` and the major-axis orientation ``psi`` of the
    local polarization ellipse. Each row is autocropped to its own intensity
    support.
    """

    if figsize is None:
        figsize = (11.0, 10.0)
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    rows = (
        (incident, incident_half_size, incident_label, 0),
        (focal, focal_half_size, focal_label, 1),
    )
    for fld, half_size, row_label, row in rows:
        plot_polarization_scalar_map(
            fld, "ellipticity", half_size=half_size, n_img=n_img,
            min_intensity_fraction=min_intensity_fraction, autocrop=False, ax=axes[row, 0],
        )
        axes[row, 0].set_title("Ángulo de elipticidad")

        plot_polarization_scalar_map(
            fld, "orientation", half_size=half_size, n_img=n_img,
            min_intensity_fraction=min_intensity_fraction, autocrop=False, ax=axes[row, 1],
        )
        axes[row, 1].set_title("Orientación del eje mayor")

        if autocrop:
            _autocrop_row(axes[row, :], fld, half_size, n_img, min_intensity_fraction, crop_padding)

        _row_label(axes[row, 0], row_label)

    if title:
        fig.suptitle(title)
    return fig, axes
