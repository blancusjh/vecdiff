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

    Returns ``(pts, head_point, head_dir)`` in the local (ex, ey) frame:
    ``pts`` is an ``(points, 2)`` array tracing the ellipse, ``head_point`` is
    the tip of the major axis and ``head_dir`` the direction the arrowhead
    points -- the local tangent there (its sense encodes handedness), falling
    back to the major axis for linear light where that tangent vanishes.
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
    # Tangent at the major tip (theta=0): d/dtheta (xe, ye) = (0, ratio).
    head_dir = np.array([-sin_p * ratio, cos_p * ratio])
    if np.linalg.norm(head_dir) <= np.finfo(float).eps:
        head_dir = head_point.copy()  # linear light: point along the line outward
    return pts, head_point, head_dir


def _polar_to_cartesian_basis(vectors, cx, cy):
    phi = np.arctan2(cy, cx)
    radial = np.array([np.cos(phi), np.sin(phi)])
    azimuthal = np.array([-np.sin(phi), np.cos(phi)])
    return vectors[:, :1] * radial + vectors[:, 1:] * azimuthal


def _curve_segments(curve):
    return np.stack([curve, np.roll(curve, -1, axis=0)], axis=1)


def _arrowhead_triangle(tip, direction, length, width):
    """Return a 3-vertex filled arrowhead centred at ``tip``.

    A single clean filled triangle (drawn as a polygon) reads as a crisp head,
    unlike a pair of haloed line segments which blob at small glyph sizes.
    """

    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm <= np.finfo(float).eps:
        return None
    t = direction / norm
    perp = np.array([-t[1], t[0]])
    # Seat the head on the vertex (apex at the tip, base pulled inward) so it
    # never overhangs the ellipse: the glyph stays centred on its grid point and
    # the map reads as a uniform lattice.
    apex = tip
    base = tip - length * t
    return np.array([apex, base + 0.5 * width * perp, base - 0.5 * width * perp])


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
    target_ellipses: int = 20,
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
    arrow_opening_angle: float = np.deg2rad(55.0),
    arrow_length: float = 0.5,
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

        pts, head_point, head_dir = _ellipse_glyph(ex_i, ey_i, ellipse_points)
        if ellipse_mode == "polar":
            pts = _polar_to_cartesian_basis(pts, cx, cy)
            head_point = _polar_to_cartesian_basis(head_point[None, :], cx, cy)[0]
            head_dir = _polar_to_cartesian_basis(head_dir[None, :], cx, cy)[0]

        center = np.array([cx, cy])
        curve = glyph_extent * pts + center
        segments = _curve_segments(curve)
        figure_segments.append(segments)

        head_length = arrow_length * glyph_extent
        head = _arrowhead_triangle(
            glyph_extent * head_point + center,
            head_dir,
            head_length,
            head_width_ratio * head_length,
        )
        if head is not None:
            head_polys.append(head)

        if color_by_phase:
            c = (phase_i + np.pi) / (2.0 * np.pi)
            colors.append(np.full(segments.shape[0], c))
            if head is not None:
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
            plt.colorbar(lc, ax=ax, label="Fase normalizada")

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
    plt.colorbar(im, ax=ax, label=label)
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
            plt.colorbar(quiver, ax=ax, label=r"$|E_y|^2 / (|E_x|^2 + |E_y|^2)$")
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
    """

    from .polarization import polarization_from_components, polarization_map_from_field

    # Always sample the field on a Cartesian mesh for the raster background; the
    # glyph positions come from this mesh (Cartesian) or an even polar layout.
    bg_x, bg_y, bg_pol = polarization_map_from_field(field, half_size=half_size, n_img=n_img)
    if sampling == "cartesian":
        xx, yy, pol = bg_x, bg_y, bg_pol
    elif sampling == "polar":
        from scipy.interpolate import RegularGridInterpolator

        n_rings = max(1, int(kwargs.pop("n_rings", 12)))
        angular_spacing = max(float(kwargs.pop("angular_spacing", 1.0)), np.finfo(float).eps)
        r_max = float(half_size) if half_size is not None else float(np.max(np.hypot(bg_x, bg_y)))
        dr = r_max / (n_rings + 0.5)
        # Always include an on-axis (r=0) glyph, then the concentric rings.
        xg, yg = [np.array([0.0])], [np.array([0.0])]
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
        colorbar = plt.colorbar(im, ax=ax, label=r"$|E_x|^2 + |E_y|^2$")
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
        plt.colorbar(im, ax=ax, label=r"$|E_y|^2 / (|E_x|^2 + |E_y|^2)$")
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
