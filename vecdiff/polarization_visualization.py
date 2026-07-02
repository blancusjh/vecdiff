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
            plt.colorbar(lc, ax=ax, label="Normalized phase")

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
