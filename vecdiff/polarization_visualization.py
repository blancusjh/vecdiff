"""Visualization helpers for polarization maps."""

from __future__ import annotations

from typing import Any, Literal, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, PowerNorm

from .polarization import PolarizationData


def _polarization_curve(ex, ey, points, scale):
    t = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    exp_it = np.exp(1j * t)
    curve = scale * np.column_stack([np.real(ex * exp_it), np.real(ey * exp_it)])
    tangent = scale * np.column_stack([np.real(1j * ex * exp_it), np.real(1j * ey * exp_it)])
    return curve, tangent


def _polar_to_cartesian_basis(vectors, cx, cy):
    phi = np.arctan2(cy, cx)
    radial = np.array([np.cos(phi), np.sin(phi)])
    azimuthal = np.array([-np.sin(phi), np.cos(phi)])
    return vectors[:, :1] * radial + vectors[:, 1:] * azimuthal


def _curve_segments(curve):
    return np.stack([curve, np.roll(curve, -1, axis=0)], axis=1)


def _arrowhead(point, tangent, length, opening_angle):
    norm = np.linalg.norm(tangent)
    if norm <= np.finfo(float).eps:
        return np.empty((0, 2, 2))

    angle = np.arctan2(-tangent[1], -tangent[0])
    half = 0.5 * opening_angle
    directions = np.array(
        [
            [np.cos(angle - half), np.sin(angle - half)],
            [np.cos(angle + half), np.sin(angle + half)],
        ]
    )
    tails = point + length * directions
    tips = np.repeat(point[None, :], 2, axis=0)
    return np.stack([tails, tips], axis=1)


def _arrow_index(curve, angle):
    curve_angle = np.arctan2(curve[:, 1], curve[:, 0])
    error = np.abs(np.angle(np.exp(1j * (curve_angle - angle))))
    return int(np.argmin(error))


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


def plot_polarization_map(
    x: np.ndarray,
    y: np.ndarray,
    pol: PolarizationData,
    stride: int | None = None,
    target_ellipses: int = 14,
    max_ellipses: int | None = None,
    scale: float | None = None,
    ellipse_points: int = 72,
    min_intensity_fraction: float = 0.01,
    color_by_phase: bool = False,
    phase_cmap: str = "twilight_shifted",
    phase_colorbar: bool = True,
    scale_by_intensity: bool = True,
    intensity_scale_mode: Literal["linear", "log", "power"] = "power",
    intensity_scale_gamma: float = 0.5,
    min_ellipse_scale: float = 0.30,
    arrow_opening_angle: float = np.deg2rad(55.0),
    arrow_length: float = 0.25,
    curve_kwargs: Mapping[str, Any] | None = None,
    arrowhead_kwargs: Mapping[str, Any] | None = None,
    ellipse_mode: Literal["polar", "cartesian"] = "polar",
    ax=None,
):
    """Draw local polarization ellipses over a sampled plane.

    ``ellipse_mode="polar"`` interprets ``pol.ex`` and ``pol.ey`` as local
    radial and azimuthal components.  ``ellipse_mode="cartesian"`` preserves
    the previous behavior and interprets them as x and y components.

    Samples are concentrated in the region above ``min_intensity_fraction`` of
    the peak intensity when ``stride`` is omitted.  The threshold is evaluated
    on ``S0 = |Ex|² + |Ey|²``; locations below it are omitted because their
    polarization state is not visually meaningful.

    When ``scale_by_intensity`` is enabled, ``intensity_scale_mode`` controls
    how the relative amplitude changes ellipse size.  ``"linear"`` preserves
    the default behavior, ``"power"`` uses ``relative_amplitude**gamma``, and
    ``"log"`` uses a normalized ``log1p(gamma * relative_amplitude)`` curve.
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
    if max_ellipses is not None and np.count_nonzero(valid) > max_ellipses:
        valid_amp = amp[valid]
        keep_count = max(1, int(max_ellipses))
        threshold_index = max(0, valid_amp.size - keep_count)
        adaptive_threshold = np.partition(valid_amp, threshold_index)[threshold_index]
        valid &= amp >= adaptive_threshold

    if scale is None:
        dx = np.nanmedian(np.abs(np.diff(xs[0]))) if xs.shape[1] > 1 else np.ptp(x)
        dy = np.nanmedian(np.abs(np.diff(ys[:, 0]))) if ys.shape[0] > 1 else np.ptp(y)
        scale = 0.46 * min(float(dx), float(dy))

    figure_segments = []
    arrow_segments = []
    colors = []
    arrow_colors = []

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

        local_scale = scale / (amp_i + np.finfo(float).eps)
        if scale_by_intensity:
            relative_amp = amp_i / amp_max
            size_factor = _intensity_scale_factor(
                relative_amp,
                intensity_scale_mode,
                intensity_scale_gamma,
                min_ellipse_scale,
            )
            local_scale *= size_factor
        else:
            size_factor = 1.0

        curve, tangent = _polarization_curve(ex_i, ey_i, ellipse_points, local_scale)
        if ellipse_mode == "polar":
            curve = _polar_to_cartesian_basis(curve, cx, cy)
            tangent = _polar_to_cartesian_basis(tangent, cx, cy)
        arrow_at = _arrow_index(curve, phase_i)
        curve = curve + np.array([cx, cy])

        segments = _curve_segments(curve)
        figure_segments.append(segments)

        arrows = _arrowhead(
            curve[arrow_at],
            tangent[arrow_at],
            arrow_length * scale * size_factor,
            arrow_opening_angle,
        )
        if arrows.size:
            arrow_segments.append(arrows)

        if color_by_phase:
            c = (phase_i + np.pi) / (2.0 * np.pi)
            colors.append(np.full(segments.shape[0], c))
            if arrows.size:
                arrow_colors.append(np.full(arrows.shape[0], c))

    if not figure_segments:
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))
        ax.set_aspect("equal", adjustable="box")
        return ax

    figure_segments = np.concatenate(figure_segments, axis=0)
    arrow_segments = np.concatenate(arrow_segments, axis=0) if arrow_segments else np.empty((0, 2, 2))

    if color_by_phase:
        lc_kwargs = _line_kwargs(curve_kwargs, linewidth=0.45, zorder=3.0)
        lc_kwargs.pop("colors", None)
        cmap = lc_kwargs.pop("cmap", phase_cmap)
        lc = LineCollection(figure_segments, array=np.concatenate(colors), cmap=cmap, **lc_kwargs)
        ax.add_collection(lc)
        if phase_colorbar:
            plt.colorbar(lc, ax=ax, label="Normalized phase")

        if arrow_segments.size:
            ah_kwargs = _line_kwargs(arrowhead_kwargs, linewidth=0.55, zorder=4.0)
            ah_kwargs.pop("colors", None)
            ax.add_collection(
                LineCollection(arrow_segments, array=np.concatenate(arrow_colors), cmap=cmap, **ah_kwargs)
            )
    else:
        lc_kwargs = _line_kwargs(curve_kwargs, linewidth=0.70, color="black", zorder=3.0)
        ax.add_collection(LineCollection(figure_segments, **lc_kwargs))
        if arrow_segments.size:
            ah_kwargs = _line_kwargs(arrowhead_kwargs, linewidth=0.85, color=lc_kwargs.get("colors", "black"), zorder=4.0)
            ax.add_collection(LineCollection(arrow_segments, **ah_kwargs))

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
    target_arrows: int = 14,
    length: float | None = None,
    arrow_length_fraction: float = 1.0,
    min_intensity_fraction: float = 0.01,
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
    This makes the default suitable for dense polarization maps without
    per-example tuning.
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
        "color": "0.82",
        "alpha": 0.7,
        "width": 0.0038,
        "headwidth": 3.6,
        "headlength": 4.2,
        "headaxislength": 3.8,
        "zorder": 4.0,
    }
    if quiver_kwargs:
        kwargs.update(dict(quiver_kwargs))

    if color_by_cross_fraction:
        kwargs.pop("color", None)
        kwargs.setdefault("cmap", cross_fraction_cmap)
        kwargs.setdefault("norm", Normalize(vmin=0.0, vmax=1.0))
        quiver = ax.quiver(xs[valid], ys[valid], u[valid], v[valid], cross_fraction[valid], **kwargs)
        if cross_fraction_colorbar:
            plt.colorbar(quiver, ax=ax, label=r"$|E_y|^2 / (|E_x|^2 + |E_y|^2)$")
    else:
        ax.quiver(xs[valid], ys[valid], u[valid], v[valid], **kwargs)
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
    """Plot field polarization with Cartesian or native polar glyph sampling."""

    from .polarization import polarization_from_components, polarization_map_from_field

    # Always use a Cartesian mesh for the raster background, but optionally
    # retain the native (q, phi) mesh for glyph placement.
    bg_x, bg_y, bg_pol = polarization_map_from_field(field, half_size=half_size, n_img=n_img)
    if sampling == "cartesian":
        xx, yy, pol = bg_x, bg_y, bg_pol
    elif sampling == "polar":
        if field.grid.type != "polar":
            raise ValueError("sampling='polar' requires a field sampled on a polar grid.")
        radial = np.asarray(field.grid.r, dtype=float)
        keep = np.ones_like(radial, dtype=bool) if half_size is None else radial <= float(half_size)
        xx = np.asarray(field.grid.X)[:, keep]
        yy = np.asarray(field.grid.Y)[:, keep]
        pol = polarization_from_components(
            np.asarray(field.x)[:, keep],
            np.asarray(field.y)[:, keep],
        )
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
        plot_polarization_map(xx, yy, pol, ax=ax, **kwargs)
    elif glyph == "quiver":
        plot_polarization_quiver(xx, yy, pol, ax=ax, **kwargs)
    else:
        raise ValueError("glyph must be 'ellipse' or 'quiver'.")
    return ax, pol
