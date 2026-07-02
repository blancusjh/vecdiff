"""Time-harmonic animation of sampled fields.

A monochromatic field oscillates in time as ``E(x, y) e^{-i omega t}``, so the
*physical* (real) field vector at each point is
``Re[E(x, y) e^{-i omega t}]``.  As ``omega t`` sweeps one optical cycle the
vector traces the local polarization ellipse.  Animating it over a period gives
the familiar "living" picture of a diffraction pattern: a static intensity
background with a quiver whose arrows rotate and pulse in step with the phase.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .polarization import polarization_map_from_field

__all__ = ["harmonic_field_frames", "animate_harmonic_field", "save_animation"]


def harmonic_field_frames(field, *, half_size=None, n_img=240):
    """Sample a field on a Cartesian mesh for harmonic animation.

    Returns ``(xx, yy, Ex, Ey, intensity)``: the mesh, the complex Cartesian
    components resampled on it (works for Cartesian and polar input grids
    through :func:`polarization_map_from_field`), and the static intensity
    ``|Ex|^2 + |Ey|^2``.
    """
    xx, yy, pol = polarization_map_from_field(field, half_size=half_size, n_img=n_img)
    intensity = np.abs(pol.ex) ** 2 + np.abs(pol.ey) ** 2
    return xx, yy, pol.ex, pol.ey, intensity


def animate_harmonic_field(
    field,
    *,
    half_size=None,
    n_img=240,
    n_frames=60,
    n_periods=1,
    target_arrows=32,
    cmap="afmhot",
    intensity_gamma=0.4,
    arrow_scale=1.3,
    arrow_norm=0.4,
    ax=None,
    title="Campo instantáneo",
    caption=None,
):
    """Animate the instantaneous real field ``Re[E e^{-i omega t}]`` over the cycle.

    A static intensity background (``|E|^2``) is overlaid with a quiver of the
    instantaneous real field vector, updated frame by frame as ``omega t`` runs
    over ``n_periods`` optical periods in ``n_frames`` steps.

    Parameters
    ----------
    field : Field
        Any field exposing Cartesian components ``x``/``y`` and a ``grid``.
    half_size : float, optional
        Half-width of the displayed window (in grid units).  ``None`` uses the
        field's native extent.
    target_arrows : int
        Approximate number of quiver arrows across the window; the mesh is
        subsampled to about this density.
    arrow_scale : float
        Peak arrow length in units of the arrow spacing.
    arrow_norm : float
        Amplitude compression for the arrow lengths, in ``[0, 1]``: the arrow
        length is proportional to the instantaneous projection times
        ``amplitude^(arrow_norm - 1)``.  ``1`` shows the true instantaneous
        magnitude (weak rings almost invisible); ``0`` shows pure direction
        (uniform arrows); the default ``0.4`` keeps the faint rings legible
        while the bright core does not dominate.
    intensity_gamma : float
        Display gamma for the background (``intensity ** gamma`` after
        normalization); ``< 1`` brightens the faint rings.
    caption : str, optional
        Static figure sub-title, e.g. the optical-system parameters.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    xx, yy, Ex, Ey, intensity = harmonic_field_frames(field, half_size=half_size, n_img=n_img)

    axis_x = xx[0, :]
    axis_y = yy[:, 0]
    extent = [float(axis_x[0]), float(axis_x[-1]), float(axis_y[0]), float(axis_y[-1])]

    imax = float(intensity.max())
    bg = (intensity / imax) ** float(intensity_gamma) if imax > 0 else intensity

    # Subsample the mesh to ~target_arrows arrows across the window.
    stride = max(1, int(round(n_img / max(int(target_arrows), 1))))
    sl = slice(stride // 2, None, stride)
    qx, qy = xx[sl, sl], yy[sl, sl]
    qEx, qEy = Ex[sl, sl], Ey[sl, sl]

    # Amplitude-compressed arrow lengths: multiply the instantaneous projection
    # by ``env^(arrow_norm - 1)`` so faint rings still show visible arrows while
    # the bright core does not dominate.  ``env`` is clipped away from zero so
    # deep nulls do not blow the weight up.
    env = np.hypot(np.abs(qEx), np.abs(qEy))
    env_max = float(env.max()) or 1.0
    env_safe = np.clip(env, 1e-2 * env_max, None)
    weight = env_safe ** (arrow_norm - 1.0)

    spacing = float(np.mean(np.diff(axis_x))) * stride
    peak_disp = env_max ** float(arrow_norm)
    quiver_scale = peak_disp / (arrow_scale * spacing)

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    fig = ax.figure

    im = ax.imshow(bg, extent=extent, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$|E|^2$ (norm.)")

    q = ax.quiver(qx, qy, np.real(qEx) * weight, np.real(qEy) * weight,
                  color="black", pivot="middle", angles="xy",
                  scale_units="xy", scale=quiver_scale, width=0.003)
    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel(r"$x/\lambda$")
    ax.set_ylabel(r"$y/\lambda$")
    title_artist = ax.set_title(f"{title}  |  $\\omega t = 0.00$")
    if caption:
        fig.suptitle(caption, fontsize=8)

    omega_t = np.linspace(0.0, 2.0 * np.pi * n_periods, n_frames, endpoint=False)

    def update(frame):
        phase = np.exp(-1.0j * omega_t[frame])
        q.set_UVC(np.real(qEx * phase) * weight, np.real(qEy * phase) * weight)
        title_artist.set_text(f"{title}  |  $\\omega t = {omega_t[frame]:.2f}$")
        return q, title_artist

    return FuncAnimation(fig, update, frames=n_frames, interval=1000.0 / 24.0, blit=False)


def save_animation(anim, path, *, fps=24, dpi=90, colors=128):
    """Save a :class:`FuncAnimation` to ``path`` (``.gif`` via Pillow, else ffmpeg).

    For GIFs the frames are re-quantized to an adaptive ``colors``-entry palette
    and optimized, which shrinks the file several-fold versus the raw writer
    output (smooth ring gradients otherwise compress poorly).
    """
    path = str(path)
    if not path.lower().endswith(".gif"):
        anim.save(path, fps=fps, dpi=dpi)
        return path

    from matplotlib.animation import PillowWriter
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)

    from PIL import Image, ImageSequence
    with Image.open(path) as gif:
        frames = [f.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT)
                  for f in ImageSequence.Iterator(gif)]
        duration = gif.info.get("duration", int(1000 / fps))
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0,
                   duration=duration, optimize=True, disposal=2)
    return path
