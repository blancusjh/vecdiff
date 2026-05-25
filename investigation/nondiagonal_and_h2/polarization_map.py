import numpy as np
import matplotlib.pyplot as plt


def _stokes_from_jones(ex, ey):
    ex = np.asarray(ex, dtype=complex)
    ey = np.asarray(ey, dtype=complex)
    s0 = np.abs(ex) ** 2 + np.abs(ey) ** 2
    s1 = np.abs(ex) ** 2 - np.abs(ey) ** 2
    s2 = 2.0 * np.real(ex * np.conj(ey))
    s3 = -2.0 * np.imag(ex * np.conj(ey))
    return s0, s1, s2, s3


def plot_polarization_map(
    xx,
    yy,
    ex,
    ey,
    title="Polarization Map",
    step=24,
    max_curves=900,
    min_intensity_frac=0.008,
    phase_ref=0.0,
    curve_color=None,
    arrow_color=None,
    save_path=None,
    ax=None,
    background=None,
    background_cmap="RdBu_r",
    scale_by_intensity=True,
    size_scale=1.0,
    show=True,
):
    s0, s1, s2, s3 = _stokes_from_jones(ex, ey)
    s0max = np.max(s0) + 1e-30

    n = xx.shape[0]
    ds = max(1, n // step)
    xs = xx[::ds, ::ds]
    ys = yy[::ds, ::ds]
    s0s = s0[::ds, ::ds]
    s1s = s1[::ds, ::ds]
    s2s = s2[::ds, ::ds]
    s3s = s3[::ds, ::ds]

    valid = s0s > (min_intensity_frac * s0max)
    if np.count_nonzero(valid) > max_curves:
        keep = np.flatnonzero(valid)
        stride = int(np.ceil(keep.size / max_curves))
        mask = np.zeros_like(valid, dtype=bool)
        mask.flat[keep[::stride]] = True
        valid = mask

    dx = float(np.median(np.diff(xs[0, :]))) if xs.shape[1] > 1 else float(np.max(xx) - np.min(xx))
    dy = float(np.median(np.diff(ys[:, 0]))) if ys.shape[0] > 1 else float(np.max(yy) - np.min(yy))
    cell = max(min(abs(dx), abs(dy)), 1e-30)
    base = 0.55 * cell * float(size_scale)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.figure

    if background is not None:
        b = np.asarray(background, dtype=float)
        vmax = np.max(np.abs(b)) + 1e-30
        ax.imshow(
            b,
            extent=[float(np.min(xx)), float(np.max(xx)), float(np.min(yy)), float(np.max(yy))],
            origin="lower",
            cmap=background_cmap,
            aspect="equal",
            vmin=-vmax,
            vmax=vmax,
            alpha=0.85,
        )

    # Choose visible defaults for both light and dark backgrounds
    if curve_color is None or arrow_color is None:
        fc = ax.get_facecolor()
        lum = 0.2126 * fc[0] + 0.7152 * fc[1] + 0.0722 * fc[2]
        if curve_color is None:
            curve_color = "lightgray" if lum < 0.45 else "black"
        if arrow_color is None:
            arrow_color = curve_color

    t = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=True)
    exs = ex[::ds, ::ds]
    eys = ey[::ds, ::ds]

    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            if not valid[i, j]:
                continue

            e0x = exs[i, j]
            e0y = eys[i, j]
            if scale_by_intensity:
                amp = np.sqrt(np.clip(s0s[i, j] / s0max, 0.0, 1.0))
                scale = base * amp
            else:
                scale = base
            if scale <= 0.0:
                continue

            # Normalize Jones vector to preserve polarization shape without huge loops.
            norm = np.sqrt(np.abs(e0x) ** 2 + np.abs(e0y) ** 2) + 1e-30
            exu = e0x / norm
            eyu = e0y / norm

            ex_curve = np.real(exu * np.exp(1j * t))
            ey_curve = np.real(eyu * np.exp(1j * t))

            px = xs[i, j] + scale * ex_curve
            py = ys[i, j] + scale * ey_curve
            ax.plot(px, py, color=curve_color, lw=1.2, alpha=0.98)

            # Arrow overlaid on the polarization curve (tangent segment on-curve).
            k = int((phase_ref % (2.0 * np.pi)) / (2.0 * np.pi) * (t.size - 2))
            du = px[k + 1] - px[k]
            dv = py[k + 1] - py[k]
            # Make the phase marker correspond to the arrow center, not the tip.
            ax_x = px[k] - 0.5 * du
            ax_y = py[k] - 0.5 * dv

            ax.arrow(
                ax_x,
                ax_y,
                du,
                dv,
                length_includes_head=True,
                head_width=0.42 * scale,
                head_length=0.50 * scale,
                fc=arrow_color,
                ec=arrow_color,
                alpha=0.98,
                lw=0.9,
                zorder=3,
            )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(float(np.min(xx)), float(np.max(xx)))
    ax.set_ylim(float(np.min(yy)), float(np.max(yy)))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
