import numpy as np
import matplotlib.pyplot as plt


def apply_style(dark_background=False):
    if dark_background:
        plt.style.use("dark_background")


def radial_to_2d(profile_q, q, rho):
    profile_q = np.asarray(profile_q)
    return np.interp(rho, q, np.abs(profile_q), left=float(np.abs(profile_q[0])), right=0.0)


def radial_to_2d_abs(profile_q, q, rho):
    return radial_to_2d(profile_q, q, rho)


def compress_intensity(I, mode="gamma", gamma=0.6, k=12.0):
    I = np.asarray(I, dtype=float)
    if mode == "linear":
        return I, "I"
    if mode == "sqrt":
        return np.sqrt(np.maximum(I, 0.0)), "sqrt(I)"
    if mode == "asinh":
        return np.arcsinh(k * np.maximum(I, 0.0)), f"asinh({k} I)"
    if mode == "log":
        return np.log10(np.maximum(I, 0.0) + 1e-12), "log10(I + 1e-12)"
    return np.maximum(I, 0.0) ** gamma, f"I^{gamma}"


def plot_intensity_maps(maps, q_view, display_mode="gamma", title=None, ncols=3, save_path=None, show=True):
    n = len(maps)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, I) in zip(axes, maps):
        if name == "I_total - I_scalar":
            vmax = np.max(np.abs(I)) + 1e-30
            im = ax.imshow(
                I,
                extent=[-q_view, q_view, -q_view, q_view],
                origin="lower",
                cmap="RdBu_r",
                aspect="equal",
                vmin=-vmax,
                vmax=vmax,
            )
            cb_label = "Delta I"
        else:
            I_plot, cb_label = compress_intensity(I, mode=display_mode)
            vmax_pos = float(np.nanmax(I_plot))
            if not np.isfinite(vmax_pos) or vmax_pos <= 0.0:
                vmax_pos = 1.0
            im = ax.imshow(
                I_plot,
                extent=[-q_view, q_view, -q_view, q_view],
                origin="lower",
                cmap="hot",
                aspect="equal",
                vmin=0.0,
                vmax=vmax_pos,
            )
        ax.set_title(name)
        ax.set_xlabel("x [q]")
        ax.set_ylabel("y [q]")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cb_label)

    for ax in axes[n:]:
        ax.axis("off")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


def _regularize_for_quiver(ex, ey):
    mag = np.sqrt(ex**2 + ey**2)
    ref = np.percentile(mag, 95) + 1e-30
    gain = np.tanh(mag / ref) / (mag + 1e-30)
    return ex * gain, ey * gain


def plot_vector_fields(
    xx,
    yy,
    Ex_v,
    Ey_v,
    q_view,
    R,
    a,
    Ex_inc,
    Ey_inc,
    vector_plot_mode="stream",
):
    step_quiver = max(1, xx.shape[0] // 24)
    step_stream = max(1, xx.shape[0] // 96)

    xp = np.linspace(-R, R, xx.shape[0])
    xxp, yyp = np.meshgrid(xp, xp)
    pupil = np.sqrt(xxp**2 + yyp**2) <= a

    xxp_q = xxp[::step_quiver, ::step_quiver]
    yyp_q = yyp[::step_quiver, ::step_quiver]
    pupil_q = pupil[::step_quiver, ::step_quiver]
    ex_in_q = np.where(pupil_q, Ex_inc, 0.0)
    ey_in_q = np.where(pupil_q, Ey_inc, 0.0)

    xxp_st = xxp[::step_stream, ::step_stream]
    yyp_st = yyp[::step_stream, ::step_stream]
    pupil_st = pupil[::step_stream, ::step_stream]
    ex_in_st = np.where(pupil_st, Ex_inc, np.nan)
    ey_in_st = np.where(pupil_st, Ey_inc, np.nan)

    xx_q = xx[::step_quiver, ::step_quiver]
    yy_q = yy[::step_quiver, ::step_quiver]
    ex_out_q = Ex_v[::step_quiver, ::step_quiver]
    ey_out_q = Ey_v[::step_quiver, ::step_quiver]
    ex_out_q, ey_out_q = _regularize_for_quiver(ex_out_q, ey_out_q)

    xx_st = xx[::step_stream, ::step_stream]
    yy_st = yy[::step_stream, ::step_stream]
    ex_out_st = Ex_v[::step_stream, ::step_stream]
    ey_out_st = Ey_v[::step_stream, ::step_stream]
    ref_st = np.percentile(np.sqrt(ex_out_st**2 + ey_out_st**2), 95) + 1e-30
    ex_out_st = ex_out_st / ref_st
    ey_out_st = ey_out_st / ref_st
    tiny = np.sqrt(ex_out_st**2 + ey_out_st**2) < 1e-4
    ex_out_st = np.where(tiny, np.nan, ex_out_st)
    ey_out_st = np.where(tiny, np.nan, ey_out_st)

    figv, axv = plt.subplots(1, 2, figsize=(12, 5))

    if vector_plot_mode == "stream":
        speed_in = np.sqrt(np.nan_to_num(ex_in_st, nan=0.0) ** 2 + np.nan_to_num(ey_in_st, nan=0.0) ** 2)
        axv[0].streamplot(
            xxp_st[0, :], yyp_st[:, 0], ex_in_st, ey_in_st,
            color=speed_in, cmap="Blues", density=1.6, linewidth=1.1, arrowsize=0.9
        )
    else:
        axv[0].quiver(xxp_q, yyp_q, ex_in_q, ey_in_q, color="tab:blue", pivot="mid", scale=6.5, scale_units="xy", width=0.006)

    axv[0].set_title("Incident field vectors (pupil plane)")
    axv[0].set_xlabel("x [mm]")
    axv[0].set_ylabel("y [mm]")
    axv[0].set_aspect("equal")
    axv[0].set_xlim(-R, R)
    axv[0].set_ylim(-R, R)

    if vector_plot_mode == "stream":
        speed_out = np.sqrt(np.nan_to_num(ex_out_st, nan=0.0) ** 2 + np.nan_to_num(ey_out_st, nan=0.0) ** 2)
        axv[1].streamplot(
            xx_st[0, :], yy_st[:, 0], ex_out_st, ey_out_st,
            color=speed_out, cmap="Reds", density=1.8, linewidth=1.0, arrowsize=0.9
        )
    else:
        axv[1].quiver(xx_q, yy_q, ex_out_q, ey_out_q, color="tab:red", pivot="mid", scale=0.8, scale_units="xy", width=0.0075)

    axv[1].set_title("Output field vectors (Ex, Ey)")
    axv[1].set_xlabel("x [q]")
    axv[1].set_ylabel("y [q]")
    axv[1].set_aspect("equal")
    axv[1].set_xlim(-q_view, q_view)
    axv[1].set_ylim(-q_view, q_view)

    figv.suptitle("Vector fields: incident vs output")
    plt.tight_layout()
    plt.show()


def plot_vector_comparison_panels(panels, mode="stream", suptitle=None):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 5))
    axes = np.atleast_1d(axes)

    for ax, panel in zip(axes, panels):
        x = panel["x"]
        y = panel["y"]
        ex = panel["ex"]
        ey = panel["ey"]
        title = panel.get("title", "")
        xlabel = panel.get("xlabel", "x")
        ylabel = panel.get("ylabel", "y")
        cmap = panel.get("cmap", "viridis")
        density = panel.get("density", 1.8)
        linewidth = panel.get("linewidth", 1.0)
        arrowsize = panel.get("arrowsize", 0.9)
        quiver_scale = panel.get("quiver_scale", 40)
        color = panel.get("color", None)
        xlim = panel.get("xlim", None)
        ylim = panel.get("ylim", None)
        regularize = panel.get("regularize_for_quiver", False)
        tiny_cut = panel.get("tiny_cut", 1e-4)

        ex = np.asarray(ex, float)
        ey = np.asarray(ey, float)

        if mode == "stream":
            ref = np.percentile(np.sqrt(ex**2 + ey**2), 95) + 1e-30
            ex_p = ex / ref
            ey_p = ey / ref
            tiny = np.sqrt(ex_p**2 + ey_p**2) < tiny_cut
            ex_p = np.where(tiny, np.nan, ex_p)
            ey_p = np.where(tiny, np.nan, ey_p)
            speed = np.sqrt(np.nan_to_num(ex_p, nan=0.0) ** 2 + np.nan_to_num(ey_p, nan=0.0) ** 2)
            ax.streamplot(
                x[0, :], y[:, 0], ex_p, ey_p,
                color=speed if color is None else color,
                cmap=cmap if color is None else None,
                density=density,
                linewidth=linewidth,
                arrowsize=arrowsize,
            )
        else:
            ex_p = ex
            ey_p = ey
            if regularize:
                ex_p, ey_p = _regularize_for_quiver(ex_p, ey_p)
            ax.quiver(x, y, ex_p, ey_p, color=color or "tab:blue", pivot="mid", scale=quiver_scale)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()
