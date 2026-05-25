import numpy as np
import matplotlib.pyplot as plt

from vecdiff.fresnel import FresnelOvoidParax
from vecdiff.hankel import HankelTransform


# ── optical system ────────────────────────────────────────────────────────────
n0 = 1.0
ni = 1.5
z0 = -10.0
zi = 6.0
R = 2.6
lam = 532e-6

PARAM_STR = f"""n0={n0}, ni={ni},
z0={z0}, zi={zi},
R={R} mm
(paraxial Fresnel)"""

N_R = 1000
N_Q = 1000
N_IMG = 512
VECTOR_PLOT_MODE = "stream"
DARK_BACKGROUND = True
PROFILE_N = 2
EX_INC = 1.0
EY_INC = 0.0

HT = HankelTransform.transform_array


def radial_to_2d(profile_q, q, rho):
    return np.interp(rho, q, np.abs(profile_q), left=float(np.abs(profile_q[0])), right=0.0)


def propagate(r, q, tp, ts, E):
    h0 = HT(0, r, (tp + ts) * E, q)
    h2 = HT(2, r, (tp - ts) * E, q)
    return h0, h2


def compress_intensity(I, mode="gamma", gamma=0.6, k=12.0):
    I = np.asarray(I, dtype=float)
    if mode == "linear":
        return I, "I"
    if mode == "sqrt":
        return np.sqrt(I), "sqrt(I)"
    if mode == "asinh":
        return np.arcsinh(k * I), f"asinh({k} I)"
    if mode == "log":
        return np.log10(I + 1e-12), "log10(I + 1e-12)"
    return I**gamma, f"I^{gamma}"


def main(display_mode="gamma", vector_plot_mode="quiver"):
    if DARK_BACKGROUND:
        plt.style.use("dark_background")

    r = np.linspace(0.0, R, N_R)
    q_max = ni / (lam * zi) * R
    q = q_max * np.linspace(0.0, 1.0, N_Q) ** 2.0

    ts, tp = FresnelOvoidParax(n0, ni, z0, zi).coeffs(r)
    ts = np.asarray(ts, float)
    tp = np.asarray(tp, float)

    a = 0.95 * R
    Eamp = ((r / (a + 1e-30)) ** PROFILE_N) * (r <= a)

    q_view = 0.0045 * q_max
    xs = np.linspace(-q_view, q_view, N_IMG)
    xx, yy = np.meshgrid(xs, xs)
    rho = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)

    # Scalar reference: same procedure, but ts=tp=1
    ts_scalar = np.ones_like(ts)
    tp_scalar = np.ones_like(tp)
    h0s, h2s = propagate(r, q, tp_scalar, ts_scalar, Eamp)
    # h2s should be zero because (tp-ts)=0
    I_scalar_2d = radial_to_2d(np.abs(2.0 * π * h0s) ** 2, q, rho)

    # Hankel terms for basis responses (X and Y use the same radial amplitude)
    h0x, h2x = propagate(r, q, tp, ts, Eamp)
    h0y, h2y = propagate(r, q, tp, ts, Eamp)

    h0x_2d = radial_to_2d(h0x, q, rho)
    h2x_2d = radial_to_2d(h2x, q, rho)
    h0y_2d = radial_to_2d(h0y, q, rho)
    h2y_2d = radial_to_2d(h2y, q, rho)

    c2 = np.cos(2.0 * phi)
    s2 = np.sin(2.0 * phi)

    # Matrix M_xy per point (using your matrix form)
    M11 = h0x_2d - c2 * h2x_2d
    M12 = -s2 * h2y_2d
    M21 = -s2 * h2x_2d
    M22 = h0y_2d + c2 * h2y_2d

    # Incident field vector E_inc = [Ex_inc, Ey_inc]^T
    Ex_inc = float(EX_INC)
    Ey_inc = float(EY_INC)

    # Result from matrix multiplication: [Ex, Ey]^T = M_xy @ E_inc
    Ex_v = M11 * Ex_inc + M12 * Ey_inc
    Ey_v = M21 * Ex_inc + M22 * Ey_inc

    # Er and Ez
    Er_v = Ex_v * np.cos(phi) + Ey_v * np.sin(phi)
    Ez_v = (rho / zi) * Er_v

    I_Ex = Ex_v**2
    I_Ey = Ey_v**2
    I_Er = Er_v**2
    I_Ez = Ez_v**2
    I_total = I_Ex + I_Ey + I_Ez
    I_diff = I_total - I_scalar_2d

    maps = [
        ("Scalar |E|^2", I_scalar_2d),
        ("|Ex|^2", I_Ex),
        ("|Ey|^2", I_Ey),
        ("|Er|^2", I_Er),
        ("|Ez|^2", I_Ez),
        ("Vector total |E|^2", I_total),
        ("I_total - I_scalar", I_diff),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.ravel()

    for ax, (title, I) in zip(axes, maps):
        if title == "I_total - I_scalar":
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
            im = ax.imshow(
                I_plot,
                extent=[-q_view, q_view, -q_view, q_view],
                origin="lower",
                cmap="hot",
                aspect="equal",
                vmin=0.0,
            )
        ax.set_title(title)
        ax.set_xlabel("x [q]")
        ax.set_ylabel("y [q]")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cb_label)

    for ax in axes[len(maps):]:
        ax.axis("off")

    fig.suptitle(PARAM_STR)
    plt.tight_layout()
    plt.show()

    # Vector-field plots: incident (aperture plane) vs output (q plane)
    step_quiver = max(1, N_IMG // 24)
    step_stream = max(1, N_IMG // 96)

    # Incident field over pupil aperture in x-y [mm]
    xp = np.linspace(-R, R, N_IMG)
    xxp, yyp = np.meshgrid(xp, xp)
    rhop = np.sqrt(xxp**2 + yyp**2)
    pupil = rhop <= a
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

    # Output field over q-plane, with magnitude compression for readability
    xx_q = xx[::step_quiver, ::step_quiver]
    yy_q = yy[::step_quiver, ::step_quiver]
    ex_out_q = Ex_v[::step_quiver, ::step_quiver]
    ey_out_q = Ey_v[::step_quiver, ::step_quiver]
    mag_out = np.sqrt(ex_out_q**2 + ey_out_q**2)
    ref = np.percentile(mag_out, 95) + 1e-30
    gain = np.tanh(mag_out / ref) / (mag_out + 1e-30)
    ex_out_q = ex_out_q * gain
    ey_out_q = ey_out_q * gain
    xx_st = xx[::step_stream, ::step_stream]
    yy_st = yy[::step_stream, ::step_stream]
    ex_out_st = Ex_v[::step_stream, ::step_stream]
    ey_out_st = Ey_v[::step_stream, ::step_stream]
    mag_out_st = np.sqrt(ex_out_st**2 + ey_out_st**2)
    ref_st = np.percentile(mag_out_st, 95) + 1e-30
    ex_out_st = ex_out_st / ref_st
    ey_out_st = ey_out_st / ref_st
    tiny = np.sqrt(ex_out_st**2 + ey_out_st**2) < 1e-4
    ex_out_st = np.where(tiny, np.nan, ex_out_st)
    ey_out_st = np.where(tiny, np.nan, ey_out_st)

    figv, axv = plt.subplots(1, 2, figsize=(12, 5))
    if vector_plot_mode == "stream":
        speed_in = np.sqrt(np.nan_to_num(ex_in_st, nan=0.0)**2 + np.nan_to_num(ey_in_st, nan=0.0)**2)
        axv[0].streamplot(
            xxp_st[0, :],
            yyp_st[:, 0],
            ex_in_st,
            ey_in_st,
            color=speed_in,
            cmap="Blues",
            density=1.6,
            linewidth=1.1,
            arrowsize=0.9,
        )
    else:
        axv[0].quiver(xxp_q, yyp_q, ex_in_q, ey_in_q, color="tab:blue", pivot="mid", scale=50)
    axv[0].set_title("Incident field vectors (pupil plane)")
    axv[0].set_xlabel("x [mm]")
    axv[0].set_ylabel("y [mm]")
    axv[0].set_aspect("equal")
    axv[0].set_xlim(-R, R)
    axv[0].set_ylim(-R, R)

    if vector_plot_mode == "stream":
        speed_out = np.sqrt(np.nan_to_num(ex_out_st, nan=0.0)**2 + np.nan_to_num(ey_out_st, nan=0.0)**2)
        axv[1].streamplot(
            xx_st[0, :],
            yy_st[:, 0],
            ex_out_st,
            ey_out_st,
            color=speed_out,
            cmap="hot",
            density=5,
            linewidth=1.0,
            arrowsize=0.9,
        )
    else:
        axv[1].quiver(xx_q, yy_q, ex_out_q, ey_out_q, color="tab:red", pivot="mid", scale=20)
    axv[1].set_title("Output field vectors (Ex, Ey, regularized)")
    axv[1].set_xlabel("x [q]")
    axv[1].set_ylabel("y [q]")
    axv[1].set_aspect("equal")
    axv[1].set_xlim(-q_view, q_view)
    axv[1].set_ylim(-q_view, q_view)

    figv.suptitle("Vector fields: incident vs output")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(display_mode="gamma", vector_plot_mode=VECTOR_PLOT_MODE)
