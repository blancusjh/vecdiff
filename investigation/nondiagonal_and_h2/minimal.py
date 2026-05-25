import numpy as np
import matplotlib.pyplot as plt

from vecdiff.fresnel import FresnelOvoidParax
from vecdiff.hankel import HankelTransform
from viz_utils import (
    apply_style,
    radial_to_2d_abs,
    plot_intensity_maps,
    plot_vector_fields,
)


# ── optical system ────────────────────────────────────────────────────────────
n0 = 1.0
ni = 1.5
z0 = -10.0
zi = 6.0
R = 2.6
lam = 532e-6



PROFILE_N = 0
EX_INC = 1.0
EY_INC = 0.0


N_R = 1000
N_Q = 1000
N_IMG = 512


PARAM_STR = f"""n0={n0}, ni={ni},\nz0={z0}, zi={zi},\nR={R} mm\n(paraxial Fresnel)"""
VECTOR_PLOT_MODE = "quiver"
DARK_BACKGROUND = False

DISPLAY_MODE = "gamma"

HT = HankelTransform.transform_array


def propagate(r, q, tp, ts, E):
    h0 = HT(0, r, (tp + ts) * E, q)
    h2 = HT(2, r, (tp - ts) * E, q)
    return h0, h2


def main(display_mode=DISPLAY_MODE, vector_plot_mode=VECTOR_PLOT_MODE):
    apply_style(DARK_BACKGROUND)

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

    ts_scalar = np.ones_like(ts)
    tp_scalar = np.ones_like(tp)
    h0s, _ = propagate(r, q, tp_scalar, ts_scalar, Eamp)
    I_scalar_2d = radial_to_2d_abs(np.abs(2.0 * np.pi * h0s) ** 2, q, rho)

    h0x, h2x = propagate(r, q, tp, ts, Eamp)
    h0y, h2y = propagate(r, q, tp, ts, Eamp)

    h0x_2d = radial_to_2d_abs(h0x, q, rho)
    h2x_2d = radial_to_2d_abs(h2x, q, rho)
    h0y_2d = radial_to_2d_abs(h0y, q, rho)
    h2y_2d = radial_to_2d_abs(h2y, q, rho)

    c2 = np.cos(2.0 * phi)
    s2 = np.sin(2.0 * phi)

    M11 = h0x_2d - c2 * h2x_2d
    M12 = -s2 * h2y_2d
    M21 = -s2 * h2x_2d
    M22 = h0y_2d + c2 * h2y_2d

    Ex_inc = float(EX_INC)
    Ey_inc = float(EY_INC)
    pupil_on_q = rho <= a
    ex_inc_q = np.where(pupil_on_q, Ex_inc, 0.0)
    ey_inc_q = np.where(pupil_on_q, Ey_inc, 0.0)

    Ex_v = M11 * Ex_inc + M12 * Ey_inc
    Ey_v = M21 * Ex_inc + M22 * Ey_inc

    Er_v = Ex_v * np.cos(phi) + Ey_v * np.sin(phi)
    Ez_v = (rho / zi) * Er_v

    I_Ex = Ex_v**2
    I_Ey = Ey_v**2
    I_Er = Er_v**2
    I_Ez = Ez_v**2
    I_total = I_Ex + I_Ey + I_Ez
    maps = [
        ("Incident |Ex|^2", ex_inc_q**2),
        ("Incident |Ey|^2", np.maximum(ey_inc_q**2, 0.0)),
        ("Scalar |E|^2", I_scalar_2d),
        ("Vector total |E|^2", I_total),
        ("|Ex|^2", I_Ex),
        ("|Ey|^2", I_Ey),
        ("|Ez|^2", I_Ez),
    ]

    plot_intensity_maps(maps, q_view=q_view, display_mode=display_mode, title=PARAM_STR, ncols=3)
    plot_vector_fields(xx, yy, Ex_v, Ey_v, q_view, R, a, Ex_inc, Ey_inc, vector_plot_mode=vector_plot_mode)

    # Difference vector field (Output - Incident) with explicit visibility
    ex_inc_q = np.where(pupil_on_q, Ex_inc, 0.0)
    ey_inc_q = np.where(pupil_on_q, Ey_inc, 0.0)
    ex_diff = Ex_v - ex_inc_q
    ey_diff = Ey_v - ey_inc_q
    dmag = np.sqrt(ex_diff**2 + ey_diff**2)

    figd, axd = plt.subplots(1, 1, figsize=(6.4, 5.6))
    im = axd.imshow(
        dmag,
        extent=[-q_view, q_view, -q_view, q_view],
        origin="lower",
        cmap="magma",
        aspect="equal",
        vmin=0.0,
    )
    plt.colorbar(im, ax=axd, fraction=0.046, pad=0.04, label="|ΔE|")

    step = max(1, N_IMG // 28)
    xg = xx[::step, ::step]
    yg = yy[::step, ::step]
    exg = ex_diff[::step, ::step]
    eyg = ey_diff[::step, ::step]
    mg = np.sqrt(exg**2 + eyg**2)
    ref = np.percentile(mg, 95) + 1e-30
    exn = exg / ref
    eyn = eyg / ref
    mn = np.sqrt(exn**2 + eyn**2)
    mask = mn < 0.08
    exn = np.where(mask, np.nan, exn)
    eyn = np.where(mask, np.nan, eyn)

    axd.quiver(
        xg,
        yg,
        exn,
        eyn,
        color="cyan",
        pivot="mid",
        scale_units="xy",
        scale=6.0,
        width=0.0045,
    )
    axd.set_title("Difference field: Output - Incident")
    axd.set_xlabel("x [q]")
    axd.set_ylabel("y [q]")
    axd.set_xlim(-q_view, q_view)
    axd.set_ylim(-q_view, q_view)
    axd.set_aspect("equal")
    plt.tight_layout()
    out_png = "difference_field_output_minus_incident.png"
    figd.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"Saved: {out_png}")

    # Polarization-only difference map (orientation change, independent of amplitude)
    s1_out = Ex_v**2 - Ey_v**2
    s2_out = 2.0 * Ex_v * Ey_v
    psi_out = 0.5 * np.arctan2(s2_out, s1_out)

    s1_inc = Ex_inc**2 - Ey_inc**2
    s2_inc = 2.0 * Ex_inc * Ey_inc
    psi_inc = 0.5 * np.arctan2(s2_inc, s1_inc)

    dpsi = psi_out - psi_inc
    dpsi = (dpsi + 0.5 * np.pi) % np.pi - 0.5 * np.pi

    et_mag = np.sqrt(Ex_v**2 + Ey_v**2)
    tiny_pol = et_mag < (0.03 * np.max(et_mag))
    dpsi_deg = np.degrees(np.where(tiny_pol, np.nan, dpsi))

    figp, axp = plt.subplots(1, 1, figsize=(6.4, 5.6))
    imp = axp.imshow(
        dpsi_deg,
        extent=[-q_view, q_view, -q_view, q_view],
        origin="lower",
        cmap="RdBu",
        aspect="equal",
        vmin=-90.0,
        vmax=90.0,
    )
    plt.colorbar(imp, ax=axp, fraction=0.046, pad=0.04, label="Delta polarization angle [deg]")
    axp.set_title("Polarization orientation difference (Output vs Incident)")
    axp.set_xlabel("x [q]")
    axp.set_ylabel("y [q]")
    axp.set_xlim(-q_view, q_view)
    axp.set_ylim(-q_view, q_view)
    axp.set_aspect("equal")
    plt.tight_layout()
    out_png_pol = "polarization_orientation_difference.png"
    figp.savefig(out_png_pol, dpi=220, bbox_inches="tight")
    print(f"Saved: {out_png_pol}")

    plt.show()


if __name__ == "__main__":
    main()
