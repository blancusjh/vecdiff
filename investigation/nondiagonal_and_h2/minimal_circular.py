import numpy as np
import matplotlib.pyplot as plt

from vecdiff.CartesianSurfaces import CartesianSurface
from vecdiff.fresnel import FresnelOvoidParax
from vecdiff.fields import FieldCircular
from vecdiff.propagation import HT, propagate_to_focal_plane_through_diopter

from viz_utils import apply_style, plot_intensity_maps, radial_to_2d_abs
from polarization_map import plot_polarization_map


# Optical system
n0 = 1.0
ni = 1.5
z0 = -10.0
zi = 6.0
R = 2.6
lam = 532e-6

N_R = 1000
N_Q = 1000
N_IMG = 512
PROFILE_N = 0

# Incident circular amplitudes (scalars)
EL_INC = 0.0
ER_INC = 1.0

DISPLAY_MODE = "gamma"
DARK_BACKGROUND = False
SAVE_PNG = True
POL_STEP = 12
POL_MAX_CURVES = 5000
POL_LIGHT_ON_HOT = "lightgray"
DIFF_POL_STEP = 24
DIFF_POL_MAX_CURVES = 12000
DIFF_POL_SIZE_SCALE = 0.45

PARAM_STR = f"""Circular case\nn0={n0}, ni={ni}, z0={z0}, zi={zi}, R={R} mm"""


def circular_to_cartesian(EL, ER):
    Ex = (EL + ER) / np.sqrt(2.0)
    Ey = -1j * (EL - ER) / np.sqrt(2.0)
    return Ex, Ey


def radial_to_2d_complex(profile_q, q, rho):
    re = np.interp(rho, q, np.real(profile_q), left=float(np.real(profile_q[0])), right=0.0)
    im = np.interp(rho, q, np.imag(profile_q), left=float(np.imag(profile_q[0])), right=0.0)
    return re + 1j * im


def main(display_mode=DISPLAY_MODE):
    apply_style(DARK_BACKGROUND)

    r = np.linspace(0.0, R, N_R)
    q_max = ni / (lam * zi) * R
    q = q_max * np.linspace(0.0, 1.0, N_Q) ** 2.0

    a = 0.95 * R
    amp = ((r / (a + 1e-30)) ** PROFILE_N) * (r <= a)

    # Build circular input field
    field_in = FieldCircular(EL_INC * amp, ER_INC * amp)

    # Use the library propagation API
    diopter = CartesianSurface(n0=n0, ni=ni, z0=z0, zi=zi)
    observation = {"r": r, "q": q, "varphi": 0.0}
    field_out = propagate_to_focal_plane_through_diopter(field_in, diopter, observation)
    EL = np.asarray(field_out.L)
    ER = np.asarray(field_out.R)

    # Fallback for current FresnelOvoid NaN path: keep example functional.
    if not (np.isfinite(EL).any() and np.isfinite(ER).any()):
        ts, tp = FresnelOvoidParax(n0, ni, z0, zi).coeffs(r)
        ts = np.asarray(ts, float)
        tp = np.asarray(tp, float)
        h0L = HT(0, r, (tp + ts) * (EL_INC * amp), q)
        h2L = HT(2, r, (tp - ts) * (ER_INC * amp), q)
        h0R = HT(0, r, (tp + ts) * (ER_INC * amp), q)
        h2R = HT(2, r, (tp - ts) * (EL_INC * amp), q)
        EL = 2.0 * np.pi * h0L - 2.0 * np.pi * h2L
        ER = -2.0 * np.pi * h2R + 2.0 * np.pi * h0R

    EL = np.nan_to_num(np.asarray(EL), nan=0.0, posinf=0.0, neginf=0.0)
    ER = np.nan_to_num(np.asarray(ER), nan=0.0, posinf=0.0, neginf=0.0)

    q_view = 0.0045 * q_max
    xs = np.linspace(-q_view, q_view, N_IMG)
    xx, yy = np.meshgrid(xs, xs)
    rho = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)

    # Radial-to-2D maps
    I_L = radial_to_2d_abs(EL, q, rho) ** 2
    I_R = radial_to_2d_abs(ER, q, rho) ** 2

    Ex_r, Ey_r = circular_to_cartesian(EL, ER)
    Ex2d_c = radial_to_2d_complex(Ex_r, q, rho)
    Ey2d_c = radial_to_2d_complex(Ey_r, q, rho)
    Ex2d = np.abs(Ex2d_c)
    Ey2d = np.abs(Ey2d_c)

    # Incident field on x-y pupil grid for polarization-map comparison
    xp = np.linspace(-R, R, N_IMG)
    xxp, yyp = np.meshgrid(xp, xp)
    rhop = np.sqrt(xxp**2 + yyp**2)
    pupil = rhop <= a

    ex_inc_c = np.full_like(xxp, (EL_INC + ER_INC) / np.sqrt(2.0), dtype=complex)
    ey_inc_c = np.full_like(xxp, -1j * (EL_INC - ER_INC) / np.sqrt(2.0), dtype=complex)
    ex_inc_c = np.where(pupil, ex_inc_c, 0.0 + 0.0j)
    ey_inc_c = np.where(pupil, ey_inc_c, 0.0 + 0.0j)

    # Difference of normalized fields: E_out_norm - E_inc_norm
    norm_out = np.sqrt(np.abs(Ex2d_c) ** 2 + np.abs(Ey2d_c) ** 2) + 1e-30
    ex_out_n = Ex2d_c / norm_out
    ey_out_n = Ey2d_c / norm_out

    norm_inc = np.sqrt(np.abs(ex_inc_c) ** 2 + np.abs(ey_inc_c) ** 2) + 1e-30
    ex_inc_n = ex_inc_c / norm_inc
    ey_inc_n = ey_inc_c / norm_inc

    ex_diff_c = ex_out_n - ex_inc_n
    ey_diff_c = ey_out_n - ey_inc_n

    Er2d = Ex2d * np.cos(phi) + Ey2d * np.sin(phi)
    Ez2d = (rho / zi) * Er2d

    I_Ex = Ex2d ** 2
    I_Ey = Ey2d ** 2
    I_Ez = Ez2d ** 2
    I_total = I_Ex + I_Ey + I_Ez

    I_inc_ref = np.where(rho <= a, np.abs((EL_INC + ER_INC) / np.sqrt(2.0)) ** 2 + np.abs(-1j * (EL_INC - ER_INC) / np.sqrt(2.0)) ** 2, 0.0)
    I_diff_out_minus_inc = I_total - I_inc_ref

    maps = [
        ("|EL|^2", I_L),
        ("|ER|^2", I_R),
        ("|Ex|^2", I_Ex),
        ("|Ey|^2", I_Ey),
        ("|Ez|^2", I_Ez),
        ("Circular total |E|^2", I_total),
    ]

    intensity_png = "circular_intensity_maps.png" if SAVE_PNG else None
    plot_intensity_maps(
        maps,
        q_view=q_view,
        display_mode=display_mode,
        title=PARAM_STR,
        ncols=3,
        save_path=intensity_png,
        show=True,
    )

    figv, axv = plt.subplots(1, 2, figsize=(12, 5))
    i_inc_pupil = np.abs(ex_inc_c) ** 2 + np.abs(ey_inc_c) ** 2
    im0 = axv[0].imshow(
        i_inc_pupil,
        extent=[-R, R, -R, R],
        origin="lower",
        cmap="hot",
        aspect="equal",
    )
    plt.colorbar(im0, ax=axv[0], fraction=0.046, pad=0.04, label="Incident |E|^2")
    plot_polarization_map(
        xxp, yyp, ex_inc_c, ey_inc_c,
        title="Incident field polarization over scalar intensity",
        min_intensity_frac=0.004,
        step=POL_STEP,
        max_curves=POL_MAX_CURVES,
        save_path=None,
        curve_color=POL_LIGHT_ON_HOT,
        arrow_color=POL_LIGHT_ON_HOT,
        ax=axv[0],
        show=False,
    )

    im1 = axv[1].imshow(
        I_total,
        extent=[float(np.min(xx)), float(np.max(xx)), float(np.min(yy)), float(np.max(yy))],
        origin="lower",
        cmap="hot",
        aspect="equal",
    )
    plt.colorbar(im1, ax=axv[1], fraction=0.046, pad=0.04, label="Output |E|^2")
    plot_polarization_map(
        xx, yy, Ex2d_c, Ey2d_c,
        title="Output field polarization over scalar intensity",
        min_intensity_frac=0.004,
        step=POL_STEP,
        max_curves=POL_MAX_CURVES,
        save_path=None,
        curve_color=POL_LIGHT_ON_HOT,
        arrow_color=POL_LIGHT_ON_HOT,
        ax=axv[1],
        show=False,
    )
    figv.suptitle("Polarization overlays: incident vs output")
    plt.tight_layout()
    if SAVE_PNG:
        figv.savefig("circular_vector_fields.png", dpi=220, bbox_inches="tight")
        print("Saved: circular_vector_fields.png")

    # Third figure: focused comparison on difference (intensity + polarization difference)
    overlay_png = "circular_overlay_diffpol_on_deltaI_hot.png" if SAVE_PNG else None
    fig_ov, ax_ov = plt.subplots(1, 2, figsize=(12.5, 5.2))
    dI_pos = np.maximum(I_diff_out_minus_inc, 0.0)
    ref_dI = np.percentile(dI_pos, 99.5) + 1e-30
    dI_vis = np.arcsinh(14.0 * dI_pos / ref_dI)
    extent = [float(np.min(xx)), float(np.max(xx)), float(np.min(yy)), float(np.max(yy))]
    im_ov0 = ax_ov[0].imshow(
        dI_vis,
        extent=extent,
        origin="lower",
        cmap="hot",
        aspect="equal",
    )
    plt.colorbar(im_ov0, ax=ax_ov[0], fraction=0.046, pad=0.04, label="asinh(14 * (I_out - I_in)_+ / p99.5)")
    ax_ov[0].set_title("Intensity difference only")
    ax_ov[0].set_xlabel("x [q]")
    ax_ov[0].set_ylabel("y [q]")

    im_ov1 = ax_ov[1].imshow(
        dI_vis,
        extent=extent,
        origin="lower",
        cmap="hot",
        aspect="equal",
    )
    plt.colorbar(im_ov1, ax=ax_ov[1], fraction=0.046, pad=0.04, label="asinh(14 * (I_out - I_in)_+ / p99.5)")
    plot_polarization_map(
        xx, yy, ex_diff_c, ey_diff_c,
        title="Delta I + polarization difference",
        min_intensity_frac=0.002,
        step=DIFF_POL_STEP,
        max_curves=DIFF_POL_MAX_CURVES,
        save_path=None,
        curve_color=POL_LIGHT_ON_HOT,
        arrow_color=POL_LIGHT_ON_HOT,
        scale_by_intensity=False,
        size_scale=DIFF_POL_SIZE_SCALE,
        ax=ax_ov[1],
        show=False,
    )
    fig_ov.suptitle("Difference between input and output")
    plt.tight_layout()
    if SAVE_PNG:
        fig_ov.savefig(overlay_png, dpi=220, bbox_inches="tight")
        print(f"Saved: {overlay_png}")
    plt.show()


if __name__ == "__main__":
    main()
