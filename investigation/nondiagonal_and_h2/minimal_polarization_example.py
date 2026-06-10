import numpy as np
import matplotlib.pyplot as plt

from vecdiff.fields import Field
from vecdiff.fresnel import FresnelOvoidParax
from vecdiff.grid import Grid
from vecdiff.hankel import HankelTransform
from vecdiff.polarization import polarization_map_from_field
from vecdiff.polarization_visualization import plot_polarization_quiver
from viz_utils import apply_style, compress_intensity, plot_intensity_maps, radial_to_2d_abs


# ── Optical System ────────────────────────────────────────────────────────────


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


PARAM_STR = f"""n0={n0}, ni={ni},
z0={z0}, zi={zi},
R={R} mm
(paraxial Fresnel)"""
DARK_BACKGROUND = False
DISPLAY_MODE = "gamma"
POL_STRIDE = 16

HT = HankelTransform.transform_array


def propagate(r, q, tp, ts, E):
    h0 = HT(0, r, (tp + ts) * E, q)
    h2 = HT(2, r, (tp - ts) * E, q)
    return h0, h2


def main(display_mode=DISPLAY_MODE):
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

    plot_intensity_maps(
        maps,
        q_view=q_view,
        display_mode=display_mode,
        title=PARAM_STR,
        ncols=3,
        show=False,
    )

    field = Field.from_cartesian(Ex_v, Ey_v, Grid.from_cartesian(xx, yy), symmetric=False)
    x_pol, y_pol, pol = polarization_map_from_field(field)

    I_plot, cb_label = compress_intensity(I_total, mode=display_mode)
    figp, axp = plt.subplots(1, 1, figsize=(6.4, 5.6))
    im = axp.imshow(
        I_plot,
        extent=[-q_view, q_view, -q_view, q_view],
        origin="lower",
        cmap="hot",
        aspect="equal",
        vmin=0.0,
        vmax=float(np.nanmax(I_plot)),
    )
    plt.colorbar(im, ax=axp, fraction=0.046, pad=0.04, label=cb_label)

    plot_polarization_quiver(
        x_pol,
        y_pol,
        pol,
        stride=POL_STRIDE,
        min_intensity_fraction=0.004,
        scale_by_intensity=False,
        ax=axp,
    )

    axp.set_title("Vector total |E|^2 + polarization quiver")
    axp.set_xlabel("x [q]")
    axp.set_ylabel("y [q]")
    axp.set_xlim(-q_view, q_view)
    axp.set_ylim(-q_view, q_view)
    axp.set_aspect("equal")
    plt.tight_layout()
    out_png = "minimal_polarization_quiver.png"
    figp.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
