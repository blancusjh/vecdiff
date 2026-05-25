import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from vecdiff.fresnel import FresnelOvoidParax
from vecdiff.hankel import HankelTransform
from vecdiff.view import radial_map


# ===================== PARAMETERS =====================
lam = 532e-6
n0 = 1.0
ni = 1.5
z0 = 5.0
zi = 10.0
varphi = 0.0

N = 4000
R = 1.0
a = 0.7
focus_refined = True
focus_power = 2.5
focus_window_frac = 0.18
N_img = 500
LOG_SCALE_INTENSITIES = True
# ======================================================


HT = HankelTransform.transform_array


def make_grid(max_val, n, refined, power):
    if refined:
        u = np.linspace(0.0, 1.0, n)
        return max_val * u**power
    return np.linspace(0.0, max_val, n)


def coeffs(model, rr):
    if model == "paraxial":
        fpx = FresnelOvoidParax(n0, ni, z0, zi)
        ts, tp = fpx.coeffs(rr)
        return np.asarray(ts, dtype=float), np.asarray(tp, dtype=float)

    raise ValueError(f"Unknown model: {model}")


def transmitted_terms(ts, tp, rr, qq, e_l, e_r, phase):
    h0_l = HT(0, rr, (tp + ts) * e_l, qq)
    h2_l = HT(2, rr, (tp - ts) * e_r, qq)
    h0_r = HT(0, rr, (tp + ts) * e_r, qq)
    h2_r = HT(2, rr, (tp - ts) * e_l, qq)

    c_h0_l = 2.0 * π * h0_l
    c_h2_l = -2.0 * π * np.conj(phase) * h2_l
    c_h0_r = 2.0 * π * h0_r
    c_h2_r = -2.0 * π * phase * h2_r

    e_lp = c_h0_l + c_h2_l
    e_rp = c_h0_r + c_h2_r

    return {
        "C_H0_L": c_h0_l,
        "C_H2_L": c_h2_l,
        "C_H0_R": c_h0_r,
        "C_H2_R": c_h2_r,
        "ELp": e_lp,
        "ERp": e_rp,
        "I_total": np.abs(e_lp) ** 2 + np.abs(e_rp) ** 2,
        "I_H0_only": np.abs(c_h0_l) ** 2 + np.abs(c_h0_r) ** 2,
    }


def show_map_row(images, titles, extent, suptitle, cmap="hot", log_flags=None):
    fig, ax = plt.subplots(1, len(images), figsize=(4.9 * len(images), 4.5), constrained_layout=True)
    if len(images) == 1:
        ax = [ax]
    if log_flags is None:
        log_flags = [False] * len(images)

    for k, img in enumerate(images):
        if log_flags[k]:
            positive = img[img > 0]
            vmin = float(np.min(positive)) if positive.size > 0 else 1e-15
            vmax = float(np.max(img)) if np.max(img) > 0 else vmin * 10.0
            im = ax[k].imshow(
                img,
                extent=extent,
                origin="lower",
                cmap=cmap,
                norm=LogNorm(vmin=vmin, vmax=vmax),
            )
        else:
            im = ax[k].imshow(img, extent=extent, origin="lower", cmap=cmap)
            im.set_clim(float(np.min(img)), float(np.max(img)))
        ax[k].set_title(titles[k])
        ax[k].set_xlabel(r"$x$")
        ax[k].set_ylabel(r"$y$")
        ax[k].set_aspect("equal")
        fig.colorbar(im, ax=ax[k], fraction=0.046, pad=0.04)

    fig.suptitle(suptitle)
    plt.show()


def main():
    q_max = ni / (lam * zi) * R
    r = make_grid(R, N, focus_refined, focus_power)
    q = make_grid(q_max, N, focus_refined, focus_power)
    q_view = focus_window_frac * q_max

    # Input circular components
    ap = (r <= a).astype(float)
    EL = np.ones_like(r, dtype=complex) * ap
    ER = np.ones_like(r, dtype=complex) * ap

    phase = exp(2.0j * varphi)

    ts_px, tp_px = coeffs("paraxial", r)
    out_px = transmitted_terms(ts_px, tp_px, r, q, EL, ER, phase)

    # Stage 1: input maps
    i_in = np.abs(EL) ** 2 + np.abs(ER) ** 2
    el_in_img, extent_r = radial_map(np.abs(EL), r, a, N_img)
    er_in_img, _ = radial_map(np.abs(ER), r, a, N_img)
    i_in_img, _ = radial_map(i_in, r, a, N_img)

    show_map_row(
        [el_in_img, er_in_img, i_in_img],
        [r"Entrada $|E_L|$", r"Entrada $|E_R|$", "Intensidad de entrada"],
        extent_r,
        "1) Campos circulares de entrada",
        log_flags=[False, False, LOG_SCALE_INTENSITIES],
    )

    # Stage 2: transmitted scalar maps
    el_px_img, extent_q = radial_map(np.abs(out_px["ELp"]), q, q_view, N_img)
    er_px_img, _ = radial_map(np.abs(out_px["ERp"]), q, q_view, N_img)

    show_map_row(
        [el_px_img, er_px_img],
        [
            r"$|E'_L|$",
            r"$|E'_R|$",
        ],
        extent_q,
        "2) Campos transmitidos",
    )

    # Stage 3: H0/H2/sum decomposition for each transmitted component
    ch0_l_img, _ = radial_map(np.abs(out_px["C_H0_L"]), q, q_view, N_img)
    ch2_l_img, _ = radial_map(np.abs(out_px["C_H2_L"]), q, q_view, N_img)
    el_sum_img, _ = radial_map(np.abs(out_px["ELp"]), q, q_view, N_img)

    show_map_row(
        [ch0_l_img, ch2_l_img, el_sum_img],
        [r"$|C_{H0,L}|$", r"$|C_{H2,L}|$", r"$|E'_L|=|C_{H0,L}+C_{H2,L}|$"],
        extent_q,
        "3) Descomposición de $E'_L$ en H0, H2 y suma",
    )

    ch0_r_img, _ = radial_map(np.abs(out_px["C_H0_R"]), q, q_view, N_img)
    ch2_r_img, _ = radial_map(np.abs(out_px["C_H2_R"]), q, q_view, N_img)
    er_sum_img, _ = radial_map(np.abs(out_px["ERp"]), q, q_view, N_img)

    show_map_row(
        [ch0_r_img, ch2_r_img, er_sum_img],
        [r"$|C_{H0,R}|$", r"$|C_{H2,R}|$", r"$|E'_R|=|C_{H0,R}+C_{H2,R}|$"],
        extent_q,
        "4) Descomposición de $E'_R$ en H0, H2 y suma",
    )

    # Stage 4: H2 impact on final intensity
    i_h0_only_px = out_px["I_H0_only"]
    i_total_px = out_px["I_total"]
    d_i_h2 = np.abs(i_total_px - i_h0_only_px)

    i_h0_img, _ = radial_map(i_h0_only_px, q, q_view, N_img)
    i_total_img, _ = radial_map(i_total_px, q, q_view, N_img)
    d_i_h2_img, _ = radial_map(d_i_h2, q, q_view, N_img)

    show_map_row(
        [i_h0_img, i_total_img, d_i_h2_img],
        [
            r"$I_{H0}$ (sin H2)",
            r"$I_{total}$ (H0 + H2)",
            r"$|I_{total}-I_{H0}|$ (efecto de H2)",
        ],
        extent_q,
        "5) Efecto de H2 en la intensidad final",
        log_flags=[LOG_SCALE_INTENSITIES, LOG_SCALE_INTENSITIES, LOG_SCALE_INTENSITIES],
    )

if __name__ == "__main__":
    main()
