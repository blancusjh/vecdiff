"""Study 4 - Resolution inversion: resolved by the scalar model, not by the vectorial one.

Goal: find a two-feature configuration whose images are *distinguishable in
the scalar model* (t_minus = 0) but *indistinguishable in the vectorial
model*, both for two lines and for the canonical case of two circular
features (discs), and map the separation window where this inversion holds.

Mechanism.  Along the separation axis the co-polar image field is A -+ B
(sign given by cos(2 phi) = +-1): the mixing kernel B interferes at first
order with the scalar field A.  For separation *perpendicular* to the
incident polarization the interference fills the valley between the two
feature images; parallel to it, it deepens the valley.  The effect scales
with t_minus/t_plus at the pupil edge on the second diopter, so a fast exit
diopter and a wide pupil (pushing the pupil edge toward D2's grazing branch)
strengthen it.

Two systems are studied, both air->glass->air conjugate-oval cascades:

* ``alto_indice``: n_i = 2.4 (a diamond/rutile-like high-index contrast) with a
  wide r_a = 10 mm pupil -- the strongest, clearest inversion.
* ``vidrio``: n_i = 1.5 (ordinary optical glass), a physically common contrast.
  Its lower t_minus/t_plus ceiling ((n_i-n0)/(n_i+n0) = 0.2 vs 0.41) makes the
  effect weaker, so the pupil radius is set to the largest value whose
  Fourier-plane image still lands inside D2's physical (pre-grazing) branch.

Distinguishability criterion: valley contrast C >= C_TH.  For each feature type
and orientation the script scans the separation, locates ``sep*(scalar)`` and
``sep*(vectorial)``, and reports the inversion window between them.  A showcase
separation inside the window is rendered as images + profiles + polarization
maps.  Outputs (suffixed by system tag) go to
``output/study_4_resolution_inversion/``.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import common
import imaging_common as ic
from imaging_common import LAM

from vecdiff import FieldCartesian, Grid
from vecdiff.polarization import polarization_map_from_field
from vecdiff.polarization_visualization import plot_field_polarization

# Distinguishability threshold: the Rayleigh criterion is a ~26 % dip, i.e. a
# valley contrast C = (peak-valley)/(peak+valley) ~ 0.15.  "Resolved" means
# C >= C_TH; the inversion is C_scalar >= C_TH > C_vectorial.
C_TH = 0.15
DISC_DIAM = 0.30e-3  # mm, canonical small circular features
LINE_LEN_F = 1.5     # line length in units of sep
# Edge-weighted pupil (r/a)^EDGE_P concentrates the energy near the grazing rim
# where t_minus/t_plus peaks, amplifying the polarization mixing so the
# inversion is visually obvious (scalar clearly split, vectorial fused).
EDGE_P = 4.0
S_FRACS = np.linspace(0.50, 0.80, 13)  # separations in units of d_Airy

out = common.output_dir(__file__)


def physical_pupil_radius(d1, d2, frac=0.97):
    """Largest r_a whose Fourier-plane pupil edge stays inside D2's oval.

    The pupil sits on D2's front surface at radius ``LAM*zi1/(2 pi ni1)*Kc``;
    it must not exceed D2's grazing radius, else the marginal rays fall on the
    non-physical back branch of the conjugate oval.
    """
    scale = LAM * d1["zi"] / (2.0 * np.pi * d1["ni"])
    r_graze = common.grazing_radius(**d2)
    f = lambda ra: scale * ic.Kc_of(ra, d1) - frac * r_graze
    return float(brentq(f, 0.3, 20.0))


# ------------------------------------------------------------------ #
#  Systems: keep the strong high-index example and add a realistic one #
# ------------------------------------------------------------------ #

def make_systems():
    high_d1 = dict(ic.D1)                                   # n_i = 2.4
    high_d2 = dict(n0=ic.D1["ni"], ni=1.0, z0=-ic.D1["zi"], zi=0.6)
    glass_d1 = dict(n0=1.0, ni=1.5, z0=-6.0, zi=3.6)        # ordinary glass
    glass_d2 = dict(n0=1.5, ni=1.0, z0=-glass_d1["zi"], zi=0.6)
    return [
        {"tag": "alto_indice", "label": r"alto índice ($n_i$=2.4, tipo diamante)",
         "d1": high_d1, "d2": high_d2, "r_a": 10.0, "edge_p": EDGE_P},
        {"tag": "vidrio", "label": r"vidrio ($n_i$=1.5)",
         "d1": glass_d1, "d2": glass_d2,
         "r_a": round(physical_pupil_radius(glass_d1, glass_d2), 2), "edge_p": EDGE_P},
    ]


def make_mask(feature, theta, sep):
    if feature == "discs":
        return ic.two_disc_mask(theta, sep, DISC_DIAM)
    return ic.two_line_mask(theta, sep, 0.35 * sep, LINE_LEN_F * sep)


def crossing(s, c, level):
    """First upward crossing of ``level`` by linear interpolation."""
    c = np.asarray(c)
    for i in range(len(c) - 1):
        if c[i] < level <= c[i + 1]:
            return float(s[i] + (s[i + 1] - s[i]) * (level - c[i]) / (c[i + 1] - c[i]))
    return np.nan


ORIENTATIONS = {"sep_x": 0.0, "sep_y": 0.5 * np.pi}


def run_system(system):
    d1, d2, r_a, edge_p = system["d1"], system["d2"], system["r_a"], system["edge_p"]
    tag, label = system["tag"], system["label"]
    mag = ic.mag_of(d2, d1)
    d_airy = 2.0 * ic.airy_radius(r_a, d1)
    caption = ic.system_caption(r_a, d1=d1, d2=d2) + rf"  ·  pupila $(r/a)^{{{edge_p:g}}}$"
    print(f"\n=== sistema '{tag}': {label} ===")
    print(f"r_a={r_a} mm, d_Airy={d_airy*1e3:.3f} um, M={mag:.2f}, edge_p={edge_p}")

    # -- separation scan ------------------------------------------------
    rows, curves = [], {}
    for feature in ("discs", "lines"):
        for name, theta in ORIENTATIONS.items():
            c_sca, c_vec = [], []
            for s_frac in S_FRACS:
                sep = float(s_frac * d_airy)
                mask = make_mask(feature, theta, sep)
                E_s = ic.image(mask, r_a, False, edge_p=edge_p, d1=d1, d2=d2)
                E_v = ic.image(mask, r_a, True, edge_p=edge_p, d1=d1, d2=d2)
                Cs, _, _ = ic.valley_contrast(E_s, theta, sep, mag=mag)
                Cv, _, _ = ic.valley_contrast(E_v, theta, sep, mag=mag)
                c_sca.append(Cs)
                c_vec.append(Cv)
                rows.append({"system": tag, "feature": feature, "orientation": name,
                             "sep_over_dAiry": s_frac, "sep_um": sep * 1e3,
                             "C_scalar": Cs, "C_vectorial": Cv})
            curves[(feature, name)] = (np.array(c_sca), np.array(c_vec))
            s_sca = crossing(S_FRACS, c_sca, C_TH)
            s_vec = crossing(S_FRACS, c_vec, C_TH)
            window = s_vec - s_sca if np.isfinite(s_sca) and np.isfinite(s_vec) else np.nan
            print(f"{feature:5s} {name}:  sep*(esc)={s_sca:.3f}, sep*(vec)={s_vec:.3f}, "
                  f"ventana={window:+.3f} dAiry")

    with (out / f"inversion_scan_{tag}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # -- Figure 1: contrast curves and inversion window -----------------
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), constrained_layout=True)
    for axi, feature in zip(axes, ("discs", "lines")):
        for nm, color in (("sep_x", "#1f77b4"), ("sep_y", "#d62728")):
            c_sca, c_vec = curves[(feature, nm)]
            axi.plot(S_FRACS, c_sca, "--", color=color, marker="o", ms=4, label=f"escalar, {nm}")
            axi.plot(S_FRACS, c_vec, "-", color=color, marker="s", ms=4, label=f"vectorial, {nm}")
        c_sca, c_vec = curves[(feature, "sep_y")]
        s1 = crossing(S_FRACS, c_sca, C_TH)
        s2 = crossing(S_FRACS, c_vec, C_TH)
        if np.isfinite(s1) and np.isfinite(s2) and s2 > s1:
            axi.axvspan(s1, s2, color="gold", alpha=0.35, label=r"ventana $C_{esc}\geq C_{th}>C_{vec}$")
        axi.axhline(C_TH, color="k", lw=0.8, ls=":")
        axi.axhline(0.0, color="k", lw=0.8)
        axi.set_xlabel(r"separación [$d_{Airy}$]")
        axi.set_ylabel("contraste del valle C")
        title = ("discos (dos puntos canónicos)" if feature == "discs"
                 else f"dos líneas (long. {LINE_LEN_F}·sep)")
        axi.set_title(f"{title}, umbral Rayleigh C = {C_TH}")
        axi.grid(True, alpha=0.3)
        axi.legend(fontsize=8)
    fig.suptitle(rf"Contraste del valle vs separación — {label}  ·  "
                 r"pol. incidente lineal $\hat{x}$" + "\n" + caption, fontsize=10)
    fig.savefig(out / f"fig1_inversion_window_{tag}.png", dpi=220)
    plt.close(fig)

    # -- showcase: the separation of clearest inversion (max C_sca - C_vec
    #    subject to scalar resolved and vectorial not) ------------------
    def pick_showcase(feature):
        c_sca, c_vec = curves[(feature, "sep_y")]
        ok = (c_vec < C_TH) & (c_sca >= C_TH)
        if not ok.any():
            return None
        return float(S_FRACS[np.argmax(np.where(ok, c_sca - c_vec, -np.inf))])

    for feature in ("discs", "lines"):
        s_frac = pick_showcase(feature)
        if s_frac is None:
            print(f"{feature}: sin ventana de inversión en el rango escaneado")
            continue
        _render_showcase(system, feature, s_frac, mag, d_airy, caption)

    return {"tag": tag, "label": label, "r_a": r_a, "mag": mag, "d_airy": d_airy,
            "curves": curves}


def _render_showcase(system, feature, s_frac, mag, d_airy, caption):
    d1, d2, r_a, tag, edge_p = (system["d1"], system["d2"], system["r_a"],
                                system["tag"], system["edge_p"])
    sep = s_frac * d_airy
    mask = make_mask(feature, 0.5 * np.pi, sep)
    E_s = ic.image(mask, r_a, False, edge_p=edge_p, d1=d1, d2=d2)
    E_v = ic.image(mask, r_a, True, edge_p=edge_p, d1=d1, d2=d2)
    Cs, u_s, p_s = ic.valley_contrast(E_s, 0.5 * np.pi, sep, mag=mag)
    Cv, u_v, p_v = ic.valley_contrast(E_v, 0.5 * np.pi, sep, mag=mag)
    label = "discos" if feature == "discs" else "líneas"

    I_s = np.abs(E_s.x) ** 2 + np.abs(E_s.y) ** 2
    I_v = np.abs(E_v.x) ** 2 + np.abs(E_v.y) ** 2
    vmax = max(I_s.max(), I_v.max())
    xr, yr = E_s.grid.X[0, :] / LAM, E_s.grid.Y[:, 0] / LAM
    extent = [xr[0], xr[-1], yr[0], yr[-1]]
    lim = 1.6 * mag * sep / LAM

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.3), constrained_layout=True)
    half_L = 0.5 * ic.L
    axes[0].imshow(mask, extent=[v / LAM for v in (-half_L, half_L, -half_L, half_L)],
                   origin="lower", cmap="gray")
    axes[0].set_title("Máscara (objeto)")
    axes[0].set_xlim(-lim / mag, lim / mag)
    axes[0].set_ylim(-lim / mag, lim / mag)
    axes[1].imshow((I_s / vmax) ** 0.8, extent=extent, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Imagen escalar ($t_-=0$): C = {Cs:.3f}")
    axes[2].imshow((I_v / vmax) ** 0.8, extent=extent, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Imagen vectorial: C = {Cv:.3f}")
    for axi in axes[1:3]:
        axi.set_xlim(-lim, lim)
        axi.set_ylim(-lim, lim)
    for axi in axes[:3]:
        axi.set_xlabel(r"$x/\lambda$")
        axi.set_ylabel(r"$y/\lambda$")

    ref = p_s.max()
    axes[3].plot(u_s / (mag * sep), p_s / ref, "k--", label=f"escalar (C={Cs:.3f})")
    axes[3].plot(u_v / (mag * sep), p_v / ref, "-", color="#d62728", label=f"vectorial (C={Cv:.3f})")
    for s in (-0.5, 0.5):
        axes[3].axvline(s, color="gray", lw=0.7, alpha=0.6)
    axes[3].set_xlabel(r"$u / (M \cdot sep)$")
    axes[3].set_ylabel("I (norm.)")
    axes[3].set_title(r"Perfil a lo largo de la separación ($\parallel \hat{y}$)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    fig.suptitle(rf"{label} — {system['label']}: sep = {s_frac:.2f} $d_{{Airy}}$ = {sep*1e3:.3f} µm  ·  "
                 r"separación $\parallel \hat{y}$" + "\n" + caption, fontsize=10)
    fig.savefig(out / f"fig2_showcase_{feature}_{tag}.png", dpi=200)
    plt.close(fig)
    print(f"{feature}: showcase sep={s_frac:.2f} dAiry (C_sca={Cs:.3f}, C_vec={Cv:.3f})")

    _render_polarization(system, feature, s_frac, sep, lim, vmax, E_s, E_v, caption)


def _render_polarization(system, feature, s_frac, sep, lim, vmax, E_s, E_v, caption):
    tag = system["tag"]
    label = "discos" if feature == "discs" else "líneas"

    E_s_lam, E_v_lam = (
        FieldCartesian(x=E.x / np.sqrt(vmax), y=E.y / np.sqrt(vmax),
                       grid=Grid.from_cartesian(E.grid.X / LAM, E.grid.Y / LAM), symmetric=False)
        for E in (E_s, E_v)
    )

    pol_maps = {}
    for name, E_lam in (("escalar", E_s_lam), ("vectorial", E_v_lam)):
        xx, yy, pol = polarization_map_from_field(E_lam, half_size=1.05 * lim, n_img=340)
        pol_maps[name] = (xx, yy, pol)

    _, _, pol_v = pol_maps["vectorial"]
    bright_v = pol_v.s0 > 0.02 * pol_v.s0.max()
    iy_lim = float(np.abs(pol_v.ey).max() ** 2)
    psi_lim = max(5.0, float(np.nanpercentile(
        np.abs(np.degrees(np.where(bright_v, pol_v.psi, np.nan))), 99)))
    chi_lim = max(1.0, 1.2 * float(np.nanmax(
        np.abs(np.degrees(np.where(bright_v, pol_v.chi, np.nan))))))

    fig, axes = plt.subplots(2, 4, figsize=(17.5, 7.8), constrained_layout=True)
    for row_i, name in enumerate(("escalar", "vectorial")):
        xx, yy, pol = pol_maps[name]
        ext = [xx.min(), xx.max(), yy.min(), yy.max()]
        bright = pol.s0 > 0.02 * pol.s0.max()
        f_cross_img = float((np.abs(pol.ey) ** 2).sum() / (pol.s0.sum() + 1e-300))
        row_tag = "escalar ($t_-=0$)" if name == "escalar" else name

        im = axes[row_i, 0].imshow((np.abs(pol.ex) ** 2) ** 0.8, extent=ext, origin="lower",
                                   cmap="gray", vmin=0, vmax=1)
        axes[row_i, 0].set_title(rf"{row_tag}: $|E_x|^2$ (copolar)")
        fig.colorbar(im, ax=axes[row_i, 0], fraction=0.046, pad=0.04)
        im = axes[row_i, 1].imshow(np.abs(pol.ey) ** 2, extent=ext, origin="lower",
                                   cmap="magma", vmin=0.0, vmax=iy_lim)
        axes[row_i, 1].set_title(rf"$|E_y|^2$ (cruzada), $f_{{cross}}$ = {f_cross_img:.1e}")
        fig.colorbar(im, ax=axes[row_i, 1], fraction=0.046, pad=0.04)
        im = axes[row_i, 2].imshow(np.where(bright, np.degrees(pol.psi), np.nan), extent=ext,
                                   origin="lower", cmap="RdBu_r", vmin=-psi_lim, vmax=psi_lim)
        axes[row_i, 2].set_title(r"orientación $\psi$ [$^\circ$]")
        fig.colorbar(im, ax=axes[row_i, 2], fraction=0.046, pad=0.04)
        im = axes[row_i, 3].imshow(np.where(bright, np.degrees(pol.chi), np.nan), extent=ext,
                                   origin="lower", cmap="RdBu_r", vmin=-chi_lim, vmax=chi_lim)
        axes[row_i, 3].set_title(r"elipticidad $\chi$ [$^\circ$]")
        fig.colorbar(im, ax=axes[row_i, 3], fraction=0.046, pad=0.04)
        for axi in axes[row_i, :]:
            axi.set_xlim(-lim, lim)
            axi.set_ylim(-lim, lim)
            axi.set_xlabel(r"$x/\lambda$")
            axi.set_ylabel(r"$y/\lambda$")
    fig.suptitle(rf"Mapas de polarización (Stokes)  ·  {label} — {system['label']}, "
                 rf"sep = {s_frac:.2f} $d_{{Airy}}$" + "\n" + caption, fontsize=10)
    fig.savefig(out / f"fig4_stokes_{feature}_{tag}.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.4), constrained_layout=True)
    for axi, (name, E_lam) in zip(axes, (("escalar ($t_-=0$)", E_s_lam), ("vectorial", E_v_lam))):
        plot_field_polarization(E_lam, half_size=1.05 * lim, n_img=300, sampling="cartesian",
                                target_ellipses=17, min_intensity_fraction=0.01, ax=axi)
        axi.set_title(name)
        axi.set_xlim(-lim, lim)
        axi.set_ylim(-lim, lim)
        axi.set_xlabel(r"$x/\lambda$")
        axi.set_ylabel(r"$y/\lambda$")
    fig.suptitle(rf"Mapa de polarización  ·  {label} — {system['label']}, "
                 rf"sep = {s_frac:.2f} $d_{{Airy}}$" + "\n" + caption, fontsize=10)
    fig.savefig(out / f"fig6_polmap_{feature}_{tag}.png", dpi=200)
    plt.close(fig)
    print(f"{feature}: polarización -> fig4_stokes / fig6_polmap ({tag})")


for _system in make_systems():
    run_system(_system)

print(f"\nDone. Outputs in {out}")
