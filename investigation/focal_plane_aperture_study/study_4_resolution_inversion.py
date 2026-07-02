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
with t_minus/t_plus at the pupil edge on the second diopter, so the study
uses a fast exit diopter (D2 with zi = 0.6) and a wide pupil (r_a = 10 mm),
which places the Fourier-plane pupil edge (~1.3 mm) close to D2's grazing
branch while keeping the whole pupil physical (grazing radius 1.46 mm).

Distinguishability criterion: valley contrast C >= C_TH (a 5 % dip).  For
each feature type and orientation the script scans the separation, locates
the thresholds ``sep*(scalar)`` and ``sep*(vectorial)``, and reports the
inversion window between them.  A showcase separation inside the window is
rendered as images + profiles.  Outputs go to
``output/study_4_resolution_inversion/``.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

import common
import imaging_common as ic
from imaging_common import LAM

from vecdiff import FieldCartesian, Grid
from vecdiff.polarization import polarization_map_from_field
from vecdiff.polarization_visualization import plot_field_polarization

R_A = 10.0
D2_FAST = dict(n0=ic.D1["ni"], ni=1.0, z0=-ic.D1["zi"], zi=0.6)
MAG = ic.mag_of(D2_FAST)

# Minimum valley depth considered "distinguishable".  Long lines average the
# cos(2 phi) mixing term along their length, so the study uses short lines
# (length 1.5*sep) and, because their coherent contrast transition is much
# steeper than for discs, a Rayleigh-like 15 % criterion instead of 5 %.
C_TH = {"discs": 0.05, "lines": 0.15}
DISC_DIAM = 0.30e-3  # mm, canonical small circular features
LINE_LEN_F = 1.5     # line length in units of sep
S_FRACS = np.linspace(0.55, 0.80, 11)  # separations in units of d_Airy

D_AIRY = 2.0 * ic.airy_radius(R_A)

out = common.output_dir(__file__)
print(f"r_a = {R_A} mm, d_Airy = {D_AIRY*1e3:.3f} um, M = {MAG:.2f}, umbrales = {C_TH}")


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


# ------------------------------------------------------------------ #
#  Separation scan                                                     #
# ------------------------------------------------------------------ #

orientations = {"sep_x": 0.0, "sep_y": 0.5 * np.pi}
rows, curves = [], {}

for feature in ("discs", "lines"):
    for name, theta in orientations.items():
        c_sca, c_vec = [], []
        for s_frac in S_FRACS:
            sep = float(s_frac * D_AIRY)
            mask = make_mask(feature, theta, sep)
            E_s = ic.image(mask, R_A, False, d2=D2_FAST)
            E_v = ic.image(mask, R_A, True, d2=D2_FAST)
            Cs, _, _ = ic.valley_contrast(E_s, theta, sep, mag=MAG)
            Cv, _, _ = ic.valley_contrast(E_v, theta, sep, mag=MAG)
            c_sca.append(Cs)
            c_vec.append(Cv)
            rows.append({"feature": feature, "orientation": name,
                         "sep_over_dAiry": s_frac, "sep_um": sep * 1e3,
                         "C_scalar": Cs, "C_vectorial": Cv})
        curves[(feature, name)] = (np.array(c_sca), np.array(c_vec))
        s_sca = crossing(S_FRACS, c_sca, C_TH[feature])
        s_vec = crossing(S_FRACS, c_vec, C_TH[feature])
        window = s_vec - s_sca if np.isfinite(s_sca) and np.isfinite(s_vec) else np.nan
        rows_txt = (f"{feature:5s} {name}:  sep*(escalar) = {s_sca:.3f} dAiry, "
                    f"sep*(vectorial) = {s_vec:.3f} dAiry, ventana = {window:+.3f} dAiry")
        print(rows_txt)

with (out / "inversion_scan.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# ------------------------------------------------------------------ #
#  Figure 1: contrast curves and inversion window                      #
# ------------------------------------------------------------------ #

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), constrained_layout=True)
for axi, feature in zip(axes, ("discs", "lines")):
    for name, color in (("sep_x", "#1f77b4"), ("sep_y", "#d62728")):
        c_sca, c_vec = curves[(feature, name)]
        axi.plot(S_FRACS, c_sca, "--", color=color, marker="o", ms=4,
                 label=f"escalar, {name}")
        axi.plot(S_FRACS, c_vec, "-", color=color, marker="s", ms=4,
                 label=f"vectorial, {name}")
    # inversion window for the perpendicular orientation
    c_sca, c_vec = curves[(feature, "sep_y")]
    s1 = crossing(S_FRACS, c_sca, C_TH[feature])
    s2 = crossing(S_FRACS, c_vec, C_TH[feature])
    if np.isfinite(s1) and np.isfinite(s2) and s2 > s1:
        axi.axvspan(s1, s2, color="gold", alpha=0.35,
                    label=r"ventana $C_{esc} \geq C_{th} > C_{vec}$")
    axi.axhline(C_TH[feature], color="k", lw=0.8, ls=":")
    axi.axhline(0.0, color="k", lw=0.8)
    axi.set_xlabel(r"separación [$d_{Airy}$]")
    axi.set_ylabel("contraste del valle C")
    title = ("discos (dos puntos canónicos)" if feature == "discs"
             else f"dos líneas (long. {LINE_LEN_F}·sep)")
    axi.set_title(f"{title}, umbral C = {C_TH[feature]}")
    axi.grid(True, alpha=0.3)
    axi.legend(fontsize=8)
fig.suptitle(
    r"Contraste del valle vs separación  ·  polarización incidente lineal $\hat{x}$" + "\n"
    + ic.system_caption(R_A, d2=D2_FAST), fontsize=10)
fig.savefig(out / "fig1_inversion_window.png", dpi=220)
plt.close(fig)

# ------------------------------------------------------------------ #
#  Figures 2-3: showcase inside the window (sep perpendicular to pol.) #
# ------------------------------------------------------------------ #

def pick_showcase(feature):
    """Scanned separation with the clearest inversion: C_vec < C_TH < C_sca."""
    c_sca, c_vec = curves[(feature, "sep_y")]
    ok = (c_vec < C_TH[feature]) & (c_sca > C_TH[feature])
    if not ok.any():
        return None
    idx = np.argmax(np.where(ok, c_sca - c_vec, -np.inf))
    return float(S_FRACS[idx])


for feature, figname in (("discs", "fig2_showcase_discs.png"),
                         ("lines", "fig3_showcase_lines.png")):
    s_frac = pick_showcase(feature)
    if s_frac is None:
        print(f"{feature}: sin separación de inversión en el rango escaneado")
        continue
    sep = s_frac * D_AIRY
    mask = make_mask(feature, 0.5 * np.pi, sep)
    E_s = ic.image(mask, R_A, False, d2=D2_FAST)
    E_v = ic.image(mask, R_A, True, d2=D2_FAST)
    Cs, u_s, p_s = ic.valley_contrast(E_s, 0.5 * np.pi, sep, mag=MAG)
    Cv, u_v, p_v = ic.valley_contrast(E_v, 0.5 * np.pi, sep, mag=MAG)

    I_s = np.abs(E_s.x) ** 2 + np.abs(E_s.y) ** 2
    I_v = np.abs(E_v.x) ** 2 + np.abs(E_v.y) ** 2
    vmax = max(I_s.max(), I_v.max())
    xr = E_s.grid.X[0, :] / LAM
    yr = E_s.grid.Y[:, 0] / LAM
    extent = [xr[0], xr[-1], yr[0], yr[-1]]
    lim = 1.6 * MAG * sep / LAM

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.3), constrained_layout=True)
    half_L = 0.5 * ic.L
    im0 = axes[0].imshow(mask, extent=[v / LAM for v in (-half_L, half_L, -half_L, half_L)],
                         origin="lower", cmap="gray")
    axes[0].set_title("Máscara (objeto)")
    axes[0].set_xlim(-lim / MAG, lim / MAG)
    axes[0].set_ylim(-lim / MAG, lim / MAG)
    im1 = axes[1].imshow((I_s / vmax) ** 0.8, extent=extent, origin="lower",
                         cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Imagen escalar ($t_-=0$): C = {Cs:.3f}")
    im2 = axes[2].imshow((I_v / vmax) ** 0.8, extent=extent, origin="lower",
                         cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Imagen vectorial: C = {Cv:.3f}")
    for axi in axes[1:3]:
        axi.set_xlim(-lim, lim)
        axi.set_ylim(-lim, lim)
    for axi in axes[:3]:
        axi.set_xlabel(r"$x/\lambda$")
        axi.set_ylabel(r"$y/\lambda$")

    ref = p_s.max()
    axes[3].plot(u_s / (MAG * sep), p_s / ref, "k--", label=f"escalar (C={Cs:.3f})")
    axes[3].plot(u_v / (MAG * sep), p_v / ref, "-", color="#d62728",
                 label=f"vectorial (C={Cv:.3f})")
    for s in (-0.5, 0.5):
        axes[3].axvline(s, color="gray", lw=0.7, alpha=0.6)
    axes[3].set_xlabel(r"$u / (M \cdot sep)$")
    axes[3].set_ylabel("I (norm.)")
    axes[3].set_title(r"Perfil a lo largo de la separación ($\parallel \hat{y}$)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    label = "discos" if feature == "discs" else "líneas"
    fig.suptitle(
        rf"{label}: sep = {s_frac:.2f} $d_{{Airy}}$ = {sep*1e3:.3f} µm  ·  "
        r"pol. incidente lineal $\hat{x}$, separación $\parallel \hat{y}$" + "\n"
        + ic.system_caption(R_A, d2=D2_FAST), fontsize=10)
    fig.savefig(out / figname, dpi=200)
    plt.close(fig)
    print(f"{feature}: showcase en sep = {s_frac:.2f} dAiry "
          f"(C_sca = {Cs:.3f}, C_vec = {Cv:.3f})")

    # -------------------------------------------------------------- #
    #  Polarization analysis of the showcase images                    #
    #  (built on the package polarization maps:                        #
    #   polarization_map_from_field + plot_polarization_map)           #
    # -------------------------------------------------------------- #
    half_win = 1.05 * lim * LAM  # analysis half-window in grid units (mm)

    # Fields in wavelength units with peak-normalized amplitude, so the map
    # backgrounds and glyph overlays share a common readable scale.
    E_s_lam, E_v_lam = (
        FieldCartesian(x=E.x / np.sqrt(vmax), y=E.y / np.sqrt(vmax),
                       grid=Grid.from_cartesian(E.grid.X / LAM, E.grid.Y / LAM),
                       symmetric=False)
        for E in (E_s, E_v)
    )

    # Stokes maps resampled on the analysis window with the package helper.
    pol_maps = {}
    for tag, E_lam in (("escalar", E_s_lam), ("vectorial", E_v_lam)):
        xx, yy, pol = polarization_map_from_field(E_lam, half_size=1.05 * lim, n_img=340)
        pol_maps[tag] = (xx, yy, pol)

    _, _, pol_v = pol_maps["vectorial"]
    bright_v = pol_v.s0 > 0.02 * pol_v.s0.max()
    iy_lim = float(np.abs(pol_v.ey).max() ** 2)
    psi_lim = max(5.0, float(np.nanpercentile(
        np.abs(np.degrees(np.where(bright_v, pol_v.psi, np.nan))), 99)))
    chi_lim = max(1.0, 1.2 * float(np.nanmax(
        np.abs(np.degrees(np.where(bright_v, pol_v.chi, np.nan))))))

    polfig = "fig4_polarization_discs.png" if feature == "discs" else "fig5_polarization_lines.png"
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 7.8), constrained_layout=True)

    for row_i, tag in enumerate(("escalar", "vectorial")):
        xx, yy, pol = pol_maps[tag]
        ext = [xx.min(), xx.max(), yy.min(), yy.max()]
        bright = pol.s0 > 0.02 * pol.s0.max()
        f_cross_img = float((np.abs(pol.ey) ** 2).sum() / (pol.s0.sum() + 1e-300))
        row_tag = "escalar ($t_-=0$)" if tag == "escalar" else tag

        im = axes[row_i, 0].imshow((np.abs(pol.ex) ** 2) ** 0.8, extent=ext, origin="lower",
                                   cmap="gray", vmin=0, vmax=1)
        axes[row_i, 0].set_title(rf"{row_tag}: $|E_x|^2$ (copolar)")
        fig.colorbar(im, ax=axes[row_i, 0], fraction=0.046, pad=0.04)

        im = axes[row_i, 1].imshow(np.abs(pol.ey) ** 2, extent=ext, origin="lower",
                                   cmap="magma", vmin=0.0, vmax=iy_lim)
        axes[row_i, 1].set_title(rf"$|E_y|^2$ (cruzada), $f_{{cross}}$ = {f_cross_img:.1e}")
        fig.colorbar(im, ax=axes[row_i, 1], fraction=0.046, pad=0.04)

        psi = np.where(bright, np.degrees(pol.psi), np.nan)
        im = axes[row_i, 2].imshow(psi, extent=ext, origin="lower",
                                   cmap="RdBu_r", vmin=-psi_lim, vmax=psi_lim)
        axes[row_i, 2].set_title(r"orientación $\psi$ [$^\circ$]")
        fig.colorbar(im, ax=axes[row_i, 2], fraction=0.046, pad=0.04)

        chi = np.where(bright, np.degrees(pol.chi), np.nan)
        im = axes[row_i, 3].imshow(chi, extent=ext, origin="lower",
                                   cmap="RdBu_r", vmin=-chi_lim, vmax=chi_lim)
        axes[row_i, 3].set_title(r"elipticidad $\chi$ [$^\circ$]")
        fig.colorbar(im, ax=axes[row_i, 3], fraction=0.046, pad=0.04)

        for axi in axes[row_i, :]:
            axi.set_xlim(-lim, lim)
            axi.set_ylim(-lim, lim)
            axi.set_xlabel(r"$x/\lambda$")
            axi.set_ylabel(r"$y/\lambda$")

    fig.suptitle(
        rf"Mapas de polarización (Stokes) de la imagen  ·  {label}, sep = {s_frac:.2f} $d_{{Airy}}$  ·  "
        r"pol. incidente lineal $\hat{x}$" + "\n" + ic.system_caption(R_A, d2=D2_FAST),
        fontsize=10)
    fig.savefig(out / polfig, dpi=180)
    plt.close(fig)
    print(f"{feature}: mapas de Stokes -> {polfig}")

    # Dedicated polarization-ellipse maps (plot_polarization_map via
    # plot_field_polarization: intensity background + local ellipse glyphs).
    mapfig = "fig6_polmap_discs.png" if feature == "discs" else "fig7_polmap_lines.png"
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.4), constrained_layout=True)
    for axi, (tag, E_lam) in zip(axes, (("escalar ($t_-=0$)", E_s_lam),
                                        ("vectorial", E_v_lam))):
        plot_field_polarization(
            E_lam, half_size=1.05 * lim, n_img=300, sampling="cartesian",
            target_ellipses=17, min_intensity_fraction=0.01,
            ax=axi,
        )
        axi.set_title(tag)
        axi.set_xlim(-lim, lim)
        axi.set_ylim(-lim, lim)
        axi.set_xlabel(r"$x/\lambda$")
        axi.set_ylabel(r"$y/\lambda$")
    fig.suptitle(
        rf"Mapa de polarización de la imagen  ·  {label}, sep = {s_frac:.2f} $d_{{Airy}}$  ·  "
        r"pol. incidente lineal $\hat{x}$" + "\n" + ic.system_caption(R_A, d2=D2_FAST),
        fontsize=10)
    fig.savefig(out / mapfig, dpi=200)
    plt.close(fig)
    print(f"{feature}: mapa de elipses -> {mapfig}")

print(f"Done. Outputs in {out}")
