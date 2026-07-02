"""Study 1 - Focal-plane field vs aperture size, scalar (t_minus = 0) vs vectorial.

For the baseline diopter (n0=1, ni=1.5, z0=-10, zi=6, lambda=532 nm) and an
x-polarized uniform pupil of increasing radius ``a`` this script computes the
exact-Fresnel focal kernels A (co-polar) and B (cross-polar) and tracks:

* f_cross: fraction of the focal energy carried by the cross-polarized field,
* spot radii (HWHM, first minimum, r50) along phi = 0 / 90 deg and for the
  scalar reference, and the resulting ellipticity of the bright spot,
* throughput of the diopter,
* the small-aperture power law of f_cross,
* 2D intensity maps and radial cuts for selected apertures.

Outputs go to ``output/study_1_aperture_sweep/``.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

import common
from common import N0, NI, Z0, ZI, TAU

N_R = 901
N_Q = 700
Q_SPAN = 30.0  # q_max = Q_SPAN / a  (~7 Airy rings)

out = common.output_dir(__file__)


def compute_case(a):
    r = np.linspace(0.0, a, N_R)
    q = np.linspace(0.0, Q_SPAN / a, N_Q)
    t_plus, t_minus = common.exact_coeffs(r)
    pupil = np.ones_like(r, dtype=complex)

    A, B = common.focal_kernels(r, pupil, t_plus, t_minus, q)
    alpha_obj, theta_inc, theta_img = common.geometry_angles(a)
    row = {"a": a, "alpha_obj_deg": alpha_obj, "theta_inc_deg": theta_inc,
           "theta_img_deg": theta_img}
    row.update(common.energy_report(A, B, q))
    row.update(common.spot_metrics(A, B, q))

    input_energy = TAU * np.trapezoid(np.abs(pupil) ** 2 * r, r)
    row["throughput"] = row["energy_total"] / (TAU**2 * input_energy)
    return row, (r, q, A, B)


def to_lambda(q):
    return common.q_to_focal_lambda(np.asarray(q))


# ------------------------------------------------------------------ #
#  Sweep                                                               #
# ------------------------------------------------------------------ #

# The oval is a closed surface; past the grazing radius the parametrization
# runs along its back face, which is not a physical first-surface pupil.
r_graze = common.grazing_radius()
apertures = np.linspace(0.4, 0.99 * r_graze, 32)
print(f"grazing radius = {r_graze:.3f} mm -> sweeping a in [{apertures[0]:.2f}, {apertures[-1]:.2f}] mm")

rows, kernels = [], {}
for a in apertures:
    row, data = compute_case(float(a))
    rows.append(row)
    kernels[float(a)] = data
    print(f"a={a:5.2f}  alpha={row['alpha_obj_deg']:5.1f} deg  f_cross={row['f_cross']:.3e}  "
          f"HWHM x/y={row['hwhm_x']:.3f}/{row['hwhm_y']:.3f}  ellip={row['ellipticity']:.4f}")

with (out / "sweep_results.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

col = {k: np.array([row[k] for row in rows]) for k in rows[0]}

# Small-aperture power law of f_cross vs tan(alpha_obj).
tan_alpha = np.tan(np.radians(col["alpha_obj_deg"]))
n_fit = 10
slope, intercept = np.polyfit(np.log(tan_alpha[:n_fit]), np.log(col["f_cross"][:n_fit]), 1)
print(f"small-aperture fit: f_cross ~ tan(alpha)^{slope:.2f}")

# ------------------------------------------------------------------ #
#  Figure 1: metrics vs aperture                                       #
# ------------------------------------------------------------------ #

fig, ax = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)

ax[0, 0].semilogy(col["a"], col["f_cross"], "o-", color="#1f77b4", label=r"$f_{cross}$ (energía)")
ax[0, 0].semilogy(col["a"], col["peak_cross_over_peak_total"], "s-", color="#ff7f0e",
                  label=r"pico $|E_y|^2$ / pico $I$")
ax[0, 0].set_xlabel("a [mm]")
ax[0, 0].set_title("Componente cruzada vs apertura")
ax[0, 0].grid(True, which="both", alpha=0.3)
ax[0, 0].legend()

lam_hx = to_lambda(col["hwhm_x"])
lam_hy = to_lambda(col["hwhm_y"])
lam_hs = to_lambda(col["hwhm_scalar"])
ax[0, 1].plot(col["a"], lam_hx, "o-", color="#d62728", label=r"HWHM $\varphi=0$ (vect)")
ax[0, 1].plot(col["a"], lam_hy, "s-", color="#2ca02c", label=r"HWHM $\varphi=90^\circ$ (vect)")
ax[0, 1].plot(col["a"], lam_hs, "k--", label="HWHM escalar")
airy = to_lambda(1.61633 / col["a"])  # HWHM of the ideal Airy pattern
ax[0, 1].plot(col["a"], airy, ":", color="gray", label="Airy ideal (sin $t_+$)")
ax[0, 1].set_xlabel("a [mm]")
ax[0, 1].set_ylabel(r"radio del spot [$\lambda$]")
ax[0, 1].set_title("Radio del punto brillante")
ax[0, 1].set_yscale("log")
ax[0, 1].grid(True, which="both", alpha=0.3)
ax[0, 1].legend()

ax[1, 0].plot(col["a"], col["ellipticity"], "o-", color="#9467bd",
              label=r"HWHM$_x$/HWHM$_y$")
ax[1, 0].plot(col["a"], col["first_min_x"] / col["first_min_y"], "s-", color="#8c564b",
              label=r"1er mín $x$ / 1er mín $y$")
ax[1, 0].axhline(1.0, color="k", lw=1, ls="--")
ax[1, 0].set_xlabel("a [mm]")
ax[1, 0].set_title("Anisotropía (elipticidad) del spot vectorial")
ax[1, 0].grid(True, alpha=0.3)
ax[1, 0].legend()

ax[1, 1].plot(col["a"], col["throughput"], "o-", color="#17becf", label="throughput")
ax[1, 1].plot(col["a"], to_lambda(col["r50"]) / to_lambda(col["r50_scalar"]), "s-",
              color="#e377c2", label=r"$r_{50}$ vect / $r_{50}$ escalar")
ax[1, 1].set_xlabel("a [mm]")
ax[1, 1].set_title("Transmisión y radio de energía encerrada")
ax[1, 1].grid(True, alpha=0.3)
ax[1, 1].legend()

fig.suptitle(rf"Barrido de apertura | $n_0$={N0}, $n_i$={NI}, $z_0$={Z0}, $z_i$={ZI} (Fresnel exacto) | "
             r"polarización incidente lineal $\hat{x}$")
fig.savefig(out / "fig1_metrics_vs_aperture.png", dpi=220)
plt.close(fig)

# ------------------------------------------------------------------ #
#  Figure 2: scaling law                                               #
# ------------------------------------------------------------------ #

fig, ax = plt.subplots(figsize=(6.4, 5), constrained_layout=True)
ax.loglog(tan_alpha, col["f_cross"], "o", color="#1f77b4", label="numérico (exacto)")
ax.loglog(tan_alpha[:n_fit], np.exp(intercept) * tan_alpha[:n_fit] ** slope, "r-",
          label=rf"ajuste: $f_{{cross}} \propto \tan^{{{slope:.2f}}}\alpha$")
ax.set_xlabel(r"$\tan\alpha_{obj} = a/|z_0|$")
ax.set_ylabel(r"$f_{cross}$")
ax.set_title("Ley de escala de la componente cruzada")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.savefig(out / "fig2_scaling_law.png", dpi=220)
plt.close(fig)

# ------------------------------------------------------------------ #
#  Figures 3+: 2D maps and radial cuts at selected apertures           #
# ------------------------------------------------------------------ #

selected = [apertures[3], apertures[len(apertures) // 2], apertures[-1]]

for a in selected:
    a = float(min(kernels, key=lambda k: abs(k - a)))
    r, q, A, B = kernels[a]
    row = rows[int(np.argmin(np.abs(apertures - a)))]
    half = 3.0 * row["first_min_scalar"]
    maps, extent = common.maps_2d(A, B, q, half)
    ext_lam = [to_lambda(e) for e in extent]

    I_vec, I_sca = maps["I_vec"], maps["I_scalar"]
    delta = I_vec - I_sca
    vmax = max(I_vec.max(), I_sca.max())

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), constrained_layout=True)
    im0 = axes[0].imshow(I_sca / vmax, extent=ext_lam, origin="lower", cmap="hot", vmin=0, vmax=1)
    axes[0].set_title(r"$I_{escalar}$  ($t_-=0$)")
    im1 = axes[1].imshow(I_vec / vmax, extent=ext_lam, origin="lower", cmap="hot", vmin=0, vmax=1)
    axes[1].set_title(r"$I_{vectorial}$")
    dmax = np.abs(delta).max() / vmax
    im2 = axes[2].imshow(delta / vmax, extent=ext_lam, origin="lower", cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    axes[2].set_title(r"$\Delta I = I_{vec} - I_{esc}$")
    im3 = axes[3].imshow(maps["I_cross"] / maps["I_cross"].max(), extent=ext_lam, origin="lower", cmap="magma")
    axes[3].set_title(r"$|E_y|^2$ (cruzada, norm.)")
    for axi, im in zip(axes, (im0, im1, im2, im3)):
        axi.set_xlabel(r"$x/\lambda$")
        axi.set_ylabel(r"$y/\lambda$")
        fig.colorbar(im, ax=axi, fraction=0.046, pad=0.04)
    fig.suptitle(
        rf"a = {a:.2f} mm  ($\alpha_{{obj}}$ = {row['alpha_obj_deg']:.1f}$^\circ$, "
        rf"$\theta_{{img}}$ = {row['theta_img_deg']:.1f}$^\circ$)  |  "
        rf"$f_{{cross}}$ = {row['f_cross']:.2e}, elipticidad = {row['ellipticity']:.3f}  |  "
        r"pol. incidente lineal $\hat{x}$"
    )
    tag = f"{a:.2f}".replace(".", "p")
    fig.savefig(out / f"fig3_maps_a{tag}.png", dpi=200)
    plt.close(fig)

    # radial cuts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    lam_q = to_lambda(q)
    I_x = np.abs(A - B) ** 2
    I_y = np.abs(A + B) ** 2
    I_45 = np.abs(A) ** 2 + np.abs(B) ** 2
    I_s = np.abs(A) ** 2
    norm = I_s[0]
    ax1.plot(lam_q, I_s / norm, "k--", label="escalar")
    ax1.plot(lam_q, I_x / norm, color="#d62728", label=r"vect $\varphi=0$")
    ax1.plot(lam_q, I_y / norm, color="#2ca02c", label=r"vect $\varphi=90^\circ$")
    ax1.plot(lam_q, I_45 / norm, color="#1f77b4", label=r"vect $\varphi=45^\circ$")
    ax1.set_xlabel(r"$\rho/\lambda$")
    ax1.set_ylabel(r"$I/I_{esc}(0)$")
    ax1.set_title(f"Cortes radiales, a = {a:.2f} mm")
    ax1.set_xlim(0, to_lambda(2.5 * row["first_min_scalar"]))
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.semilogy(lam_q, np.abs(A) ** 2 / norm, color="#1f77b4", label=r"$|A|^2$ (copolar)")
    ax2.semilogy(lam_q, np.abs(B) ** 2 / norm, color="#ff7f0e", label=r"$|B|^2$ (cruzada)")
    ax2.set_xlabel(r"$\rho/\lambda$")
    ax2.set_title("Kernels radiales")
    ax2.set_xlim(0, to_lambda(q[-1]))
    ax2.set_ylim(1e-8, 2)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    fig.savefig(out / f"fig4_cuts_a{tag}.png", dpi=200)
    plt.close(fig)

print(f"Done. Outputs in {out}")
