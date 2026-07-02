"""Study 2 - Maximizing the cross-polarized component of the focal field.

Levers explored (exact Fresnel, physical branch of the oval only):

1. Index contrast ``ni``: at grazing incidence ``t_minus/t_plus`` tends to the
   ceiling ``rho_max = (ni - n0) / (ni + n0)``, which bounds the achievable
   cross fraction roughly by ``f <= rho_max^2 / (2 (1 + rho_max^2))``.
2. Conjugate geometry ``(z0, zi)``: expected to be a weak lever - the ceiling
   depends on the indices only.
3. Pupil engineering at full aperture: central obstruction (annulus with inner
   fraction ``eps``) and edge-weighted amplitude profiles ``(r/a)^p``, both of
   which concentrate the pupil energy where ``t_minus/t_plus`` is largest.

Every configuration records the cross fraction, throughput (the price paid),
peak ratio and spot metrics.  Outputs go to ``output/study_2_maximize_cross/``.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

import common
from common import N0, TAU

N_R = 701
N_Q = 450
Q_SPAN = 30.0

out = common.output_dir(__file__)

NI_LIST = [1.33, 1.5, 1.8, 2.2, 2.6, 3.0]
EPS_LIST = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95]
P_LIST = [0, 2, 4, 8]


def evaluate(r, q, M0, M2, t_plus, t_minus, pupil):
    """Metrics of one pupil configuration reusing precomputed Hankel matrices."""
    A = TAU * (M0 @ (t_plus * pupil))
    B = TAU * (M2 @ (t_minus * pupil))
    row = common.energy_report(A, B, q)
    row.update(common.spot_metrics(A, B, q))
    input_energy = TAU * np.trapezoid(np.abs(pupil) ** 2 * r, r)
    row["throughput"] = row["energy_total"] / (TAU**2 * input_energy + 1e-300)
    return row, A, B


def case_grid(ni, z0, zi, a_frac=0.98):
    """Radial/frequency grids, Hankel matrices and coefficients for one geometry."""
    a = a_frac * common.grazing_radius(ni=ni, z0=z0, zi=zi)
    r = np.linspace(0.0, a, N_R)
    q = np.linspace(0.0, Q_SPAN / a, N_Q)
    M0 = common.hankel_matrix(0, r, q)
    M2 = common.hankel_matrix(2, r, q)
    t_plus, t_minus = common.exact_coeffs(r, ni=ni, z0=z0, zi=zi)
    return a, r, q, M0, M2, t_plus, t_minus


rows = []


def record(ni, z0, zi, a, eps, p, row):
    entry = {"ni": ni, "z0": z0, "zi": zi, "a": a, "eps": eps, "p": p}
    entry.update({k: row[k] for k in
                  ("f_cross", "peak_cross_over_peak_total", "throughput",
                   "hwhm_x", "hwhm_y", "ellipticity", "energy_total")})
    rows.append(entry)
    return entry


# ------------------------------------------------------------------ #
#  Part A - index contrast and geometry (uniform full pupil)           #
# ------------------------------------------------------------------ #

print("== Part A: index contrast (uniform pupil, a = 0.98 r_graze) ==")
best_uniform = {}
for ni in NI_LIST:
    a, r, q, M0, M2, tpl, tmn = case_grid(ni, common.Z0, common.ZI)
    pupil = np.ones_like(r, dtype=complex)
    row, _, _ = evaluate(r, q, M0, M2, tpl, tmn, pupil)
    record(ni, common.Z0, common.ZI, a, 0.0, 0, row)
    best_uniform[ni] = row["f_cross"]
    rho_max = (ni - N0) / (ni + N0)
    print(f"ni={ni:4}  a={a:5.2f}  f_cross={row['f_cross']:.3e}  "
          f"techo local rho_max^2/(2(1+rho_max^2))={rho_max**2 / (2 * (1 + rho_max**2)):.3e}")

print("== Part A2: geometry variations at ni=1.5 ==")
for z0 in (-3.0, -5.0, -10.0):
    for zi in (3.0, 6.0, 10.0):
        a, r, q, M0, M2, tpl, tmn = case_grid(1.5, z0, zi)
        row, _, _ = evaluate(r, q, M0, M2, tpl, tmn, np.ones_like(r, dtype=complex))
        record(1.5, z0, zi, a, 0.0, 0, row)
        print(f"z0={z0:6}  zi={zi:5}  a={a:5.2f}  f_cross={row['f_cross']:.3e}")

# ------------------------------------------------------------------ #
#  Part B - pupil engineering (annulus x edge weighting)               #
# ------------------------------------------------------------------ #

print("== Part B: pupil engineering ==")
curves = {}
best = None
for ni in (1.5, 2.2, 3.0):
    a, r, q, M0, M2, tpl, tmn = case_grid(ni, common.Z0, common.ZI)
    curves[ni] = {"eps": [], "f": [], "T": []}
    for eps in EPS_LIST:
        for p in P_LIST:
            pupil = ((r >= eps * a) & (r <= a)).astype(complex)
            if p > 0:
                pupil *= (r / a) ** p
            row, A, B = evaluate(r, q, M0, M2, tpl, tmn, pupil)
            entry = record(ni, common.Z0, common.ZI, a, eps, p, row)
            if best is None or entry["f_cross"] > best[0]["f_cross"]:
                best = (entry, A.copy(), B.copy(), q.copy())
            if p == 0:
                curves[ni]["eps"].append(eps)
                curves[ni]["f"].append(row["f_cross"])
                curves[ni]["T"].append(row["throughput"])
    fmax = max(x["f_cross"] for x in rows if x["ni"] == ni and not (x["eps"] == 0 and x["p"] == 0))
    print(f"ni={ni}: uniforme f_cross={best_uniform[ni]:.3e} -> max con pupila diseñada {fmax:.3e}")

with (out / "maximize_results.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

best_entry, A_best, B_best, q_best = best
print("Mejor configuración:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in best_entry.items()})

# ------------------------------------------------------------------ #
#  Figure 1: ceiling law and pupil-engineering curves                  #
# ------------------------------------------------------------------ #

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

ni_arr = np.array(NI_LIST)
f_uni = np.array([best_uniform[ni] for ni in NI_LIST])
f_eng = np.array([max(x["f_cross"] for x in rows if x["ni"] == ni) for ni in NI_LIST])
rho = (ni_arr - N0) / (ni_arr + N0)
ceiling = rho**2 / (2.0 * (1.0 + rho**2))
axes[0].semilogy(ni_arr, f_uni, "o-", label="pupila uniforme")
axes[0].semilogy([1.5, 2.2, 3.0], [max(x["f_cross"] for x in rows if x["ni"] == ni and (x["eps"] > 0 or x["p"] > 0)) for ni in (1.5, 2.2, 3.0)],
                 "s-", label="mejor pupila diseñada")
axes[0].semilogy(ni_arr, ceiling, "k--", label=r"techo $\rho_{max}^2/(2(1+\rho_{max}^2))$")
axes[0].set_xlabel(r"$n_i$")
axes[0].set_ylabel(r"$f_{cross}$")
axes[0].set_title("Contraste de índices y techo teórico")
axes[0].grid(True, which="both", alpha=0.3)
axes[0].legend()

for ni, c in curves.items():
    axes[1].semilogy(c["eps"], c["f"], "o-", label=rf"$n_i$={ni}")
axes[1].set_xlabel(r"obstrucción central $\epsilon$ (pupila anular, $p=0$)")
axes[1].set_ylabel(r"$f_{cross}$")
axes[1].set_title("Efecto de la obstrucción anular")
axes[1].grid(True, which="both", alpha=0.3)
axes[1].legend()

for ni in (1.5, 2.2, 3.0):
    pts = [(x["throughput"], x["f_cross"]) for x in rows if x["ni"] == ni and x["z0"] == common.Z0 and x["zi"] == common.ZI]
    T, F = zip(*sorted(pts))
    axes[2].semilogy(T, F, "o", ms=4, label=rf"$n_i$={ni}")
axes[2].set_xlabel("throughput")
axes[2].set_ylabel(r"$f_{cross}$")
axes[2].set_title("Compromiso eficiencia vs componente cruzada")
axes[2].grid(True, which="both", alpha=0.3)
axes[2].legend()

fig.suptitle("Maximización de la componente cruzada (Fresnel exacto, rama física del óvalo)")
fig.savefig(out / "fig1_maximization.png", dpi=220)
plt.close(fig)

# ------------------------------------------------------------------ #
#  Figure 2: focal maps of the best configuration                      #
# ------------------------------------------------------------------ #

row_best = common.spot_metrics(A_best, B_best, q_best)
half = 3.0 * (row_best["first_min_scalar"] if np.isfinite(row_best["first_min_scalar"]) else q_best[-1] / 6)
maps, extent = common.maps_2d(A_best, B_best, q_best, half)
zi_best = best_entry["zi"]
ni_best = best_entry["ni"]
ext_lam = [e * zi_best / (TAU * ni_best) for e in extent]

I_vec, I_sca, I_cross = maps["I_vec"], maps["I_scalar"], maps["I_cross"]
vmax = max(I_vec.max(), I_sca.max())

fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), constrained_layout=True)
panels = [
    (I_sca / vmax, "hot", r"$I_{escalar}$ ($t_-=0$)", (0, 1)),
    (I_vec / vmax, "hot", r"$I_{vectorial}$", (0, 1)),
    ((I_vec - I_sca) / vmax, "RdBu_r", r"$\Delta I$", None),
    (I_cross / I_cross.max(), "magma", r"$|E_y|^2$ (cruzada, norm.)", (0, 1)),
]
for axi, (img, cmap, title, clim) in zip(axes, panels):
    if clim is None:
        m = np.abs(img).max()
        im = axi.imshow(img, extent=ext_lam, origin="lower", cmap=cmap, vmin=-m, vmax=m)
    else:
        im = axi.imshow(img, extent=ext_lam, origin="lower", cmap=cmap, vmin=clim[0], vmax=clim[1])
    axi.set_title(title)
    axi.set_xlabel(r"$x/\lambda$")
    axi.set_ylabel(r"$y/\lambda$")
    fig.colorbar(im, ax=axi, fraction=0.046, pad=0.04)
fig.suptitle(
    rf"Mejor configuración: $n_i$={best_entry['ni']}, $\epsilon$={best_entry['eps']}, "
    rf"$p$={best_entry['p']}, a={best_entry['a']:.2f} mm  |  "
    rf"$f_{{cross}}$={best_entry['f_cross']:.3f}, throughput={best_entry['throughput']:.3f}"
)
fig.savefig(out / "fig2_best_config_maps.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ #
#  Figure 3: cross-channel shape, linear vs circular input             #
# ------------------------------------------------------------------ #

# Same kernels: for a circular (L) input the cross channel is |E_R|^2 = |B|^2
# (a ring carrying a charge-2 vortex), while for a linear input it is the
# sin^2(2 phi) clover.  The cross ENERGY fraction is identical in both bases.
axis = np.linspace(-half, half, 460)
X, Y = np.meshgrid(axis, axis)
rho_g = np.hypot(X, Y)
B2 = np.interp(rho_g, q_best, B_best.real) + 1j * np.interp(rho_g, q_best, B_best.imag)
phi_g = np.arctan2(Y, X)
clover = np.abs(np.sin(2.0 * phi_g) * B2) ** 2
ring = np.abs(B2) ** 2

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True)
ext_lam2 = [e * zi_best / (TAU * ni_best) for e in (-half, half, -half, half)]
for axi, img, title in zip(axes, (clover, ring),
                           (r"entrada lineal $\hat{x}$: $|E_y|^2 \propto \sin^2 2\varphi\, |B|^2$",
                            r"entrada circular $L$: $|E_R|^2 = |B|^2$ (vórtice $m=2$)")):
    im = axi.imshow(img / img.max(), extent=ext_lam2, origin="lower", cmap="magma")
    axi.set_title(title, fontsize=10)
    axi.set_xlabel(r"$x/\lambda$")
    axi.set_ylabel(r"$y/\lambda$")
    fig.colorbar(im, ax=axi, fraction=0.046, pad=0.04)
fig.suptitle("Forma del canal cruzado según la base de polarización (misma energía)")
fig.savefig(out / "fig3_cross_channel_shape.png", dpi=200)
plt.close(fig)

print(f"Done. Outputs in {out}")
