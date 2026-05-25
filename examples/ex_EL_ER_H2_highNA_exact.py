import numpy as np
import matplotlib.pyplot as plt
from vecdiff.fresnel import FresnelOvoid, FresnelOvoidParax
from vecdiff.hankel import HankelTransform


# ---------------- CONFIG ----------------
lam = 532e-6
n0 = 1.0
ni = 2.2
z0 = 2.0
zi = 8.0

R = 2.0
a = 0.95 * R
N = 1080

USE_PARAXIAL = False  # default exact, set True for paraxial
FOCUS_REFINED = True
FOCUS_POWER = 2.3

# Keep both components active; symmetric edge weighting helps H2 significance.
PROFILE_POWER = 0  # EL=ER=(r/a)^PROFILE_POWER inside aperture

varphi = 0.0
phase = exp(2.0j * varphi)
# ----------------------------------------


def make_grid(max_val, n, refined, power):
    if refined:
        u = np.linspace(0.0, 1.0, n)
        return max_val * u**power
    return np.linspace(0.0, max_val, n)


def sanitize_at_origin(arr, r):
    arr = np.asarray(arr, dtype=float)
    bad = ~np.isfinite(arr)
    if np.any(bad):
        good = ~bad
        arr[bad] = np.interp(r[bad], r[good], arr[good])
    return arr


def nonnegative_vmax(img, q=99.5):
    finite = np.asarray(img)[np.isfinite(img)]
    if finite.size == 0:
        return 1.0
    vmax = float(np.percentile(finite, q))
    if vmax <= 0.0:
        vmax = float(np.max(finite))
    if vmax <= 0.0:
        vmax = 1.0
    return vmax


q_max = ni / (lam * zi) * R
r = make_grid(R, N, FOCUS_REFINED, FOCUS_POWER)
q = make_grid(q_max, N, FOCUS_REFINED, FOCUS_POWER)

# Fresnel coefficients
if USE_PARAXIAL:
    F = FresnelOvoidParax(n0, ni, z0, zi)
    ts, tp = F.coeffs(r)
    fresnel_label = "Paraxial"
else:
    F = FresnelOvoid(n0=n0, z0=z0, ni=ni, zi=zi)
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = F.ts(r)
        tp = F.tp(r)
    fresnel_label = "Exact"

ts = sanitize_at_origin(ts, r)
tp = sanitize_at_origin(tp, r)


# INPUT FIELDS

ap = (r <= a).astype(float)
profile = (r / a) ** PROFILE_POWER

EL = profile * ap
ER = profile * ap * 0 



field_expr = f"EL(r)=ER(r)=(r/a)^{PROFILE_POWER} for r<=a, else 0"

HT = HankelTransform.transform_array

C_L_H0 = 2.0 * π * HT(0, r, (tp + ts) * EL, q)
C_L_H1 = -2.0 * π * phase * HT(1, r, (tp - ts) * ER, q)
E_L = C_L_H0 + C_L_H1

C_R_H0 = 2.0 * π * HT(0, r, (tp + ts) * ER, q)
C_R_H2 = -2.0 * π * phase * HT(2, r, (tp - ts) * EL, q)
E_R = C_R_H0 + C_R_H2

I_full = np.abs(E_L) ** 2 + np.abs(E_R) ** 2
I_noH2 = np.abs(E_L) ** 2 + np.abs(C_R_H0) ** 2

h2_peak_ratio = np.max(np.abs(C_R_H2)) / (np.max(np.abs(C_R_H0)) + 1e-15)
h2_energy_ratio = np.trapezoid(np.abs(C_R_H2) ** 2 * q, q) / (
    np.trapezoid(np.abs(C_R_H0) ** 2 * q, q) + 1e-15
)
delta_intensity = np.trapezoid(np.abs(I_full - I_noH2) * q, q) / (
    np.trapezoid(I_full * q, q) + 1e-15
)

print(f"Fresnel: {fresnel_label}")
print(f"H2/H0 peak ratio in E'_R: {h2_peak_ratio:.4f}")
print(f"H2/H0 energy ratio in E'_R: {h2_energy_ratio:.4f}")
print(f"Relative intensity impact of H2: {delta_intensity:.4f}")

# Input-field view (what EL and ER were chosen)
fig0, ax0 = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
ax0.plot(r, np.real(EL), color="tab:blue", linewidth=2, label=r"$E_L(r)$")
ax0.plot(r, np.real(ER), color="tab:orange", linewidth=2, linestyle="--", label=r"$E_R(r)$")
ax0.axvline(a, color="k", linestyle=":", linewidth=1.5, label=r"$r=a$")
ax0.set_xlabel(r"$r$")
ax0.set_ylabel("Amplitude")
ax0.set_title("Chosen Input Components")
ax0.grid(True, alpha=0.3)
ax0.legend()
ax0.text(
    0.02,
    0.96,
    field_expr,
    transform=ax0.transAxes,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# 1D comparisons
fig1, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

ax[0].plot(q, np.abs(C_L_H0), color="tab:blue", label=r"$|C^{(L)}_{H0}|$", linewidth=2)
ax[0].plot(q, np.abs(C_L_H1), color="tab:orange", label=r"$|C^{(L)}_{H1}|$", linewidth=2)
ax[0].plot(q, np.abs(E_L), color="tab:green", linestyle="--", label=r"$|E'_L|$", linewidth=2)
ax[0].set_title(r"$E'_L$: H$_0$ vs H$_1$")
ax[0].set_xlabel(r"$q$")
ax[0].set_ylabel("Amplitude")
ax[0].grid(True, alpha=0.3)
ax[0].legend()

ax[1].plot(q, np.abs(C_R_H0), color="tab:blue", label=r"$|C^{(R)}_{H0}|$", linewidth=2)
ax[1].plot(q, np.abs(C_R_H2), color="tab:orange", label=r"$|C^{(R)}_{H2}|$", linewidth=2)
ax[1].plot(q, np.abs(E_R), color="tab:green", linestyle="--", label=r"$|E'_R|$", linewidth=2)
ax[1].set_title(r"$E'_R$: H$_0$ vs H$_2$")
ax[1].set_xlabel(r"$q$")
ax[1].set_ylabel("Amplitude")
ax[1].grid(True, alpha=0.3)
ax[1].legend()

fig1.suptitle(
    f"Focus-Refined High-NA comparison ({fresnel_label} Fresnel, a/R={a/R:.2f}, p={PROFILE_POWER}, power={FOCUS_POWER})",
    y=1.03,
)

# 2D maps (tighter focused window)
Q_view = 0.04 * q_max
N_img = 1000
x = np.linspace(-Q_view, Q_view, N_img)
y = np.linspace(-Q_view, Q_view, N_img)
X, Y = np.meshgrid(x, y)
R_img = np.sqrt(X**2 + Y**2)
extent = [-Q_view, Q_view, -Q_view, Q_view]

EL_img = np.interp(R_img, q, np.abs(E_L), left=np.abs(E_L)[0], right=np.abs(E_L)[-1])
ER_img = np.interp(R_img, q, np.abs(E_R), left=np.abs(E_R)[0], right=np.abs(E_R)[-1])
I_img = np.interp(R_img, q, I_full, left=I_full[0], right=I_full[-1])
H0_img = np.interp(R_img, q, np.abs(C_R_H0), left=np.abs(C_R_H0)[0], right=np.abs(C_R_H0)[-1])
H2_img = np.interp(R_img, q, np.abs(C_R_H2), left=np.abs(C_R_H2)[0], right=np.abs(C_R_H2)[-1])
H2_over_H0_img = H2_img / (H0_img + 1e-15)

# Robust upper limits to preserve contrast while keeping zero mapped to zero.
vmax_EL = nonnegative_vmax(EL_img)
vmax_ER = nonnegative_vmax(ER_img)
vmax_I = nonnegative_vmax(I_img)
vmax_H0 = nonnegative_vmax(H0_img)
vmax_H2 = nonnegative_vmax(H2_img)
vmax_ratio = nonnegative_vmax(H2_over_H0_img)

fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

im0 = ax2[0].imshow(EL_img, extent=extent, origin="lower", cmap="hot")
im0.set_clim(0.0, vmax_EL)
ax2[0].set_title(r"$|E'_L|$")
ax2[0].set_xlabel(r"$x$")
ax2[0].set_ylabel(r"$y$")
ax2[0].set_aspect("equal")
fig2.colorbar(im0, ax=ax2[0], fraction=0.046, pad=0.04)

im1 = ax2[1].imshow(ER_img, extent=extent, origin="lower", cmap="hot")
im1.set_clim(0.0, vmax_ER)
ax2[1].set_title(r"$|E'_R|$")
ax2[1].set_xlabel(r"$x$")
ax2[1].set_ylabel(r"$y$")
ax2[1].set_aspect("equal")
fig2.colorbar(im1, ax=ax2[1], fraction=0.046, pad=0.04)

im2 = ax2[2].imshow(I_img, extent=extent, origin="lower", cmap="hot")
im2.set_clim(0.0, vmax_I)
ax2[2].set_title(r"Intensity $|E'_L|^2 + |E'_R|^2$")
ax2[2].set_xlabel(r"$x$")
ax2[2].set_ylabel(r"$y$")
ax2[2].set_aspect("equal")
fig2.colorbar(im2, ax=ax2[2], fraction=0.046, pad=0.04)
fig2.suptitle(
    f"Focus-Refined Maps ({fresnel_label} Fresnel) | {field_expr}",
    y=1.04,
)

# Scalar maps for H0 and H2 contributions in E'_R
fig3, ax3 = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

im30 = ax3[0].imshow(H0_img, extent=extent, origin="lower", cmap="hot")
im30.set_clim(0.0, vmax_H0)
ax3[0].set_title(r"$|C^{(R)}_{H0}|$")
ax3[0].set_xlabel(r"$x$")
ax3[0].set_ylabel(r"$y$")
ax3[0].set_aspect("equal")
fig3.colorbar(im30, ax=ax3[0], fraction=0.046, pad=0.04)

im31 = ax3[1].imshow(H2_img, extent=extent, origin="lower", cmap="hot")
im31.set_clim(0.0, vmax_H2)
ax3[1].set_title(r"$|C^{(R)}_{H2}|$")
ax3[1].set_xlabel(r"$x$")
ax3[1].set_ylabel(r"$y$")
ax3[1].set_aspect("equal")
fig3.colorbar(im31, ax=ax3[1], fraction=0.046, pad=0.04)

im32 = ax3[2].imshow(H2_over_H0_img, extent=extent, origin="lower", cmap="viridis")
im32.set_clim(0.0, vmax_ratio)
ax3[2].set_title(r"$|C^{(R)}_{H2}| / |C^{(R)}_{H0}|$")
ax3[2].set_xlabel(r"$x$")
ax3[2].set_ylabel(r"$y$")
ax3[2].set_aspect("equal")
fig3.colorbar(im32, ax=ax3[2], fraction=0.046, pad=0.04)

fig3.suptitle(
    f"H0 vs H2 Scalar Contributions (q-view={Q_view/q_max:.2f} q_max)",
    y=1.03,
)

plt.show()
