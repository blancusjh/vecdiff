"""
Three vector-diffraction studies — Circular and Cartesian polarisation modes.

Study 1 — Non-diagonal contamination
    Single-component illumination (pure EL or pure Ex).
    The H₂ cross-coupling term transfers energy to the originally-zero output
    component.  We measure the leaked fraction and the resulting spot widening.

Study 2 — H₂ vs H₀ contribution
    H₀[(tp+ts)·E]  →  J₀(0)=1  →  bright centre.
    H₂[(tp-ts)·E]  →  J₂(0)=0  →  dark centre, broadens the spot.
    Sweep: marginal angle (vary a/R) and input profile shape.
    Profiles: flat, r/a, (r/a)², (r/a)⁴, spherical wave 1/√(z₀²+r²).

Study 3 — Source-distance (z₀) sweep
    At fixed geometry (a/R=0.95, flat, zᵢ=2.0), vary z₀.
    ξ ∝ (z₀−zᵢ)²/z₀² controls the H₂ amplitude.

PARAMETER CHOICE
    n₀=1.0, nᵢ=1.5, z₀=1.0, zᵢ=2.0, R=1.6 mm.
    θ_max ≈ 37° at a/R=0.95.  r_crit = √(γ₀/|n₀ξ|) = 2.0 mm > R,
    so tp remains positive across the entire aperture — the paraxial Fresnel
    approximation holds throughout.

NUMERICAL NOTE
    The Hankel transform uses Simpson's rule over the r-grid.  A focus-refined
    r-grid (clustered near r=0) undersamples the aperture edge and makes the
    H₂ integral noisy.  This script uses a UNIFORM r-grid for integration.
    The q-grid is focus-refined to resolve the PSF central peak.

Propagation formulas written explicitly (not imported from vecdiff.propagation).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from vecdiff.fresnel import FresnelOvoidParax
from vecdiff.hankel import HankelTransform


# ── optical system ────────────────────────────────────────────────────────────
n0  = 1.0     # incident refractive index
ni  = 1.5     # transmitted refractive index
z0  = 1.0     # source distance [mm]  — r_crit = 2.0 mm > R, tp > 0 everywhere
zi  = 2.0     # image distance  [mm]  — θ_max ≈ 37° at a/R=0.95
R   = 1.6     # aperture radius [mm]
lam = 532e-6  # wavelength      [mm]

PARAM_STR = f"n₀={n0}, nᵢ={ni}, z₀={z0}, zᵢ={zi}, R={R} mm  (paraxial Fresnel)"

# ── sweeps ────────────────────────────────────────────────────────────────────
A_FRACS   = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95]

# z₀ sweep: r_crit > 0.95·R for all entries (computed offline: need z₀ > 0.864)
Z0_VALUES = [0.90, 1.0, 1.2, 1.5, 1.8]

PROFILES = {          # profile shapes tested at a/R = 0.95
    "flat"   : 0,
    "r/a"    : 1,
    "(r/a)²" : 2,
    "(r/a)⁴" : 4,
    "sph"    : -1,   # spherical wave: ∝ 1/√(z₀²+r²), normalised to peak=1
}

MODES = ["circular", "cartesian"]

# ── numerical ─────────────────────────────────────────────────────────────────
N_R         = 2000    # UNIFORM r-grid: accurate Simpson quadrature at aperture edge
N_Q         = 1200    # focus-refined q-grid: resolves PSF central peak
FOCUS_POWER = 2.3     # q = q_max · u^FOCUS_POWER  (clusters near q=0)

# 2-D map viewing window: ≈ 4 first-Airy-radii at a/R=0.95
# q1 = 3.83/a = 3.83/(0.95·1.6) = 2.52 mm⁻¹
# q_max = ni/(lam·zi)·R = 2256 mm⁻¹
# VIEW_FRAC = 4·q1/q_max = 0.0045 → q_view = 10.1 mm⁻¹
VIEW_FRAC = 0.0045
N_IMG     = 512

# Subset of A_FRACS used in the spot-size annotated figure
SPOT_AFS  = [0.35, 0.65, 0.80, 0.95]

OUT_DIR = Path(__file__).resolve().parent
HT = HankelTransform.transform_array


# ─────────────────────────────────────────────────────────────────────────────

def make_r_grid():
    """Uniform grid — accurate Simpson quadrature for aperture-edge integrands."""
    return np.linspace(0.0, R, N_R)


def make_q_grid(q_max):
    """Focus-refined grid — dense near q=0 to resolve PSF central peak."""
    return q_max * np.linspace(0.0, 1.0, N_Q) ** FOCUS_POWER


def sanitize(arr, r):
    arr = np.asarray(arr, dtype=float)
    bad = ~np.isfinite(arr)
    if bad.any():
        arr[bad] = np.interp(r[bad], r[~bad], arr[~bad])
    return arr


def input_field(r, a, power, z0_sph):
    mask = (r <= a).astype(float)
    if power == 0:
        return mask
    if power == -1:
        amp = 1.0 / np.sqrt(z0_sph ** 2 + r ** 2)
        amp /= amp.max() + 1e-30
        return amp * mask
    return (r / a) ** power * mask


# ── propagation (Circular, pure EL) ──────────────────────────────────────────
#   EL' = 2π H₀[(tp+ts)·EL]            (diagonal, bright centre)
#   ER' = −2π e^{-2iφ} H₂[(tp-ts)·EL]  (non-diagonal, dark centre)
#
# ── propagation (Cartesian, pure Ex) ─────────────────────────────────────────
#   Ex' = H₀[(tp+ts)·Ex] − cos(2φ) H₂[(tp-ts)·Ex]
#   Ey' =               − sin(2φ) H₂[(tp-ts)·Ex]   (non-diagonal)
#
# ── E_z (longitudinal, paraxial) ─────────────────────────────────────────────
#   E_z_surface = (r/zᵢ)·tp·|E|/√2   propagated via H₁

def propagate(r, q, tp, ts, E, mode, _zi):
    h0 = HT(0, r, (tp + ts) * E, q)
    h2 = HT(2, r, (tp - ts) * E, q)
    Ez_surface_r = tp * (r / _zi) * (np.abs(E) / np.sqrt(2))
    Ez_env = 2 * π * HT(1, r, Ez_surface_r, q)
    if mode == "circular":
        return 2 * π * h0, 2 * π * h2, Ez_env, Ez_surface_r
    return h0, h2, Ez_env, Ez_surface_r


def radial_to_2d(profile, q, rho):
    return np.interp(rho, q, np.abs(profile),
                     left=float(np.abs(profile[0])), right=0.0)


def make_2d_mesh(q_half):
    xs = np.linspace(-q_half, q_half, N_IMG)
    xx, yy = np.meshgrid(xs, xs)
    return np.sqrt(xx**2 + yy**2), np.arctan2(yy, xx), [-q_half, q_half, -q_half, q_half]


def intensities_2d(H0_env, H2_env, Ez_env, q, mode, rho, phi):
    h0 = radial_to_2d(H0_env, q, rho)
    h2 = radial_to_2d(H2_env, q, rho)
    ez = radial_to_2d(Ez_env,  q, rho)
    if mode == "circular":
        I_diag    = h0 ** 2
        I_nondiag = h2 ** 2
        I_H0only  = I_diag
        I_Ez      = ez ** 2
    else:
        Ex = h0 - np.cos(2 * phi) * h2
        Ey =    - np.sin(2 * phi) * h2
        I_diag    = Ex ** 2
        I_nondiag = Ey ** 2
        I_H0only  = h0 ** 2
        I_Ez      = ez ** 2 * np.cos(phi) ** 2
    return I_diag + I_nondiag + I_Ez, I_diag, I_nondiag, I_H0only


def half_power_radius(I_1d, q):
    dq  = np.gradient(q)
    cum = np.cumsum(I_1d * q * dq)
    if cum[-1] <= 0:
        return np.nan
    return q[min(np.searchsorted(cum, 0.5 * cum[-1]), len(q) - 1)]


def energies_from_r(r, tp, ts, E, Ez_surface_r, mode):
    """Parseval — avoids q-grid sampling bias for H₂ energy."""
    fac = (2 * π) ** 2 if mode == "circular" else 1.0
    E_H0 = fac          * float(np.trapezoid(np.abs((tp + ts) * E) ** 2 * r, r))
    E_H2 = fac          * float(np.trapezoid(np.abs((tp - ts) * E) ** 2 * r, r))
    E_Ez = (2*π)**2 * float(np.trapezoid(np.abs(Ez_surface_r)    ** 2 * r, r))
    return E_H0, E_H2, E_Ez


def metrics_1d(H0_env, H2_env, Ez_env, q, mode, E_H0, E_H2, E_Ez):
    I_h0 = np.abs(H0_env) ** 2
    I_h2 = np.abs(H2_env) ** 2
    I_ez = np.abs(Ez_env)  ** 2
    E_total   = E_H0 + E_H2 + E_Ez
    E_nondiag = E_H2 if mode == "circular" else 0.5 * E_H2
    E_diag    = E_total - E_nondiag
    I_H0only_1d = I_h0
    I_full_1d   = I_h0 + I_h2 + I_ez
    w_H0only = half_power_radius(I_H0only_1d, q)
    w_full   = half_power_radius(I_full_1d,   q)
    return {
        "eta_nondiag" : E_nondiag / (E_diag   + 1e-30),
        "eta_H2"      : E_H2      / (E_total  + 1e-30),
        "eta_Ez"      : E_Ez      / (E_total  + 1e-30),
        "w_H0only"    : w_H0only,
        "w_full"      : w_full,
        "width_ratio" : w_full / (w_H0only + 1e-30),
        "I_H0only_1d" : I_H0only_1d,
        "I_full_1d"   : I_full_1d,
    }


def compute_one(a_frac, prof_power, mode, z0_val=None, zi_val=None):
    _z0 = z0_val if z0_val is not None else z0
    _zi = zi_val if zi_val is not None else zi
    a     = a_frac * R
    q_max = ni / (lam * _zi) * R
    r     = make_r_grid()
    q     = make_q_grid(q_max)
    ts_, tp_ = FresnelOvoidParax(n0, ni, _z0, _zi).coeffs(r)
    tp = sanitize(np.asarray(tp_, float), r)
    ts = sanitize(np.asarray(ts_, float), r)
    E  = input_field(r, a, prof_power, _z0)
    H0_env, H2_env, Ez_env, Ez_surface_r = propagate(r, q, tp, ts, E, mode, _zi)
    E_H0, E_H2, E_Ez = energies_from_r(r, tp, ts, E, Ez_surface_r, mode)
    m = metrics_1d(H0_env, H2_env, Ez_env, q, mode, E_H0, E_H2, E_Ez)
    return {
        "theta"  : np.degrees(np.arctan(a / _zi)),
        "a_frac" : a_frac,
        "z0"     : _z0,
        "zi"     : _zi,
        "q_max"  : q_max,
        "q"      : q,
        "H0_env" : H0_env,
        "H2_env" : H2_env,
        "Ez_env" : Ez_env,
        **m,
    }


def compute_scalar(a_frac):
    """
    Scalar Fraunhofer–Hankel diffraction: ts=tp=1, H₀ only, no H₂, no E_z.

    With ts=tp=1 the Fresnel transmission is uniform (no dioptric boundary).
    The H₂ driver (tp−ts)=0 vanishes exactly, giving a pure Airy-type PSF.
    """
    a     = a_frac * R
    q_max = ni / (lam * zi) * R
    r     = make_r_grid()
    q     = make_q_grid(q_max)
    E     = input_field(r, a, 0, z0)          # flat aperture, same as vectorial
    # tp+ts = 2, tp−ts = 0  →  pure H₀
    H0_env = 2.0 * π * HT(0, r, 2.0 * E, q)
    I_1d   = np.abs(H0_env) ** 2
    return {
        "theta"  : np.degrees(np.arctan(a / zi)),
        "q"      : q,
        "q_max"  : q_max,
        "H0_env" : H0_env,
        "I_1d"   : I_1d,
        "w"      : half_power_radius(I_1d, q),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

angle_data = {}
print("Sweep A — marginal angle  (flat, vary a/R) …")
total = len(MODES) * len(A_FRACS);  done = 0
for mode in MODES:
    for af in A_FRACS:
        d = compute_one(af, 0, mode)
        angle_data[(mode, af)] = d;  done += 1
        print(f"  [{done}/{total}] {mode:10s}  a/R={af:.2f}  "
              f"θ={d['theta']:.1f}°  η_H2={d['eta_H2']:.4f}  "
              f"η_nd={d['eta_nondiag']:.4f}  wr={d['width_ratio']:.4f}")

profile_data = {}
print(f"\nSweep B — profiles  (a/R=0.95) …")
total = len(MODES) * len(PROFILES);  done = 0
for mode in MODES:
    for pname, ppower in PROFILES.items():
        d = compute_one(0.95, ppower, mode)
        profile_data[(mode, pname)] = d;  done += 1
        print(f"  [{done}/{total}] {mode:10s}  {pname:8s}  "
              f"η_H2={d['eta_H2']:.4f}  η_nd={d['eta_nondiag']:.4f}  "
              f"wr={d['width_ratio']:.4f}")

z0_data = {}
print(f"\nSweep C — z₀ sweep  (a/R=0.95, flat, zᵢ={zi}) …")
total = len(MODES) * len(Z0_VALUES);  done = 0
for mode in MODES:
    for z0v in Z0_VALUES:
        d = compute_one(0.95, 0, mode, z0_val=z0v)
        z0_data[(mode, z0v)] = d;  done += 1
        print(f"  [{done}/{total}] {mode:10s}  z₀={z0v:.2f}  "
              f"η_H2={d['eta_H2']:.4f}  wr={d['width_ratio']:.4f}")

print("\nData collection complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

CMAP_INT  = "hot"
CMAP_DIFF = "RdBu_r"

n_prof = len(PROFILES)
n_afr  = len(A_FRACS)
n_z0   = len(Z0_VALUES)

PROF_COLORS = [plt.cm.tab10(i)                              for i in range(n_prof)]
AF_COLORS   = plt.cm.viridis(np.linspace(0.15, 0.90, n_afr))
Z0_COLORS   = plt.cm.plasma( np.linspace(0.15, 0.90, n_z0))

MODE_LABELS    = {"circular": "Circular  (pure EL)",  "cartesian": "Cartesian  (pure Ex)"}
DIAG_LABELS    = {"circular": r"|EL'|²",              "cartesian": r"|Ex'|²"}
NONDIAG_LABELS = {"circular": r"|ER'|²  (non-diag)",  "cartesian": r"|Ey'|²  (non-diag)"}


def norm_vmax(img, pct=99.5):
    v = float(np.percentile(img[np.isfinite(img)], pct))
    return v if v > 0 else 1.0


def show_map(ax, img, extent, title, cmap=CMAP_INT, vmin=0, vmax=None):
    vmax = vmax or norm_vmax(img)
    im = ax.imshow(img, extent=extent, origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x [q]", fontsize=7);  ax.set_ylabel("y [q]", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def draw_spot_annotation(ax, w, color, y_ann, q_view, label_str):
    """
    Draws a caliper-style dimension-line annotation:

        |←—————  label  —————→|

    The measurement label is centred on the arrow in a white rounded box,
    interrupting the line cleanly (technical-drawing convention).
    Vertical serif ticks are placed at both endpoints for precision.
    """
    if not (np.isfinite(w) and 0 < w <= q_view):
        return
    t = 0.028 * q_view                          # serif tick half-height
    # Double-headed arrow spanning the full measured diameter
    ax.annotate("", xy=(w, y_ann), xytext=(-w, y_ann),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.35,
                               mutation_scale=8, shrinkA=0, shrinkB=0))
    # Vertical serif ticks at both ends
    for x_e in (-w, w):
        ax.plot([x_e, x_e], [y_ann - t, y_ann + t],
                color=color, lw=1.8, solid_capstyle="butt", zorder=5)
    # Measurement label — white box sits on the arrow line, interrupting it
    ax.text(0, y_ann, label_str,
            ha="center", va="center",
            color=color, fontsize=7.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=color,
                      alpha=0.96, lw=1.0),
            zorder=6)


# Shared geometry for the "steepest angle" overview maps
_HIGH_AF     = A_FRACS[-1]
_d_steep     = angle_data[("circular", _HIGH_AF)]
_theta_steep = _d_steep["theta"]
_q_half      = VIEW_FRAC * _d_steep["q_max"]
_rho, _phi, _ext = make_2d_mesh(_q_half)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 0 — Fresnel coefficient diagnostics
# ─────────────────────────────────────────────────────────────────────────────

_r_diag = make_r_grid()
_F = FresnelOvoidParax(n0, ni, z0, zi)
_ts_d, _tp_d = [np.asarray(x, float) for x in _F.coeffs(_r_diag)]

fig0, ax0 = plt.subplots(1, 2, figsize=(12, 4.5))

ax0[0].plot(_r_diag / R, _tp_d, lw=2, color="tab:blue",   label=r"$t_p$")
ax0[0].plot(_r_diag / R, _ts_d, lw=2, color="tab:orange", label=r"$t_s$")
ax0[0].axhline(0, color="k", lw=0.7, linestyle=":")
for af in A_FRACS:
    ax0[0].axvline(af, color="gray", lw=0.4, alpha=0.6)
ax0[0].set_xlabel("r / R");  ax0[0].set_ylabel("value")
ax0[0].set_title(r"$t_p$,  $t_s$  vs radius  (grey: a/R sweep values)")
ax0[0].legend(fontsize=9);  ax0[0].grid(True, alpha=0.3)

ax0[1].plot(_r_diag / R, _tp_d + _ts_d, lw=2, color="tab:green", label=r"$t_p+t_s$  [H₀ driver]")
ax0[1].plot(_r_diag / R, _tp_d - _ts_d, lw=2, color="tab:red",   label=r"$t_p-t_s$  [H₂ driver]")
ax0[1].axhline(0, color="k", lw=0.7, linestyle=":")
for af in A_FRACS:
    ax0[1].axvline(af, color="gray", lw=0.4, alpha=0.6)
ax0[1].set_xlabel("r / R");  ax0[1].set_ylabel("value")
ax0[1].set_title(r"$(t_p \pm t_s)$  vs radius")
ax0[1].legend(fontsize=9);  ax0[1].grid(True, alpha=0.3)

fig0.suptitle(f"Fig 0 — Fresnel coefficient check\n{PARAM_STR}\n"
              r"$t_p$ remains positive ($>0$) for all $r \leq R$ — paraxial model valid",
              fontsize=10)
fig0.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# Fig 0-B — Incident field profiles
# ─────────────────────────────────────────────────────────────────────────────
# One wide 1-D panel showing all five profiles normalised to peak, plus five
# small 2-D spatial maps so the reader sees exactly what each illumination
# pattern looks like on the entrance pupil.
# ─────────────────────────────────────────────────────────────────────────────

_a_ref  = 0.95 * R
_r_prof = make_r_grid()

_MATH_LABELS = {
    "flat"   : r"$E(r) = 1$",
    "r/a"    : r"$E(r) = r/a$",
    "(r/a)²" : r"$E(r) = (r/a)^2$",
    "(r/a)⁴" : r"$E(r) = (r/a)^4$",
    "sph"    : r"$E(r) \propto 1/\sqrt{z_0^2+r^2}$",
}

_profile_E = {name: input_field(_r_prof, _a_ref, pw, z0)
              for name, pw in PROFILES.items()}

fig0b, ax0b = plt.subplots(
    1, 6, figsize=(18, 4.6), facecolor="white",
    gridspec_kw={"width_ratios": [3, 1, 1, 1, 1, 1], "wspace": 0.38})

# ── 1-D radial profiles ───────────────────────────────────────────────────────
ax_1d = ax0b[0]
for ki, (name, pw) in enumerate(PROFILES.items()):
    E_p = _profile_E[name]
    E_n = E_p / (E_p.max() + 1e-30)
    ax_1d.plot(_r_prof / R, E_n, lw=2.5, color=PROF_COLORS[ki],
               label=_MATH_LABELS[name] + f"  ({name})")
ax_1d.axvline(_a_ref / R, color="k", lw=1.4, ls="--", alpha=0.65,
              label=r"aperture edge  $r = a$")
ax_1d.set_xlabel(r"$r\,/\,R$", fontsize=10)
ax_1d.set_ylabel(r"$E(r)\,/\,E_{\rm max}$", fontsize=10)
ax_1d.set_title(r"All input profiles  ($a/R = 0.95$, normalised to peak)", fontsize=9)
ax_1d.set_xlim(0, 1.0);  ax_1d.set_ylim(-0.05, 1.13)
ax_1d.legend(fontsize=8.5, loc="upper left");  ax_1d.grid(True, alpha=0.3)

# ── 2-D spatial maps ──────────────────────────────────────────────────────────
_xs_p        = np.linspace(-R, R, N_IMG)
_XX_p, _YY_p = np.meshgrid(_xs_p, _xs_p)
_RR_p        = np.sqrt(_XX_p ** 2 + _YY_p ** 2)
_ext_p       = [-R, R, -R, R]

for ki, (name, pw) in enumerate(PROFILES.items()):
    ax = ax0b[ki + 1]
    E_p  = _profile_E[name]
    E_n  = E_p / (E_p.max() + 1e-30)
    E_2d = np.interp(_RR_p, _r_prof, E_n, left=float(E_n[0]), right=0.0)
    im   = ax.imshow(E_2d, extent=_ext_p, origin="lower",
                     cmap="hot", vmin=0, vmax=1.0, aspect="equal")
    ax.set_title(_MATH_LABELS[name] + f"\n({name})", fontsize=8)
    ax.set_xlabel(r"$x$ [mm]", fontsize=7.5)
    ax.set_ylabel(r"$y$ [mm]", fontsize=7.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig0b.suptitle(
    r"Fig 0-B — Incident field profiles on the entrance pupil  ($a/R = 0.95$)" + "\n"
    f"{PARAM_STR}",
    fontsize=10)
fig0b.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# STUDY 1 — Non-diagonal contamination
# ─────────────────────────────────────────────────────────────────────────────

fig1a, ax1a = plt.subplots(1, 2, figsize=(11, 4.5))
for col, mode in enumerate(MODES):
    ax = ax1a[col]
    thetas = [angle_data[(mode, af)]["theta"]       for af in A_FRACS]
    etas   = [angle_data[(mode, af)]["eta_nondiag"] for af in A_FRACS]
    ax.plot(thetas, etas, "o-", lw=2, color="tab:blue")
    ax.set_xlabel("Marginal angle  θ  [deg]")
    ax.set_ylabel(r"$\eta_\mathrm{nd}$ = $E_\mathrm{non-diag}$ / $E_\mathrm{diag}$")
    ax.set_title(f"Non-diagonal contamination — {mode}")
    ax.grid(True, alpha=0.3)
fig1a.suptitle(f"Study 1 — Non-diagonal energy fraction vs marginal angle\n{PARAM_STR}", fontsize=10)
fig1a.tight_layout()

fig1b, ax1b = plt.subplots(1, 2, figsize=(11, 4.5))
for col, mode in enumerate(MODES):
    ax = ax1b[col]
    for pi, (pname, _) in enumerate(PROFILES.items()):
        ax.bar(pi, profile_data[(mode, pname)]["eta_nondiag"],
               color=PROF_COLORS[pi], alpha=0.8, label=pname)
    ax.set_xticks(range(n_prof))
    ax.set_xticklabels(list(PROFILES.keys()), fontsize=8)
    ax.set_ylabel(r"$\eta_\mathrm{nd}$")
    ax.set_title(f"Non-diagonal fraction by profile — {mode}")
    ax.set_xlabel("profile  (sph = 1/√(z₀²+r²))", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
fig1b.suptitle(f"Study 1 — Non-diagonal fraction by profile\n{PARAM_STR}", fontsize=10)
fig1b.tight_layout()

fig1c, ax1c = plt.subplots(2, 4, figsize=(16, 7))
for row, mode in enumerate(MODES):
    d = angle_data[(mode, _HIGH_AF)]
    I_full, I_diag, I_nondiag, I_H0only = intensities_2d(
        d["H0_env"], d["H2_env"], d["Ez_env"], d["q"], mode, _rho, _phi)
    vd = norm_vmax(I_diag)
    show_map(ax1c[row, 0], I_diag,    _ext, DIAG_LABELS[mode])
    show_map(ax1c[row, 1], I_nondiag, _ext, NONDIAG_LABELS[mode])
    show_map(ax1c[row, 2], I_full,    _ext, r"$I_\mathrm{full}$  (diag+nd+$E_z$)")
    ratio = np.where(I_diag > 1e-12 * vd, (I_diag - I_nondiag) / (I_diag + 1e-30), 0.0)
    show_map(ax1c[row, 3], ratio, _ext,
             r"$(I_\mathrm{diag} - I_\mathrm{nd})\,/\,I_\mathrm{diag}$", cmap="hot")
    ax1c[row, 0].set_ylabel(MODE_LABELS[mode], fontsize=8)
fig1c.suptitle(
    f"Study 1 — 2-D maps  (a/R={_HIGH_AF}, θ≈{_theta_steep:.1f}°, flat)\n{PARAM_STR}", fontsize=10)
fig1c.tight_layout()

fig1d, ax1d = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for col, mode in enumerate(MODES):
    ax = ax1d[col]
    thetas = [angle_data[(mode, af)]["theta"]       for af in A_FRACS]
    ratios = [angle_data[(mode, af)]["width_ratio"] for af in A_FRACS]
    ax.plot(thetas, ratios, "o-", lw=2, color="tab:orange")
    ax.axhline(1.0, color="k", lw=0.8, linestyle=":")
    ax.set_xlabel("Marginal angle  θ  [deg]")
    ax.set_ylabel(r"$w_\mathrm{full}$ / $w_\mathrm{H0only}$")
    ax.set_title(f"Bright-zone widening — {mode}")
    ax.grid(True, alpha=0.3)
fig1d.suptitle(f"Study 1 — Bright-zone widening vs marginal angle\n{PARAM_STR}", fontsize=10)
fig1d.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# STUDY 2 — H₂ vs H₀
# ─────────────────────────────────────────────────────────────────────────────

fig2a, ax2a = plt.subplots(1, 2, figsize=(11, 4.5))
for col, mode in enumerate(MODES):
    ax = ax2a[col]
    thetas = [angle_data[(mode, af)]["theta"]  for af in A_FRACS]
    etas   = [angle_data[(mode, af)]["eta_H2"] for af in A_FRACS]
    ax.plot(thetas, etas, "o-", lw=2, color="tab:red")
    ax.set_xlabel("Marginal angle  θ  [deg]")
    ax.set_ylabel(r"$\eta_\mathrm{H2}$ = $E_\mathrm{H2}$ / $E_\mathrm{total}$")
    ax.set_title(f"H₂ energy fraction — {mode}")
    ax.grid(True, alpha=0.3)
fig2a.suptitle(f"Study 2 — H₂ energy fraction vs marginal angle\n{PARAM_STR}", fontsize=10)
fig2a.tight_layout()

fig2b, ax2b = plt.subplots(1, 2, figsize=(11, 4.5))
for col, mode in enumerate(MODES):
    ax = ax2b[col]
    for pi, (pname, _) in enumerate(PROFILES.items()):
        ax.bar(pi, profile_data[(mode, pname)]["eta_H2"],
               color=PROF_COLORS[pi], alpha=0.8, label=pname)
    ax.set_xticks(range(n_prof))
    ax.set_xticklabels(list(PROFILES.keys()), fontsize=8)
    ax.set_ylabel(r"$\eta_\mathrm{H2}$")
    ax.set_title(f"H₂ fraction by profile — {mode}")
    ax.set_xlabel("profile  (sph = 1/√(z₀²+r²))", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
fig2b.suptitle(f"Study 2 — H₂ fraction by profile\n{PARAM_STR}", fontsize=10)
fig2b.tight_layout()

# ── Fig 2-C: normalised radial PSF with per-case xlim and half-power markers ─

fig2c, ax2c = plt.subplots(2, n_afr, figsize=(3.2 * n_afr, 7), sharey="row")
for col, af in enumerate(A_FRACS):
    # Airy-disk scale for this aperture: q1 = 3.83/a
    q1_case = 3.83 / (af * R)
    xlim_q  = 5.0 * q1_case
    for row, mode in enumerate(MODES):
        ax = ax2c[row, col]
        d  = angle_data[(mode, af)]
        q_ = d["q"]
        mk = q_ <= xlim_q
        I0 = d["I_H0only_1d"]
        If = d["I_full_1d"]
        I0n = I0 / (I0[mk].max() + 1e-30)
        Ifn = If / (If[mk].max() + 1e-30)
        ax.plot(q_[mk], I0n[mk], lw=2, color="tab:blue",   label="H₀ only")
        ax.plot(q_[mk], Ifn[mk], lw=2, color="tab:orange", ls="--", label="Full")
        ax.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.7)
        w0, wf = d["w_H0only"], d["w_full"]
        if np.isfinite(w0) and w0 <= xlim_q:
            ax.axvline(w0, color="tab:blue",   lw=1.5, alpha=0.7)
        if np.isfinite(wf) and wf <= xlim_q:
            ax.axvline(wf, color="tab:orange", lw=1.5, ls="--", alpha=0.7)
        ax.set_xlim(0, xlim_q)
        ax.set_title(f"θ={d['theta']:.1f}°\nwr={d['width_ratio']:.3f}", fontsize=7)
        if col == 0:
            ax.set_ylabel(f"{mode}\nI / I_peak", fontsize=8)
        if row == 1:
            ax.set_xlabel("q  [mm⁻¹]", fontsize=8)
        if col == n_afr - 1 and row == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
fig2c.suptitle(
    "Study 2 — Normalised PSF: H₀-only (blue solid) vs Full (orange dashed)\n"
    "Vertical bars = half-power radii  |  wr = w_full / w_H0only\n"
    f"{PARAM_STR}", fontsize=9)
fig2c.tight_layout()

# ── Fig 2-D: 2-D maps at steepest angle ──────────────────────────────────────

fig2d, ax2d = plt.subplots(2, 3, figsize=(13, 7))
for row, mode in enumerate(MODES):
    d = angle_data[(mode, _HIGH_AF)]
    I_full, I_diag, I_nondiag, I_H0only = intensities_2d(
        d["H0_env"], d["H2_env"], d["Ez_env"], d["q"], mode, _rho, _phi)
    diff = I_full - I_H0only
    vs = norm_vmax(np.maximum(I_H0only, I_full))
    vd = norm_vmax(np.abs(diff))
    show_map(ax2d[row, 0], I_H0only, _ext, r"$I_\mathrm{H0only}$  (scalar-like)", vmax=vs)
    show_map(ax2d[row, 1], I_full,   _ext, r"$I_\mathrm{full}$  (H₀+H₂+$E_z$)",  vmax=vs)
    show_map(ax2d[row, 2], diff,     _ext, r"$I_\mathrm{full} - I_\mathrm{H0only}$",
             cmap=CMAP_DIFF, vmin=-vd, vmax=vd)
    ax2d[row, 0].set_ylabel(MODE_LABELS[mode], fontsize=8)
fig2d.suptitle(
    f"Study 2 — 2-D maps  (a/R={_HIGH_AF}, θ≈{_theta_steep:.1f}°)\n{PARAM_STR}", fontsize=10)
fig2d.tight_layout()

# ── Fig 2-E: summary curves ───────────────────────────────────────────────────

fig2e, ax2e = plt.subplots(2, 2, figsize=(11, 8))
for col, mode in enumerate(MODES):
    thetas = [angle_data[(mode, af)]["theta"]       for af in A_FRACS]
    wr     = [angle_data[(mode, af)]["width_ratio"] for af in A_FRACS]
    eh2    = [angle_data[(mode, af)]["eta_H2"]      for af in A_FRACS]
    end    = [angle_data[(mode, af)]["eta_nondiag"] for af in A_FRACS]
    ax2e[0, col].plot(thetas, wr, "o-", lw=2, color="tab:orange")
    ax2e[0, col].axhline(1.0, color="k", lw=0.8, ls=":")
    ax2e[0, col].set_ylabel(r"$w_\mathrm{full}$ / $w_\mathrm{H0only}$")
    ax2e[0, col].set_title(f"Spot widening — {mode}")
    ax2e[1, col].plot(thetas, eh2, "o-",  lw=2, color="tab:red",  label=r"$\eta_\mathrm{H2}$")
    ax2e[1, col].plot(thetas, end, "s--", lw=2, color="tab:blue", label=r"$\eta_\mathrm{nd}$")
    ax2e[1, col].set_ylabel("Energy fraction")
    ax2e[1, col].set_title(f"H₂ and non-diagonal fractions — {mode}")
    ax2e[1, col].legend(fontsize=8)
    for r2 in range(2):
        ax2e[r2, col].set_xlabel("Marginal angle  θ  [deg]")
        ax2e[r2, col].grid(True, alpha=0.3)
fig2e.suptitle(
    f"Study 2 — Summary: spot widening and energy fractions vs marginal angle\n{PARAM_STR}",
    fontsize=10)
fig2e.tight_layout()

# ── Fig 2-F: spot-size annotated 2-D PSF ─────────────────────────────────────
# For each selected a/R and both polarisation modes, show I_full with:
#   • Solid circle at the H₀-only 50%-EE radius  (sky-blue)
#   • Dashed circle at the full-PSF 50%-EE radius (amber)
#   • Caliper-style dimension lines: |←— 2w —→| with serif ticks and
#     a labelled white box sitting centred on each measurement bar.
# Viewing window is scaled per panel to keep ≈ 3.5 Airy radii visible.

_C_H0   = "#3BB8F0"   # sky-blue  — H₀-only circle & annotation
_C_FULL = "#F0993B"   # amber     — full-PSF circle & annotation

n_spot = len(SPOT_AFS)
fig2f, ax2f = plt.subplots(2, n_spot, figsize=(4.8 * n_spot, 10.5),
                            facecolor="white")

for row, mode in enumerate(MODES):
    for col, af in enumerate(SPOT_AFS):
        ax = ax2f[row, col]
        d  = angle_data[(mode, af)]
        q_ = d["q"]

        # Per-panel view: 3.5 × first-Airy-radius so circles are prominent
        q1c = 3.83 / (af * R)
        qv  = 3.5 * q1c
        rho2, phi2, ext2 = make_2d_mesh(qv)
        I_full, _, _, _ = intensities_2d(
            d["H0_env"], d["H2_env"], d["Ez_env"], q_, mode, rho2, phi2)

        im = ax.imshow(I_full, extent=ext2, origin="lower",
                       cmap=CMAP_INT, vmin=0, vmax=norm_vmax(I_full),
                       aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        w0 = d["w_H0only"];  wf = d["w_full"];  wr = d["width_ratio"]

        # Half-power radius circles
        if np.isfinite(w0) and w0 <= qv:
            ax.add_patch(plt.Circle((0, 0), w0, color=_C_H0,
                                    fill=False, lw=2.2, ls="-",  zorder=4))
        if np.isfinite(wf) and wf <= qv:
            ax.add_patch(plt.Circle((0, 0), wf, color=_C_FULL,
                                    fill=False, lw=2.2, ls="--", zorder=4))

        # Caliper annotations at two y-levels in the dark outer region
        y_h0   = -0.60 * qv
        y_full = -0.78 * qv
        draw_spot_annotation(ax, w0, _C_H0,   y_h0,   qv,
                             fr"$2w_{{H_0}} = {2*w0:.3f}$")
        draw_spot_annotation(ax, wf, _C_FULL,  y_full, qv,
                             fr"$2w_{{\rm f}} = {2*wf:.3f}$")

        # Delicate cross-hair at the optical axis
        ax.axhline(0, color="white", lw=0.6, ls=":", alpha=0.40, zorder=3)
        ax.axvline(0, color="white", lw=0.6, ls=":", alpha=0.40, zorder=3)

        # Title: θ, a/R on first line; width ratio on second
        ax.set_title(
            fr"$\theta = {d['theta']:.1f}^\circ$   "
            fr"$a/R = {af}$"
            "\n"
            r"$w_{\rm f}/w_{H_0} = $" + f"{wr:.3f}",
            fontsize=8.5, pad=5)
        ax.set_xlabel(r"$x\;\left[\mathrm{mm}^{-1}\right]$", fontsize=8)
        if col == 0:
            ax.set_ylabel(
                f"{MODE_LABELS[mode]}\n"
                r"$y\;\left[\mathrm{mm}^{-1}\right]$",
                fontsize=8)

# Shared legend below the grid
from matplotlib.lines import Line2D
_legend_items = [
    Line2D([0], [0], color=_C_H0,   lw=2.2, ls="-",
           label=r"$w_{H_0}$ — 50%-enc.-energy radius, $H_0$-only PSF"),
    Line2D([0], [0], color=_C_FULL, lw=2.2, ls="--",
           label=r"$w_{\rm f}$ — 50%-enc.-energy radius, full PSF ($H_0+H_2+E_z$)"),
]
fig2f.legend(handles=_legend_items, loc="lower center", ncol=2,
             fontsize=9, frameon=True, framealpha=0.96,
             edgecolor="#CCCCCC", bbox_to_anchor=(0.5, 0.0))

fig2f.suptitle(
    r"Study 2 — Spot-size annotated PSF  ($I_{\rm full}$ map)" + "\n"
    r"$|\!\longleftarrow\,2w_{1/2}\,\longrightarrow\!|$"
    r"  caliper lines show 50%-encircled-energy diameter" + "\n"
    f"{PARAM_STR}",
    fontsize=10)
fig2f.tight_layout(rect=[0, 0.055, 1, 1])


# ─────────────────────────────────────────────────────────────────────────────
# STUDY 3 — Source-distance (z₀) sweep
# ─────────────────────────────────────────────────────────────────────────────

Z0_STR = f"n₀={n0}, nᵢ={ni}, zᵢ={zi}, R={R} mm, z₀ varies — paraxial Fresnel"

fig3a, ax3a = plt.subplots(1, 2, figsize=(11, 4.5))

for col, mode in enumerate(MODES):
    ax = ax3a[col]
    ax.plot(Z0_VALUES, [z0_data[(mode, v)]["eta_H2"] for v in Z0_VALUES],
            "o-", lw=2, color="tab:red")
    ax.set_xlabel("Source distance  z₀  [mm]")
    ax.set_ylabel(r"$\eta_\mathrm{H2}$")
    ax.set_title(f"H₂ fraction vs z₀ — {mode}")
    ax.grid(True, alpha=0.3)


    
fig3a.suptitle(f"Study 3 — H₂ fraction vs source distance\n{Z0_STR}", fontsize=10)
fig3a.tight_layout()

fig3b, ax3b = plt.subplots(1, 2, figsize=(11, 4.5))
for col, mode in enumerate(MODES):
    ax = ax3b[col]
    ax.plot(Z0_VALUES, [z0_data[(mode, v)]["width_ratio"] for v in Z0_VALUES],
            "o-", lw=2, color="tab:orange")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Source distance  z₀  [mm]")
    ax.set_ylabel(r"$w_\mathrm{full}$ / $w_\mathrm{H0only}$")
    ax.set_title(f"Spot widening vs z₀ — {mode}")
    ax.grid(True, alpha=0.3)
fig3b.suptitle(f"Study 3 — Spot widening vs source distance\n{Z0_STR}", fontsize=10)
fig3b.tight_layout()

# Normalised radial profiles per z₀ value
_qr  = z0_data[(MODES[0], Z0_VALUES[0])]["q"]
_qmr = z0_data[(MODES[0], Z0_VALUES[0])]["q_max"]
_q1r = 3.83 / (0.95 * R)
_mk  = _qr <= 5.0 * _q1r

fig3c, ax3c = plt.subplots(1, 2, figsize=(12, 5))
for col, mode in enumerate(MODES):
    ax = ax3c[col]
    d0  = z0_data[(mode, z0)]           # H₀-only reference at default z₀
    I0  = d0["I_H0only_1d"]
    ax.plot(_qr[_mk], I0[_mk] / (I0[_mk].max() + 1e-30),
            lw=2, color="k", ls=":", label=f"H₀ only  (z₀={z0})")
    for ki, z0v in enumerate(Z0_VALUES):
        d   = z0_data[(mode, z0v)]
        If  = d["I_full_1d"]
        wr  = d["width_ratio"]
        ax.plot(_qr[_mk], If[_mk] / (If[_mk].max() + 1e-30),
                lw=2, color=Z0_COLORS[ki],
                label=f"z₀={z0v:.2f}  wr={wr:.3f}")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("q  [mm⁻¹]");  ax.set_ylabel("I / I_peak")
    ax.set_title(f"Normalised PSF vs z₀ — {mode}")
    ax.legend(fontsize=7);  ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5.0 * _q1r)
fig3c.suptitle(f"Study 3 — PSF evolution with z₀  (a/R=0.95, flat)\n{Z0_STR}", fontsize=10)
fig3c.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Scalar vs Vectorial theory comparison
# ─────────────────────────────────────────────────────────────────────────────
# Scalar   : ts=tp=1, I = |H₀[E]|²                  (Fraunhofer/Airy limit)
# Vectorial: physical Fresnel ts(r),tp(r), I = |H₀|² + |H₂|² + |E_z|²
# Mode: circular polarisation (pure EL input).
# Layout: 3 rows × n_spot columns
#   Row 0 — Scalar 2-D PSF map  (normalised to scalar peak per column)
#   Row 1 — Vectorial 2-D PSF map  (same vmax → broadening is visible)
#   Row 2 — Normalised 1-D radial profiles overlaid
# ─────────────────────────────────────────────────────────────────────────────

print("\nFig 4 — computing scalar references (ts=tp=1) …")
scalar_data_f4 = {af: compute_scalar(af) for af in SPOT_AFS}

_C_SC  = "#3A78C9"   # sapphire  — scalar
_C_VEC = "#D9531E"   # vermilion — vectorial

n_s4   = len(SPOT_AFS)
fig4, ax4 = plt.subplots(3, n_s4,
                          figsize=(4.8 * n_s4, 14.5),
                          facecolor="white")

for col, af in enumerate(SPOT_AFS):
    ds  = scalar_data_f4[af]
    dv  = angle_data[("circular", af)]

    q_s = ds["q"]
    q_v = dv["q"]
    q1c = 3.83 / (af * R)
    qv  = 3.5 * q1c

    rho2, phi2, ext2 = make_2d_mesh(qv)

    # ── 2-D intensity maps ────────────────────────────────────────────
    I_sc_2d = radial_to_2d(ds["H0_env"], q_s, rho2) ** 2

    I_v_2d, _, _, _ = intensities_2d(
        dv["H0_env"], dv["H2_env"], dv["Ez_env"], q_v, "circular", rho2, phi2)

    # Normalise both rows to the scalar peak so broadening is visually plain
    vmax_col = norm_vmax(I_sc_2d)

    for ax_row, img in [(ax4[0, col], I_sc_2d),
                         (ax4[1, col], I_v_2d)]:
        im = ax_row.imshow(img / vmax_col, extent=ext2, origin="lower",
                           cmap=CMAP_INT, vmin=0, vmax=1.0, aspect="equal")
        ax_row.axhline(0, color="white", lw=0.5, ls=":", alpha=0.35, zorder=3)
        ax_row.axvline(0, color="white", lw=0.5, ls=":", alpha=0.35, zorder=3)
        plt.colorbar(im, ax=ax_row, fraction=0.046, pad=0.04)

    # Half-power circles
    w_sc  = ds["w"]
    w_vec = dv["w_full"]
    if np.isfinite(w_sc)  and w_sc  <= qv:
        ax4[0, col].add_patch(
            plt.Circle((0, 0), w_sc,  color=_C_SC,  fill=False,
                        lw=2.2, ls="-",  zorder=4))
    if np.isfinite(w_vec) and w_vec <= qv:
        ax4[1, col].add_patch(
            plt.Circle((0, 0), w_vec, color=_C_VEC, fill=False,
                        lw=2.2, ls="--", zorder=4))

    # Caliper dimension-lines
    draw_spot_annotation(ax4[0, col], w_sc,  _C_SC,
                         -0.65 * qv, qv,
                         fr"$2w_\mathrm{{sc}}={2*w_sc:.3f}$")
    draw_spot_annotation(ax4[1, col], w_vec, _C_VEC,
                         -0.65 * qv, qv,
                         fr"$2w_\mathrm{{vec}}={2*w_vec:.3f}$")

    # Column title (on row 0 only): angle, aperture fraction, spot-size ratio
    wr_f4 = w_vec / (w_sc + 1e-30)
    ax4[0, col].set_title(
        fr"$\theta = {ds['theta']:.1f}^\circ$   $a/R = {af}$" + "\n"
        r"$w_\mathrm{vec}/w_\mathrm{sc}=$" + f"{wr_f4:.3f}",
        fontsize=8.5, pad=6)

    # Axis labels
    ax4[0, col].set_xlabel(r"$x\;[\mathrm{mm}^{-1}]$", fontsize=7.5)
    ax4[1, col].set_xlabel(r"$x\;[\mathrm{mm}^{-1}]$", fontsize=7.5)
    if col == 0:
        ax4[0, col].set_ylabel(
            r"Scalar  ($t_s=t_p=1$,  $H_0$ only)" + "\n"
            r"$y\;[\mathrm{mm}^{-1}]$", fontsize=8)
        ax4[1, col].set_ylabel(
            r"Vectorial  ($H_0+H_2+E_z$,  Fresnel)" + "\n"
            r"$y\;[\mathrm{mm}^{-1}]$", fontsize=8)

    # ── 1-D normalised profile comparison ────────────────────────────
    ax = ax4[2, col]
    mk_s = q_s <= qv
    mk_v = q_v <= qv

    I_sc_n  = ds["I_1d"]       / (ds["I_1d"][mk_s].max()       + 1e-30)
    I_vec_1d = dv["I_full_1d"]
    I_vec_n  = I_vec_1d        / (I_vec_1d[mk_v].max()          + 1e-30)

    ax.plot(q_s[mk_s], I_sc_n[mk_s],  lw=2.2, color=_C_SC,  ls="-",
            label="Scalar")
    ax.plot(q_v[mk_v], I_vec_n[mk_v], lw=2.2, color=_C_VEC, ls="--",
            label="Vectorial")
    ax.axhline(0.5, color="gray", lw=0.7, ls=":", alpha=0.7)

    # Half-power vertical markers
    if np.isfinite(w_sc)  and w_sc  <= qv:
        ax.axvline(w_sc,  color=_C_SC,  lw=1.4, ls="-",  alpha=0.85)
    if np.isfinite(w_vec) and w_vec <= qv:
        ax.axvline(w_vec, color=_C_VEC, lw=1.4, ls="--", alpha=0.85)

    # Measured spot-size text box (top-right corner)
    ax.text(0.97, 0.96,
            fr"$2w_{{\rm sc}}  = {2*w_sc:.3f}$" + "\n"
            fr"$2w_{{\rm vec}} = {2*w_vec:.3f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#AAAAAA", alpha=0.92, lw=0.9))

    ax.set_xlim(0, qv)
    ax.set_ylim(-0.06, 1.12)
    ax.set_xlabel(r"$q\;[\mathrm{mm}^{-1}]$", fontsize=8)
    ax.set_title("Radial profiles", fontsize=8)
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel(r"$I / I_\mathrm{peak}$", fontsize=8)
        ax.legend(fontsize=8, loc="upper right")

# Figure-level legend
_leg_f4 = [
    Line2D([0], [0], color=_C_SC,  lw=2.2, ls="-",
           label=r"Scalar — $t_s=t_p=1$,  $I=\left|H_0[E]\right|^2$"),
    Line2D([0], [0], color=_C_VEC, lw=2.2, ls="--",
           label=r"Vectorial — paraxial Fresnel,  $I=\left|H_0\right|^2+\left|H_2\right|^2+\left|E_z\right|^2$"),
]
fig4.legend(handles=_leg_f4, loc="lower center", ncol=2,
            fontsize=9, frameon=True, framealpha=0.96,
            edgecolor="#CCCCCC", bbox_to_anchor=(0.5, 0.0))

fig4.suptitle(
    r"Fig 4 — Scalar vs Vectorial diffraction through the ovoid diopter" + "\n"
    r"Scalar: uniform transmission ($t_s=t_p=1$) — Vectorial: physical Fresnel coefficients" + "\n"
    r"Both rows share the scalar peak as colour scale  $\Rightarrow$  "
    r"vectorial broadening is directly visible" + "\n"
    f"Circular polarisation (pure $E_L$)  |  {PARAM_STR}",
    fontsize=10)
fig4.tight_layout(rect=[0, 0.045, 1, 1])


plt.show()
