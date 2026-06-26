import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from vecdiff.field_reconstruction import hankel_terms
from vecdiff.fresnel import FresnelOvoid, FresnelOvoidParax
from vecdiff.view import radial_map


def aperture_from_alpha(alpha_deg, z0):
    return z0 * math.tan(math.radians(alpha_deg))


def iq_integral(y, q):
    return float(np.trapezoid(y * q, q))


def sanitize_coeff(arr, x):
    a = np.asarray(arr, dtype=float)
    x = np.asarray(x, dtype=float)
    m = np.isfinite(a)
    if np.all(m):
        return a
    if np.sum(m) < 2:
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    out = a.copy()
    out[~m] = np.interp(x[~m], x[m], a[m])
    return out


def make_grids(r_max, q_max, n_samples, refined=True, power=2.2):
    if refined:
        u = np.linspace(0.0, 1.0, n_samples)
        r = r_max * u**power
    else:
        r = np.linspace(0.0, r_max, n_samples)
    q = np.linspace(0.0, q_max, n_samples)
    return r, q


def compute_coeffs(model, n0, ni, z0, zi, r):
    if model == "paraxial":
        f = FresnelOvoidParax(n0=n0, ni=ni, z0=z0, zi=zi)
        ts, tp = f.coeffs(r)
    else:
        f = FresnelOvoid(n0=n0, z0=z0, ni=ni, zi=zi)
        ts, tp = f.ts(r), f.tp(r)
    ts = sanitize_coeff(ts, r)
    tp = sanitize_coeff(tp, r)
    return ts, tp


def intensities(r, q, ts, tp, e1):
    z = np.zeros_like(e1)
    terms_v = hankel_terms(r=r, q=q, ts=ts, tp=tp, e1=e1, e2=z, polarization="circular")
    i_total = np.abs(terms_v["E_LL"]) ** 2 + np.abs(terms_v["E_RL"]) ** 2

    one = np.ones_like(ts)
    terms_s = hankel_terms(r=r, q=q, ts=one, tp=one, e1=e1, e2=z, polarization="circular")
    i_scalar = np.abs(terms_s["E_LL"]) ** 2
    return i_total, i_scalar


def ensure_positive_for_log(img):
    eps = 1e-16
    return np.maximum(img, eps)


def save_case_figures(case_dir, alpha_deg, ni, q, i_scalar, i_total, n_img, q_view):
    case_dir.mkdir(parents=True, exist_ok=True)
    delta = i_total - i_scalar
    abs_delta = np.abs(delta)

    img_sc, ext = radial_map(i_scalar, q, q_view, n_img)
    img_tt, _ = radial_map(i_total, q, q_view, n_img)
    img_dd, _ = radial_map(abs_delta, q, q_view, n_img)

    # linear
    vmax_i = max(float(np.percentile(np.r_[img_sc.flatten(), img_tt.flatten()], 99.8)), 1e-12)
    vmax_d = max(float(np.percentile(img_dd, 99.8)), 1e-12)

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    im0 = ax[0].imshow(img_sc, extent=ext, origin="lower", cmap="hot", vmin=0.0, vmax=vmax_i)
    im1 = ax[1].imshow(img_tt, extent=ext, origin="lower", cmap="hot", vmin=0.0, vmax=vmax_i)
    im2 = ax[2].imshow(img_dd, extent=ext, origin="lower", cmap="magma", vmin=0.0, vmax=vmax_d)
    ax[0].set_title(r"$I_{scalar}$")
    ax[1].set_title(r"$I_{total}$")
    ax[2].set_title(r"$|\Delta I|$")
    for a in ax:
        a.set_xlabel(r"$q_x$")
        a.set_ylabel(r"$q_y$")
        a.set_aspect("equal")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    fig.suptitle(f"Linear maps | alpha={alpha_deg:.1f} deg | ni={ni}")
    fig.savefig(case_dir / "maps_linear.png", dpi=240)
    plt.close(fig)

    # log
    img_sc_l = ensure_positive_for_log(img_sc)
    img_tt_l = ensure_positive_for_log(img_tt)
    img_dd_l = ensure_positive_for_log(img_dd)

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    im0 = ax[0].imshow(img_sc_l, extent=ext, origin="lower", cmap="hot", norm=LogNorm(vmin=np.min(img_sc_l), vmax=np.max(img_sc_l)))
    im1 = ax[1].imshow(img_tt_l, extent=ext, origin="lower", cmap="hot", norm=LogNorm(vmin=np.min(img_tt_l), vmax=np.max(img_tt_l)))
    im2 = ax[2].imshow(img_dd_l, extent=ext, origin="lower", cmap="magma", norm=LogNorm(vmin=np.min(img_dd_l), vmax=np.max(img_dd_l)))
    ax[0].set_title(r"$I_{scalar}$ (log)")
    ax[1].set_title(r"$I_{total}$ (log)")
    ax[2].set_title(r"$|\Delta I|$ (log)")
    for a in ax:
        a.set_xlabel(r"$q_x$")
        a.set_ylabel(r"$q_y$")
        a.set_aspect("equal")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    fig.suptitle(f"Log maps | alpha={alpha_deg:.1f} deg | ni={ni}")
    fig.savefig(case_dir / "maps_log.png", dpi=240)
    plt.close(fig)


def save_global_plots(out_dir, rows, alphas, nis):
    # central factor vs alpha per ni
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ni in nis:
        rs = [r for r in rows if abs(r["ni"] - ni) < 1e-12]
        rs.sort(key=lambda d: d["alpha_deg"])
        ax[0].plot([r["alpha_deg"] for r in rs], [r["F_c"] for r in rs], marker="o", label=f"ni={ni}")
        ax[1].plot([r["alpha_deg"] for r in rs], [r["D_c"] for r in rs], marker="o", label=f"ni={ni}")
    ax[0].set_title(r"Central factor $F_c = I_{total}(0)/I_{scalar}(0)$")
    ax[1].set_title(r"Central deficit $D_c = 1-F_c$")
    for a in ax:
        a.set_xlabel(r"$\alpha$ [deg]")
        a.grid(True, alpha=0.3)
        a.legend(fontsize=8)
    ax[0].set_ylabel("F_c")
    ax[1].set_ylabel("D_c")
    fig.savefig(out_dir / "central_factor_vs_alpha.png", dpi=240)
    plt.close(fig)

    # heatmaps
    A = np.array(alphas, dtype=float)
    N = np.array(nis, dtype=float)
    Z1 = np.full((len(N), len(A)), np.nan)
    Z2 = np.full((len(N), len(A)), np.nan)
    for i, ni in enumerate(N):
        for j, a in enumerate(A):
            m = next((r for r in rows if abs(r["ni"] - ni) < 1e-12 and abs(r["alpha_deg"] - a) < 1e-12), None)
            if m is not None:
                Z1[i, j] = m["D_c"]
                Z2[i, j] = m["abs_delta_energy"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    im1 = ax[0].imshow(Z1, origin="lower", aspect="auto", cmap="viridis", extent=[A.min(), A.max(), N.min(), N.max()])
    im2 = ax[1].imshow(Z2, origin="lower", aspect="auto", cmap="magma", extent=[A.min(), A.max(), N.min(), N.max()])
    ax[0].set_title("Heatmap D_c")
    ax[1].set_title("Heatmap abs_delta_energy")
    for a in ax:
        a.set_xlabel(r"$\alpha$ [deg]")
        a.set_ylabel("ni")
    fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
    fig.savefig(out_dir / "heatmaps_metrics.png", dpi=240)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Organized study of non-scalar contribution with pure and central metrics.")
    p.add_argument("--output-dir", default="outputs/acceptance_effect_clean_pure_default/study")
    p.add_argument("--model", choices=["paraxial", "exact"], default="paraxial")
    p.add_argument("--n0", type=float, default=1.0)
    p.add_argument("--ni-list", type=float, nargs="+", default=[1.8, 2.2, 2.6])
    p.add_argument("--z0", type=float, default=2.0)
    p.add_argument("--zi", type=float, default=10.0)
    p.add_argument("--lam", type=float, default=532e-6)
    p.add_argument("--alpha-list", type=float, nargs="+", default=[10.0, 20.0, 30.0, 40.0])
    p.add_argument("--profile-power", type=float, default=2.0)
    p.add_argument("--r-max", type=float, default=4.0)
    p.add_argument("--n-samples", type=int, default=900)
    p.add_argument("--n-img", type=int, default=280)
    p.add_argument("--q-view-factor", type=float, default=3.0)
    p.add_argument("--save-case-figures", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = vars(args)
    (out_dir / "parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    rows = []

    for ni in args.ni_list:
        q_max = ni / (args.lam * args.zi) * args.r_max
        r, q = make_grids(args.r_max, q_max, args.n_samples, refined=True, power=2.2)
        ts, tp = compute_coeffs(args.model, args.n0, ni, args.z0, args.zi, r)

        for alpha_deg in args.alpha_list:
            a_raw = aperture_from_alpha(alpha_deg, args.z0)
            a_eff = float(np.clip(a_raw, 0.0, args.r_max))
            mask = (r <= a_eff).astype(float)
            if a_eff > 0:
                profile = np.where(mask > 0, (r / a_eff) ** args.profile_power, 0.0)
            else:
                profile = np.zeros_like(r)
            e1 = (mask * profile).astype(complex)

            i_total, i_scalar = intensities(r, q, ts, tp, e1)
            delta = i_total - i_scalar

            i0_total = float(i_total[0])
            i0_scalar = float(i_scalar[0])
            F_c = i0_total / (i0_scalar + 1e-30)
            D_c = 1.0 - F_c

            abs_delta_energy = iq_integral(np.abs(delta), q)
            signed_delta_energy = iq_integral(delta, q)
            rel_l1_delta = abs_delta_energy / (iq_integral(i_scalar, q) + 1e-30)

            NA_obj = args.n0 * math.sin(math.radians(alpha_deg))
            NA_proxy_ni = ni * math.sin(math.radians(alpha_deg))

            row = {
                "ni": ni,
                "alpha_deg": alpha_deg,
                "NA_obj": NA_obj,
                "NA_proxy_ni": NA_proxy_ni,
                "a_eff": a_eff,
                "I0_scalar": i0_scalar,
                "I0_total": i0_total,
                "F_c": F_c,
                "D_c": D_c,
                "abs_delta_energy": abs_delta_energy,
                "signed_delta_energy": signed_delta_energy,
                "rel_l1_delta": rel_l1_delta,
            }
            rows.append(row)

            if args.save_case_figures:
                q1 = 3.8317059702075125 / max(a_eff, 1e-12)
                q_view = min(float(np.max(q)), float(args.q_view_factor * q1))
                case_name = f"ni_{ni:.2f}_alpha_{alpha_deg:.1f}".replace(".", "p")
                save_case_figures(out_dir / "cases" / case_name, alpha_deg, ni, q, i_scalar, i_total, args.n_img, q_view)

    rows.sort(key=lambda d: d["D_c"], reverse=True)

    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    save_global_plots(out_dir, rows, args.alpha_list, args.ni_list)

    print("Study complete")
    print(f"Output: {out_dir}")
    print("Top 5 by D_c (central deficit):")
    for r in rows[:5]:
        print(
            f"ni={r['ni']:.2f}, alpha={r['alpha_deg']:.1f}, D_c={r['D_c']:.4f}, "
            f"F_c={r['F_c']:.4f}, rel_l1_delta={r['rel_l1_delta']:.4f}"
        )


if __name__ == "__main__":
    main()
