import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from vecdiff.field_reconstruction import hankel_terms, make_observation_grid, reconstruct_2d_from_terms
from vecdiff.fresnel import FresnelOvoid, FresnelOvoidParax


def aperture_from_alpha(alpha_deg, z0):
    return z0 * math.tan(math.radians(alpha_deg))


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


def make_r_q(r_max, q_max, n_samples, refined=True, power=2.2):
    if refined:
        u = np.linspace(0.0, 1.0, n_samples)
        r = r_max * u**power
    else:
        r = np.linspace(0.0, r_max, n_samples)
    q = np.linspace(0.0, q_max, n_samples)
    return r, q


def compute_ts_tp(model, n0, ni, z0, zi, r):
    if model == "paraxial":
        f = FresnelOvoidParax(n0=n0, ni=ni, z0=z0, zi=zi)
        ts, tp = f.coeffs(r)
    else:
        f = FresnelOvoid(n0=n0, z0=z0, ni=ni, zi=zi)
        ts, tp = f.ts(r), f.tp(r)
    ts = sanitize_coeff(ts, r)
    tp = sanitize_coeff(tp, r)
    return ts, tp


def build_input_profile(r, a_eff, profile_power):
    mask = (r <= a_eff).astype(float)
    if a_eff > 0:
        prof = np.where(mask > 0, (r / a_eff) ** profile_power, 0.0)
    else:
        prof = np.zeros_like(r)
    return (mask * prof).astype(complex)


def intensity_maps_for_polarization(polarization, r, q, ts, tp, e1, rho, varphi):
    zero = np.zeros_like(e1)

    terms_total = hankel_terms(r=r, q=q, ts=ts, tp=tp, e1=e1, e2=zero, polarization=polarization)
    comp1_t, comp2_t = reconstruct_2d_from_terms(terms=terms_total, q=q, rho=rho, varphi=varphi, polarization=polarization)
    i_total = np.abs(comp1_t) ** 2 + np.abs(comp2_t) ** 2

    one = np.ones_like(ts)
    terms_scalar = hankel_terms(r=r, q=q, ts=one, tp=one, e1=e1, e2=zero, polarization=polarization)
    comp1_s, comp2_s = reconstruct_2d_from_terms(terms=terms_scalar, q=q, rho=rho, varphi=varphi, polarization=polarization)
    i_scalar = np.abs(comp1_s) ** 2 + np.abs(comp2_s) ** 2

    return i_scalar, i_total


def ee50_radius(img, extent):
    """
    Radius enclosing 50% of total intensity energy (EE50), computed directly
    from the 2D map. This is more stable than FWHM-like estimators for
    oscillatory profiles.
    """
    img = np.asarray(img, dtype=float)
    ny, nx = img.shape
    x = np.linspace(extent[0], extent[1], nx)
    y = np.linspace(extent[2], extent[3], ny)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2).ravel()
    ww = np.maximum(img.ravel(), 0.0)
    total = float(np.sum(ww))
    if total <= 0:
        return float("nan")
    idx = np.argsort(rr)
    rr_s = rr[idx]
    ww_s = ww[idx]
    c = np.cumsum(ww_s) / total
    k = int(np.searchsorted(c, 0.5))
    k = min(max(k, 0), len(rr_s) - 1)
    return float(rr_s[k])


def save_case_maps(case_dir, i_scalar, i_total, extent, title_prefix):
    case_dir.mkdir(parents=True, exist_ok=True)
    delta = i_total - i_scalar
    abs_delta = np.abs(delta)

    vmax_sc = max(float(np.max(i_scalar)), 1e-12)
    vmax_tt = max(float(np.max(i_total)), 1e-12)
    vmax_d = max(float(np.percentile(abs_delta, 99.7)), 1e-12)

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    im0 = ax[0].imshow(i_scalar, extent=extent, origin="lower", cmap="hot", vmin=0.0, vmax=vmax_sc)
    im1 = ax[1].imshow(i_total, extent=extent, origin="lower", cmap="hot", vmin=0.0, vmax=vmax_tt)
    im2 = ax[2].imshow(abs_delta, extent=extent, origin="lower", cmap="magma", vmin=0.0, vmax=vmax_d)
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
    fig.suptitle(f"{title_prefix} | linear")
    fig.savefig(case_dir / "maps_linear.png", dpi=230)
    plt.close(fig)

    eps = 1e-16
    sc_l = np.maximum(i_scalar, eps)
    tt_l = np.maximum(i_total, eps)
    dd_l = np.maximum(abs_delta, eps)

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    im0 = ax[0].imshow(sc_l, extent=extent, origin="lower", cmap="hot", norm=LogNorm(vmin=np.min(sc_l), vmax=np.max(sc_l)))
    im1 = ax[1].imshow(tt_l, extent=extent, origin="lower", cmap="hot", norm=LogNorm(vmin=np.min(tt_l), vmax=np.max(tt_l)))
    im2 = ax[2].imshow(dd_l, extent=extent, origin="lower", cmap="magma", norm=LogNorm(vmin=np.min(dd_l), vmax=np.max(dd_l)))
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
    fig.suptitle(f"{title_prefix} | log")
    fig.savefig(case_dir / "maps_log.png", dpi=230)
    plt.close(fig)


def save_fc_plots(out_dir, rows, alphas, nis, polarizations, models):
    for pol in polarizations:
        for model in models:
            sub = [r for r in rows if r["polarization"] == pol and r["model"] == model]
            if not sub:
                continue

            fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
            for ni in nis:
                rs = [r for r in sub if abs(r["ni"] - ni) < 1e-12]
                rs.sort(key=lambda d: d["alpha_deg"])
                ax[0].plot([r["alpha_deg"] for r in rs], [r["F_c"] for r in rs], marker="o", label=f"ni={ni}")
                ax[1].plot(
                    [r["alpha_deg"] for r in rs],
                    [r["spot_ratio"] for r in rs],
                    marker="o",
                    label=f"ni={ni}",
                )
            ax[0].set_title(f"F_c vs alpha | {pol} | {model}")
            ax[0].set_xlabel(r"$\alpha$ [deg]")
            ax[0].set_ylabel(r"$F_c = I_{{total}}(0)/I_{{scalar}}(0)$")
            ax[0].grid(True, alpha=0.3)
            ax[0].legend(fontsize=8)
            ax[1].set_title(f"Spot ratio vs alpha | {pol} | {model}")
            ax[1].set_xlabel(r"$\alpha$ [deg]")
            ax[1].set_ylabel(r"$w_{total}/w_{scalar}$")
            ax[1].grid(True, alpha=0.3)
            ax[1].legend(fontsize=8)
            fig.savefig(out_dir / f"Fc_vs_alpha_{pol}_{model}.png", dpi=230)
            plt.close(fig)

            A = np.array(alphas, dtype=float)
            N = np.array(nis, dtype=float)
            Z = np.full((len(N), len(A)), np.nan)
            Zs = np.full((len(N), len(A)), np.nan)
            for i, ni in enumerate(N):
                for j, a in enumerate(A):
                    m = next((r for r in sub if abs(r["ni"] - ni) < 1e-12 and abs(r["alpha_deg"] - a) < 1e-12), None)
                    if m is not None:
                        Z[i, j] = m["F_c"]
                        Zs[i, j] = m["spot_ratio"]
            fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
            im = ax[0].imshow(Z, origin="lower", aspect="auto", cmap="viridis", extent=[A.min(), A.max(), N.min(), N.max()])
            ax[0].set_title(f"Heatmap F_c | {pol} | {model}")
            ax[0].set_xlabel(r"$\alpha$ [deg]")
            ax[0].set_ylabel("ni")
            fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
            im2 = ax[1].imshow(Zs, origin="lower", aspect="auto", cmap="coolwarm", extent=[A.min(), A.max(), N.min(), N.max()])
            ax[1].set_title(f"Heatmap w_total/w_scalar | {pol} | {model}")
            ax[1].set_xlabel(r"$\alpha$ [deg]")
            ax[1].set_ylabel("ni")
            fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
            fig.savefig(out_dir / f"Fc_heatmap_{pol}_{model}.png", dpi=230)
            plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="F_c-only study for circular, polar and cartesian polarizations.")
    p.add_argument("--output-dir", default="outputs/acceptance_effect_clean_pure_default/report_fc")
    p.add_argument("--n0", type=float, default=1.0)
    p.add_argument("--z0", type=float, default=2.0)
    p.add_argument("--zi", type=float, default=10.0)
    p.add_argument("--lam", type=float, default=532e-6)
    p.add_argument("--ni-list", type=float, nargs="+", default=[1.8, 2.2, 2.6])
    p.add_argument("--alpha-list", type=float, nargs="+", default=[10.0, 20.0, 30.0, 40.0])
    p.add_argument("--profile-power", type=float, default=2.0)
    p.add_argument("--r-max", type=float, default=4.0)
    p.add_argument("--n-samples", type=int, default=700)
    p.add_argument("--n-img", type=int, default=260)
    p.add_argument("--map-view-factor", type=float, default=8.0)
    p.add_argument("--save-case-maps", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = vars(args)
    (out_dir / "parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    polarizations = ["circular", "polar", "cartesian"]
    models = ["paraxial", "exact"]
    rows = []

    for pol in polarizations:
        for model in models:
            for ni in args.ni_list:
                q_max = ni / (args.lam * args.zi) * args.r_max
                r, q = make_r_q(args.r_max, q_max, args.n_samples, refined=True, power=2.2)
                ts, tp = compute_ts_tp(model, args.n0, ni, args.z0, args.zi, r)

                for alpha in args.alpha_list:
                    a_raw = aperture_from_alpha(alpha, args.z0)
                    a_eff = float(np.clip(a_raw, 0.0, args.r_max))
                    e1 = build_input_profile(r, a_eff, args.profile_power)

                    q1 = 3.8317059702075125 / max(a_eff, 1e-12)
                    q_view = min(float(np.max(q)), float(args.map_view_factor * q1))
                    xx, yy, rho, varphi, extent = make_observation_grid(q_view, args.n_img)

                    i_scalar, i_total = intensity_maps_for_polarization(pol, r, q, ts, tp, e1, rho, varphi)
                    c = args.n_img // 2
                    i0_scalar = float(i_scalar[c, c])
                    i0_total = float(i_total[c, c])
                    Fc = i0_total / (i0_scalar + 1e-30)
                    w_s = ee50_radius(i_scalar, extent)
                    w_t = ee50_radius(i_total, extent)
                    spot_ratio = w_t / (w_s + 1e-30)

                    rows.append(
                        {
                            "polarization": pol,
                            "model": model,
                            "ni": ni,
                            "alpha_deg": alpha,
                            "NA_obj": args.n0 * math.sin(math.radians(alpha)),
                            "a_eff": a_eff,
                            "I0_scalar": i0_scalar,
                            "I0_total": i0_total,
                            "F_c": Fc,
                            "w_scalar": w_s,
                            "w_total": w_t,
                            "spot_ratio": spot_ratio,
                        }
                    )

                    if args.save_case_maps:
                        case = f"{pol}_{model}_ni{ni:.2f}_a{alpha:.1f}".replace(".", "p")
                        save_case_maps(
                            case_dir=out_dir / "cases" / case,
                            i_scalar=i_scalar,
                            i_total=i_total,
                            extent=extent,
                            title_prefix=(
                                f"{pol} | {model} | ni={ni:.2f} | alpha={alpha:.1f} | "
                                f"incident: e1=(r/a)^p, p={args.profile_power}, e2=0"
                            ),
                        )

    with (out_dir / "metrics_Fc.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    save_fc_plots(out_dir, rows, args.alpha_list, args.ni_list, polarizations, models)

    # Build compact markdown report
    rows_sorted = sorted(rows, key=lambda d: d["F_c"])
    top_low = rows_sorted[:6]
    md = []
    md.append("# Reporte F_c por Polarización")
    md.append("")
    md.append("Métrica principal: `F_c = I_total(0)/I_scalar(0)`.")
    md.append("Métrica adicional: `w_total/w_scalar` (comparación de spot size a media altura).")
    md.append("")
    md.append("## Lectura")
    md.append("- `F_c < 1`: el total vectorial tiene menor intensidad central que el escalar.")
    md.append("- `F_c ≈ 1`: diferencia central baja.")
    md.append("- `w_total/w_scalar > 1`: el spot vectorial es más ancho que el escalar.")
    md.append("- `w_total/w_scalar < 1`: el spot vectorial es más estrecho.")
    md.append("")
    md.append("## Casos con menor F_c (mayor caída central del total)")
    for r in top_low:
        md.append(
            f"- pol={r['polarization']}, model={r['model']}, ni={r['ni']:.2f}, alpha={r['alpha_deg']:.1f}°, "
            f"F_c={r['F_c']:.4f}"
        )
    md.append("")
    md.append("## Figuras")
    for pol in polarizations:
        for model in models:
            md.append(f"### {pol} | {model}")
            md.append(f"![Fc-alpha](Fc_vs_alpha_{pol}_{model}.png)")
            md.append(f"![Fc-heatmap](Fc_heatmap_{pol}_{model}.png)")
            md.append("")

    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8")

    print("F_c study complete")
    print(f"Output: {out_dir}")
    print("Lowest F_c cases:")
    for r in top_low:
        print(
            f"pol={r['polarization']}, model={r['model']}, ni={r['ni']:.2f}, alpha={r['alpha_deg']:.1f}, F_c={r['F_c']:.4f}"
        )


if __name__ == "__main__":
    main()
