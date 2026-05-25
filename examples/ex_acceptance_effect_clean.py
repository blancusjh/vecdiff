import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vecdiff.field_reconstruction import hankel_terms
from vecdiff.fresnel import FresnelOvoid, FresnelOvoidParax
from vecdiff.view import radial_map


def aperture_from_alpha(alpha_rad, z0, definition):
    if definition == "a_over_z0":
        return z0 * np.tan(alpha_rad)
    if definition == "z0_over_a":
        t = np.tan(alpha_rad)
        return z0 / np.where(np.abs(t) < 1e-12, 1e-12, t)
    raise ValueError("Invalid alpha definition")


def intensities_vector_scalar(r, q, ts, tp, e1):
    z = np.zeros_like(e1)

    t_vec = hankel_terms(r=r, q=q, tp=tp, ts=ts, e1=e1, e2=z, polarization="circular")
    i_vec = np.abs(t_vec["E_LL"]) ** 2 + np.abs(t_vec["E_RL"]) ** 2

    one = np.ones_like(ts)
    t_sc = hankel_terms(r=r, q=q, tp=one, ts=one, e1=e1, e2=z, polarization="circular")
    i_scalar = np.abs(t_sc["E_LL"]) ** 2

    return i_vec, i_scalar


def iq_integral(y, q):
    return float(np.trapezoid(y * q, q))


def half_power_radius(iq, q):
    y = np.asarray(iq, dtype=float)
    x = np.asarray(q, dtype=float)
    if y.size < 2:
        return float("nan")
    ymax = float(np.max(y))
    if ymax <= 0:
        return float("nan")
    target = 0.5 * ymax
    below = np.where(y <= target)[0]
    if below.size == 0:
        return float("nan")
    j = int(below[0])
    if j == 0:
        return float(x[0])
    x0, x1 = x[j - 1], x[j]
    y0, y1 = y[j - 1], y[j]
    if np.isclose(y1, y0):
        return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


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


def safe_exact_radius(ts_raw, tp_raw, r):
    fin = np.isfinite(ts_raw) & np.isfinite(tp_raw)
    if not np.any(fin):
        return 0.0
    return float(np.max(r[fin]))


def summarize_case(alpha_deg, a_eff, q, i_vec, i_scalar):
    delta = i_vec - i_scalar
    e_scalar = iq_integral(i_scalar, q)
    e_vec = iq_integral(i_vec, q)
    abs_delta = iq_integral(np.abs(delta), q)
    signed_delta = iq_integral(delta, q)
    i_scalar_n = i_scalar / (e_scalar + 1e-30)
    i_vec_n = i_vec / (e_vec + 1e-30)
    delta_n = i_vec_n - i_scalar_n
    abs_delta_n = iq_integral(np.abs(delta_n), q)
    w_scalar = half_power_radius(i_scalar_n, q)
    w_vec = half_power_radius(i_vec_n, q)
    peak_scalar = float(np.max(i_scalar_n))
    peak_vec = float(np.max(i_vec_n))
    return {
        "alpha_deg": float(alpha_deg),
        "a_eff": float(a_eff),
        "rel_l1_delta": float(abs_delta / (e_scalar + 1e-30)),
        "rel_l1_delta_norm": float(abs_delta_n / (iq_integral(i_scalar_n, q) + 1e-30)),
        "abs_delta_energy": float(abs_delta),
        "signed_delta_energy": float(signed_delta),
        "peak_ratio_norm": float(peak_vec / (peak_scalar + 1e-30)),
        "width_ratio_norm": float(w_vec / (w_scalar + 1e-30)),
        "peak_delta": float(np.max(delta)),
        "min_delta": float(np.min(delta)),
    }


def main():
    p = argparse.ArgumentParser(
        description="Clean example: show why vector-vs-scalar intensity differences matter versus acceptance angle."
    )
    p.add_argument("--output-dir", default="outputs/acceptance_effect_clean")

    p.add_argument("--lam", type=float, default=532e-6)
    p.add_argument("--n0", type=float, default=1.0)
    p.add_argument("--ni", type=float, default=1.5)
    p.add_argument("--z0", type=float, default=5.0)
    p.add_argument("--zi", type=float, default=10.0)
    p.add_argument("--r-max", type=float, default=5.0)

    p.add_argument("--n-samples", type=int, default=700)
    p.add_argument("--n-img", type=int, default=450)
    p.add_argument("--refined-grid", action="store_true", help="Use power-law refined grid near q=0 and r=0.")
    p.add_argument("--grid-power", type=float, default=2.5, help="Power for refined grid when --refined-grid is enabled.")
    p.add_argument("--plot-resample", type=int, default=1600, help="Dense q-samples used only for plotting curves.")
    p.add_argument(
        "--q-view-factor",
        type=float,
        default=6.0,
        help="Profile x-limit as factor of first Airy-like zero q1≈3.83/a_eff.",
    )
    p.add_argument(
        "--map-view-factor",
        type=float,
        default=3.0,
        help="2D map half-size as factor of first Airy-like zero q1≈3.83/a_eff.",
    )
    p.add_argument("--profile-power", type=float, default=0.0, help="Input radial profile: (r/a)^p inside aperture.")
    p.add_argument("--fresnel-model", choices=["paraxial", "exact"], default="paraxial")
    p.add_argument(
        "--difference-mode",
        choices=["pure", "normalized"],
        default="pure",
        help="Use pure difference ΔI=I-I_scalar or normalized difference.",
    )
    p.add_argument(
        "--exact-safe-aperture",
        action="store_true",
        help="When using exact Fresnel, clip aperture to the first numerically safe radius (before NaN/Inf coefficients).",
    )

    p.add_argument(
        "--alpha-deg",
        type=float,
        nargs="+",
        default=[5.0, 20.0, 35.0],
        help="Representative acceptance angles to compare directly.",
    )
    p.add_argument(
        "--alpha-definition",
        choices=["a_over_z0", "z0_over_a"],
        default="a_over_z0",
        help="a_over_z0: tan(alpha)=a/z0; z0_over_a: tan(alpha)=z0/a",
    )

    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.refined_grid:
        u = np.linspace(0.0, 1.0, args.n_samples)
        r = args.r_max * u**args.grid_power
    else:
        r = np.linspace(0.0, args.r_max, args.n_samples)
    q_max = args.ni / (args.lam * args.zi) * args.r_max
    if args.refined_grid:
        uq = np.linspace(0.0, 1.0, args.n_samples)
        q = q_max * uq**args.grid_power
    else:
        q = np.linspace(0.0, q_max, args.n_samples)

    if args.fresnel_model == "paraxial":
        fresnel = FresnelOvoidParax(n0=args.n0, ni=args.ni, z0=args.z0, zi=args.zi)
        ts, tp = fresnel.coeffs(r)
        r_safe_exact = float(r[-1])
    else:
        fresnel = FresnelOvoid(n0=args.n0, z0=args.z0, ni=args.ni, zi=args.zi)
        ts_raw, tp_raw = fresnel.ts(r), fresnel.tp(r)
        r_safe_exact = safe_exact_radius(ts_raw, tp_raw, r)
        ts, tp = ts_raw, tp_raw
    ts = sanitize_coeff(np.asarray(ts, dtype=float), r)
    tp = sanitize_coeff(np.asarray(tp, dtype=float), r)

    cases = []
    for a_deg in args.alpha_deg:
        a = float(aperture_from_alpha(np.deg2rad(a_deg), args.z0, args.alpha_definition))
        a_eff = float(np.clip(a, 0.0, args.r_max))
        clipped_exact = False
        if args.fresnel_model == "exact" and args.exact_safe_aperture:
            a_old = a_eff
            a_eff = min(a_eff, r_safe_exact)
            clipped_exact = (a_eff < a_old)
        ap_mask = (r <= a_eff).astype(float)
        if a_eff > 0:
            prof = np.where(ap_mask > 0, (r / a_eff) ** args.profile_power, 0.0)
        else:
            prof = np.zeros_like(r)
        ap = (ap_mask * prof).astype(complex)

        i_vec, i_scalar = intensities_vector_scalar(r, q, ts, tp, ap)
        metrics = summarize_case(a_deg, a_eff, q, i_vec, i_scalar)
        metrics["clipped"] = bool(a != a_eff)
        metrics["clipped_exact_safe"] = bool(clipped_exact)
        cases.append({"metrics": metrics, "i_vec": i_vec, "i_scalar": i_scalar})

    params = {
        "lam": args.lam,
        "n0": args.n0,
        "ni": args.ni,
        "z0": args.z0,
        "zi": args.zi,
        "r_max": args.r_max,
        "q_max": q_max,
        "n_samples": args.n_samples,
        "n_img": args.n_img,
        "refined_grid": args.refined_grid,
        "grid_power": args.grid_power,
        "plot_resample": args.plot_resample,
        "profile_power": args.profile_power,
        "fresnel_model": args.fresnel_model,
        "exact_safe_aperture": args.exact_safe_aperture,
        "r_safe_exact": r_safe_exact,
        "alpha_deg": args.alpha_deg,
        "alpha_definition": args.alpha_definition,
    }
    with (out / "parameters.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    with (out / "metrics.csv").open("w", encoding="utf-8") as f:
        f.write(
            "alpha_deg,a_eff,clipped,rel_l1_delta,rel_l1_delta_norm,"
            "abs_delta_energy,signed_delta_energy,peak_ratio_norm,width_ratio_norm,peak_delta,min_delta,clipped_exact_safe\n"
        )
        for c in cases:
            m = c["metrics"]
            f.write(
                f"{m['alpha_deg']},{m['a_eff']},{int(m['clipped'])},{m['rel_l1_delta']},{m['rel_l1_delta_norm']},"
                f"{m['abs_delta_energy']},{m['signed_delta_energy']},{m['peak_ratio_norm']},{m['width_ratio_norm']},"
                f"{m['peak_delta']},{m['min_delta']},{int(m['clipped_exact_safe'])}\n"
            )

    np.savez(
        out / "data.npz",
        r=r,
        q=q,
        ts=ts,
        tp=tp,
        i_vec=np.array([c["i_vec"] for c in cases]),
        i_scalar=np.array([c["i_scalar"] for c in cases]),
        alpha_deg=np.array([c["metrics"]["alpha_deg"] for c in cases]),
    )

    fig_p, axes_p = plt.subplots(len(cases), 2, figsize=(11.8, 4.0 * len(cases)), constrained_layout=True)
    fig_m, axes_m = plt.subplots(len(cases), 3, figsize=(14.8, 4.1 * len(cases)), constrained_layout=True)
    if len(cases) == 1:
        axes_p = np.array([axes_p])
        axes_m = np.array([axes_m])

    for i, c in enumerate(cases):
        m = c["metrics"]
        i_vec = c["i_vec"]
        i_scalar = c["i_scalar"]
        d = i_vec - i_scalar
        e_vec = iq_integral(i_vec, q)
        e_scalar = iq_integral(i_scalar, q)
        i_vec_n = i_vec / (e_vec + 1e-30)
        i_scalar_n = i_scalar / (e_scalar + 1e-30)
        d_n = i_vec_n - i_scalar_n

        ax0, ax1 = axes_p[i]
        ax2, ax3, ax4 = axes_m[i]

        # Plot-quality evaluation: recompute profiles on a dense, uniform q-grid
        # to avoid piecewise-linear artifacts from interpolation.
        q1 = 3.8317059702075125 / max(m["a_eff"], 1e-12)
        q_view = min(float(np.max(q)), float(args.q_view_factor * q1))
        q_plot = np.linspace(0.0, q_view, max(args.plot_resample, 600))
        a_eff = m["a_eff"]
        ap_mask = (r <= a_eff).astype(float)
        if a_eff > 0:
            prof = np.where(ap_mask > 0, (r / a_eff) ** args.profile_power, 0.0)
        else:
            prof = np.zeros_like(r)
        ap_plot = (ap_mask * prof).astype(complex)
        i_vec_plot_raw, i_scalar_plot_raw = intensities_vector_scalar(r, q_plot, ts, tp, ap_plot)
        e_vec_plot = iq_integral(i_vec_plot_raw, q_plot)
        e_scalar_plot = iq_integral(i_scalar_plot_raw, q_plot)
        i_vec_plot = i_vec_plot_raw / (e_vec_plot + 1e-30)
        i_scalar_plot = i_scalar_plot_raw / (e_scalar_plot + 1e-30)
        d_plot = i_vec_plot - i_scalar_plot

        if args.difference_mode == "pure":
            i_scalar_plot_disp = i_scalar_plot_raw
            i_vec_plot_disp = i_vec_plot_raw
            d_plot_disp = i_vec_plot_raw - i_scalar_plot_raw
            ylab_i = "Intensity"
            ylab_d = r"$\Delta I$"
            ttl_d = r"$\Delta I = I - I_{scalar}$"
            ttl_sc = r"2D scalar map $I_{scalar}$"
            ttl_vc = r"2D total map $I$"
            ttl_df = "2D pure difference"
        else:
            i_scalar_plot_disp = i_scalar_plot
            i_vec_plot_disp = i_vec_plot
            d_plot_disp = d_plot
            ylab_i = "Normalized intensity"
            ylab_d = r"$\Delta I^{(norm)}$"
            ttl_d = r"$\Delta I^{(norm)} = I^{(norm)} - I_{scalar}^{(norm)}$"
            ttl_sc = r"2D scalar map $I_{scalar}^{(norm)}$"
            ttl_vc = r"2D total map $I^{(norm)}$"
            ttl_df = "2D normalized difference"

        ax0.plot(q_plot, i_scalar_plot_disp, lw=2, color="#1f77b4", label=r"$I_{scalar}$")
        ax0.plot(q_plot, i_vec_plot_disp, lw=2, color="#ff7f0e", label=r"$I$")
        ax0.set_title(f"alpha={m['alpha_deg']:.1f} deg | a_eff={m['a_eff']:.3f}")
        ax0.set_xlabel("q")
        ax0.set_ylabel(ylab_i)
        ax0.grid(True, alpha=0.3)
        ax0.legend()

        ax1.plot(q_plot, d_plot_disp, lw=2, color="#d62728")
        ax1.axhline(0.0, color="k", ls="--", lw=1)
        ax1.set_title(ttl_d)
        ax1.set_xlabel("q")
        ax1.set_ylabel(ylab_d)
        ax1.grid(True, alpha=0.3)

        # Useful zoom: avoid plotting the full q_max when the lobe sits near q=0.
        ax0.set_xlim(0.0, q_view)
        ax1.set_xlim(0.0, q_view)

        q_half = min(float(np.max(q_plot)), float(args.map_view_factor * q1))
        img_sc, ext = radial_map(i_scalar_plot_disp, q_plot, q_half, args.n_img)
        img_d, _ = radial_map(d_plot_disp, q_plot, q_half, args.n_img)

        # Mask outside circular support to avoid square-boundary artifacts.
        x = np.linspace(-q_half, q_half, args.n_img)
        y = np.linspace(-q_half, q_half, args.n_img)
        xx, yy = np.meshgrid(x, y)
        rr = np.sqrt(xx**2 + yy**2)
        msk = rr <= q_half
        img_sc = np.where(msk, img_sc, np.nan)
        img_d = np.where(msk, img_d, np.nan)

        vmax_sc = max(float(np.percentile(img_sc[np.isfinite(img_sc)], 99.8)), 1e-12)
        vmax_d = max(float(np.percentile(np.abs(img_d[np.isfinite(img_d)]), 99.5)), 1e-12)

        img_vc, _ = radial_map(i_vec_plot_disp, q_plot, q_half, args.n_img)
        img_vc = np.where(msk, img_vc, np.nan)
        vmax_vc = max(float(np.percentile(img_vc[np.isfinite(img_vc)], 99.8)), 1e-12)

        im_sc = ax2.imshow(img_sc, extent=ext, origin="lower", cmap="hot", vmin=0.0, vmax=vmax_sc)
        ax2.set_title(ttl_sc)
        ax2.set_xlabel(r"$q_x$")
        ax2.set_ylabel(r"$q_y$")
        ax2.set_aspect("equal")
        fig_m.colorbar(im_sc, ax=ax2, fraction=0.046, pad=0.04)

        im_vc = ax3.imshow(img_vc, extent=ext, origin="lower", cmap="hot", vmin=0.0, vmax=vmax_vc)
        ax3.set_title(ttl_vc)
        ax3.set_xlabel(r"$q_x$")
        ax3.set_ylabel(r"$q_y$")
        ax3.set_aspect("equal")
        fig_m.colorbar(im_vc, ax=ax3, fraction=0.046, pad=0.04)

        im_d = ax4.imshow(img_d, extent=ext, origin="lower", cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d)
        if args.difference_mode == "pure":
            ax4.set_title(f"{ttl_df}\nrel_l1={m['rel_l1_delta']:.3f}")
        else:
            ax4.set_title(f"{ttl_df}\nrel_l1_norm={m['rel_l1_delta_norm']:.3f}")
        ax4.set_xlabel(r"$q_x$")
        ax4.set_ylabel(r"$q_y$")
        ax4.set_aspect("equal")
        fig_m.colorbar(im_d, ax=ax4, fraction=0.046, pad=0.04)

    fig_p.suptitle(
        "Radial profiles: scalar vs total vs difference\n"
        f"Definition: {args.alpha_definition} | mode={args.difference_mode} | model={args.fresnel_model} | p={args.profile_power} | "
        f"n0={args.n0}, ni={args.ni}, z0={args.z0}, zi={args.zi}, lambda={args.lam}",
        fontsize=11,
    )
    fig_m.suptitle(
        "Vectorial vs Scalar intensity: effect of acceptance angle\n"
        f"Definition: {args.alpha_definition} | mode={args.difference_mode} | model={args.fresnel_model} | p={args.profile_power} | "
        f"n0={args.n0}, ni={args.ni}, z0={args.z0}, zi={args.zi}, lambda={args.lam}",
        fontsize=11,
    )
    fig_p.savefig(out / "profiles_comparison.png", dpi=260)
    fig_m.savefig(out / "maps2d_comparison.png", dpi=260)
    plt.close(fig_p)
    plt.close(fig_m)

    best = max((c["metrics"] for c in cases), key=lambda x: x["rel_l1_delta"])
    print("Clean example complete")
    print(f"Output dir: {out}")
    print(
        f"Best relative difference at alpha={best['alpha_deg']:.2f} deg: "
        f"rel_l1_delta={best['rel_l1_delta']:.6f}, rel_l1_delta_norm={best['rel_l1_delta_norm']:.6f}"
    )
    for c in cases:
        m = c["metrics"]
        print(
            f"alpha={m['alpha_deg']:.1f} deg | peak_ratio_norm={m['peak_ratio_norm']:.6f} | "
            f"width_ratio_norm={m['width_ratio_norm']:.6f}"
        )


if __name__ == "__main__":
    main()
