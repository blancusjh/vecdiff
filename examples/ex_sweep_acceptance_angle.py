import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vecdiff.field_reconstruction import hankel_terms
from vecdiff.fresnel import FresnelOvoidParax
from vecdiff.view import radial_map


DEFAULT_LAMBDA = 532e-6


def acceptance_to_aperture(alpha_rad, z0, definition):
    if definition == "a_over_z0":
        # tan(alpha) = a / z0
        return z0 * np.tan(alpha_rad)
    if definition == "z0_over_a":
        # tan(alpha) = z0 / a
        t = np.tan(alpha_rad)
        eps = 1e-12
        t = np.where(np.abs(t) < eps, np.sign(t) * eps + (t == 0.0) * eps, t)
        return z0 / t
    raise ValueError("Invalid alpha definition")


def radial_intensities(r, q, ts, tp, aperture):
    terms = hankel_terms(
        r=r,
        q=q,
        tp=tp,
        ts=ts,
        e1=aperture,
        e2=np.zeros_like(aperture),
        polarization="circular",
    )
    e_ll = terms["E_LL"]
    e_rl = terms["E_RL"]
    i_vec = np.abs(e_ll) ** 2 + np.abs(e_rl) ** 2

    terms_scalar = hankel_terms(
        r=r,
        q=q,
        tp=np.ones_like(tp),
        ts=np.ones_like(ts),
        e1=aperture,
        e2=np.zeros_like(aperture),
        polarization="circular",
    )
    e_ll_scalar = terms_scalar["E_LL"]
    i_scalar = np.abs(e_ll_scalar) ** 2

    return i_vec, i_scalar


def metrics_from_intensity(q, i_vec, i_scalar):
    delta_i = i_vec - i_scalar
    e_scalar = weighted_integral(i_scalar, q)
    signed_delta = weighted_integral(delta_i, q)
    abs_delta = weighted_integral(np.abs(delta_i), q)
    rel_l1 = abs_delta / (e_scalar + 1e-30)
    return {
        "energy_scalar": e_scalar,
        "signed_delta_energy": signed_delta,
        "abs_delta_energy": abs_delta,
        "rel_l1_delta": rel_l1,
        "peak_delta": float(np.max(delta_i)),
        "min_delta": float(np.min(delta_i)),
    }


def weighted_integral(y, q):
    return float(np.trapezoid(y * q, q))


def run_sweep(
    n0,
    ni,
    z0,
    zi,
    lam,
    r_max,
    n_samples,
    alpha_deg_min,
    alpha_deg_max,
    n_alpha,
    alpha_definition,
):
    r = np.linspace(0.0, r_max, n_samples)
    q_max = ni / (lam * zi) * r_max
    q = np.linspace(0.0, q_max, n_samples)

    fresnel = FresnelOvoidParax(n0=n0, ni=ni, z0=z0, zi=zi)
    ts_full, tp_full = fresnel.coeffs(r)

    alphas_deg = np.linspace(alpha_deg_min, alpha_deg_max, n_alpha)
    rows = []

    for alpha_deg in alphas_deg:
        alpha_rad = np.deg2rad(alpha_deg)
        a = float(acceptance_to_aperture(alpha_rad, z0, alpha_definition))
        a_eff = float(np.clip(a, 0.0, r_max))

        aperture = (r <= a_eff).astype(float).astype(complex)
        i_vec, i_scalar = radial_intensities(r, q, ts_full, tp_full, aperture)
        delta_i = i_vec - i_scalar

        e_scalar = weighted_integral(i_scalar, q)
        signed_delta = weighted_integral(delta_i, q)
        abs_delta = weighted_integral(np.abs(delta_i), q)
        rel_l1 = abs_delta / (e_scalar + 1e-30)

        rows.append(
            {
                "alpha_deg": alpha_deg,
                "alpha_rad": alpha_rad,
                "a_raw": a,
                "a_eff": a_eff,
                "clipped": float(a != a_eff),
                "q_max": q_max,
                "energy_scalar": e_scalar,
                "signed_delta_energy": signed_delta,
                "abs_delta_energy": abs_delta,
                "rel_l1_delta": rel_l1,
                "peak_delta": float(np.max(delta_i)),
                "min_delta": float(np.min(delta_i)),
            }
        )

    return q, rows


def _pick_plot_indices(n_alpha):
    raw = [0, n_alpha // 2, n_alpha - 1]
    seen = []
    for idx in raw:
        if idx not in seen:
            seen.append(idx)
    return seen


def _params_dict(args):
    return {
        "n0": args.n0,
        "ni": args.ni,
        "z0": args.z0,
        "zi": args.zi,
        "lambda": args.lam,
        "r_max": args.r_max,
        "n_samples": args.n_samples,
        "n_alpha": args.n_alpha,
        "alpha_deg_min": args.alpha_deg_min,
        "alpha_deg_max": args.alpha_deg_max,
        "alpha_definition": args.alpha_definition,
        "n_img": args.n_img,
    }


def _save_params(output_dir, params):
    params_path = output_dir / "parameters.json"
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    return params_path


def _plot_sweep_summary(output_dir, rows):
    alpha = np.array([r["alpha_deg"] for r in rows], dtype=float)
    rel_l1 = np.array([r["rel_l1_delta"] for r in rows], dtype=float)
    signed = np.array([r["signed_delta_energy"] for r in rows], dtype=float)
    absd = np.array([r["abs_delta_energy"] for r in rows], dtype=float)
    a_eff = np.array([r["a_eff"] for r in rows], dtype=float)
    clipped = np.array([r["clipped"] for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(alpha, rel_l1, color="#1f77b4", lw=2)
    axes[0, 0].set_title(r"Relative difference: $\int |\Delta I| q\,dq / \int I_{scalar} q\,dq$")
    axes[0, 0].set_xlabel(r"$\alpha$ [deg]")
    axes[0, 0].set_ylabel("rel_l1_delta")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(alpha, signed, color="#d62728", lw=2, label=r"$\int \Delta I\, q\,dq$")
    axes[0, 1].plot(alpha, absd, color="#2ca02c", lw=2, label=r"$\int |\Delta I|\, q\,dq$")
    axes[0, 1].set_title(r"Integrated intensity differences")
    axes[0, 1].set_xlabel(r"$\alpha$ [deg]")
    axes[0, 1].set_ylabel("Integral value")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(alpha, a_eff, color="#9467bd", lw=2)
    axes[1, 0].set_title("Effective aperture used in simulation")
    axes[1, 0].set_xlabel(r"$\alpha$ [deg]")
    axes[1, 0].set_ylabel(r"$a_{\mathrm{eff}}$")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].step(alpha, clipped, where="mid", color="#8c564b", lw=2)
    axes[1, 1].set_title("Aperture clipping flag (1: clipped by r_max)")
    axes[1, 1].set_xlabel(r"$\alpha$ [deg]")
    axes[1, 1].set_ylabel("clipped")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].grid(True, alpha=0.3)

    fig.savefig(output_dir / "sweep_summary.png", dpi=250)
    plt.close(fig)


def _plot_profiles(output_dir, r, q, ts, tp, rows, alpha_definition):
    indices = _pick_plot_indices(len(rows))
    fig, axes = plt.subplots(len(indices), 2, figsize=(12, 4 * len(indices)), constrained_layout=True)
    if len(indices) == 1:
        axes = np.array([axes])

    for row_i, idx in enumerate(indices):
        data = rows[idx]
        aperture = (r <= data["a_eff"]).astype(float).astype(complex)
        i_vec, i_scalar = radial_intensities(r, q, ts, tp, aperture)
        delta_i = i_vec - i_scalar

        ax1 = axes[row_i, 0]
        ax1.plot(q, i_scalar, lw=2, color="#1f77b4", label=r"$I_{scalar}$")
        ax1.plot(q, i_vec, lw=2, color="#ff7f0e", label=r"$I$")
        ax1.set_title(
            f"Radial intensities | alpha={data['alpha_deg']:.2f} deg | "
            f"a_eff={data['a_eff']:.4f} | {alpha_definition}"
        )
        ax1.set_xlabel("q")
        ax1.set_ylabel("Intensity")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = axes[row_i, 1]
        ax2.plot(q, delta_i, lw=2, color="#d62728", label=r"$\Delta I = I-I_{scalar}$")
        ax2.axhline(0.0, color="k", ls="--", lw=1)
        ax2.set_title("Difference profile")
        ax2.set_xlabel("q")
        ax2.set_ylabel(r"$\Delta I$")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    fig.savefig(output_dir / "profiles_selected_alpha.png", dpi=250)
    plt.close(fig)


def _plot_2d_maps(output_dir, r, q, ts, tp, rows, n_img):
    indices = _pick_plot_indices(len(rows))
    for idx in indices:
        data = rows[idx]
        aperture = (r <= data["a_eff"]).astype(float).astype(complex)
        i_vec, i_scalar = radial_intensities(r, q, ts, tp, aperture)
        delta_i = i_vec - i_scalar

        q_half = float(0.75 * np.max(q))
        img_scalar, extent = radial_map(i_scalar, q, q_half, n_img)
        img_vec, _ = radial_map(i_vec, q, q_half, n_img)
        img_delta, _ = radial_map(delta_i, q, q_half, n_img)

        vmax_int = max(float(np.max(img_scalar)), float(np.max(img_vec)), 1e-12)
        vmax_delta = max(float(np.max(np.abs(img_delta))), 1e-12)

        fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
        im0 = ax[0].imshow(img_scalar, extent=extent, origin="lower", cmap="hot", vmin=0.0, vmax=vmax_int)
        im1 = ax[1].imshow(img_vec, extent=extent, origin="lower", cmap="hot", vmin=0.0, vmax=vmax_int)
        im2 = ax[2].imshow(
            img_delta,
            extent=extent,
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax_delta,
            vmax=vmax_delta,
        )
        ax[0].set_title(r"$I_{scalar}$")
        ax[1].set_title(r"$I$")
        ax[2].set_title(r"$\Delta I = I-I_{scalar}$")
        for axi in ax:
            axi.set_xlabel(r"$q_x$")
            axi.set_ylabel(r"$q_y$")
            axi.set_aspect("equal")
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        fig.suptitle(
            f"2D intensity maps | alpha={data['alpha_deg']:.2f} deg | a_eff={data['a_eff']:.4f}",
            fontsize=11,
        )
        out_name = f"maps_alpha_{data['alpha_deg']:.2f}".replace(".", "p") + ".png"
        fig.savefig(output_dir / out_name, dpi=250)
        plt.close(fig)


def save_outputs(output_dir, q, rows, params):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "acceptance_sweep.csv"
    header = (
        "alpha_deg,alpha_rad,a_raw,a_eff,clipped,q_max,"
        "energy_scalar,signed_delta_energy,abs_delta_energy,rel_l1_delta,peak_delta,min_delta"
    )
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(
                f"{row['alpha_deg']},{row['alpha_rad']},{row['a_raw']},{row['a_eff']},"
                f"{row['clipped']},{row['q_max']},{row['energy_scalar']},"
                f"{row['signed_delta_energy']},{row['abs_delta_energy']},{row['rel_l1_delta']},"
                f"{row['peak_delta']},{row['min_delta']}\n"
            )

    npz_path = output_dir / "acceptance_sweep.npz"
    np.savez(npz_path, q=q, rows=np.array(rows, dtype=object))

    params_path = _save_params(output_dir, params)

    return csv_path, npz_path, params_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep acceptance angle and evaluate intensity difference ΔI = I - I_scalar."
        )
    )
    parser.add_argument("--output-dir", default="outputs/acceptance_sweep")
    parser.add_argument("--n-samples", type=int, default=700)
    parser.add_argument("--n-alpha", type=int, default=25)
    parser.add_argument("--alpha-deg-min", type=float, default=3.0)
    parser.add_argument("--alpha-deg-max", type=float, default=40.0)
    parser.add_argument(
        "--alpha-definition",
        choices=["a_over_z0", "z0_over_a"],
        default="a_over_z0",
        help="a_over_z0: tan(alpha)=a/z0; z0_over_a: tan(alpha)=z0/a",
    )

    parser.add_argument("--n0", type=float, default=1.0)
    parser.add_argument("--ni", type=float, default=1.5)
    parser.add_argument("--z0", type=float, default=5.0)
    parser.add_argument("--zi", type=float, default=10.0)
    parser.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--r-max", type=float, default=0.30)
    parser.add_argument("--n-img", type=int, default=500)

    args = parser.parse_args()

    q, rows = run_sweep(
        n0=args.n0,
        ni=args.ni,
        z0=args.z0,
        zi=args.zi,
        lam=args.lam,
        r_max=args.r_max,
        n_samples=args.n_samples,
        alpha_deg_min=args.alpha_deg_min,
        alpha_deg_max=args.alpha_deg_max,
        n_alpha=args.n_alpha,
        alpha_definition=args.alpha_definition,
    )

    out_dir = Path(args.output_dir)
    params = _params_dict(args)
    csv_path, npz_path, params_path = save_outputs(out_dir, q, rows, params)

    # Recreate grids/coefs once for plotting selected cases.
    r = np.linspace(0.0, args.r_max, args.n_samples)
    fresnel = FresnelOvoidParax(n0=args.n0, ni=args.ni, z0=args.z0, zi=args.zi)
    ts, tp = fresnel.coeffs(r)
    ts = np.asarray(ts, dtype=float)
    tp = np.asarray(tp, dtype=float)

    _plot_sweep_summary(out_dir, rows)
    _plot_profiles(out_dir, r, q, ts, tp, rows, args.alpha_definition)
    _plot_2d_maps(out_dir, r, q, ts, tp, rows, args.n_img)

    best = max(rows, key=lambda d: d["rel_l1_delta"])
    print("Sweep complete")
    print(f"CSV: {csv_path}")
    print(f"NPZ: {npz_path}")
    print(f"Parameters: {params_path}")
    print(f"Figures: {out_dir / 'sweep_summary.png'}, {out_dir / 'profiles_selected_alpha.png'}, maps_alpha_*.png")
    print(
        "Max rel_l1_delta at "
        f"alpha={best['alpha_deg']:.3f} deg, a_eff={best['a_eff']:.6f}, rel_l1_delta={best['rel_l1_delta']:.6e}"
    )


if __name__ == "__main__":
    main()
