import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vecdiff.CartesianSurfaces import CartesianSurface
from vecdiff.coordinate_transformation import cartesian_to_polar, circular_to_polar
from vecdiff.fresnel import FresnelOvoidParax
from vecdiff.field_reconstruction import (
    hankel_terms,
    make_observation_grid,
    radial_to_2d,
    reconstruct_2d_from_terms,
)


def render_mode(arr, mode):
    if mode == "abs":
        return np.abs(arr)
    if mode == "real":
        return np.real(arr)
    if mode == "imag":
        return np.imag(arr)
    raise ValueError("`component_view` must be one of: 'abs', 'real', 'imag'.")


def save_case_figure(
    out_path,
    polarization,
    combination_label,
    incident_1,
    incident_2,
    propagated_1,
    propagated_2,
    incident_z,
    propagated_z,
    extent_incident,
    extent_propagated,
    labels,
    component_view,
    metadata_text,
    dpi,
    include_z_component,
):
    in1 = render_mode(incident_1, component_view)
    in2 = render_mode(incident_2, component_view)
    out1 = render_mode(propagated_1, component_view)
    out2 = render_mode(propagated_2, component_view)
    inz = render_mode(incident_z, component_view) if include_z_component else None
    outz = render_mode(propagated_z, component_view) if include_z_component else None

    inten_in = in1**2 + in2**2 + (inz**2 if include_z_component else 0.0)
    inten_out = out1**2 + out2**2 + (outz**2 if include_z_component else 0.0)

    ncols = 4 if include_z_component else 3
    fig_w = 18 if include_z_component else 14
    fig, axes = plt.subplots(2, ncols, figsize=(fig_w, 8.5), constrained_layout=True)

    panels = [(axes[0, 0], in1, rf"Incident $E_{{{labels[0]}}}$", extent_incident, r"$x$"), (axes[0, 1], in2, rf"Incident $E_{{{labels[1]}}}$", extent_incident, r"$x$")]
    if include_z_component:
        panels.append((axes[0, 2], inz, r"Incident $E_{z}$", extent_incident, r"$x$"))
        panels.append((axes[0, 3], inten_in, r"Incident $|E_x|^2 + |E_y|^2 + |E_z|^2$", extent_incident, r"$x$"))
    else:
        panels.append((axes[0, 2], inten_in, r"Incident $|E_1|^2 + |E_2|^2$", extent_incident, r"$x$"))

    panels += [(axes[1, 0], out1, rf"Propagated $E'_{{{labels[0]}}}$", extent_propagated, r"$q_x$"), (axes[1, 1], out2, rf"Propagated $E'_{{{labels[1]}}}$", extent_propagated, r"$q_x$")]
    if include_z_component:
        panels.append((axes[1, 2], outz, r"Propagated $E'_{z}$", extent_propagated, r"$q_x$"))
        panels.append((axes[1, 3], inten_out, r"Propagated $|E'_x|^2 + |E'_y|^2 + |E'_z|^2$", extent_propagated, r"$q_x$"))
    else:
        panels.append((axes[1, 2], inten_out, r"Propagated $|E'_1|^2 + |E'_2|^2$", extent_propagated, r"$q_x$"))

    for ax, img, title, extent, xlab in panels:
        vmax = float(np.max(np.abs(img)))
        if vmax == 0.0:
            vmax = 1.0
        im = ax.imshow(np.abs(img), extent=extent, origin="lower", cmap="hot", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(r"$y$" if xlab == r"$x$" else r"$q_y$")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{polarization.capitalize()} polarization | input {combination_label}", fontsize=14)
    fig.text(0.01, 0.01, metadata_text, ha="left", va="bottom", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_incident_figure(
    out_path,
    polarization,
    combination_label,
    incident_1,
    incident_2,
    incident_z,
    extent_incident,
    labels,
    component_view,
    metadata_text,
    dpi,
    include_z_component,
):
    in1 = render_mode(incident_1, component_view)
    in2 = render_mode(incident_2, component_view)
    inz = render_mode(incident_z, component_view) if include_z_component else None
    inten_in = in1**2 + in2**2 + (inz**2 if include_z_component else 0.0)

    ncols = 4 if include_z_component else 3
    fig_w = 18 if include_z_component else 14
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 4.2), constrained_layout=True)
    if ncols == 3:
        panels = [
            (axes[0], in1, rf"Incident $E_{{{labels[0]}}}$"),
            (axes[1], in2, rf"Incident $E_{{{labels[1]}}}$"),
            (axes[2], inten_in, r"Incident $|E_1|^2 + |E_2|^2$"),
        ]
    else:
        panels = [
            (axes[0], in1, rf"Incident $E_{{{labels[0]}}}$"),
            (axes[1], in2, rf"Incident $E_{{{labels[1]}}}$"),
            (axes[2], inz, r"Incident $E_{z}$"),
            (axes[3], inten_in, r"Incident $|E_x|^2 + |E_y|^2 + |E_z|^2$"),
        ]

    for ax, img, title in panels:
        vmax = float(np.max(np.abs(img)))
        if vmax == 0.0:
            vmax = 1.0
        im = ax.imshow(np.abs(img), extent=extent_incident, origin="lower", cmap="hot", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{polarization.capitalize()} polarization | input {combination_label} | Incident", fontsize=14)
    fig.text(0.01, 0.01, metadata_text, ha="left", va="bottom", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_propagated_figure(
    out_path,
    polarization,
    combination_label,
    propagated_1,
    propagated_2,
    propagated_z,
    extent_propagated,
    labels,
    component_view,
    metadata_text,
    dpi,
    include_z_component,
):
    out1 = render_mode(propagated_1, component_view)
    out2 = render_mode(propagated_2, component_view)
    outz = render_mode(propagated_z, component_view) if include_z_component else None
    inten_out = out1**2 + out2**2 + (outz**2 if include_z_component else 0.0)

    ncols = 4 if include_z_component else 3
    fig_w = 18 if include_z_component else 14
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 4.2), constrained_layout=True)
    if ncols == 3:
        panels = [
            (axes[0], out1, rf"Propagated $E'_{{{labels[0]}}}$"),
            (axes[1], out2, rf"Propagated $E'_{{{labels[1]}}}$"),
            (axes[2], inten_out, r"Propagated $|E'_1|^2 + |E'_2|^2$"),
        ]
    else:
        panels = [
            (axes[0], out1, rf"Propagated $E'_{{{labels[0]}}}$"),
            (axes[1], out2, rf"Propagated $E'_{{{labels[1]}}}$"),
            (axes[2], outz, r"Propagated $E'_{z}$"),
            (axes[3], inten_out, r"Propagated $|E'_x|^2 + |E'_y|^2 + |E'_z|^2$"),
        ]

    for ax, img, title in panels:
        vmax = float(np.max(np.abs(img)))
        if vmax == 0.0:
            vmax = 1.0
        im = ax.imshow(np.abs(img), extent=extent_propagated, origin="lower", cmap="hot", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(r"$q_x$")
        ax.set_ylabel(r"$q_y$")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{polarization.capitalize()} polarization | input {combination_label} | Propagated", fontsize=14)
    fig.text(0.01, 0.01, metadata_text, ha="left", va="bottom", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Batch export incident+propagated figures for all polarization/combinations.")
    parser.add_argument("--output-root", default="outputs/export_results", help="Root directory for exported results.")
    parser.add_argument("--component-view", choices=["abs", "real", "imag"], default="abs")
    parser.add_argument("--n-samples", type=int, default=700)
    parser.add_argument("--n-img", type=int, default=500)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf"])
    parser.add_argument("--save-npz", action="store_true", help="Save arrays and metadata per case.")
    parser.add_argument("--no-z-component", action="store_true", help="Disable reconstructed longitudinal component Ez in plots/data.")
    parser.add_argument("--same-plot", action="store_true", help="Use a single combined figure (incident + propagated). Default is separate figures.")
    args = parser.parse_args()
    include_z_component = not args.no_z_component

    lam = 532e-6
    n0 = 1.0
    ni = 1.5
    z0 = 5.0
    zi = 10.0
    r_max = 0.30
    aperture_radius = 0.22

    r = np.linspace(0.0, r_max, args.n_samples)
    q_max = ni / (lam * zi) * r_max
    q = np.linspace(0.0, q_max, args.n_samples)
    aperture = (r <= aperture_radius).astype(float).astype(complex)

    fresnel = FresnelOvoidParax(n0=n0, ni=ni, z0=z0, zi=zi)
    ts, tp = fresnel.coeffs(r)
    ovoid = CartesianSurface(n0=n0, z0=z0, ni=ni, zi=zi)

    _, _, rho_in, varphi_in, extent_in = make_observation_grid(r_max, args.n_img)
    _, _, rho_out, varphi_out, extent_out = make_observation_grid(0.80 * q_max, args.n_img)

    combinations = {
        "10": (1.0 + 0.0j, 0.0 + 0.0j, "(1, 0)"),
        "01": (0.0 + 0.0j, 1.0 + 0.0j, "(0, 1)"),
        "1i": (1.0 + 0.0j, 1.0j, "(1, i)"),
        "11": (1.0 + 0.0j, 1.0 + 0.0j, "(1, 1)"),
    }
    labels = {"circular": ("L", "R"), "cartesian": ("x", "y"), "polar": ("r", r"\phi")}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / f"run_{stamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    metadata_text = (
        f"n0={n0}, ni={ni}, z0={z0}, zi={zi}, lambda={lam}, R={r_max}, aperture={aperture_radius} | "
        "Observation axes: q_x, q_y | q = (2*pi*n/(lambda*z_i))*r | "
        "Ez = r/(z_(o,i)-z(r))*Er"
    )

    def safe_longitudinal(er_field, rho_plane, z_ref):
        sag = ovoid.z(rho_plane)
        denom = z_ref - sag
        eps = 1e-12
        denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps + (denom == 0.0) * eps, denom)
        return (rho_plane / denom) * er_field

    for polarization in ("circular", "cartesian", "polar"):
        for comb_key, (c1, c2, comb_label) in combinations.items():
            e1 = c1 * aperture
            e2 = c2 * aperture
            incident_1 = radial_to_2d(e1, r, rho_in)
            incident_2 = radial_to_2d(e2, r, rho_in)

            terms = hankel_terms(r=r, q=q, tp=tp, ts=ts, e1=e1, e2=e2, polarization=polarization)
            propagated_1, propagated_2 = reconstruct_2d_from_terms(
                terms=terms,
                q=q,
                rho=rho_out,
                varphi=varphi_out,
                polarization=polarization,
            )

            if polarization == "polar":
                er_in = incident_1
                er_out = propagated_1
            elif polarization == "cartesian":
                er_in, _ = cartesian_to_polar(incident_1, incident_2, varphi_in)
                er_out, _ = cartesian_to_polar(propagated_1, propagated_2, varphi_out)
            else:
                er_in, _ = circular_to_polar(incident_1, incident_2, varphi_in)
                er_out, _ = circular_to_polar(propagated_1, propagated_2, varphi_out)

            r_obs = rho_out * lam * zi / (2.0 * π * ni)
            incident_z = safe_longitudinal(er_in, rho_in, z0) if include_z_component else np.zeros_like(incident_1)
            propagated_z = safe_longitudinal(er_out, r_obs, zi) if include_z_component else np.zeros_like(propagated_1)

            case_dir = run_root / polarization / comb_key
            case_dir.mkdir(parents=True, exist_ok=True)

            for fmt in args.formats:
                if args.same_plot:
                    out_file = case_dir / f"{polarization}_{comb_key}_{args.component_view}_combined.{fmt}"
                    save_case_figure(
                        out_path=out_file,
                        polarization=polarization,
                        combination_label=comb_label,
                        incident_1=incident_1,
                        incident_2=incident_2,
                        propagated_1=propagated_1,
                        propagated_2=propagated_2,
                        incident_z=incident_z,
                        propagated_z=propagated_z,
                        extent_incident=extent_in,
                        extent_propagated=extent_out,
                        labels=labels[polarization],
                        component_view=args.component_view,
                        metadata_text=metadata_text,
                        dpi=args.dpi,
                        include_z_component=include_z_component,
                    )
                else:
                    out_file_in = case_dir / f"{polarization}_{comb_key}_{args.component_view}_incident.{fmt}"
                    out_file_out = case_dir / f"{polarization}_{comb_key}_{args.component_view}_propagated.{fmt}"
                    save_incident_figure(
                        out_path=out_file_in,
                        polarization=polarization,
                        combination_label=comb_label,
                        incident_1=incident_1,
                        incident_2=incident_2,
                        incident_z=incident_z,
                        extent_incident=extent_in,
                        labels=labels[polarization],
                        component_view=args.component_view,
                        metadata_text=metadata_text,
                        dpi=args.dpi,
                        include_z_component=include_z_component,
                    )
                    save_propagated_figure(
                        out_path=out_file_out,
                        polarization=polarization,
                        combination_label=comb_label,
                        propagated_1=propagated_1,
                        propagated_2=propagated_2,
                        propagated_z=propagated_z,
                        extent_propagated=extent_out,
                        labels=labels[polarization],
                        component_view=args.component_view,
                        metadata_text=metadata_text,
                        dpi=args.dpi,
                        include_z_component=include_z_component,
                    )

            if args.save_npz:
                np.savez(
                    case_dir / f"{polarization}_{comb_key}_data.npz",
                    r=r,
                    q=q,
                    tp=tp,
                    ts=ts,
                    incident_1=incident_1,
                    incident_2=incident_2,
                    incident_z=incident_z,
                    propagated_1=propagated_1,
                    propagated_2=propagated_2,
                    propagated_z=propagated_z,
                    polarization=polarization,
                    combination=comb_key,
                    combination_label=comb_label,
                    component_view=args.component_view,
                    include_z_component=include_z_component,
                    formula_q="q = (2*pi*n/(lambda*z_i))*r",
                    formula_ez="Ez = r/(z_(o,i)-z(r))*Er",
                    n0=n0,
                    ni=ni,
                    z0=z0,
                    zi=zi,
                    lam=lam,
                    r_max=r_max,
                    aperture_radius=aperture_radius,
                )

    print(f"Export complete: {run_root}")


if __name__ == "__main__":
    main()
