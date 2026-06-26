import argparse
import numpy as np
from vecdiff.fresnel import FresnelOvoidParax
from vecdiff.field_reconstruction import hankel_terms, make_observation_grid, reconstruct_2d_from_terms, radial_to_2d
from vecdiff.view import plot_field_2d_components, plot_field_2d_intensity


def main():
    parser = argparse.ArgumentParser(description="Plot 2D propagated fields with explicit angular dependence varphi(x,y).")
    parser.add_argument("--polarization", choices=["circular", "polar", "cartesian"], default="cartesian")
    parser.add_argument("--combination", choices=["all", "10", "01", "1i", "11"], default=None)
    parser.add_argument("--component-view", choices=["abs", "real", "imag"], default="abs")
    parser.add_argument("--show-incident", action="store_true", help="Plot compact incident intensity.")
    parser.add_argument(
        "--show-incident-full",
        dest="show_incident_full",
        action="store_true",
        help="Plot full incident components and intensity.",
    )
    parser.add_argument("--n-samples", type=int, default=700)
    parser.add_argument("--n-img", type=int, default=500)
    args = parser.parse_args()

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

    _, _, rho, varphi, extent = make_observation_grid(0.80 * q_max, args.n_img)
    _, _, rho_in, _, extent_in = make_observation_grid(r_max, args.n_img)

    combinations = {
        "10": (1.0 + 0.0j, 0.0 + 0.0j, "(1, 0)"),
        "01": (0.0 + 0.0j, 1.0 + 0.0j, "(0, 1)"),
        "1i": (1.0 + 0.0j, 1.0j, "(1, i)"),
        "11": (1.0 + 0.0j, 1.0 + 0.0j, "(1, 1)"),
    }

    selected = args.combination
    if selected is None:
        print(f"Polarization selected: {args.polarization}")
        print("Choose combination: all, 10, 01, 1i, 11")
        selected = input("Combination [all]: ").strip().lower() or "all"
        if selected not in {"all", "10", "01", "1i", "11"}:
            raise ValueError("Invalid combination. Use one of: all, 10, 01, 1i, 11.")

    if selected == "all":
        selected_combinations = list(combinations.values())
    else:
        selected_combinations = [combinations[selected]]

    labels = {"circular": ("L", "R"), "cartesian": ("x", "y"), "polar": ("r", r"\phi")}

    for c1, c2, label in selected_combinations:
        e1 = c1 * aperture
        e2 = c2 * aperture
        if args.show_incident or args.show_incident_full:
            in1 = radial_to_2d(e1, r, rho_in)
            in2 = radial_to_2d(e2, r, rho_in)
            if args.show_incident_full:
                plot_field_2d_components(
                    component1=in1,
                    component2=in2,
                    extent=extent_in,
                    labels=labels[args.polarization],
                    cmap="hot",
                    component_view=args.component_view,
                    title=f"{args.polarization} basis, incident input {label}",
                    field_symbol="E",
                )
            else:
                plot_field_2d_intensity(
                    component1=in1,
                    component2=in2,
                    extent=extent_in,
                    cmap="hot",
                    component_view=args.component_view,
                    title=f"{args.polarization} basis, incident intensity {label}",
                )
        terms = hankel_terms(r=r, q=q, tp=tp, ts=ts, e1=e1, e2=e2, polarization=args.polarization)
        out1, out2 = reconstruct_2d_from_terms(terms=terms, q=q, rho=rho, varphi=varphi, polarization=args.polarization)
        plot_field_2d_components(
            component1=out1,
            component2=out2,
            extent=extent,
            labels=labels[args.polarization],
            cmap="hot",
            component_view=args.component_view,
            title=f"{args.polarization} basis, input {label}",
            field_symbol="E'",
        )


if __name__ == "__main__":
    main()
