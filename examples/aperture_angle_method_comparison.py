from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

from vecdiff import CartesianSurface, Field, Grid
from vecdiff.coordinate_transformation import polar_grid_to_cartesian_grid
from vecdiff.polarization import polarization_from_components, polarization_map_from_field
from vecdiff.polarization_visualization import plot_polarization_map
from vecdiff.propagation import (
    propagate_to_focal_plane_through_diopter,
    propagate_to_focal_plane_through_diopter_fft,
)


def rel_l2(a, b):
    denom = max(np.linalg.norm(a), np.linalg.norm(b), 1e-30)
    return np.linalg.norm(a - b) / denom


def max_rel(a, b):
    scale = max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-30)
    return np.max(np.abs(a - b)) / scale


def make_incident_fields(case_name, amp_r, amp_xy, polar_grid, cart_grid):
    zeros_r = np.zeros_like(amp_r)
    zeros_xy = np.zeros_like(amp_xy)

    if case_name == "cartesian Ex":
        return (
            Field.from_cartesian(amp_r, zeros_r, polar_grid, symmetric=True),
            Field.from_cartesian(amp_xy, zeros_xy, cart_grid, symmetric=False),
        )
    if case_name == "circular ER":
        return (
            Field.from_circular(zeros_r, amp_r, polar_grid, symmetric=True),
            Field.from_circular(zeros_xy, amp_xy, cart_grid, symmetric=False),
        )
    if case_name == "polar Er":
        return (
            Field.from_polar(amp_r, zeros_r, polar_grid, symmetric=True),
            Field.from_polar(amp_xy, zeros_xy, cart_grid, symmetric=False),
        )
    raise ValueError(f"Unknown case: {case_name}")


def make_incident_polar_field(case_name, pupil, polar_grid):
    zeros = np.zeros_like(pupil)
    if case_name == "cartesian Ex":
        return Field.from_cartesian(pupil, zeros, polar_grid, symmetric=True)
    if case_name == "circular ER":
        return Field.from_circular(zeros, pupil, polar_grid, symmetric=True)
    if case_name == "polar Er":
        return Field.from_polar(pupil, zeros, polar_grid, symmetric=True)
    raise ValueError(f"Unknown case: {case_name}")


def field_intensity(field):
    return np.abs(field.x) ** 2 + np.abs(field.y) ** 2


def hankel_on_fft_grid(hankel_field, fft_grid):
    q = np.asarray(hankel_field.grid.r, dtype=float)
    varphi = np.asarray(hankel_field.grid.varphi, dtype=float)
    hx = polar_grid_to_cartesian_grid(
        hankel_field.x,
        q,
        varphi,
        fft_grid.X,
        fft_grid.Y,
        fill_value=0.0,
    )
    hy = polar_grid_to_cartesian_grid(
        hankel_field.y,
        q,
        varphi,
        fft_grid.X,
        fft_grid.Y,
        fill_value=0.0,
    )
    return hx, hy


def add_intensity(ax, image, extent, title):
    vmax = max(float(np.max(np.abs(image))), 1e-30)
    im = ax.imshow(
        np.abs(image),
        extent=extent,
        origin="lower",
        cmap="hot",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    return im


def add_difference(ax, image, extent, title):
    vmax = max(float(np.nanpercentile(image, 99.5)), 1e-30)
    im = ax.imshow(
        image,
        extent=extent,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        interpolation="lanczos",
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    return im


def add_pol(ax, x, y, ex, ey, title, *, ellipse_mode="cartesian"):
    pol = polarization_from_components(ex, ey)
    s0_vmax = max(float(np.nanpercentile(pol.s0, 99.8)), 1e-30)
    ax.imshow(
        pol.s0,
        extent=[float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))],
        origin="lower",
        cmap="hot",
        norm=PowerNorm(gamma=0.85, vmin=0.0, vmax=s0_vmax),
        interpolation="lanczos",
    )
    plot_polarization_map(
        x,
        y,
        pol,
        target_ellipses=28,
        max_ellipses=190,
        min_intensity_fraction=0.08,
        scale_by_intensity=True,
        intensity_scale_mode="power",
        arrow_length=0.14,
        color_by_phase=False,
        curve_kwargs={"linewidth": 0.45, "color": "black", "alpha": 0.9},
        arrowhead_kwargs={"linewidth": 0.8, "color": "white", "alpha": 0.95},
        ellipse_mode=ellipse_mode,
        ax=ax,
    )
    ax.set_title(title)


def native_incident_polarization_map(field, half_size, n_img):
    if getattr(field, "symmetry", None) != "polar":
        x, y, pol = polarization_map_from_field(field, half_size=half_size, n_img=n_img)
        return x, y, pol.ex, pol.ey, "cartesian"

    x = np.linspace(-half_size, half_size, n_img)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    radial_axis = np.asarray(field.grid.r, dtype=float)
    angular_axis = np.asarray(field.grid.varphi, dtype=float)
    rho = np.sqrt(xx**2 + yy**2)

    def sample_native(component):
        component = np.asarray(component)
        if component.ndim == 1:
            return np.interp(rho, radial_axis, component, left=component[0], right=0.0)
        return polar_grid_to_cartesian_grid(
            component,
            radial_axis,
            angular_axis,
            xx,
            yy,
            fill_value=0.0,
        )

    er = sample_native(field.r)
    ephi = sample_native(field.phi)
    return xx, yy, er, ephi, "polar"


def save_case_figure(
    *,
    output_path,
    case_name,
    diopter,
    wavelength,
    theta_deg,
    aperture_radius,
    waist,
    incident_field,
    hankel_field,
    fft_field,
    errors,
    q_plot_view,
):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.8))

    x_in, y_in, pol_in = polarization_map_from_field(
        incident_field,
        half_size=aperture_radius,
        n_img=260,
    )
    extent_in = [
        float(np.min(x_in)),
        float(np.max(x_in)),
        float(np.min(y_in)),
        float(np.max(y_in)),
    ]
    add_intensity(axes[0, 0], pol_in.s0, extent_in, "Incident intensity")
    pol_x, pol_y, pol_a, pol_b, pol_basis = native_incident_polarization_map(
        incident_field,
        aperture_radius,
        260,
    )
    add_pol(
        axes[1, 0],
        pol_x,
        pol_y,
        pol_a,
        pol_b,
        "Incident polarization",
        ellipse_mode=pol_basis,
    )

    q = np.asarray(hankel_field.grid.r, dtype=float)
    varphi = np.asarray(hankel_field.grid.varphi, dtype=float)
    q_plot = np.linspace(-q_plot_view, q_plot_view, 260)
    qx, qy = np.meshgrid(q_plot, q_plot, indexing="xy")
    h_x = polar_grid_to_cartesian_grid(hankel_field.x, q, varphi, qx, qy, fill_value=0.0)
    h_y = polar_grid_to_cartesian_grid(hankel_field.y, q, varphi, qx, qy, fill_value=0.0)
    h_i = np.abs(h_x) ** 2 + np.abs(h_y) ** 2
    extent_q = [-q_plot_view, q_plot_view, -q_plot_view, q_plot_view]
    add_intensity(axes[0, 1], h_i, extent_q, "Hankel intensity")
    add_pol(axes[1, 1], qx, qy, h_x, h_y, "Hankel final polarization")

    x_axis = fft_field.grid.X[0]
    y_axis = fft_field.grid.Y[:, 0]
    cols = np.flatnonzero(np.abs(x_axis) <= q_plot_view)
    rows = np.flatnonzero(np.abs(y_axis) <= q_plot_view)
    y_slice = slice(rows[0], rows[-1] + 1)
    x_slice = slice(cols[0], cols[-1] + 1)
    fx = fft_field.x[y_slice, x_slice]
    fy = fft_field.y[y_slice, x_slice]
    fgrid_x = fft_field.grid.X[y_slice, x_slice]
    fgrid_y = fft_field.grid.Y[y_slice, x_slice]
    f_i = np.abs(fx) ** 2 + np.abs(fy) ** 2
    extent_fft = [
        float(np.min(fgrid_x)),
        float(np.max(fgrid_x)),
        float(np.min(fgrid_y)),
        float(np.max(fgrid_y)),
    ]
    add_intensity(axes[0, 2], f_i, extent_fft, "FFT intensity")
    add_pol(axes[1, 2], fgrid_x, fgrid_y, fx, fy, "FFT final polarization")

    hfx, hfy = hankel_on_fft_grid(hankel_field, fft_field.grid)
    dx = np.abs(hfx[y_slice, x_slice] - fx)
    dy = np.abs(hfy[y_slice, x_slice] - fy)
    add_difference(axes[0, 3], dx, extent_fft, r"$|E_x^{Hankel}-E_x^{FFT}|$")
    add_difference(axes[1, 3], dy, extent_fft, r"$|E_y^{Hankel}-E_y^{FFT}|$")

    for ax in axes[:, 0]:
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
    for ax in axes[:, 1:].flat:
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")

    title = (
        f"{case_name}: aperture-angle sweep sample, "
        rf"$\theta_{{max}}={theta_deg:.2f}^\circ$"
    )
    fig.suptitle(title, fontsize=15)
    metadata = (
        f"Diopter: n0={diopter.n0}, ni={diopter.ni}, z0={diopter.z0}, zi={diopter.zi}; "
        f"wavelength={wavelength}; tan(theta_max)=R/|z0|; R={aperture_radius:.6g}; "
        f"pupil radius=0.95R={0.95 * aperture_radius:.6g}; incident polarization={case_name}; "
        f"rel_l2={errors['rel_l2']:.3e}; max_rel={errors['max_rel']:.3e}; "
        f"center_abs_diff={errors['center_abs_diff']:.3e}"
    )
    fig.text(
        0.01,
        0.025,
        metadata,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2.0},
    )
    fig.tight_layout(rect=(0.0, 0.065, 1.0, 0.94))
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def run_sweep():
    output_dir = Path(__file__).resolve().parent / "output" / "aperture_angle_method_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    wavelength = 532e-6
    diopter = CartesianSurface(n0=1.0, ni=1.5, z0=-10.0, zi=6.0)
    theta_degrees = np.array([5.0, 10.0, 15.0, 20.0, np.rad2deg(np.arctan(5.1 / abs(diopter.z0)))])
    cases = ["cartesian Ex", "circular ER", "polar Er"]

    n_radial = 1000
    n_phi = 100
    n_k = 800
    records = []

    for case_name in cases:
        for theta_deg in theta_degrees:
            theta = np.deg2rad(theta_deg)
            aperture_radius = abs(diopter.z0) * np.tan(theta)

            r = np.linspace(0.0, aperture_radius, n_radial)
            varphi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
            polar_grid = Grid.from_polar(r, varphi)
            pupil = (r <= 0.95 * aperture_radius).astype(float)
            incident = make_incident_polar_field(case_name, pupil, polar_grid)

            propagated_half_size = 4.0 * 3.8317059702075125 / (0.95 * aperture_radius)
            k_axis = np.linspace(-propagated_half_size, propagated_half_size, n_k)
            KX, KY = np.meshgrid(k_axis, k_axis, indexing="xy")
            kgrid = Grid.from_cartesian(KX, KY, domain="k")
            kx = kgrid.X[0]
            ky = kgrid.Y[:, 0]
            ix_nonneg = np.where(kx >= 0.0)[0]
            iy0 = int(np.argmin(np.abs(ky)))
            q_axis = kx[ix_nonneg]
            q_corner = np.sqrt(2.0) * propagated_half_size
            q_plot_radial = np.linspace(0.0, q_corner, n_k)
            q = np.unique(np.concatenate((q_axis, q_plot_radial)))

            hankel = propagate_to_focal_plane_through_diopter(incident, diopter, q)
            fft = propagate_to_focal_plane_through_diopter_fft(incident, diopter, kgrid=kgrid)

            h_axis = np.interp(q_axis, q, hankel.x[0])
            f_axis = fft.x[iy0, ix_nonneg]
            keep = q_axis <= 0.75 * q_axis[-1]
            errors = {
                "case": case_name,
                "theta_deg": theta_deg,
                "aperture_radius": aperture_radius,
                "rel_l2": rel_l2(h_axis[keep], f_axis[keep]),
                "max_rel": max_rel(h_axis[keep], f_axis[keep]),
                "center_abs_diff": abs(h_axis[0] - f_axis[0]),
            }
            records.append(errors)

            safe_case = case_name.lower().replace(" ", "_")
            figure_path = output_dir / f"{safe_case}_theta_{theta_deg:04.1f}.png"
            save_case_figure(
                output_path=figure_path,
                case_name=case_name,
                diopter=diopter,
                wavelength=wavelength,
                theta_deg=theta_deg,
                aperture_radius=aperture_radius,
                waist=0.0,
                incident_field=incident,
                hankel_field=hankel,
                fft_field=fft,
                errors=errors,
                q_plot_view=propagated_half_size,
            )
            print(
                f"{case_name:13s} theta={theta_deg:4.1f} R={aperture_radius:.6f} "
                f"rel_l2={errors['rel_l2']:.6e} max_rel={errors['max_rel']:.6e} "
                f"figure={figure_path}"
            )

    save_summary(output_dir, records, diopter, wavelength)
    return output_dir, records


def save_summary(output_dir, records, diopter, wavelength):
    cases = list(dict.fromkeys(record["case"] for record in records))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    for case_name in cases:
        subset = [record for record in records if record["case"] == case_name]
        theta = np.array([record["theta_deg"] for record in subset])
        rel = np.array([record["rel_l2"] for record in subset])
        mrel = np.array([record["max_rel"] for record in subset])
        axes[0].semilogy(theta, rel, marker="o", label=case_name)
        axes[1].semilogy(theta, mrel, marker="o", label=case_name)

    axes[0].set_title("Hankel vs FFT relative L2")
    axes[1].set_title("Hankel vs FFT max-relative error")
    for ax in axes:
        ax.set_xlabel(r"$\theta_{max}$ [deg]")
        ax.grid(True, which="both", alpha=0.35)
        ax.legend()

    fig.suptitle(
        f"Aperture-angle method comparison; "
        f"diopter n0={diopter.n0}, ni={diopter.ni}, z0={diopter.z0}, zi={diopter.zi}; "
        f"wavelength={wavelength}; tan(theta_max)=R/|z0|"
    )
    fig.savefig(output_dir / "summary_errors.png", dpi=240)
    plt.close(fig)

    csv_path = output_dir / "summary_errors.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("case,theta_deg,aperture_radius,rel_l2,max_rel,center_abs_diff\n")
        for record in records:
            f.write(
                f"{record['case']},{record['theta_deg']:.8g},"
                f"{record['aperture_radius']:.12g},{record['rel_l2']:.12e},"
                f"{record['max_rel']:.12e},{record['center_abs_diff']:.12e}\n"
            )


if __name__ == "__main__":
    output_dir, _ = run_sweep()
    print(f"Saved aperture-angle comparison outputs to {output_dir}")
