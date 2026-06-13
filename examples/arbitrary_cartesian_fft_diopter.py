from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

from vecdiff import CartesianSurface, FieldCartesian, Grid
from vecdiff.polarization import polarization_from_field
from vecdiff.polarization_visualization import plot_polarization_map


def recover_Ez_from_transverse_k_field(field, wavelength, n):
    """Recover Ez on a k-domain Cartesian field from k dot E = 0."""
    if field.grid.type != "cartesian" or field.grid.domain != "k":
        raise ValueError("field must be sampled on a Cartesian k-domain grid.")

    k0n = 2.0 * np.pi * n / wavelength
    kt2 = field.grid.X**2 + field.grid.Y**2
    kz2 = k0n**2 - kt2
    KZ = np.empty_like(field.grid.X, dtype=complex)
    propagating = kz2 >= 0.0
    KZ[propagating] = np.sqrt(kz2[propagating])
    KZ[~propagating] = np.inf

    with np.errstate(divide="ignore", invalid="ignore"):
        Ez = -(field.grid.X * field.x + field.grid.Y * field.y) / KZ
    return np.where(np.isfinite(Ez), Ez, 0.0)


def lower_half_phase_flip(grid, *components):
    """Apply a pi phase step to the y < 0 half of the provided components."""
    phase = np.where(grid.Y < 0.0, -1.0, 1.0)
    return tuple(component * phase for component in components)


def electric_dipole_field_on_G(grid, wavelength, n0, z_object, dipole, pupil_radius):
    """Full electric dipole field on G, the object-centered sphere tangent to z=0.

    The global factor 1/(4*pi*eps0) is omitted because it only rescales plots.
    """
    k = 2.0 * np.pi * n0 / wavelength
    sphere_radius = abs(float(z_object))
    rho2 = grid.R**2
    inside_G = rho2 < sphere_radius**2

    rx = grid.X
    ry = grid.Y
    rz = -np.sign(float(z_object)) * np.sqrt(np.maximum(sphere_radius**2 - rho2, 0.0))
    R = np.full_like(grid.R, sphere_radius, dtype=float)

    sx = np.where(inside_G, rx / R, 0.0)
    sy = np.where(inside_G, ry / R, 0.0)
    sz = np.where(inside_G, rz / R, 0.0)

    px, py, pz = np.asarray(dipole, dtype=complex)
    s_dot_p = sx * px + sy * py + sz * pz

    transverse_x = px - sx * s_dot_p
    transverse_y = py - sy * s_dot_p
    transverse_z = pz - sz * s_dot_p

    near_x = 3.0 * sx * s_dot_p - px
    near_y = 3.0 * sy * s_dot_p - py
    near_z = 3.0 * sz * s_dot_p - pz

    radiative = k**2 / R
    near = 1.0 / R**3 - 1.0j * k / R**2
    phase = np.exp(1.0j * k * R)
    pupil = np.exp(-(grid.R / pupil_radius) ** 16) * inside_G

    Ex = phase * (radiative * transverse_x + near * near_x) * pupil
    Ey = phase * (radiative * transverse_y + near * near_y) * pupil
    Ez = phase * (radiative * transverse_z + near * near_z) * pupil
    return Ex, Ey, Ez


def save_field_figure(field, title, path, *, x_label, y_label, extent_scale=1.0, view=None):
    Ex = np.abs(field.x)
    Ey = np.abs(field.y)
    Ez = None if field.z is None else np.abs(field.z)
    intensity = Ex**2 + Ey**2 if Ez is None else Ex**2 + Ey**2 + Ez**2
    grid = field.grid
    X = grid.X
    Y = grid.Y

    if view is not None:
        x_axis = X[0] / extent_scale
        y_axis = Y[:, 0] / extent_scale
        cols = np.flatnonzero(np.abs(x_axis) <= view)
        rows = np.flatnonzero(np.abs(y_axis) <= view)
        if cols.size and rows.size:
            x_slice = slice(cols[0], cols[-1] + 1)
            y_slice = slice(rows[0], rows[-1] + 1)
            Ex = Ex[y_slice, x_slice]
            Ey = Ey[y_slice, x_slice]
            if Ez is not None:
                Ez = Ez[y_slice, x_slice]
            intensity = intensity[y_slice, x_slice]
            X = X[y_slice, x_slice]
            Y = Y[y_slice, x_slice]

    extent = [
        float(np.min(X) / extent_scale),
        float(np.max(X) / extent_scale),
        float(np.min(Y) / extent_scale),
        float(np.max(Y) / extent_scale),
    ]

    component_vmax = max(float(np.max(Ex)), float(np.max(Ey)), 1e-30)
    if Ez is not None:
        component_vmax = max(component_vmax, float(np.max(Ez)))
    intensity_vmax = max(float(np.max(intensity)), 1e-30)
    panels = [
        (Ex, r"$|E_x|$", component_vmax, None),
        (Ey, r"$|E_y|$", component_vmax, None),
    ]
    if Ez is not None:
        ez_vmax = max(float(np.max(Ez)), 1e-30)
        panels.append((Ez, r"$|E_z|$", ez_vmax, None))
        intensity_label = r"$|E_x|^2+|E_y|^2+|E_z|^2$"
    else:
        intensity_label = r"$|E_x|^2+|E_y|^2$"
    panels.append(
        (
            intensity,
            intensity_label,
            intensity_vmax,
            PowerNorm(gamma=0.45, vmin=0.0, vmax=intensity_vmax),
        )
    )

    fig, axes = plt.subplots(1, len(panels), figsize=(4.35 * len(panels), 4.2), constrained_layout=True)
    for ax, (image, panel_title, vmax, norm) in zip(axes, panels):
        im = ax.imshow(
            image,
            extent=extent,
            origin="lower",
            cmap="magma",
            vmin=None if norm is not None else 0.0,
            vmax=None if norm is not None else vmax,
            norm=norm,
            interpolation="lanczos",
        )
        ax.set_title(panel_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.savefig(path, dpi=220)
    return fig


def cropped_field(field, view):
    x_axis = field.grid.X[0]
    y_axis = field.grid.Y[:, 0]
    cols = np.flatnonzero(np.abs(x_axis) <= view)
    rows = np.flatnonzero(np.abs(y_axis) <= view)
    if not cols.size or not rows.size:
        return field

    x_slice = slice(cols[0], cols[-1] + 1)
    y_slice = slice(rows[0], rows[-1] + 1)
    grid = Grid.from_cartesian(
        field.grid.X[y_slice, x_slice],
        field.grid.Y[y_slice, x_slice],
        domain=field.grid.domain,
        dual=field.grid.dual,
    )
    return FieldCartesian(
        field.x[y_slice, x_slice],
        field.y[y_slice, x_slice],
        grid=grid,
        symmetric=False,
    )


def save_polarization_figure(field, title, path, *, view):
    field_view = cropped_field(field, view)
    pol = polarization_from_field(field_view)
    s0_vmax = max(float(np.nanpercentile(pol.s0, 99.8)), 1e-30)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    
    # Scalar Field. 
    im = ax.imshow(
        pol.s0,
        extent=[
            float(np.min(field_view.grid.X)),
            float(np.max(field_view.grid.X)),
            float(np.min(field_view.grid.Y)),
            float(np.max(field_view.grid.Y)),
        ],
        origin="lower",
        cmap="hot",
        norm=PowerNorm(gamma=0.85, vmin=0.0, vmax=s0_vmax),
        interpolation="lanczos",
    )
    fig.colorbar(im, ax=ax, label=r"$|E_x|^2+|E_y|^2$")


    plot_polarization_map(
        field_view.grid.X,
        field_view.grid.Y,
        pol,
        target_ellipses=54,
        max_ellipses=None,
        min_intensity_fraction=0.06,
        scale_by_intensity=True,
        intensity_scale_mode="power",
        arrow_length=0.16,
        color_by_phase=False,
        curve_kwargs={"linewidth": 0.45, "color": "black", "alpha": 0.92},
        arrowhead_kwargs={"linewidth": 0.95, "color": "white", "alpha": 0.92},
        ax=ax,
    )


    ax.set_title(title)
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    fig.savefig(path, dpi=360)
    return fig


def main():
    wavelength = 0.532
    diopter = CartesianSurface(n0=1.0, ni=4.0, z0=-16.0, zi=30.0)
    pupil_radius = 13.6
    include_z_component = False

    n = 1536
    half_size = 80.0
    axis = np.linspace(-half_size, half_size, n, endpoint=False)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    grid = Grid.from_cartesian(X, Y)

    E_Gx, E_Gy, E_Gz = electric_dipole_field_on_G(
        grid,
        wavelength=wavelength,
        n0=diopter.n0,
        z_object=diopter.z0,
        dipole=(0.0, 1.0, 0.0),
        pupil_radius=pupil_radius,
    )



    (E_Gy,) = lower_half_phase_flip(grid, E_Gy)

    E_G = FieldCartesian(
        E_Gx,
        E_Gy,
        grid=grid,
        symmetric=False,
        z=E_Gz if include_z_component else None,
    )
    E_out = E_G.propagate_through_diopter(diopter.zi, diopter, method="fft")
    if include_z_component:
        E_out.z = recover_Ez_from_transverse_k_field(E_out, wavelength=wavelength, n=diopter.ni)

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)

    save_field_figure(
        E_G,
        r"Incident $E_G$ on the object-centered tangent sphere",
        output_dir / "arbitrary_cartesian_fft_diopter_incident.png",
        x_label=r"$x/\lambda$",
        y_label=r"$y/\lambda$",
        extent_scale=wavelength,
    )
    save_field_figure(
        E_out,
        r"FFT propagation of $E_G$ through a diopter",
        output_dir / "arbitrary_cartesian_fft_diopter.png",
        x_label=r"$k_x$",
        y_label=r"$k_y$",
        view=2.5,
    )
    save_polarization_figure(
        E_out,
        r"Polarization of propagated $E_G$",
        output_dir / "arbitrary_cartesian_fft_diopter_polarization.png",
        view=1.35,
    )

    plt.show()


if __name__ == "__main__":
    main()
