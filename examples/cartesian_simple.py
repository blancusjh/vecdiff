import numpy as np
import matplotlib.pyplot as plt

from vecdiff import CartesianSurface, FieldCartesian, Grid, generate_Ez_cartesian
from vecdiff.coordinate_transformation import polar_grid_to_cartesian_grid
from vecdiff.view import plot_field


def _component_to_cartesian(component, radial_axis, angular_axis, xx, yy):
    component = np.asarray(component, dtype=complex)
    rho = np.sqrt(xx**2 + yy**2)

    if component.ndim == 1:
        return np.interp(rho, radial_axis, component, left=component[0], right=0.0)

    if component.ndim == 2 and component.shape[0] == 1:
        row = component[0]
        return np.interp(rho, radial_axis, row, left=row[0], right=0.0)

    return polar_grid_to_cartesian_grid(
        component,
        radial_axis,
        angular_axis,
        xx,
        yy,
        fill_value=0.0,
    )


def plot_propagated_xyz(
    field,
    half_size,
    wavelength,
    n=1.0,
    n_img=500,
    component_view="abs",
    cmap="hot",
):
    radial_axis = np.asarray(field.grid.r, dtype=float)
    angular_axis = np.asarray(field.grid.varphi, dtype=float)

    x = np.linspace(-half_size, half_size, n_img)
    xx, yy = np.meshgrid(x, x)
    extent = [-half_size, half_size, -half_size, half_size]

    Ex = _component_to_cartesian(field.x, radial_axis, angular_axis, xx, yy)
    Ey = _component_to_cartesian(field.y, radial_axis, angular_axis, xx, yy)
    Ez = generate_Ez_cartesian(
        Ex,
        Ey,
        Grid.from_cartesian(xx, yy),
        wavelength=wavelength,
        n=n,
        method="exact",
        direction="+z",
    )

    components = [
        Ex,
        Ey,
        Ez,
    ]
    total_intensity = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

    if component_view == "abs":
        rep = np.abs
        label = "abs"
    elif component_view == "real":
        rep = np.real
        label = "real"
    elif component_view == "imag":
        rep = np.imag
        label = "imag"
    else:
        raise ValueError("component_view must be one of: 'abs', 'real', 'imag'.")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    for ax, component, title in zip(axes, components, (r"$E_x$", r"$E_y$", r"$E_z$")):
        image = rep(component)
        vmax = float(np.max(np.abs(image)))
        if vmax == 0.0:
            vmax = 1.0
        im = ax.imshow(np.abs(image), extent=extent, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(f"{label} {title}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    vmax = float(np.max(total_intensity))
    if vmax == 0.0:
        vmax = 1.0
    im = axes[-1].imshow(
        total_intensity,
        extent=extent,
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
    )
    axes[-1].set_title(r"$|E_x|^2 + |E_y|^2 + |E_z|^2$")
    axes[-1].set_xlabel(r"$x$")
    axes[-1].set_ylabel(r"$y$")
    axes[-1].set_aspect("equal")
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)

    fig.suptitle("Propagated Cartesian field components")
    return fig, axes


n0, ni = 1.0, 1.5
z0, zi = -10.0, 6.0
lam = 532e-6
R = 2.6

n_r, n_q, n_phi = 1000, 1000, 256
r = np.linspace(0.0, R, n_r)
q = (ni * R / (lam * zi)) * np.linspace(0.0, 1.0, n_q) ** 2
phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

pupil = r <= 0.95 * R
grid = Grid.from_polar(r, phi)
diopter = CartesianSurface(n0=n0, ni=ni, z0=z0, zi=zi)

E0 = FieldCartesian(x=1.0 * pupil, y=0.0 * pupil, grid=grid)
E = E0.propagate_through_diopter(zi, diopter, q)

print(f"theta = {np.rad2deg(np.arctan(R / zi)):.3f} deg")
plot_field(E0, half_size=R, title="Input Cartesian field")
propagated_half_size = 4.0 * 3.8317059702075125 / (0.95 * R)
plot_field(E, half_size=propagated_half_size, title="Propagated transverse Cartesian field")
plot_propagated_xyz(E, half_size=propagated_half_size, wavelength=lam, n=ni)
plt.show()
