import numpy as np
import matplotlib.pyplot as plt

from vecdiff import CartesianSurface, FieldCartesian, Grid, generate_Ez_cartesian
from vecdiff.view import field_cartesian_maps, plot_field


def plot_propagated_xyz(
    field,
    half_size,
    wavelength,
    n=1.0,
    n_img=500,
    component_view="abs",
    cmap="hot",
):
    x = np.linspace(-half_size, half_size, n_img)
    xx, yy = np.meshgrid(x, x)
    Ex, Ey, extent, _ = field_cartesian_maps(field, half_size=half_size, n_img=n_img)
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
R = 5.1

print(np.arctan2(R, z0) * 180/np.pi  )

n_r, n_q, n_phi = 1000, 1000, 100
r = np.linspace(0.0, R, n_r)
q = (ni * R / (lam * zi)) * np.linspace(0.0, 1.0, n_q) ** 2
phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

pupil = r <= 0.95 * R
grid = Grid.from_polar(r, phi)
diopter = CartesianSurface(n0=n0, ni=ni, z0=z0, zi=zi)

E0 = FieldCartesian(x=1.0 * pupil, y=0.0 * pupil, grid=grid)
propagated_half_size = 4.0 * 3.8317059702075125 / (0.95 * R)
k_axis = np.linspace(-propagated_half_size, propagated_half_size, 800)

KX, KY = np.meshgrid(k_axis, k_axis, indexing="xy")
kgrid = Grid.from_cartesian(KX, KY, domain="k")

E = E0.propagate_through_diopter(zi, diopter, q, method="fft", kgrid=kgrid)




print(f"theta = {np.rad2deg(np.arctan(R / zi)):.3f} deg")
plot_field(E0, half_size=R, title="Input Cartesian field")
plot_field(E, half_size=propagated_half_size, title="Propagated transverse Cartesian field")
plot_propagated_xyz(E, half_size=propagated_half_size, wavelength=lam, n=ni)
plt.show()
