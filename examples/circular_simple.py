import numpy as np
import matplotlib.pyplot as plt

from vecdiff import CartesianSurface, FieldCartesian, FieldCircular, Grid
from vecdiff.polarization_visualization import plot_field_polarization
from vecdiff.view import plot_field


def field_in_focal_wavelengths(field, zi, ni):
    scale = zi / (2.0 * np.pi * ni)
    grid = Grid.from_polar(scale * field.grid.r, field.grid.varphi)
    grid.domain = "focal/lambda"
    return FieldCartesian(field.x, field.y, grid=grid, symmetric=False)


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

E0 = FieldCircular(L=0.0 * pupil, R=1.0 * pupil, grid=grid)
E = E0.propagate_through_diopter(zi, diopter, q)
E_focal = field_in_focal_wavelengths(E, zi, ni)

input_half_size = R
propagated_half_size = 5.0

plot_field(E0, half_size=input_half_size, title="Input circular field")
fig, axes = plot_field(E_focal, half_size=propagated_half_size, title="Propagated circular field")
for ax in axes:
    ax.set_xlabel(r"$x/\lambda$")
    ax.set_ylabel(r"$y/\lambda$")

ax, _ = plot_field_polarization(E0, half_size=input_half_size, ellipse_mode="cartesian")
ax.set_title("Input circular polarization")
ax, _ = plot_field_polarization(E_focal, half_size=propagated_half_size, ellipse_mode="cartesian")
ax.set_title("Propagated circular polarization")
ax.set_xlabel(r"$x/\lambda$")
ax.set_ylabel(r"$y/\lambda$")
plt.show()
