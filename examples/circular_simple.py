import numpy as np
import matplotlib.pyplot as plt

from vecdiff import CartesianSurface, FieldCircular, Grid
from vecdiff.polarization_visualization import plot_field_polarization
from vecdiff.view import plot_field
from _common import focal_field, figure_saver

# --- Setup: right-circular pupil focused through a diopter ---
n0, ni, z0, zi = 1.0, 1.5, -10.0, 6.0
lam, R = 532e-6, 2.6
n_r, n_q, n_phi = 1000, 1000, 256

r = np.linspace(0.0, R, n_r)
q = (ni * R / (lam * zi)) * np.linspace(0.0, 1.0, n_q) ** 2
phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
grid = Grid.from_polar(r, phi)
pupil = r <= 0.95 * R
diopter = CartesianSurface(n0=n0, ni=ni, z0=z0, zi=zi)

E0 = FieldCircular(L=0.0 * pupil, R=1.0 * pupil, grid=grid)
E_focal = focal_field(E0.propagate_through_diopter(zi, diopter, q), zi / (2.0 * np.pi * ni))

# --- Figures ---
save = figure_saver(__file__)

save(plot_field(E0, half_size=R, title="Input circular field")[0], "input_field_components")
save(plot_field(E_focal, half_size=5.0, title="Propagated circular field")[0], "propagated_field_components")

ax, _ = plot_field_polarization(E0, half_size=R, sampling="polar")
ax.set_title("Input circular polarization")
save(ax, "input_polarization")

ax, _ = plot_field_polarization(E_focal, half_size=5.0, sampling="polar")
ax.set(title="Propagated circular polarization", xlabel=r"$x/\lambda$", ylabel=r"$y/\lambda$")
save(ax, "propagated_polarization")

plt.show()
