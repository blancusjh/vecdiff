import numpy as np
import matplotlib.pyplot as plt

from vecdiff import CartesianSurface, FieldCartesian, Grid
from vecdiff.polarization_visualization import plot_field_polarization
from vecdiff.view import plot_field
from _common import focal_field, figure_saver

# --- Setup: uniform x-polarized pupil focused through a diopter (Hankel) ---
n0, ni, z0, zi = 1.0, 1.5, -10.0, 6.0
lam, R = 532e-6, 2.6
n_r, n_q, n_phi = 1000, 1000, 256

r = np.linspace(0.0, R, n_r)
q = (ni * R / (lam * zi)) * np.linspace(0.0, 1.0, n_q) ** 2
phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
grid = Grid.from_polar(r, phi)
pupil = r <= 0.95 * R
diopter = CartesianSurface(n0=n0, ni=ni, z0=z0, zi=zi)

E0 = FieldCartesian(x=1.0 * pupil, y=0.0 * pupil, grid=grid)
E_focal = focal_field(E0.propagate_through_diopter(zi, diopter, q, method="hankel"), zi / ni)

# --- Figures ---
save = figure_saver(__file__)

save(plot_field(E0, half_size=R, title="Input Cartesian field")[0], "input_field_components")
save(plot_field(E_focal, half_size=20.0, title="Propagated Cartesian field")[0], "propagated_field_components")

ax, _ = plot_field_polarization(E0, half_size=R)
ax.set_title("Input Cartesian polarization")
save(ax, "input_polarization")

ax, _ = plot_field_polarization(E_focal, half_size=20.0)
ax.set(title="Propagated Cartesian polarization", xlabel=r"$x/\lambda$", ylabel=r"$y/\lambda$")
save(ax, "propagated_polarization")

# Cross-polarization diagnostic: Ex has nodal rings where Ey dominates locally
# despite the low total intensity (needs the low thresholds to reveal them).
ax, _ = plot_field_polarization(
    E_focal,
    half_size=20.0,
    background="cross_fraction",
    cross_fraction_min_intensity=1e-7,
    glyph="quiver",
    min_intensity_fraction=1e-7,
    min_cross_fraction=0.50,
    target_arrows=120,
)
ax.set(title=r"Ey-dominant polarization: $|E_y|^2 / |E|^2 \geq 0.5$", xlabel=r"$x/\lambda$", ylabel=r"$y/\lambda$")
save(ax, "propagated_cross_polarization")

plt.show()
