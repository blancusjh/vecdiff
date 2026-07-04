import numpy as np
import matplotlib.pyplot as plt

from vecdiff import CartesianSurface, FieldCartesian, Grid
from vecdiff.polarization_visualization import (
    plot_incident_and_focal_components,
    plot_incident_and_focal_polarization_map,
    plot_incident_and_focal_polarization_angles,
)
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

# Cartesian components of the incident field (top) and focal field (bottom).
fig, _ = plot_incident_and_focal_components(
    E0,
    E_focal,
    basis="cartesian",
    incident_half_size=R,
    focal_half_size=20.0,
    title="Componentes cartesianas: pupila vs. plano focal",
)
save(fig, "components")

# Local polarization ellipses side by side. The library defaults are set so
# every glyph carries a visible arrowhead -- for the linear pupil the head
# points outward along the major axis (peak field direction).
fig, _ = plot_incident_and_focal_polarization_map(
    E0,
    E_focal,
    incident_half_size=R,
    focal_half_size=20.0,
    title="Polarización: pupila vs. plano focal (cartesiano)",
)
save(fig, "polarization_map")

# Ellipticity and major-axis orientation.
fig, _ = plot_incident_and_focal_polarization_angles(
    E0,
    E_focal,
    incident_half_size=R,
    focal_half_size=20.0,
    title="Ángulos de la elipse: pupila vs. plano focal (cartesiano)",
)
save(fig, "polarization_angles")

plt.show()
