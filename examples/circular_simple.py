import numpy as np
import matplotlib.pyplot as plt

from vecdiff import CartesianSurface, FieldCircular, Grid
from vecdiff.polarization_visualization import (
    plot_incident_and_focal_components,
    plot_incident_and_focal_polarization_map,
    plot_incident_and_focal_polarization_angles,
)
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

# Circular components (EL, ER) of the incident field (top) and focal field
# (bottom). The focal field carries a Cartesian representation, so its L/R
# panels are computed on the fly from (Ex, Ey).
fig, _ = plot_incident_and_focal_components(
    E0,
    E_focal,
    basis="circular",
    incident_half_size=R,
    focal_half_size=5.0,
    title="Componentes circulares: pupila vs. plano focal",
)
save(fig, "components")

# The library defaults already carry the harmonic preset (uniform sizes,
# 45%/42 deg arrow heads). Only the polar sampling and ring count are
# problem-specific: polar layout respects the rotational symmetry of both
# pupil and focal diffraction pattern.
polarization_preset = dict(sampling="polar", n_rings=12)
fig, _ = plot_incident_and_focal_polarization_map(
    E0,
    E_focal,
    incident_half_size=R,
    focal_half_size=5.0,
    incident_polarization_kwargs=polarization_preset,
    focal_polarization_kwargs=polarization_preset,
    title="Polarización: pupila vs. plano focal (circular)",
)
save(fig, "polarization_map")

# Ellipticity and major-axis orientation.
fig, _ = plot_incident_and_focal_polarization_angles(
    E0,
    E_focal,
    incident_half_size=R,
    focal_half_size=5.0,
    title="Ángulos de la elipse: pupila vs. plano focal (circular)",
)
save(fig, "polarization_angles")

plt.show()
