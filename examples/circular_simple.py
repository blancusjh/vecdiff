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

# --- Glyph layout tuning parameters ---
# Incident pupil: sparser ring layout (it is uniformly circular, fewer rings suffice).
n_rings_incident = 20
angular_spacing_incident = 1.0

# Focal field: n_rings=23 with focal_half_size=5.0 gives dr ≈ 0.213 λ, which
# distributes the rings across the three Airy zones as:
#   spot zone    (r < 0.94 λ):   4 rings  — same as before
#   segunda zona (0.94–1.72 λ):  4 rings
#   tercera zona (1.72–2.49 λ):  3 rings
# max_radius clips ellipses beyond the third Airy zero.


n_rings_focal = 23
angular_spacing_focal = 0.75
focal_max_radius = 2.5      # ≈ third Airy zero in focal/lambda units


# Polarization Presets.


#   Incident 
incident_polarization_preset = dict(
    sampling="polar", n_rings=n_rings_incident, angular_spacing=angular_spacing_incident,
    scale = 0.06
)


#   Focal Plane

focal_polarization_preset = dict(
    sampling="polar",
    n_rings=n_rings_focal,
    angular_spacing=angular_spacing_focal,
    max_radius=focal_max_radius,
    min_intensity_fraction=0.001,
    scale = 0.06
)
fig, _ = plot_incident_and_focal_polarization_map(
    E0,
    E_focal,
    incident_half_size=R,
    focal_half_size=5.0,
    min_intensity_fraction=0.001,
    incident_polarization_kwargs=incident_polarization_preset,
    focal_polarization_kwargs=focal_polarization_preset,
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
