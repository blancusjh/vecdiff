import numpy as np
import matplotlib.pyplot as plt

from vecdiff import CartesianSurface, FieldCircular, Grid
from vecdiff.polarization_visualization import plot_field_polarization_summary
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

# Input: components, intensity, polarization ellipses, ellipticity and orientation
# in one figure. Uniform field, so a coarse polar layout is enough.
fig, _ = plot_field_polarization_summary(
    E0,
    half_size=R,
    title="Input circular field",
    polarization_kwargs=dict(sampling="polar", n_rings=10, scale_by_intensity=True),
)
save(fig, "input_summary")

# Focal plane: crop to ~4 maxima, sample finely in radius so the polarization
# deformation across each maximum is visible (>= 5 radii per lobe), and size the
# glyphs by intensity (non-linear). All panels are auto-cropped to the region
# where the pattern actually has signal.
fig, _ = plot_field_polarization_summary(
    E_focal,
    half_size=5.0,
    title="Propagated circular field",
    polarization_kwargs=dict(sampling="polar", n_rings=15, scale_by_intensity=True, min_ellipse_scale=0.5),
)
save(fig, "propagated_summary")

plt.show()
