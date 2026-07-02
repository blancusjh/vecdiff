import numpy as np
import matplotlib.pyplot as plt

from vecdiff import CartesianSurface, FieldCartesian, Grid
from vecdiff.polarization_visualization import plot_field_polarization_summary
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

# Input: components, intensity, polarization ellipses, ellipticity and orientation
# in one figure. Uniform field, so a coarse layout is enough.
fig, _ = plot_field_polarization_summary(
    E0,
    half_size=R,
    title="Input Cartesian field",
    polarization_kwargs=dict(scale_by_intensity=True),
)
save(fig, "input_summary")

# Focal plane: window reaches ~the 4th maximum. target_ellipses=36 gives ~5
# samples across the second maximum's width. A milder gamma and a lower size
# floor let the intensity scaling show as a size gradient from the bright core
# down to the faint 4th maximum. The cross-polarization panel needs very low
# thresholds to reveal the Ey-dominant nodal rings, so it stays uncropped
# while the other panels zoom to where the pattern actually has signal.
fig, _ = plot_field_polarization_summary(
    E_focal,
    half_size=20.0,
    title="Propagated Cartesian field",
    show_cross_fraction=True,
    polarization_kwargs=dict(
        target_ellipses=36,
        scale_by_intensity=True,
        intensity_scale_gamma=0.35,
        min_ellipse_scale=0.28,
        arrow_length=0.5,
        arrow_opening_angle=np.deg2rad(70.0),
    ),
    # Ex has nodal rings where Ey dominates locally despite the low total
    # intensity: needs low thresholds to reveal them.
    cross_fraction_kwargs=dict(
        cross_fraction_min_intensity=1e-7,
        min_intensity_fraction=1e-7,
        min_cross_fraction=0.50,
        target_arrows=120,
    ),
)
save(fig, "propagated_summary")

plt.show()
