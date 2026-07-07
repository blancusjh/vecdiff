
import numpy as np
import matplotlib.pyplot as plt

from vecdiff import CartesianSurface, FieldPolar, Grid
from vecdiff.polarization_visualization import (
    plot_incident_and_focal_components,
    plot_incident_and_focal_polarization_map,
)
from _common import focal_field, figure_saver

# --- Setup: radial and azimuthal pupils focused through a diopter (Hankel) ---
n0, ni, z0, zi = 1.0, 1.5, -10.0, 6.0
lam, R = 532e-6, 2.6
n_r, n_q, n_phi = 1000, 1000, 256

r = np.linspace(0.0, R, n_r)
q = (ni * R / (lam * zi)) * np.linspace(0.0, 1.0, n_q) ** 2
phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
grid = Grid.from_polar(r, phi)
pupil = r <= 0.95 * R
diopter = CartesianSurface(n0=n0, ni=ni, z0=z0, zi=zi)

save = figure_saver(__file__)

diopter_caption = (
    rf"dioptrio estigmático: $n_0$={n0:g}, $n_i$={ni:g}, "
    rf"$z_0$={z0:g} mm, $z_i$={zi:g} mm  ·  "
    rf"$\lambda$={lam * 1e6:.0f} nm  ·  $R$={R:g} mm"
)

# --- Glyph layout tuning parameters (same presets as circular_simple) ---
incident_polarization_preset = dict(
    sampling="polar",
    n_rings=15,
    angular_spacing=1.0,
    scale=0.06,
    curve_kwargs={"linewidth": 0.9},
)
focal_polarization_preset = dict(
    sampling="polar",
    n_rings=12,
    angular_spacing=1.0,
    max_radius=2.5,      # ~ third Airy zero in focal/lambda units
    min_intensity_fraction=0.001,
    scale=0.085,
    curve_kwargs={"linewidth": 0.9},
)

cases = {
    "radial": FieldPolar(r=1.0 * pupil, phi=0.0 * pupil, grid=grid),
    "azimutal": FieldPolar(r=0.0 * pupil, phi=1.0 * pupil, grid=grid),
}

for name, E0 in cases.items():
    E_focal = focal_field(
        E0.propagate_through_diopter(zi, diopter, q), zi / (2.0 * np.pi * ni)
    )

    # Cylindrical components (Er, Ephi) of the incident field (top) and focal
    # field (bottom). The transmission is diagonal in this basis, and the
    # Hankel-1 structure produces a dark center in the focal plane.
    fig, _ = plot_incident_and_focal_components(
        E0,
        E_focal,
        basis="polar",
        incident_half_size=R,
        focal_half_size=5.0,
        title=(
            f"Componentes cilíndricas: pupila vs. plano focal ({name})\n"
            + diopter_caption
        ),
    )
    save(fig, f"{name}_components")

    # Local polarization ellipses side by side.
    fig, _ = plot_incident_and_focal_polarization_map(
        E0,
        E_focal,
        incident_half_size=R,
        focal_half_size=3.0,
        min_intensity_fraction=0.001,
        incident_polarization_kwargs=incident_polarization_preset,
        focal_polarization_kwargs=focal_polarization_preset,
        title=f"Polarización: pupila vs. plano focal ({name})\n" + diopter_caption,
    )
    save(fig, f"{name}_polarization_map")

plt.show()
