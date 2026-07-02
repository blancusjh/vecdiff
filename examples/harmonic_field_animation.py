"""Time-harmonic animation of a focal field (reproduces the README GIF).

A right-circularly polarized wide pupil is focused through the stigmatic
diopter of ``circular_simple.py``.  The focal field is animated with
``vecdiff.animate_harmonic_field``: the intensity ``|E|^2`` is the static
background and the quiver shows the instantaneous *real* field vector
``Re[E e^{-i omega t}]``, which spins over the optical cycle because the input
is circularly polarized.

Running this script rewrites ``docs/assets/quiver_harmonic_readme.gif`` (the
animation shown at the top of the repository README).
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vecdiff import CartesianSurface, FieldCircular, Grid
from vecdiff import animate_harmonic_field, save_animation
from _output import example_output_dir, print_saved

# --- Focused right-circular pupil (same diopter as circular_simple.py) ---
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
E_focal = E0.propagate_through_diopter(zi, diopter, q)

# Rescale the polar output grid to focal-plane radii in wavelengths.
E_focal.grid = Grid.from_polar(E_focal.grid.r * zi / (2.0 * np.pi * ni), E_focal.grid.varphi)

# --- Animate the instantaneous field over two optical cycles ---
anim = animate_harmonic_field(
    E_focal,
    half_size=4.0,
    n_img=240,
    n_frames=72,
    n_periods=2,
    target_arrows=34,
    intensity_gamma=0.4,
    title="Campo instantáneo (entrada circular R)",
)

out_dir = example_output_dir(__file__)
gif = out_dir / "harmonic_field.gif"
save_animation(anim, gif, fps=20, dpi=85)
print_saved(gif)

# Refresh the README hero asset from the same animation.
readme_gif = Path(__file__).resolve().parent.parent / "docs" / "assets" / "quiver_harmonic_readme.gif"
save_animation(anim, readme_gif, fps=20, dpi=85)
print_saved(readme_gif)
plt.close("all")
