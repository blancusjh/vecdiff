"""Time-harmonic animation of the focal field (reproduces the README GIF).

Primary case (Cartesian, the README hero): an x-polarized wide pupil with an
edge weight (r/a)^p is focused through a high-index stigmatic diopter, the same
cross-polarization-maximizing setup as ``maximize_cross_polarization.py``.  The
focal field is real and linearly polarized everywhere, so
``Re[E e^{-i omega t}]`` *breathes*: every arrow grows, shrinks to zero and
flips over the optical cycle, tracing out the spatially varying linear
polarization -- horizontal in the core, tilting toward the diagonals in the
cross-polarized ``sin(2*phi)`` clover.

Secondary case (circular): a right-circular pupil through the same diopter.
Its focal field is not in phase, so the instantaneous vector *rotates* instead
of breathing.

Both animations are made with ``vecdiff.animate_harmonic_field`` and carry the
optical-system parameters as a caption.  Running the script rewrites the README
hero asset ``docs/assets/quiver_harmonic_readme.gif`` from the Cartesian case.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vecdiff.fresnel import FresnelOvoid
from vecdiff.field_reconstruction import hankel_terms, reconstruct_2d_from_terms, make_observation_grid
from vecdiff import CartesianSurface, FieldCartesian, FieldCircular, Grid
from vecdiff import animate_harmonic_field, save_animation
from _output import example_output_dir, print_saved

# --- Shared high-index diopter (as in maximize_cross_polarization.py) ---
n0, ni, z0, zi = 1.0, 2.4, -10.0, 6.0
lam = 532e-6  # mm
edge_p = 6.0  # pupil amplitude ~ (r/a)^edge_p
q_lambda = zi / (2.0 * np.pi * ni)  # q -> focal-plane radius [lambda]

out_dir = example_output_dir(__file__)
assets = Path(__file__).resolve().parent.parent / "docs" / "assets"
readme_gif = assets / "quiver_harmonic_readme.gif"          # Cartesian hero
readme_gif_circular = assets / "quiver_harmonic_circular.gif"  # circular companion


def grazing_aperture():
    """Aperture at 0.97x the grazing radius, where t-/t+ is largest."""
    rho = np.linspace(1e-4, 3.0 * abs(z0), 4000)
    fr = FresnelOvoid(n0=n0, ni=ni, z0=z0, zi=zi)
    with np.errstate(all="ignore"):
        cos_i, _ = fr._cosines(rho)
        finite = np.isfinite(fr.ovoid.z(rho)) & np.isfinite(fr.ovoid.r(rho))
    return 0.97 * float(rho[np.nanargmin(np.where(finite, cos_i, np.nan))])


a = grazing_aperture()
n_r, n_q = 1200, 900
r = np.linspace(0.0, a, n_r)
q = (ni * a / (lam * zi)) * np.linspace(0.0, 1.0, n_q) ** 2

fresnel = FresnelOvoid(n0=n0, ni=ni, z0=z0, zi=zi)
with np.errstate(divide="ignore", invalid="ignore"):
    tp = np.asarray(fresnel.tp(r), dtype=float)
    ts = np.asarray(fresnel.ts(r), dtype=float)
for t in (tp, ts):  # repair the r=0 indeterminacy by interpolation
    bad = ~np.isfinite(t)
    t[bad] = np.interp(r[bad], r[~bad], t[~bad])

pupil_edge = (r / a) ** edge_p
alpha_obj = np.degrees(np.arctan(a / abs(z0)))

# Two-line caption so the full system description fits the animation figure.
caption = (rf"dioptrio estigmático: $n_0$={n0:g}, $n_i$={ni:g}, $z_0$={z0:g}, $z_i$={zi:g} mm  ·  "
           rf"$\lambda$={lam*1e6:.0f} nm" + "\n"
           rf"$a$={a:.2f} mm, $\alpha_\mathrm{{obj}}$={alpha_obj:.1f}°, pupila $(r/a)^{{{edge_p:g}}}$")

# ---------------------------------------------------------------------
# Primary: Cartesian (linear x) focal field, cross component maximized.
# ---------------------------------------------------------------------

half_size = 1.4  # focal-plane half-window, wavelengths
xx, yy, rho2d, varphi2d, _ = make_observation_grid(half_size, 420)
terms_cart = hankel_terms(r, q, tp, ts, e1=pupil_edge, e2=np.zeros_like(pupil_edge), polarization="cartesian")
Ex, Ey = reconstruct_2d_from_terms(terms_cart, q * q_lambda, rho2d, varphi2d, polarization="cartesian")
E_cart = FieldCartesian(x=Ex, y=Ey, grid=Grid.from_cartesian(xx, yy), symmetric=False)

anim = animate_harmonic_field(
    E_cart, n_img=380, n_frames=60, n_periods=2, target_arrows=30,
    intensity_gamma=0.45, vmin=0.0, vmax=1.0, arrow_norm=0.45,
    title=r"Campo instantáneo cartesiano (pol. lineal $\hat{x}$)",
    caption=caption + r"  ·  pol. incidente lineal $\hat{x}$",
)
save_animation(anim, out_dir / "harmonic_field_cartesian.gif", fps=20, dpi=80)
print_saved(out_dir / "harmonic_field_cartesian.gif")
save_animation(anim, readme_gif, fps=20, dpi=80)  # README hero asset
print_saved(readme_gif)
plt.close("all")

# ---------------------------------------------------------------------
# Secondary: circular (R) focal field -- the instantaneous vector rotates.
# ---------------------------------------------------------------------

n_phi = 256
phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
grid_c = Grid.from_polar(r, phi)
diopter = CartesianSurface(n0=n0, ni=ni, z0=z0, zi=zi)
E0 = FieldCircular(L=0.0 * pupil_edge, R=1.0 * pupil_edge, grid=grid_c)
E_circ = E0.propagate_through_diopter(zi, diopter, q)
E_circ.grid = Grid.from_polar(E_circ.grid.r * q_lambda, E_circ.grid.varphi)

anim_c = animate_harmonic_field(
    E_circ, half_size=half_size, n_img=280, n_frames=60, n_periods=2, target_arrows=30,
    intensity_gamma=0.45, vmin=0.0, vmax=1.0, arrow_norm=0.45,
    title=r"Campo instantáneo circular (pol. incidente $R$)",
    caption=caption + r"  ·  pol. incidente circular $R$",
)
save_animation(anim_c, out_dir / "harmonic_field_circular.gif", fps=20, dpi=80)
print_saved(out_dir / "harmonic_field_circular.gif")
save_animation(anim_c, readme_gif_circular, fps=20, dpi=80)  # README companion asset
print_saved(readme_gif_circular)
plt.close("all")
