"""Focal-plane field vs aperture: scalar (t- = 0) vs vectorial focusing.

A uniform, x-polarized pupil is focused through a single stigmatic diopter at
a wide aperture, close to the physical acceptance limit of the Cartesian oval.
The *scalar* field keeps only the isotropic Fresnel transmission t+ = (tp+ts)/2
(t- = 0, no polarization mixing); the *vectorial* field uses the full
transverse operator, which mixes the incident x polarization into a y
component through the t- = (tp-ts)/2 term.

At high aperture the vectorial focal spot becomes visibly anisotropic
(quadrupolar, cos(2*phi)) compared to the circularly symmetric scalar spot,
and a four-lobe cross-polarized "clover" appears with zero energy on-axis.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from references.legacy.vecdiff import CartesianSurface, focal_channel_weights
from references.legacy.vecdiff.field_reconstruction import hankel_terms, reconstruct_2d_from_terms, make_observation_grid
from references.legacy.vecdiff import FieldCartesian, Grid
from references.legacy.vecdiff.polarization_visualization import plot_field_polarization
from _output import example_output_dir, print_saved

# ---------------------------------------------------------------------
# Optical configuration: same stigmatic diopter as circular_simple.py,
# apertured close to the grazing-incidence limit of the Cartesian oval.
# ---------------------------------------------------------------------

n0, ni, z0, zi = 1.0, 1.5, -10.0, 6.0
lam = 532e-6  # mm
surface = CartesianSurface(n0=n0, ni=ni, z0=z0, zi=zi)
a = 0.97 * surface.aperture_limit  # mm, just inside grazing incidence

n_r, n_q = 1200, 900
r = np.linspace(0.0, a, n_r)
q = (ni * a / (lam * zi)) * np.linspace(0.0, 1.0, n_q) ** 2  # focus-refined grid

# Transfer eigenvalues times the pupil-mapping Jacobian, and the variable the
# focal transform integrates over.
tp, ts, _, u = focal_channel_weights(surface, r)
t_plus = 0.5 * (tp + ts)

pupil = np.ones_like(r)

# Vectorial: exact (tp, ts).  Scalar: t- = 0, i.e. tp = ts = t_plus.
terms_vec = hankel_terms(u, q, tp, ts, e1=pupil, e2=np.zeros_like(pupil), polarization="cartesian")
terms_sca = hankel_terms(u, q, t_plus, t_plus, e1=pupil, e2=np.zeros_like(pupil), polarization="cartesian")

half_size = 1.2  # focal-plane half-window, wavelengths
xx, yy, rho, varphi, extent = make_observation_grid(half_size, 500)
q_lambda = zi / (2.0 * np.pi * ni)  # q -> focal-plane radius [lambda]

Ex_v, Ey_v = reconstruct_2d_from_terms(terms_vec, q * q_lambda, rho, varphi, polarization="cartesian")
Ex_s, Ey_s = reconstruct_2d_from_terms(terms_sca, q * q_lambda, rho, varphi, polarization="cartesian")

I_scalar = np.abs(Ex_s) ** 2 + np.abs(Ey_s) ** 2
I_vector = np.abs(Ex_v) ** 2 + np.abs(Ey_v) ** 2
I_cross = np.abs(Ey_v) ** 2
vmax = I_scalar.max()

alpha_obj = np.degrees(np.arctan(a / abs(z0)))
f_cross = float(np.sum(I_cross) / np.sum(I_vector))
print(f"alpha_obj = {alpha_obj:.1f} deg, f_cross = {f_cross:.4f}")

caption = (rf"dioptrio estigmático: $n_0$={n0:g}, $n_i$={ni:g}, $z_0$={z0:g}, $z_i$={zi:g} mm  ·  "
           rf"$\lambda$={lam*1e6:.0f} nm  ·  $a$={a:g} mm, $\alpha_\mathrm{{obj}}$={alpha_obj:.1f}°  ·  "
           r"pol. incidente lineal $\hat{x}$")

out_dir = example_output_dir(__file__)

# ---------------------------------------------------------------------
# Figure 1: 2D focal-plane maps
# ---------------------------------------------------------------------

fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), constrained_layout=True)
panels = [
    (I_scalar / vmax, "hot", (0, 1), r"$I_\mathrm{escalar}$ ($t_-=0$)"),
    (I_vector / vmax, "hot", (0, 1), r"$I_\mathrm{vectorial}$"),
    ((I_vector - I_scalar) / vmax, "RdBu_r", None, r"$\Delta I = I_\mathrm{vec} - I_\mathrm{esc}$"),
    (I_cross / I_cross.max(), "magma", (0, 1), r"$|E_y|^2$ (cruzada, norm.)"),
]
for axi, (img, cmap, clim, title) in zip(axes, panels):
    if clim is None:
        m = np.abs(img).max()
        im = axi.imshow(img, extent=extent, origin="lower", cmap=cmap, vmin=-m, vmax=m)
    else:
        im = axi.imshow(img, extent=extent, origin="lower", cmap=cmap, vmin=clim[0], vmax=clim[1])
    axi.set_title(title)
    axi.set_xlabel(r"$x/\lambda$")
    axi.set_ylabel(r"$y/\lambda$")
    fig.colorbar(im, ax=axi, fraction=0.046, pad=0.04)
fig.suptitle("Mapas del plano focal\n" + caption, fontsize=10)
path = out_dir / "focal_plane_maps.png"
fig.savefig(path, dpi=200)
print_saved(path)
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 2: radial cuts at phi = 0, 45, 90 deg vs the scalar reference
# ---------------------------------------------------------------------

phi_cuts = {"escalar": None, r"vect. $\varphi=0°$": 0.0, r"vect. $\varphi=45°$": np.pi / 4, r"vect. $\varphi=90°$": np.pi / 2}
q_axis = np.linspace(0.0, half_size, 400)

fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
Is_r, _ = reconstruct_2d_from_terms(terms_sca, q * q_lambda, q_axis, np.zeros_like(q_axis), polarization="cartesian")
ax.plot(q_axis, np.abs(Is_r) ** 2 / vmax, "k--", label="escalar")
for label, phi0 in phi_cuts.items():
    if phi0 is None:
        continue
    Exr, Eyr = reconstruct_2d_from_terms(terms_vec, q * q_lambda, q_axis, np.full_like(q_axis, phi0), polarization="cartesian")
    ax.plot(q_axis, (np.abs(Exr) ** 2 + np.abs(Eyr) ** 2) / vmax, label=label)
ax.set_xlabel(r"$\rho/\lambda$")
ax.set_ylabel(r"$I/I_\mathrm{esc}(0)$")
ax.set_title("Cortes radiales")
ax.grid(True, alpha=0.3)
ax.legend()
fig.suptitle(caption, fontsize=8)
path = out_dir / "radial_cuts.png"
fig.savefig(path, dpi=200)
print_saved(path)
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 3: polarization map on the focal plane (vectorial field).
# Polar glyph layout suits the radially structured diffraction pattern.
# ---------------------------------------------------------------------

E_focal = FieldCartesian(x=Ex_v, y=Ey_v, grid=Grid.from_cartesian(xx, yy), symmetric=False)
ax, _ = plot_field_polarization(E_focal, half_size=half_size, sampling="polar", n_rings=12,
                                scale_by_intensity=True, min_ellipse_scale=0.5)
ax.set(title="Mapa de polarización en el plano focal (vectorial)", xlabel=r"$x/\lambda$", ylabel=r"$y/\lambda$")
ax.figure.suptitle(caption, fontsize=8)
path = out_dir / "focal_plane_polarization.png"
ax.figure.savefig(path, dpi=200, bbox_inches="tight")
print_saved(path)
plt.close(ax.figure)
