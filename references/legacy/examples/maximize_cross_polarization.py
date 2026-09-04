"""Maximizing the cross-polarized component of the focal field.

The cross-polarized kernel is driven by t- = (tp-ts)/2, which peaks at
grazing incidence and is bounded there by (ni-n0)/(ni+n0).  A high-index
diopter therefore has more cross-polarization to give than an ordinary-glass
one, and an *edge-weighted* pupil amplitude (r/a)^p concentrates the incident
energy where that ratio is largest, trading throughput for cross-polarized
fraction.  This example compares a uniform pupil against an edge-weighted one
on the same high-index diopter, then shows that the cross-polarized channel
has the same energy but a different shape for linear vs. circular incident
polarization (a sin^2(2*phi) clover vs. a charge-2 vortex ring).
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
# High-index diopter, edge-weighted pupil (Toraldo-type apodization).
# ---------------------------------------------------------------------

n0, ni, z0, zi = 1.0, 2.4, -10.0, 6.0
lam = 532e-6  # mm
edge_p = 6.0  # pupil amplitude ~ (r/a)^edge_p

n_r, n_q = 1200, 900

# t- = (tp-ts)/2 peaks at grazing incidence (cos_i -> 0), well before the
# Cartesian oval's parametrization stops being finite (which happens further
# out, on its closed back branch).  aperture_limit is that true grazing radius,
# expressed in the transverse aperture coordinate.
surface = CartesianSurface(n0=n0, ni=ni, z0=z0, zi=zi)
r_graze = surface.aperture_limit
a = 0.97 * r_graze

r = np.linspace(0.0, a, n_r)
q = (ni * a / (lam * zi)) * np.linspace(0.0, 1.0, n_q) ** 2

tp, ts, _, u = focal_channel_weights(surface, r)

pupil_uniform = np.ones_like(r)
pupil_edge = (r / a) ** edge_p

# t-/t+ -> rho_max at grazing incidence; concentrating the whole pupil in an
# infinitesimally thin ring right at that radius is the asymptotic ceiling on
# f_cross for any pupil shape, rho_max^2/(1+rho_max^2).  A finite-power edge
# weight like (r/a)^p approaches, but never reaches, this bound.
rho_max_theory = (ni - n0) / (ni + n0)
ceiling = rho_max_theory**2 / (1.0 + rho_max_theory**2)
print(f"a = {a:.2f} mm, edge_p = {edge_p:g}, techo asintótico (anillo rasante) f_cross -> {ceiling:.3f}")


def cross_fraction(pupil):
    terms = hankel_terms(u, q, tp, ts, e1=pupil, e2=np.zeros_like(pupil), polarization="cartesian")
    e_total = np.trapezoid((np.abs(terms["H0x"]) ** 2 + np.abs(terms["H2x"]) ** 2) * q, q)
    e_cross = np.trapezoid(np.abs(terms["H2x"]) ** 2 * q, q)
    return float(e_cross / e_total), terms


f_uniform, _ = cross_fraction(pupil_uniform)
f_edge, terms_edge = cross_fraction(pupil_edge)
print(f"f_cross: pupila uniforme = {f_uniform:.4f}, pupila con peso de borde = {f_edge:.4f}")

caption = (rf"dioptrio estigmático: $n_0$={n0:g}, $n_i$={ni:g}, $z_0$={z0:g}, $z_i$={zi:g} mm  ·  "
           rf"$\lambda$={lam*1e6:.0f} nm  ·  $a$={a:.2f} mm, pupila $(r/a)^{{{edge_p:g}}}$  ·  "
           r"pol. incidente lineal $\hat{x}$")

out_dir = example_output_dir(__file__)
half_size = 0.9
xx, yy, rho, varphi, extent = make_observation_grid(half_size, 500)
q_lambda = zi / (2.0 * np.pi * ni)

Ex_v, Ey_v = reconstruct_2d_from_terms(terms_edge, q * q_lambda, rho, varphi, polarization="cartesian")
t_plus = 0.5 * (tp + ts)
terms_sca = hankel_terms(u, q, t_plus, t_plus, e1=pupil_edge, e2=np.zeros_like(pupil_edge), polarization="cartesian")
Ex_s, Ey_s = reconstruct_2d_from_terms(terms_sca, q * q_lambda, rho, varphi, polarization="cartesian")

I_scalar = np.abs(Ex_s) ** 2 + np.abs(Ey_s) ** 2
I_vector = np.abs(Ex_v) ** 2 + np.abs(Ey_v) ** 2
I_cross = np.abs(Ey_v) ** 2
vmax = I_scalar.max()

# ---------------------------------------------------------------------
# Figure 1: 2D focal-plane maps with the edge-weighted, cross-maximizing pupil
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
fig.suptitle(rf"Pupila con peso de borde: $f_\mathrm{{cross}}$={f_edge:.3f} "
             rf"(vs. {f_uniform:.3f} con pupila uniforme)" + "\n" + caption, fontsize=10)
path = out_dir / "cross_polarization_maps.png"
fig.savefig(path, dpi=200)
print_saved(path)
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 2: cross-channel shape for linear vs. circular incident polarization
# (same energy, different spatial pattern)
# ---------------------------------------------------------------------

# Circular (L) input pupil, in the circular representation: e1 is the L
# component, e2 the R component.  The induced cross channel E_R is then
# purely radial (E_R = exp(-2i*phi)*E_RL, a unit-magnitude phase factor times
# a Hankel transform of order 2 that vanishes at q=0) -- a clean vortex ring.
terms_circ = hankel_terms(u, q, tp, ts, e1=pupil_edge, e2=np.zeros_like(pupil_edge), polarization="circular")
_, E_R = reconstruct_2d_from_terms(terms_circ, q * q_lambda, rho, varphi, polarization="circular")
I_R = np.abs(E_R) ** 2

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True)
im0 = axes[0].imshow(I_cross / I_cross.max(), extent=extent, origin="lower", cmap="magma")
axes[0].set_title(r"entrada lineal $\hat{x}$: trébol $\sin^2(2\varphi)$")
im1 = axes[1].imshow(I_R / I_R.max(), extent=extent, origin="lower", cmap="magma")
axes[1].set_title(r"entrada circular $L$: anillo-vórtice $m=2$")
for axi, im in zip(axes, (im0, im1)):
    axi.set_xlabel(r"$x/\lambda$")
    axi.set_ylabel(r"$y/\lambda$")
    fig.colorbar(im, ax=axi, fraction=0.046, pad=0.04)
fig.suptitle("Forma del canal cruzado según la base de polarización (misma energía)\n" + caption, fontsize=10)
path = out_dir / "cross_channel_shape.png"
fig.savefig(path, dpi=200)
print_saved(path)
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 3: polarization map on the focal plane with the maximized cross
# component.  Polar glyph layout suits the radially structured pattern.
# ---------------------------------------------------------------------

E_focal = FieldCartesian(x=Ex_v, y=Ey_v, grid=Grid.from_cartesian(xx, yy), symmetric=False)
ax, _ = plot_field_polarization(E_focal, half_size=half_size, sampling="polar", n_rings=12,
                                scale_by_intensity=True, min_ellipse_scale=0.5)
ax.set(title="Mapa de polarización en el plano focal (pupila de borde)", xlabel=r"$x/\lambda$", ylabel=r"$y/\lambda$")
ax.figure.suptitle(caption, fontsize=8)
path = out_dir / "focal_plane_polarization.png"
ax.figure.savefig(path, dpi=200, bbox_inches="tight")
print_saved(path)
plt.close(ax.figure)
