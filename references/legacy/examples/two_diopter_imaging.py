"""Two-diopter imaging: scalar (t- = 0) vs vectorial, resolution vs orientation.

A two-line mask, x-polarized, is imaged by a pair of conjugate diopters (air
-> glass -> air) with a circular pupil stop in the Fourier plane between
them -- the vecdiff equivalent of a simple 4f imaging system.  The *scalar*
image applies only the isotropic transmission t+ at each diopter (t- = 0);
the *vectorial* image applies the full transverse operator.  Both share the
same pupil, so any difference in the image comes from the polarization
mixing term.

Comparing the two-line separation aligned with vs. perpendicular to the
incident polarization shows that the vectorial correction is *directional*:
near the Rayleigh limit it can raise or lower the image contrast depending on
the mask orientation relative to the incident polarization.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from references.legacy.vecdiff import CartesianSurface, FieldCartesian, Grid
from references.legacy.vecdiff.propagation import fresnel_coefficients_on_grid
from references.legacy.vecdiff.polarization_visualization import plot_field_polarization
from _output import example_output_dir, print_saved

# ---------------------------------------------------------------------
# Optical configuration: two conjugate diopters, air -> glass -> air.
# ---------------------------------------------------------------------

lam = 532e-6  # mm
D1 = dict(n0=1.0, ni=2.4, z0=-6.0, zi=3.6)
D2 = dict(n0=D1["ni"], ni=1.0, z0=-D1["zi"], zi=3.6)

r_a = 6.5  # mm, pupil-stop radius (sets the numerical aperture)
alpha_max = np.arctan(r_a / abs(D1["z0"]))
NA = D1["n0"] * np.sin(alpha_max)
Kc = (2.0 * np.pi / lam) * NA
d_Airy = 2.0 * 3.8317059702075125 / Kc
pupil_radius = lam * D1["zi"] * Kc / (2.0 * np.pi * D1["ni"])

sep = 0.9 * d_Airy
line_w = 0.35 * sep
line_len = 3.0 * sep

N, L, Npad = 641, 16.0 * sep, 1024
x = np.linspace(-L / 2.0, L / 2.0, N)
grid = Grid.from_axes(x, x)
X, Y = grid.X, grid.Y

print(f"r_a = {r_a} mm, alpha_max = {np.degrees(alpha_max):.1f} deg, NA = {NA:.3f}, "
      f"d_Airy = {d_Airy*1e3:.4f} um, sep/d_Airy = {sep/d_Airy:.2f}")


def two_line_mask(horizontal):
    """Two lines separated along x (horizontal) or y (vertical)."""
    T = np.zeros_like(X)
    for sign in (-1.0, 1.0):
        if horizontal:
            u, v, c = X, Y, sign * 0.5 * sep
            mask = (np.abs(u - c) <= 0.5 * line_w) & (np.abs(v) <= 0.5 * line_len)
        else:
            u, v, c = Y, X, sign * 0.5 * sep
            mask = (np.abs(u - c) <= 0.5 * line_w) & (np.abs(v) <= 0.5 * line_len)
        T[mask] = 1.0
    return T


def scalar_limit(field, diopter):
    """Apply the isotropic t+ = (tp+ts)/2 factor (the t- = 0 scalar limit)."""
    support = (np.abs(field.x) > 0.0) | (np.abs(field.y) > 0.0)
    tp, ts = fresnel_coefficients_on_grid(field.grid.R, diopter, support=support)
    t_plus = 0.5 * (tp + ts)
    return FieldCartesian(x=t_plus * field.x, y=t_plus * field.y, grid=field.grid, symmetric=False)


def propagate(field, diopter, vectorial):
    if vectorial:
        return field.propagate_through_diopter(
            diopter.zi, diopter, method="fft", output="focal", wavelength=lam,
            kgrid=field.grid.kgrid(Npad), transmission="vectorial")
    return scalar_limit(field, diopter).propagate_through_diopter(
        diopter.zi, diopter, method="fft", output="focal", wavelength=lam,
        kgrid=field.grid.kgrid(Npad), transmission="identity")


def image(mask, vectorial):
    field0 = FieldCartesian(x=mask.astype(complex), y=np.zeros_like(mask, dtype=complex), grid=grid, symmetric=False)
    diopter1 = CartesianSurface(**D1)
    field1 = propagate(field0, diopter1, vectorial).with_circular_aperture(pupil_radius)
    diopter2 = CartesianSurface(**D2)
    return propagate(field1, diopter2, vectorial)


caption = (rf"D1: $n_0$={D1['n0']:g}$\to${D1['ni']:g}, $z_0$={D1['z0']:g}, $z_i$={D1['zi']:g} mm  ·  "
           rf"D2: $z_i$={D2['zi']:g} mm  ·  $\lambda$={lam*1e6:.0f} nm  ·  "
           rf"$r_a$={r_a:g} mm, $\alpha_\mathrm{{max}}$={np.degrees(alpha_max):.1f}°  ·  "
           r"pol. incidente lineal $\hat{x}$")

out_dir = example_output_dir(__file__)

# ---------------------------------------------------------------------
# Figure 1: image gallery, both orientations
# ---------------------------------------------------------------------

orientations = {"separación horizontal ($\\parallel \\hat{x}$)": True,
                "separación vertical ($\\perp \\hat{x}$)": False}
results = {}

fig, axes = plt.subplots(2, 4, figsize=(17, 8.4), constrained_layout=True)
for row, (label, horizontal) in enumerate(orientations.items()):
    mask = two_line_mask(horizontal)
    E_s = image(mask, vectorial=False)
    E_v = image(mask, vectorial=True)
    I_s = np.abs(E_s.x) ** 2 + np.abs(E_s.y) ** 2
    I_v = np.abs(E_v.x) ** 2 + np.abs(E_v.y) ** 2
    I_cross = np.abs(E_v.y) ** 2
    results[horizontal] = dict(mask=mask, E_s=E_s, E_v=E_v, I_s=I_s, I_v=I_v)

    lim = 2.2 * sep / lam
    ext_in = [v / lam for v in (-L / 2, L / 2, -L / 2, L / 2)]
    ext_out = [E_s.grid.X[0, 0] / lam, E_s.grid.X[0, -1] / lam, E_s.grid.Y[0, 0] / lam, E_s.grid.Y[-1, 0] / lam]
    vmax = max(I_s.max(), I_v.max())

    panels = [(mask, ext_in, "gray", "Máscara", (0, 1)),
              (I_s / vmax, ext_out, "gray", r"$I_\mathrm{escalar}$", (0, 1)),
              (I_v / vmax, ext_out, "gray", r"$I_\mathrm{vectorial}$", (0, 1)),
              (I_cross / (I_cross.max() + 1e-30), ext_out, "magma", r"$|E_y|^2$ cruzada", (0, 1))]
    for col, (img, ext, cmap, title, clim) in enumerate(panels):
        axi = axes[row, col]
        im = axi.imshow(img, extent=ext, origin="lower", cmap=cmap, vmin=clim[0], vmax=clim[1])
        axi.set_title(f"{title}" if col > 0 else f"{title}\n{label}", fontsize=10)
        axi.set_xlim(-lim, lim)
        axi.set_ylim(-lim, lim)
        axi.set_xlabel(r"$x/\lambda$")
        axi.set_ylabel(r"$y/\lambda$")
        fig.colorbar(im, ax=axi, fraction=0.046, pad=0.04)
fig.suptitle("Imágenes de dos líneas: escalar vs vectorial, por orientación\n" + caption, fontsize=10)
path = out_dir / "image_gallery.png"
fig.savefig(path, dpi=180)
print_saved(path)
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 2: profile cuts across the separation, both orientations
# ---------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
for axi, (label, horizontal) in zip(axes, orientations.items()):
    r = results[horizontal]
    E_s, E_v = r["E_s"], r["E_v"]
    xr = E_s.grid.X[0, :]
    yr = E_s.grid.Y[:, 0]
    mid = len(xr) // 2
    if horizontal:
        u = xr / lam
        Is_line = (np.abs(E_s.x[mid, :]) ** 2 + np.abs(E_s.y[mid, :]) ** 2)
        Iv_line = (np.abs(E_v.x[mid, :]) ** 2 + np.abs(E_v.y[mid, :]) ** 2)
    else:
        u = yr / lam
        Is_line = (np.abs(E_s.x[:, mid]) ** 2 + np.abs(E_s.y[:, mid]) ** 2)
        Iv_line = (np.abs(E_v.x[:, mid]) ** 2 + np.abs(E_v.y[:, mid]) ** 2)
    ref = Is_line.max()
    axi.plot(u, Is_line / ref, "k--", label="escalar")
    axi.plot(u, Iv_line / ref, "-", color="#d62728", label="vectorial")
    for s in (-0.5 * sep / lam, 0.5 * sep / lam):
        axi.axvline(s, color="gray", lw=0.7, alpha=0.6)
    axi.set_xlim(-2.2 * sep / lam, 2.2 * sep / lam)
    axi.set_xlabel(r"posición$/\lambda$")
    axi.set_ylabel("I (norm.)")
    axi.set_title(label, fontsize=10)
    axi.grid(True, alpha=0.3)
    axi.legend()
fig.suptitle("Perfiles a través de las dos líneas\n" + caption, fontsize=10)
path = out_dir / "profiles.png"
fig.savefig(path, dpi=200)
print_saved(path)
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 3: polarization map on the image plane (vectorial, vertical sep.)
# ---------------------------------------------------------------------

E_v = results[False]["E_v"]
lim = 2.2 * sep / lam
E_v_lam = FieldCartesian(x=E_v.x, y=E_v.y, grid=Grid.from_cartesian(E_v.grid.X / lam, E_v.grid.Y / lam), symmetric=False)
ax, _ = plot_field_polarization(E_v_lam, half_size=lim, sampling="cartesian",
                                target_ellipses=17, scale_by_intensity=True, min_ellipse_scale=0.5,
                                min_intensity_fraction=0.01)
ax.set(title="Mapa de polarización en el plano imagen (vectorial, separación vertical)",
       xlabel=r"$x/\lambda$", ylabel=r"$y/\lambda$")
ax.figure.suptitle(caption, fontsize=8)
path = out_dir / "image_plane_polarization.png"
ax.figure.savefig(path, dpi=200, bbox_inches="tight")
print_saved(path)
plt.close(ax.figure)
