import numpy as np
import matplotlib.pyplot as plt

from references.legacy.vecdiff import (
    CartesianSurface,
    FieldCartesian,
    FieldCircular,
    FieldPolar,
    Grid,
    focal_channel_weights,
)
from references.legacy.vecdiff.hankel import HankelTransform
from references.legacy.vecdiff.longitudinal import generate_Ez_cartesian
from references.legacy.vecdiff.view import sample_component_pair_on_cartesian_mesh
from _common import focal_field, figure_saver

# --- Setup: same diopter as polar_simple / circular_simple ---
n0, ni, z0, zi = 1.0, 1.5, -10.0, 6.0
lam = 532e-6
diopter_for_aperture = CartesianSurface(n0=n0, ni=ni, z0=z0, zi=zi)
# Just inside grazing incidence, in the transverse radius on the surface --
# which is the aperture coordinate.
R = 0.97 * diopter_for_aperture.aperture_limit
n_r, n_q, n_phi = 1000, 4800, 256

r = np.linspace(0.0, R, n_r)
# Uniform focal grid out to 24 lambda: the FFT window below must be large
# enough that the truncated Airy tails do not ring back into the center
# (with half_win=8 the reconstruction error at the peak is ~3%; with 16 it
# drops to ~2e-3).
s_max = 24.0
q = (2.0 * np.pi * ni / zi) * np.linspace(0.0, s_max, n_q)
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

# The longitudinal channel is exact: lam_z of Eq. (63) is
# A tp sin(alpha_i)/cos(alpha_0), and transversality on the image sphere gives
# E_z = tan(alpha_i) E_r.
tp, ts, wz_tp, u = focal_channel_weights(diopter, r)
with np.errstate(divide="ignore", invalid="ignore"):
    w_z = np.divide(wz_tp, tp, out=np.zeros_like(tp), where=np.abs(tp) > 0.0)
tp = tp * pupil
ts = ts * pupil

s = q * zi / (2.0 * np.pi * ni)  # focal radius in units of lambda


def HT(order, f_r):
    return HankelTransform.transform_array(order, u, f_r, q)


cases = {
    "radial": FieldPolar(r=1.0 * pupil, phi=0.0 * pupil, grid=grid),
    "azimutal": FieldPolar(r=0.0 * pupil, phi=1.0 * pupil, grid=grid),
    "lineal": FieldCartesian(x=1.0 * pupil, y=0.0 * pupil, grid=grid),
    "circular": FieldCircular(L=0.0 * pupil, R=1.0 * pupil, grid=grid),
}

# Closed radial forms of |E_z| on the focal axis s (same 2*pi Hankel
# convention as vecdiff.propagation): m=0 channel for radial incidence,
# m=1 channel for linear and circular incidence, exactly zero for azimuthal.
Hz_radial = 2.0 * np.pi * np.abs(HT(0, w_z * tp * pupil))
Hz_m1 = 2.0 * np.pi * np.abs(HT(1, w_z * tp * pupil))
closed_form = {
    "radial": lambda rho, varphi: np.interp(rho, s, Hz_radial),
    "azimutal": lambda rho, varphi: np.zeros_like(rho),
    "lineal": lambda rho, varphi: np.interp(rho, s, Hz_m1) * np.abs(np.cos(varphi)),
    "circular": lambda rho, varphi: np.interp(rho, s, Hz_m1) / np.sqrt(2.0),
}

# --- Package pipeline: sample the focal transverse field on a fine Cartesian
# window (in lambda units) and reconstruct E_z by angular spectrum. ---
half_win, n_win = 16.0, 2048
x_win = np.linspace(-half_win, half_win, n_win)
xx, yy = np.meshgrid(x_win, x_win)
rho, varphi = np.hypot(xx, yy), np.arctan2(yy, xx)

Ez_maps, Et_maps = {}, {}
for name, E0 in cases.items():
    E_f = focal_field(
        E0.propagate_through_diopter(zi, diopter, q), zi / (2.0 * np.pi * ni)
    )
    Ex, Ey, _ = sample_component_pair_on_cartesian_mesh(
        E_f.x, E_f.y, E_f.grid, half_size=half_win, n_img=n_win
    )
    Ez_maps[name] = generate_Ez_cartesian(Ex, Ey, (xx, yy), wavelength=1.0, n=ni)
    Et_maps[name] = np.abs(Ex) ** 2 + np.abs(Ey) ** 2

# --- Validation: package pipeline vs closed radial forms ---
ref_peak = float(np.max(np.abs(Ez_maps["radial"])))
central = rho < 3.0
for name in cases:
    ana = closed_form[name](rho, varphi)
    num = np.abs(Ez_maps[name])
    err = float(np.max(np.abs(num - ana)[central])) / ref_peak
    print(f"{name:9s}: max |E_z(FFT)| = {np.max(num):.4f},  "
          f"max ||E_z|-forma cerrada| / pico radial = {err:.2e}")
    assert err < 1e-2, f"validación fallida para {name}"

# --- Figure 1: |E_z| for the four incident polarizations, shared scale ---
titles = {
    "radial": "radial",
    "azimutal": "azimutal",
    "lineal": r"lineal $\hat{\mathbf{x}}$",
    "circular": "circular",
}
half_show = 3.0
sel = (np.abs(x_win) <= half_show)
extent_show = [-half_show, half_show, -half_show, half_show]
vmax = max(float(np.max(np.abs(Ez_maps[n])[np.ix_(sel, sel)])) for n in cases)

fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.4), constrained_layout=True)
for ax, name in zip(axes, cases):
    img = np.abs(Ez_maps[name])[np.ix_(sel, sel)]
    im = ax.imshow(img, extent=extent_show, origin="lower", cmap="hot",
                   vmin=0.0, vmax=vmax, aspect="equal")
    ax.set_title(titles[name])
    ax.set_xlabel(r"$x/\lambda$")
axes[0].set_ylabel(r"$y/\lambda$")
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label=r"$|E_z|$")
fig.suptitle(
    "Componente longitudinal $|E_z|$ en el plano focal\n" + diopter_caption
)
save(fig, "Ez_casos")

# --- Figure 2: radial incidence, transverse ring vs total intensity ---
Er_prof = 2.0 * np.pi * np.abs(HT(1, tp * pupil))
I_perp = Er_prof**2
I_z = Hz_radial**2
I_tot = I_perp + I_z
norm = float(np.max(I_perp))

rho_show = np.hypot(*np.meshgrid(x_win[sel], x_win[sel]))
I_perp_map = np.interp(rho_show, s, I_perp)
I_z_map = np.interp(rho_show, s, I_z)
I_tot_map = np.interp(rho_show, s, I_tot)
vmax_int = float(np.max(I_tot_map))

panels = (
    (r"transversal $|E_r|^2$", I_perp_map),
    (r"longitudinal $|E_z|^2$", I_z_map),
    (r"total $|E_r|^2 + |E_z|^2$", I_tot_map),
)
fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4), constrained_layout=True)
for ax, (title, img) in zip(axes, panels):
    im = ax.imshow(img / img.max(), extent=extent_show, origin="lower",
                   cmap="hot", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel(r"$x/\lambda$")
    ax.text(0.03, 0.97, f"máx = {float(img.max()) / vmax_int:.3f} · máx total",
            transform=ax.transAxes, color="w", va="top", fontsize=8)
axes[0].set_ylabel(r"$y/\lambda$")
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
fig.suptitle(
    "Incidencia radial: la componente longitudinal llena el centro oscuro\n"
    + diopter_caption
)
save(fig, "radial_intensidad_total")

print(f"radial: I_z(0)/max(I_perp) = {I_z[0] / norm:.3f}")
print(f"radial: I_tot(0)/max(I_tot) = {I_tot[0] / float(np.max(I_tot)):.3f}")

plt.show()
