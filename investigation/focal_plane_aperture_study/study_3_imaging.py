"""Study 3 - Image formation: scalar (t_minus = 0) vs vectorial vs aperture.

A two-line mask, x-polarized, is imaged by a 4f-like system of two conjugate
diopters (D1: air -> glass, D2: glass -> air, the geometry of the former
``resolution_two_features`` example).  A circular pupil of radius ``Kc(r_a)``
is applied in the Fourier plane between the diopters; sweeping the aperture
radius ``r_a`` sweeps the numerical aperture of the system.

The *scalar* system applies only the isotropic transmission ``t_plus`` at each
diopter (t_minus = 0); the *vectorial* system applies the full transverse
operator.  Both share the same pupil, so every difference in the images comes
from the polarization mixing term.

For each aperture and for the two mask orientations (line separation parallel
and perpendicular to the incident polarization) the script records the
two-line valley contrast and the energy fraction in the cross-polarized image
channel.  Outputs go to ``output/study_3_imaging/``.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from vecdiff import CartesianSurface, FieldCartesian, Grid
from vecdiff.propagation import fresnel_coefficients_on_grid

import common

LAM = 532e-6  # mm

D1 = dict(n0=1.0, ni=2.4, z0=-6.0, zi=3.6)
D2 = dict(n0=D1["ni"], ni=1.0, z0=-D1["zi"], zi=3.6)

R_A_LIST = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.5]  # pupil radii in mm
GALLERY_R_A = [3.0, 6.5]
R_A_REF = 4.0                          # aperture defining the line separation

N = 641
L = 0.02  # mm
N_K = 1024

out = common.output_dir(__file__)


def Kc_of(r_a):
    alpha = np.arctan(r_a / abs(D1["z0"]))
    return D1["n0"] * (2.0 * np.pi / LAM) * np.sin(alpha)


# Two-line mask: separation 0.9 * d_Airy at the reference aperture.
r_airy_ref = 3.8317059702075125 / Kc_of(R_A_REF)
SEP = 0.9 * 2.0 * r_airy_ref
LINE_W = 0.35 * SEP
LINE_LEN = 3.0 * SEP

x = np.linspace(-L / 2.0, L / 2.0, N)
X, Y = np.meshgrid(x, x)
DX = float(x[1] - x[0])


def two_line_mask(theta):
    """Binary mask of two lines whose separation axis makes ``theta`` with x."""
    ux, uy = np.cos(theta), np.sin(theta)
    vx, vy = -uy, ux
    T = np.zeros_like(X)
    for sign in (-1.0, 1.0):
        cx, cy = sign * 0.5 * SEP * ux, sign * 0.5 * SEP * uy
        u = (X - cx) * ux + (Y - cy) * uy
        v = (X - cx) * vx + (Y - cy) * vy
        T[(np.abs(u) <= 0.5 * LINE_W) & (np.abs(v) <= 0.5 * LINE_LEN)] = 1.0
    return T


def kgrid_from_spacing(n, dx):
    k = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    KX, KY = np.meshgrid(k, k)
    return Grid.from_cartesian(KX, KY, domain="k")


def apply_t_plus(field, diopter):
    """Scalar limit: multiply both components by t_plus = (tp + ts)/2."""
    support = (np.abs(field.x) > 0.0) | (np.abs(field.y) > 0.0)
    tp, ts = fresnel_coefficients_on_grid(field.grid.R, diopter, support=support)
    t_plus = 0.5 * (tp + ts)
    return FieldCartesian(x=t_plus * field.x, y=t_plus * field.y,
                          grid=field.grid, symmetric=False)


def stage1(mask, vectorial):
    """Mask plane -> Fourier plane through D1 (no pupil yet)."""
    E0 = FieldCartesian(x=mask.astype(complex), y=np.zeros_like(mask, dtype=complex),
                        grid=Grid.from_cartesian(X, Y), symmetric=False)
    diopter = CartesianSurface(**D1)
    if vectorial:
        return E0.propagate_through_diopter(
            z=diopter.zi, ovoid=diopter, method="fft",
            kgrid=kgrid_from_spacing(N_K, DX), transmission="vectorial")
    E0s = apply_t_plus(E0, diopter)
    return E0s.propagate_through_diopter(
        z=diopter.zi, ovoid=diopter, method="fft",
        kgrid=kgrid_from_spacing(N_K, DX), transmission="identity")


def stage2(E1, r_a, vectorial):
    """Pupil in the Fourier plane, then D2 to the image plane."""
    kx = E1.grid.X[0, :]
    ky = E1.grid.Y[:, 0]
    KX, KY = np.meshgrid(kx, ky)
    pupil = (KX**2 + KY**2) <= Kc_of(r_a) ** 2

    scale = LAM * D1["zi"] / (2.0 * np.pi * D1["ni"])
    XF, YF = np.meshgrid(scale * kx, scale * ky)
    Ef = FieldCartesian(x=E1.x * pupil, y=E1.y * pupil,
                        grid=Grid.from_cartesian(XF, YF), symmetric=False)

    dxf = scale * float(kx[1] - kx[0])
    diopter = CartesianSurface(**D2)
    if vectorial:
        return Ef.propagate_through_diopter(
            z=diopter.zi, ovoid=diopter, method="fft",
            kgrid=kgrid_from_spacing(N_K, dxf), output="focal",
            wavelength=LAM, transmission="vectorial")
    Efs = apply_t_plus(Ef, diopter)
    return Efs.propagate_through_diopter(
        z=diopter.zi, ovoid=diopter, method="fft",
        kgrid=kgrid_from_spacing(N_K, dxf), output="focal",
        wavelength=LAM, transmission="identity")


# Transverse magnification of the two-diopter cascade (two Fourier stages).
MAG = (D2["zi"] * D1["ni"]) / (D1["zi"] * D2["ni"])


def _bilinear(I, xr, yr, px, py):
    ix = np.clip(np.searchsorted(xr, px) - 1, 0, xr.size - 2)
    iy = np.clip(np.searchsorted(yr, py) - 1, 0, yr.size - 2)
    wx = np.clip((px - xr[ix]) / (xr[ix + 1] - xr[ix]), 0.0, 1.0)
    wy = np.clip((py - yr[iy]) / (yr[iy + 1] - yr[iy]), 0.0, 1.0)
    return ((1 - wy) * ((1 - wx) * I[iy, ix] + wx * I[iy, ix + 1])
            + wy * ((1 - wx) * I[iy + 1, ix] + wx * I[iy + 1, ix + 1]))


def two_line_contrast(E, theta):
    """Valley contrast of the two-line image along the separation axis.

    The image is magnified by ``MAG``: the line centers sit at
    ``u = +-0.5 * MAG * SEP``.  Peaks and valley are searched in windows
    around their nominal positions to stay robust against coherent ringing.
    """
    I = np.abs(E.x) ** 2 + np.abs(E.y) ** 2
    xr = E.grid.X[0, :]
    yr = E.grid.Y[:, 0]
    u = np.linspace(-1.0, 1.0, 801) * (1.1 * MAG * SEP)
    profile = _bilinear(I, xr, yr, u * np.cos(theta), u * np.sin(theta))

    u_line = 0.5 * MAG * SEP
    peak_win = np.abs(np.abs(u) - u_line) <= 0.3 * u_line
    valley_win = np.abs(u) <= 0.4 * u_line
    peaks = 0.5 * (profile[peak_win & (u < 0)].max() + profile[peak_win & (u > 0)].max())
    valley = profile[valley_win].min()
    contrast = (peaks - valley) / (peaks + valley + 1e-300)
    return float(contrast), u, profile


def image_energies(E):
    I_co = np.abs(E.x) ** 2
    I_cross = np.abs(E.y) ** 2
    return float(I_co.sum()), float(I_cross.sum())


# ------------------------------------------------------------------ #
#  Sweep                                                               #
# ------------------------------------------------------------------ #

orientations = {"sep_x": 0.0, "sep_y": 0.5 * np.pi}
masks = {name: two_line_mask(theta) for name, theta in orientations.items()}

rows = []
images = {}
profiles = {}
for name, theta in orientations.items():
    E1 = {"vec": stage1(masks[name], vectorial=True),
          "sca": stage1(masks[name], vectorial=False)}
    for r_a in R_A_LIST:
        I_sca_ref = None
        for model in ("sca", "vec"):
            E2 = stage2(E1[model], r_a, vectorial=(model == "vec"))
            contrast, u, prof = two_line_contrast(E2, theta)
            e_co, e_cross = image_energies(E2)
            I_total = np.abs(E2.x) ** 2 + np.abs(E2.y) ** 2
            if model == "sca":
                I_sca_ref = I_total
                rel_l1 = 0.0
            else:
                # Global scalar/vectorial image difference (normalization-free).
                s = I_sca_ref / I_sca_ref.sum()
                v = I_total / I_total.sum()
                rel_l1 = float(np.abs(v - s).sum())
            rows.append({
                "orientation": name, "r_a": r_a, "model": model,
                "alpha_deg": np.degrees(np.arctan(r_a / abs(D1["z0"]))),
                "sep_over_dAiry": SEP / (2.0 * 3.8317059702075125 / Kc_of(r_a)),
                "contrast": contrast,
                "f_cross_image": e_cross / (e_co + e_cross + 1e-300),
                "rel_l1_vs_scalar": rel_l1,
            })
            images[(name, r_a, model)] = E2
            profiles[(name, r_a, model)] = (u, prof)
            print(f"{name}  r_a={r_a:4.1f}  {model}  contraste={contrast:+.4f}  "
                  f"f_cross_img={rows[-1]['f_cross_image']:.2e}  relL1={rel_l1:.3e}")

with (out / "imaging_results.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# ------------------------------------------------------------------ #
#  Figure 1: contrast and image cross fraction vs aperture             #
# ------------------------------------------------------------------ #

def _sel(name, model):
    return [r for r in rows if r["orientation"] == name and r["model"] == model]


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
styles = {("sep_x", "sca"): ("#1f77b4", "--", "escalar, sep ∥ x"),
          ("sep_x", "vec"): ("#1f77b4", "-", "vectorial, sep ∥ x"),
          ("sep_y", "sca"): ("#d62728", "--", "escalar, sep ∥ y"),
          ("sep_y", "vec"): ("#d62728", "-", "vectorial, sep ∥ y")}
for (name, model), (color, ls, label) in styles.items():
    sel = _sel(name, model)
    ax1.plot([r["r_a"] for r in sel], [r["contrast"] for r in sel],
             ls, color=color, marker="o", label=label)
ax1.axhline(0.0, color="k", lw=0.8)
ax1.set_xlabel(r"radio de pupila $r_a$ [mm]")
ax1.set_ylabel("contraste del valle")
ax1.set_title(f"Resolución de dos líneas (sep = {SEP*1e3:.2f} µm fija)")
ax1.grid(True, alpha=0.3)
ax1.legend()

for name, color in (("sep_x", "#1f77b4"), ("sep_y", "#d62728")):
    c_v = np.array([r["contrast"] for r in _sel(name, "vec")])
    c_s = np.array([r["contrast"] for r in _sel(name, "sca")])
    ra = [r["r_a"] for r in _sel(name, "vec")]
    ax2.plot(ra, c_v - c_s, "o-", color=color, label=f"{name}")
ax2.axhline(0.0, color="k", lw=0.8)
ax2.set_xlabel(r"radio de pupila $r_a$ [mm]")
ax2.set_ylabel(r"$C_{vec} - C_{esc}$")
ax2.set_title("Cambio de contraste por el término vectorial")
ax2.grid(True, alpha=0.3)
ax2.legend()

for name, color in (("sep_x", "#1f77b4"), ("sep_y", "#d62728")):
    sel = _sel(name, "vec")
    ax3.semilogy([r["r_a"] for r in sel], [r["f_cross_image"] for r in sel],
                 "o-", color=color, label=rf"$f_{{cross}}$ imagen, {name}")
    ax3.semilogy([r["r_a"] for r in sel], [r["rel_l1_vs_scalar"] for r in sel],
                 "s--", color=color, label=rf"$L_1(I_{{vec}}, I_{{esc}})$, {name}")
ax3.set_xlabel(r"radio de pupila $r_a$ [mm]")
ax3.set_title("Diferencia global escalar/vectorial")
ax3.grid(True, which="both", alpha=0.3)
ax3.legend(fontsize=8)
fig.suptitle("Formación de imágenes: escalar ($t_-=0$) vs vectorial")
fig.savefig(out / "fig1_contrast_vs_aperture.png", dpi=220)
plt.close(fig)

# ------------------------------------------------------------------ #
#  Figure 2: image galleries at the largest aperture                   #
# ------------------------------------------------------------------ #

def norm(I):
    return I / (I.max() + 1e-300)


for name in orientations:
  for r_a in GALLERY_R_A:
    E_s = images[(name, r_a, "sca")]
    E_v = images[(name, r_a, "vec")]
    I_s = norm(np.abs(E_s.x) ** 2 + np.abs(E_s.y) ** 2)
    I_v = norm(np.abs(E_v.x) ** 2 + np.abs(E_v.y) ** 2)
    I_cross = np.abs(E_v.y) ** 2 / (np.abs(E_v.x) ** 2 + np.abs(E_v.y) ** 2).max()
    delta = I_v - I_s

    xr = E_s.grid.X[0, :] / LAM
    yr = E_s.grid.Y[:, 0] / LAM
    extent = [xr[0], xr[-1], yr[0], yr[-1]]
    lim = 1.6 * MAG * SEP / LAM

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2), constrained_layout=True)
    panels = [
        (masks[name], [v / LAM for v in (-L / 2, L / 2, -L / 2, L / 2)], "gray", "Máscara", (0, 1)),
        (I_s ** 0.8, extent, "gray", "Imagen escalar", (0, 1)),
        (I_v ** 0.8, extent, "gray", "Imagen vectorial", (0, 1)),
        (delta, extent, "RdBu_r", r"$\Delta I$", None),
        (I_cross, extent, "magma", r"$|E_y|^2$ (cruzada)", None),
    ]
    for axi, (img, ext, cmap, title, clim) in zip(axes, panels):
        if clim is None and cmap == "RdBu_r":
            m = np.abs(img).max()
            im = axi.imshow(img, extent=ext, origin="lower", cmap=cmap, vmin=-m, vmax=m)
        elif clim is None:
            im = axi.imshow(img / (img.max() + 1e-300), extent=ext, origin="lower", cmap=cmap)
        else:
            im = axi.imshow(img, extent=ext, origin="lower", cmap=cmap, vmin=clim[0], vmax=clim[1])
        axi.set_title(title)
        axi.set_xlabel(r"$x/\lambda$")
        axi.set_ylabel(r"$y/\lambda$")
        axi.set_xlim(-lim, lim)
        axi.set_ylim(-lim, lim)
        fig.colorbar(im, ax=axi, fraction=0.046, pad=0.04)
    fig.suptitle(f"Imágenes con r_a = {r_a} mm, orientación {name}")
    fig.savefig(out / f"fig2_gallery_{name}_ra{str(r_a).replace('.', 'p')}.png", dpi=200)
    plt.close(fig)

# ------------------------------------------------------------------ #
#  Figure 3: line profiles across apertures                            #
# ------------------------------------------------------------------ #

fig, axes = plt.subplots(2, len(R_A_LIST), figsize=(4.0 * len(R_A_LIST), 7.6),
                         constrained_layout=True, sharey="row")
for row_i, name in enumerate(orientations):
    for col_i, r_a in enumerate(R_A_LIST):
        ax = axes[row_i, col_i]
        u_s, p_s = profiles[(name, r_a, "sca")]
        u_v, p_v = profiles[(name, r_a, "vec")]
        ref = p_s.max() + 1e-300
        ax.plot(u_s / (MAG * SEP), p_s / ref, "k--", label="escalar")
        ax.plot(u_v / (MAG * SEP), p_v / ref, "-", color="#d62728", label="vectorial")
        for s in (-0.5, 0.5):
            ax.axvline(s, color="gray", lw=0.7, alpha=0.6)
        if row_i == 0:
            ax.set_title(f"r_a = {r_a} mm")
        if col_i == 0:
            ax.set_ylabel(f"{name}\nI (norm. al pico escalar)")
        ax.set_xlabel(r"$u / (M \cdot sep)$")
        ax.grid(True, alpha=0.3)
axes[0, 0].legend()
fig.suptitle("Perfiles a través de las dos líneas: escalar vs vectorial")
fig.savefig(out / "fig3_profiles.png", dpi=200)
plt.close(fig)

print(f"Done. Outputs in {out}")
