"""Resolution inversion: two points resolved by the scalar model, fused by the vectorial one.

Two circular features (the canonical two-point resolution objects) are imaged
by a pair of conjugate diopters with a fast exit surface and an edge-weighted
pupil (r/a)^8.  The edge weight plays two roles at once: it is a
Toraldo-type super-resolving apodization (narrowing the scalar spot), and it
concentrates the pupil energy near grazing incidence on the exit diopter,
where the polarization-mixing ratio t-/t+ peaks.

With the separation perpendicular to the incident x polarization, the mixing
term interferes constructively in the valley between the two image spots:
the *scalar* image (t- = 0) shows two clearly split maxima while the
*vectorial* image fills the valley and fuses them.  The same inversion is
shown for a high-index diopter (n_i = 2.4, diamond-like) and for ordinary
glass (n_i = 1.5), where it is weaker but survives.

The renders use a zoom FFT (a k-grid restricted to the displayed window), so
the sub-wavelength images are densely sampled instead of inheriting the
coarse full-field pixel size.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vecdiff import CartesianSurface, FieldCartesian, Grid
from vecdiff.fresnel import FresnelOvoid
from vecdiff.propagation import fresnel_coefficients_on_grid
from vecdiff.polarization_visualization import plot_field_polarization
from _output import example_output_dir, print_saved

lam = 532e-6  # mm
EDGE_P = 8.0       # pupil amplitude ~ (r/a)^EDGE_P
SEP_FRAC = 0.56    # separation in Airy diameters: inside the inversion window
DISC_DIAM = 0.30e-3  # mm, canonical small circular features
N, N_K, N_ZOOM = 641, 1024, 480

out_dir = example_output_dir(__file__)


def grazing_radius(diopter_params):
    """Pupil radius where incidence on the oval becomes grazing (cos_i -> 0)."""
    fresnel = FresnelOvoid(**diopter_params)
    rho = np.linspace(1e-4, 3.0 * abs(diopter_params["z0"]), 4000)
    with np.errstate(all="ignore"):
        cos_i, _ = fresnel._cosines(rho)
        finite = np.isfinite(fresnel.ovoid.z(rho)) & np.isfinite(fresnel.ovoid.r(rho))
    return float(rho[np.nanargmin(np.where(finite, cos_i, np.nan))])


def make_system(ni_glass, r_a=None):
    """Conjugate pair air -> glass -> air with a fast exit surface."""
    d1 = dict(n0=1.0, ni=ni_glass, z0=-6.0, zi=3.6)
    d2 = dict(n0=ni_glass, ni=1.0, z0=-d1["zi"], zi=0.6)
    if r_a is None:
        # Largest pupil whose Fourier-plane image stays inside D2's physical
        # (pre-grazing) branch.
        scale = lam * d1["zi"] / (2.0 * np.pi * d1["ni"])
        k_edge = 0.97 * grazing_radius(d2) / scale
        sin_alpha = min(k_edge * lam / (2.0 * np.pi * d1["n0"]), 0.999)
        r_a = abs(d1["z0"]) * np.tan(np.arcsin(sin_alpha))
    return d1, d2, float(r_a)


def scalar_limit(field, diopter):
    """Apply the isotropic t+ = (tp+ts)/2 factor (the t- = 0 scalar limit)."""
    support = (np.abs(field.x) > 0.0) | (np.abs(field.y) > 0.0)
    tp, ts = fresnel_coefficients_on_grid(field.grid.R, diopter, support=support)
    t_plus = 0.5 * (tp + ts)
    return FieldCartesian(x=t_plus * field.x, y=t_plus * field.y, grid=field.grid, symmetric=False)


def run_system(tag, label, ni_glass, r_a=None):
    d1, d2, r_a = make_system(ni_glass, r_a)
    alpha_max = np.arctan(r_a / abs(d1["z0"]))
    Kc = d1["n0"] * (2.0 * np.pi / lam) * np.sin(alpha_max)
    d_airy = 2.0 * 3.8317059702075125 / Kc
    sep = SEP_FRAC * d_airy
    mag = (d2["zi"] * d1["ni"]) / (d1["zi"] * d2["ni"])
    pupil_radius = lam * d1["zi"] * Kc / (2.0 * np.pi * d1["ni"])

    # Object: two discs separated along y (perpendicular to the x polarization).
    L = 24.0 * sep
    x = np.linspace(-L / 2.0, L / 2.0, N)
    grid = Grid.from_axes(x, x)
    mask = np.zeros(grid.shape)
    for sign in (-1.0, 1.0):
        mask[grid.X**2 + (grid.Y - sign * 0.5 * sep) ** 2 <= (0.5 * DISC_DIAM) ** 2] = 1.0

    diopter1, diopter2 = CartesianSurface(**d1), CartesianSurface(**d2)
    lim_lam = 1.6 * mag * sep / lam  # image-plane half-window in wavelengths

    # Zoomed output k-grid: N_ZOOM points covering only the displayed window.
    scale2 = lam * d2["zi"] / (2.0 * np.pi * d2["ni"])
    k_zoom = np.linspace(-1.05 * lim_lam * lam / scale2, 1.05 * lim_lam * lam / scale2, N_ZOOM)
    KXz, KYz = np.meshgrid(k_zoom, k_zoom)
    kgrid_zoom = Grid.from_cartesian(KXz, KYz, domain="k")

    def image(vectorial):
        field0 = FieldCartesian(x=mask.astype(complex), y=np.zeros_like(mask, dtype=complex),
                                grid=grid, symmetric=False)
        transmission = "vectorial" if vectorial else "identity"
        field0 = field0 if vectorial else scalar_limit(field0, diopter1)
        field1 = field0.propagate_through_diopter(
            diopter1.zi, diopter1, method="fft", output="focal", wavelength=lam,
            kgrid=grid.kgrid(N_K), transmission=transmission)
        # Edge-weighted circular pupil in the Fourier plane of D1.
        weight = np.where(field1.grid.R <= pupil_radius,
                          (field1.grid.R / pupil_radius) ** EDGE_P, 0.0)
        field1 = FieldCartesian(x=weight * field1.x, y=weight * field1.y,
                                grid=field1.grid, symmetric=False)
        field1 = field1 if vectorial else scalar_limit(field1, diopter2)
        return field1.propagate_through_diopter(
            diopter2.zi, diopter2, method="fft", output="focal", wavelength=lam,
            kgrid=kgrid_zoom, transmission=transmission)

    E_s, E_v = image(False), image(True)
    I_s = np.abs(E_s.x) ** 2 + np.abs(E_s.y) ** 2
    I_v = np.abs(E_v.x) ** 2 + np.abs(E_v.y) ** 2
    vmax = max(I_s.max(), I_v.max())

    # Valley contrast along the separation axis (x = 0 column).
    yr = E_s.grid.Y[:, 0]
    mid = N_ZOOM // 2
    contrasts = {}
    for name, I in (("escalar", I_s), ("vectorial", I_v)):
        cut = I[:, mid]
        u_line = 0.5 * mag * sep
        peaks = 0.5 * (cut[(yr < 0) & (np.abs(-yr - u_line) < 0.3 * u_line)].max()
                       + cut[(yr > 0) & (np.abs(yr - u_line) < 0.3 * u_line)].max())
        valley = cut[np.abs(yr) <= 0.4 * u_line].min()
        contrasts[name] = float((peaks - valley) / (peaks + valley))
    print(f"{tag}: r_a={r_a:.2f} mm, sep={sep*1e3:.3f} um ({SEP_FRAC:g} d_Airy), "
          f"C_escalar={contrasts['escalar']:.3f}, C_vectorial={contrasts['vectorial']:.3f}")

    caption = (rf"dioptrios conjugados: $n_0$=1, $n_i$={ni_glass:g}  ·  "
               rf"$z_0$={d1['z0']:g}, $z_{{i1}}$={d1['zi']:g}, $z_{{i2}}$={d2['zi']:g} mm  ·  "
               rf"$\lambda$={lam*1e6:.0f} nm  ·  $r_a$={r_a:.2f} mm, "
               rf"$\alpha_\mathrm{{max}}$={np.degrees(alpha_max):.1f}°, pupila $(r/a)^{{{EDGE_P:g}}}$  ·  "
               r"pol. incidente lineal $\hat{x}$, separación $\parallel \hat{y}$")

    # Figure: mask, scalar image, vectorial image, profile along the separation.
    ext = [E_s.grid.X[0, 0] / lam, E_s.grid.X[0, -1] / lam,
           E_s.grid.Y[0, 0] / lam, E_s.grid.Y[-1, 0] / lam]
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.3), constrained_layout=True)

    half_mask = lim_lam / mag * lam
    xf = np.linspace(-half_mask, half_mask, N_ZOOM)
    XF, YF = np.meshgrid(xf, xf)
    mask_fine = np.zeros_like(XF)
    for sign in (-1.0, 1.0):
        mask_fine[XF**2 + (YF - sign * 0.5 * sep) ** 2 <= (0.5 * DISC_DIAM) ** 2] = 1.0
    axes[0].imshow(mask_fine, extent=[v / lam for v in (-half_mask, half_mask, -half_mask, half_mask)],
                   origin="lower", cmap="gray")
    axes[0].set_title("Máscara (objeto)")
    axes[1].imshow((I_s / vmax) ** 0.8, extent=ext, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(rf"Imagen escalar ($t_-=0$): C = {contrasts['escalar']:.3f}")
    axes[2].imshow((I_v / vmax) ** 0.8, extent=ext, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(rf"Imagen vectorial: C = {contrasts['vectorial']:.3f}")
    for axi in axes[1:3]:
        axi.set_xlim(-lim_lam, lim_lam)
        axi.set_ylim(-lim_lam, lim_lam)
    for axi in axes[:3]:
        axi.set_xlabel(r"$x/\lambda$")
        axi.set_ylabel(r"$y/\lambda$")

    ref = I_s[:, mid].max()
    axes[3].plot(yr / (mag * sep), I_s[:, mid] / ref, "k--", label=f"escalar (C={contrasts['escalar']:.3f})")
    axes[3].plot(yr / (mag * sep), I_v[:, mid] / ref, "-", color="#d62728",
                 label=f"vectorial (C={contrasts['vectorial']:.3f})")
    for s in (-0.5, 0.5):
        axes[3].axvline(s, color="gray", lw=0.7, alpha=0.6)
    axes[3].set_xlim(-1.1, 1.1)
    axes[3].set_xlabel(r"$y / (M \cdot sep)$")
    axes[3].set_ylabel("I (norm.)")
    axes[3].set_title(r"Perfil a lo largo de la separación")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.suptitle(rf"Dos puntos, {label}: sep = {SEP_FRAC:g} $d_\mathrm{{Airy}}$ = {sep*1e3:.3f} µm"
                 + "\n" + caption, fontsize=10)
    path = out_dir / f"inversion_{tag}.png"
    fig.savefig(path, dpi=200)
    print_saved(path)
    plt.close(fig)

    return E_v, lim_lam, caption


E_v_high, lim_high, caption_high = run_system("high_index", r"alto índice ($n_i$=2.4)", 2.4, r_a=10.0)
run_system("glass", r"vidrio ($n_i$=1.5)", 1.5)

# ---------------------------------------------------------------------
# Polarization map on the image plane (high-index vectorial image)
# ---------------------------------------------------------------------

E_lam = FieldCartesian(x=E_v_high.x, y=E_v_high.y,
                       grid=Grid.from_cartesian(E_v_high.grid.X / lam, E_v_high.grid.Y / lam),
                       symmetric=False)
ax, _ = plot_field_polarization(E_lam, half_size=lim_high, sampling="cartesian",
                                target_ellipses=17, scale_by_intensity=True, min_ellipse_scale=0.5,
                                min_intensity_fraction=0.01)
ax.set(title="Mapa de polarización en el plano imagen (vectorial, alto índice)",
       xlabel=r"$x/\lambda$", ylabel=r"$y/\lambda$")
ax.figure.suptitle(caption_high, fontsize=8)
path = out_dir / "image_plane_polarization.png"
ax.figure.savefig(path, dpi=200, bbox_inches="tight")
print_saved(path)
plt.close(ax.figure)
