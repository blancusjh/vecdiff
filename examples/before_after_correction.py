"""What the geometric factor and the pupil mapping change, side by side.

Two comparisons, both against ``geometry="none"`` -- the weighting the package
applied before this correction, kept as a switch so the difference is
measurable rather than remembered:

1. **Focal spot** of the glass dioptre at a near-grazing pupil, with the
   Franz / Stratton-Chu reference overlaid so it is clear which of the two is
   right rather than merely which is different.
2. **Lithography relay**: the circuit-mask image through the two-dioptre ArF
   system, where the change is a resolution change rather than a scale change.

Run from the repository root::

    uv run python examples/before_after_correction.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.special import jv

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vecdiff import CartesianSurface  # noqa: E402
from vecdiff.pupil_mapping import debye_prefactor  # noqa: E402
from vecdiff.reference import focal_field_reference  # noqa: E402
from vecdiff.transfer import focal_channel_weights  # noqa: E402

from _output import example_output_dir, print_saved  # noqa: E402

π = np.pi


# ------------------------------------------------------------------ #
#  Focal spot                                                          #
# ------------------------------------------------------------------ #

def focal_map(surface, wavelength, aperture, geometry, rho, phi, *, n_r=4001):
    """Focal-plane |E|^2 map for one weighting, on an absolute amplitude scale."""
    mapping = "sine" if geometry == "full" else "identity"
    r = np.linspace(0.0, aperture, int(n_r))
    w_p, w_s, w_z, u = focal_channel_weights(surface, r, geometry=geometry, mapping=mapping)

    lam_plus = 0.5 * (w_p + w_s)
    lam_minus = 0.5 * (w_p - w_s)

    k_i = 2.0 * π * surface.ni / wavelength
    # The channels are radial, so evaluate them on a 1D q profile and
    # interpolate onto the observation points.  Building the kernel at every
    # pixel of a 2D map would be a q-samples by pupil-samples matrix.
    rho = np.asarray(rho, dtype=float)
    q_max = k_i * float(np.max(rho)) / abs(surface.zi)
    q = np.linspace(0.0, q_max, 1200)

    def hankel(order, g):
        integrand = g[None, :] * jv(order, q[:, None] * u[None, :]) * u[None, :]
        return simpson(integrand, x=u, axis=1)

    h0, h2, h1 = hankel(0, lam_plus), hankel(2, lam_minus), hankel(1, w_z)

    q_of = k_i * rho / abs(surface.zi)
    prefactor = debye_prefactor(surface, wavelength)
    H0 = np.interp(q_of, q, h0)
    H2 = np.interp(q_of, q, h2)
    H1 = np.interp(q_of, q, h1)

    Ex = prefactor * (H0 - np.cos(2.0 * phi) * H2)
    Ey = prefactor * (-np.sin(2.0 * phi) * H2)
    Ez = 1j * prefactor * np.cos(phi) * H1
    return Ex, Ey, Ez


def focal_comparison(out_dir):
    n0, ni, z0, zi = 1.0, 1.5, -10.0, 6.0
    lam = 532e-6
    surface = CartesianSurface(n0=n0, ni=ni, z0=z0, zi=zi)
    aperture = 0.97 * surface.aperture_limit
    edge = surface.ray_geometry(np.array([aperture]))

    half = 1.6
    n = 401
    axis = np.linspace(-half, half, n) * lam
    X, Y = np.meshgrid(axis, axis)
    rho, phi = np.hypot(X, Y), np.arctan2(Y, X)

    fields = {}
    for tag, geometry in (("antes", "none"), ("después", "full")):
        Ex, Ey, Ez = focal_map(surface, lam, aperture, geometry, rho, phi)
        fields[tag] = dict(
            I=np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2,
            I_t=np.abs(Ex) ** 2 + np.abs(Ey) ** 2,
            Iz=np.abs(Ez) ** 2,
            Ex=Ex,
        )

    cut_rho = np.linspace(0.0, half, 241) * lam
    zeros = np.zeros_like(cut_rho)
    obs = np.stack([cut_rho, zeros, np.full_like(cut_rho, zi)], axis=-1)
    print("   integrando la referencia de Franz ...")
    _, E_ref = focal_field_reference(
        surface, lam, lambda r: np.ones_like(r),
        aperture=aperture, observation=obs, n_r=2400, n_phi=192, chunk=24_000,
    )
    ref_cut = np.abs(E_ref[:, 0]) ** 2

    cuts = {}
    for tag, geometry in (("antes", "none"), ("después", "full")):
        Ex, _, _ = focal_map(
            surface, lam, aperture, geometry, cut_rho, np.zeros_like(cut_rho)
        )
        cuts[tag] = np.abs(Ex) ** 2

    print(f"   apertura r = {aperture:.4f} mm, NA_i = {ni*edge.sin_ai[0]:.3f}")
    for tag in ("antes", "después"):
        print(f"   {tag:<8} pico |Ex|^2 / referencia = {cuts[tag][0]/ref_cut[0]:.5f}")

    vmax = max(f["I"].max() for f in fields.values())
    extent = [-half, half, -half, half]

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.0))
    for row, tag in enumerate(("antes", "después")):
        f = fields[tag]
        for col, (key, title) in enumerate(
            (("I_t", r"transversal $|E_x|^2+|E_y|^2$"),
             ("Iz", r"longitudinal $|E_z|^2$"),
             ("I", r"total $|\mathbf{E}|^2$")),
        ):
            ax = axes[row, col]
            im = ax.imshow(f[key], extent=extent, origin="lower", cmap="hot",
                           vmin=0.0, vmax=vmax)
            ax.set_title(f"{title} — {tag}", fontsize=10)
            ax.set_xlabel(r"$x/\lambda$")
            if col == 0:
                ax.set_ylabel(r"$y/\lambda$")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        "Escala absoluta común. «antes» = pesos de Fresnel desnudos, sin mapeo; "
        f"«después» = $A(Q)$ + proyección meridional + mapeo seno.  NA$_i$={ni*edge.sin_ai[0]:.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / "focal_spot_before_after.png"
    fig.savefig(path, dpi=140)
    print_saved(path)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    s = cut_rho / lam
    ax = axes[0]
    ax.plot(s, ref_cut, "k", lw=3.2, alpha=0.35, label="referencia exacta (Franz)")
    ax.plot(s, cuts["antes"], lw=1.5, label="antes")
    ax.plot(s, cuts["después"], lw=1.5, label="después")
    ax.set(xlabel=r"$\rho/\lambda$", ylabel=r"$|E_x|^2$ (escala absoluta)",
           title="Corte radial, amplitud absoluta")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for tag in ("antes", "después"):
        ax.plot(s, cuts[tag] / cuts[tag].max(), lw=1.5, label=tag)
    ax.plot(s, ref_cut / ref_cut.max(), "k", lw=3.2, alpha=0.35, label="referencia")
    ax.set(xlabel=r"$\rho/\lambda$", ylabel="intensidad normalizada",
           title="Forma del lóbulo (normalizada): el error no es un factor de escala",
           yscale="log", ylim=(1e-5, 2.0))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "focal_cut_before_after.png"
    fig.savefig(path, dpi=140)
    print_saved(path)


# ------------------------------------------------------------------ #
#  Lithography relay                                                   #
# ------------------------------------------------------------------ #

def lithography_comparison(out_dir):
    import lithography as litho

    results = {}
    for tag, geometry in (("antes", "none"), ("después", "full")):
        print(f"   relevo de litografía, {tag} ...")
        results[tag] = litho.run_lithography_example(geometry=geometry)

    ref = results["después"]
    xr, yr = ref["xr"], ref["yr"]
    win = ref["output_zoom"]
    extent = litho.image_extent_over_lambda(xr, yr)

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.4))
    for row, tag in enumerate(("antes", "después")):
        res = results[tag]
        panels = (
            ("Is", "escalar ($t_p=t_s$)"),
            ("Iv", r"vectorial $\hat{x}$"),
            ("Icross", r"canal cruzado $|E_y|^2$"),
        )
        for col, (key, title) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(litho.display_positive(res[key]), extent=extent,
                      origin="lower", cmap="inferno")
            ax.set_title(f"{title} — {tag}", fontsize=10)
            ax.set_xlim(win["xmin"] / litho.lam, win["xmax"] / litho.lam)
            ax.set_ylim(win["ymin"] / litho.lam, win["ymax"] / litho.lam)
            ax.set_xlabel(r"$x/\lambda$")
            if col == 0:
                ax.set_ylabel(r"$y/\lambda$")

    for tag in ("antes", "después"):
        print(f"   {tag:<8} f_cross = {results[tag]['cross_fraction']:.5f}")

    fig.suptitle(
        "Relevo de litografía ArF (193 nm, dos dioptrios): máscara de circuito en el plano imagen",
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / "lithography_before_after.png"
    fig.savefig(path, dpi=140)
    print_saved(path)

    # Radial-average power spectrum: where the resolution actually went.
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for tag in ("antes", "después"):
        res = results[tag]
        I = res["Iv"] - res["Iv"].mean()
        S = np.abs(np.fft.fftshift(np.fft.fft2(I))) ** 2
        ny, nx = S.shape
        ky = np.fft.fftshift(np.fft.fftfreq(ny, d=abs(res["yr"][1] - res["yr"][0])))
        kx = np.fft.fftshift(np.fft.fftfreq(nx, d=abs(res["xr"][1] - res["xr"][0])))
        KX, KY = np.meshgrid(kx, ky)
        K = np.hypot(KX, KY) * litho.lam
        bins = np.linspace(0.0, 2.0, 90)
        idx = np.digitize(K.ravel(), bins)
        prof = np.array([S.ravel()[idx == i].mean() if np.any(idx == i) else np.nan
                         for i in range(1, len(bins))])
        centers = 0.5 * (bins[1:] + bins[:-1])
        ax.semilogy(centers, prof / np.nanmax(prof), lw=1.6, label=tag)
    ax.set(xlabel=r"frecuencia espacial $k\lambda/2\pi$",
           ylabel="potencia radialmente promediada (normalizada)",
           title="Contenido espectral de la imagen vectorial")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = out_dir / "lithography_spectrum_before_after.png"
    fig.savefig(path, dpi=140)
    print_saved(path)


def main():
    out_dir = example_output_dir(__file__)
    print("punto focal:")
    focal_comparison(out_dir)
    print("litografía:")
    lithography_comparison(out_dir)


if __name__ == "__main__":
    main()
