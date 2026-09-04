"""What limits the aperture of a two-surface stigmatic element, and whether the model holds there.

Three things can stop a ray through such an element, and which one binds decides
what the design can deliver:

* the two faces meeting, which is set by the axial separation and, through the
  sags, by the conjugates;
* grazing incidence on the entrance surface, where the incidence angle reaches
  ninety degrees at a finite radius;
* the critical angle on the exit surface, where the *transmitted* angle reaches
  ninety degrees first, at ``sin(theta_0) = ni / n0``.

The third is the hard one.  Past it no oval exists that would send the ray to
the image point -- Snell has no real solution -- so a dense-to-rare exit surface
cannot be continued however the conjugates are chosen.  A fused-silica-to-air
exit saturates near ``NA = 0.74``; reaching beyond that needs a higher-index
last element and an immersion medium, which is what the second configuration
here uses.

At those apertures the ray-optical chain has no right to be believed on its
own, so both exit surfaces are checked against the Franz / Stratton-Chu
integral, which is an exact Maxwell field in the image medium.

Run from the repository root::

    uv run python examples/aperture_limits.py
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

from references.legacy.vecdiff import CartesianSurface, relay_throughput  # noqa: E402
from references.legacy.vecdiff.pupil_mapping import debye_prefactor  # noqa: E402
from references.legacy.vecdiff.reference import focal_field_reference  # noqa: E402
from references.legacy.vecdiff.transfer import focal_channel_weights  # noqa: E402

from _output import example_output_dir, print_saved  # noqa: E402

π = np.pi

LAM = 193e-6
N_LUAG = 2.14
N_WATER = 1.437

DRY = dict(
    label="sílice → aire (seco)",
    first=dict(n0=1.0, ni=1.5602, z0=-4.0, zi=2.0),
    second=dict(n0=1.5602, ni=1.0, z0=2.0 - 5.0, zi=0.25 * 2.0 / 1.5602),
    separation=5.0,
)
IMMERSION = dict(
    label="LuAG → agua (inmersión)",
    first=dict(n0=1.0, ni=N_LUAG, z0=-4.0, zi=8.0),
    second=dict(n0=N_LUAG, ni=N_WATER, z0=8.0 - 50.0, zi=0.8),
    separation=50.0,
)


def model_cut(surface, wavelength, aperture, rho, *, n_r=5001):
    """Focal cut along x from the package's radial channels, absolute scale."""
    r = np.linspace(0.0, aperture, int(n_r))
    w_p, w_s, w_z, u = focal_channel_weights(surface, r)
    lam_plus = 0.5 * (w_p + w_s)
    lam_minus = 0.5 * (w_p - w_s)

    k_i = 2.0 * π * surface.ni / wavelength
    q = k_i * np.asarray(rho, dtype=float) / abs(surface.zi)

    def hankel(order, g):
        integrand = g[None, :] * jv(order, q[:, None] * u[None, :]) * u[None, :]
        return simpson(integrand, x=u, axis=1)

    prefactor = debye_prefactor(surface, wavelength)
    Ex = prefactor * (hankel(0, lam_plus) - hankel(2, lam_minus))
    Ez = 1j * prefactor * hankel(1, w_z)
    return Ex, Ez


def analyse(design):
    s1 = CartesianSurface(**design["first"])
    s2 = CartesianSurface(**design["second"])
    t = relay_throughput(s1, s2, design["separation"])
    return s1, s2, t


def main():
    out_dir = example_output_dir(__file__)

    print(f"{'':<28} {'NA obj':>8} {'NA img':>8} {'reducción':>10}  limita")
    print("-" * 78)
    designs = {}
    for key, design in (("dry", DRY), ("immersion", IMMERSION)):
        s1, s2, t = analyse(design)
        designs[key] = (design, s1, s2, t)
        print(
            f"{design['label']:<28} {t['na_object']:>8.4f} {t['na_image']:>8.4f} "
            f"{t['reduction']:>9.2f}x  {t['limited_by']}"
        )
    print()
    for key, (design, s1, s2, t) in designs.items():
        b = t["budget"]
        print(f"{design['label']}:")
        print(f"   caras se cortan en r = {b.intersection_radius:.4f} mm"
              f"{'' if b.surfaces_intersect else '  (fuera del dominio: el elemento no se cierra)'}")
        print(f"   límite de refracción  D1 = {b.first_limit:.4f} mm, D2 = {b.second_limit:.4f} mm")
        print(f"   radio usado           D1 = {t['radius_first']:.4f} mm, D2 = {t['radius_second']:.4f} mm")
    print()

    # -- validate the exit surface of each design against Franz ------- #
    rho = np.linspace(0.0, 3.0, 121) * LAM
    results = {}
    for key, (design, _s1, s2, t) in designs.items():
        aperture = 0.999 * min(t["radius_second"], s2.aperture_limit)
        print(f"referencia exacta para {design['label']} (NA={t['na_image']:.3f}) ...")
        obs = np.stack(
            [rho, np.zeros_like(rho), np.full_like(rho, float(s2.zi))], axis=-1
        )
        _, E_ref = focal_field_reference(
            s2, LAM, lambda r: np.ones_like(r),
            aperture=aperture, observation=obs,
            n_r=2600, n_phi=224, chunk=24_000,
        )
        Ex_model, _ = model_cut(s2, LAM, aperture, rho)
        results[key] = dict(
            label=design["label"], na=t["na_image"],
            ref=E_ref[:, 0], model=Ex_model, aperture=aperture,
        )
        ratio = Ex_model[0] / E_ref[0, 0]
        prof_m = np.abs(Ex_model) / np.abs(Ex_model).max()
        prof_r = np.abs(E_ref[:, 0]) / np.abs(E_ref[:, 0]).max()
        rms = float(np.sqrt(np.mean((prof_m - prof_r) ** 2)))
        print(f"   |modelo/exacto| en el pico = {abs(ratio):.6f}, "
              f"fase = {np.degrees(np.angle(ratio)):+.3f}°, RMS del perfil = {rms:.2e}")
    print()

    # -- figure -------------------------------------------------------- #
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    s = rho / LAM

    ax = axes[0]
    for key, style in (("dry", "--"), ("immersion", "-")):
        R = results[key]
        ax.plot(s, np.abs(R["ref"]) ** 2, "k", ls=style, lw=3.0, alpha=0.3)
        ax.plot(s, np.abs(R["model"]) ** 2, ls=style, lw=1.6,
                label=f"{R['label']}  (NA={R['na']:.2f})")
    ax.set(xlabel=r"$\rho/\lambda$", ylabel=r"$|E_x|^2$ (escala absoluta)",
           title="Punto focal de la superficie de salida\n(gris: campo exacto de Franz)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for key, style in (("dry", "--"), ("immersion", "-")):
        R = results[key]
        prof = np.abs(R["model"]) ** 2
        ax.plot(s, prof / prof.max(), ls=style, lw=1.6,
                label=f"{R['label']}  (NA={R['na']:.2f})")
    ax.set(xlabel=r"$\rho/\lambda$", ylabel="intensidad normalizada",
           title="La inmersión concentra más", yscale="log", ylim=(1e-5, 2.0))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for key, style in (("dry", "--"), ("immersion", "-")):
        R = results[key]
        ax.plot(s, np.abs(R["model"]) / np.abs(R["ref"]), ls=style, lw=1.6,
                label=f"{R['label']}  (NA={R['na']:.2f})")
    ax.axhline(1.0, color="k", ls=":", lw=1.0)
    ax.set(xlabel=r"$\rho/\lambda$", ylabel="modelo / exacto",
           title="Acuerdo con Maxwell, a NA alta", ylim=(0.9, 1.1))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Superficie de salida estigmática a 193 nm: qué apertura alcanza cada "
        "configuración, y si el modelo la describe",
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / "aperture_limits.png"
    fig.savefig(path, dpi=140)
    print_saved(path)
    return path


if __name__ == "__main__":
    main()
