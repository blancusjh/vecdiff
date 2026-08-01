"""Arbitrate the diopter transfer conventions against an exact Maxwell field.

The ray-optical chain ``G -> G' -> focal plane`` has three places where a
convention has to be chosen, and choosing wrong is invisible on axis:

* whether the geometric amplitude factor ``A(Q) = |z0| li / (|zi| l0)`` belongs
  in the transfer operator at all;
* whether the radial eigenvalue carries the meridional projection
  ``cos(alpha_i) / cos(alpha_0)``;
* which pupil mapping the focal-plane transform integrates over -- the sine
  mapping ``u = zi sin(alpha_i)`` of the Debye reduction, or the perspective
  mapping ``u = zi tan(alpha_i)`` of Sec. 12 of the note.

Rather than argue, this compares every variant against the Franz /
Stratton-Chu integral, which is an exact Maxwell field in the image medium.

Run from the repository root::

    uv run python examples/reference_vs_model.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.special import jv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vecdiff import CartesianSurface  # noqa: E402
from vecdiff.pupil_mapping import debye_prefactor  # noqa: E402
from vecdiff.reference import focal_field_reference  # noqa: E402
from vecdiff.transfer import focal_channel_weights  # noqa: E402

π = np.pi

OUTPUT = Path(__file__).resolve().parent / "output"


# ------------------------------------------------------------------ #
#  Model variants, driven through the package itself                   #
# ------------------------------------------------------------------ #

def model_focal_cut(surface, wavelength, pupil, variant, rho, *, aperture, n_r=6001):
    """Focal-plane cut along x predicted by one model variant.

    Goes through :func:`vecdiff.transfer.focal_channel_weights`, so what is
    plotted is the package's own weighting rather than a parallel
    reimplementation.  The absolute Debye prefactor is applied, which puts the
    comparison on amplitude and phase, not just shape.
    """
    r = np.linspace(0.0, aperture, int(n_r))
    w_p, w_s, w_z, u = focal_channel_weights(
        surface, r, mapping=variant["mapping"]
    )

    f = np.asarray(pupil(r), dtype=complex)
    lam_plus = 0.5 * (w_p + w_s)
    lam_minus = 0.5 * (w_p - w_s)

    k_i = 2.0 * π * surface.ni / wavelength
    zi = abs(surface.zi)
    q = k_i * np.asarray(rho, dtype=float) / zi

    def hankel(order, g):
        integrand = g[None, :] * jv(order, q[:, None] * u[None, :]) * u[None, :]
        return simpson(integrand, x=u, axis=1)

    h0 = hankel(0, lam_plus * f)
    h2 = hankel(2, lam_minus * f)
    h1 = hankel(1, w_z * f)

    prefactor = debye_prefactor(surface, wavelength)
    Ex = prefactor * (h0 - h2)          # phi_P = 0 on the x cut
    Ez = 1j * prefactor * h1
    return Ex, Ez


VARIANTS = {
    "mapeo seno (Debye)": dict(mapping="sine"),
    "mapeo tangente (perspectivo)": dict(mapping="tangent"),
}


# ------------------------------------------------------------------ #
#  Driver                                                              #
# ------------------------------------------------------------------ #

def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n0, ni, z0, zi = 1.0, 1.5, -10.0, 6.0
    lam = 532e-6  # mm
    surface = CartesianSurface(n0=n0, ni=ni, z0=z0, zi=zi)
    a = 0.999 * surface.aperture_limit

    geom_edge = surface.ray_geometry(np.array([a]))
    print(f"system: n0={n0}, ni={ni}, z0={z0}, zi={zi}, lambda={lam*1e6:.0f} nm")
    print(f"aperture limit (grazing) r = {surface.aperture_limit:.6f} mm")
    print(f"image-side NA at the edge  = {ni*geom_edge.sin_ai[0]:.4f}")
    print(f"A at the edge              = {geom_edge.A[0]:.5f}")
    print(f"cos(ai)/cos(a0) at the edge= {(geom_edge.cos_ai/geom_edge.cos_a0)[0]:.5f}")
    print()

    def pupil(r):
        return np.ones_like(r)

    half_width = 3.0
    n_obs = 121
    rho = np.linspace(0.0, half_width, n_obs) * lam

    print("running the Franz/Stratton-Chu reference ...")
    obs = np.stack([rho, np.zeros_like(rho), np.full_like(rho, zi)], axis=-1)

    def progress(done, total):
        print(f"   sources {done}/{total}", end="\r", flush=True)

    _, E_ref = focal_field_reference(
        surface,
        lam,
        pupil,
        aperture=a,
        observation=obs,
        n_r=3000,
        n_phi=256,
        chunk=24_000,
        progress=progress,
    )
    print(" " * 40, end="\r")
    Ex_ref, Ez_ref = E_ref[:, 0], E_ref[:, 2]

    results = {}
    for name, variant in VARIANTS.items():
        results[name] = model_focal_cut(surface, lam, pupil, variant, rho, aperture=a)

    # -- report -------------------------------------------------- #
    print(f"{'variante':<28} {'|Ex(0)|/ref':>12} {'fase(0)':>10} {'RMS perfil':>12}")
    print("-" * 66)
    ref_peak = np.abs(Ex_ref).max()
    ref_prof = np.abs(Ex_ref) / ref_peak
    for name, (Ex, _Ez) in results.items():
        ratio = Ex[0] / Ex_ref[0]
        prof = np.abs(Ex) / np.abs(Ex).max()
        rms = float(np.sqrt(np.mean((prof - ref_prof) ** 2)))
        print(f"{name:<28} {abs(ratio):>12.5f} {np.degrees(np.angle(ratio)):>9.2f}d {rms:>12.2e}")
    print()

    # The longitudinal channel is an independent check: it comes from a
    # different Hankel order and never enters the transverse comparison.
    peak_z = np.abs(Ez_ref).max()
    print(f"{'variante':<28} {'max|Ez| / ref':>14}")
    print("-" * 44)
    for name, (_Ex, Ez) in results.items():
        print(f"{name:<28} {np.abs(Ez).max() / peak_z:>14.5f}")
    print()

    # -- figure ---------------------------------------------------- #
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(21.0, 5.0))
    s = rho / lam

    ax = axes[0]
    ax.plot(s, np.abs(Ex_ref) ** 2, "k", lw=3.0, alpha=0.35, label="referencia (Franz)")
    for name, (Ex, _) in results.items():
        ax.plot(s, np.abs(Ex) ** 2, lw=1.4, label=name)
    ax.set_xlabel(r"$\rho/\lambda$")
    ax.set_ylabel(r"$|E_x|^2$  (escala absoluta)")
    ax.set_title("Perfil focal, amplitud absoluta")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(s, ref_prof**2, "k", lw=3.0, alpha=0.35, label="referencia (Franz)")
    for name, (Ex, _) in results.items():
        ax.plot(s, (np.abs(Ex) / np.abs(Ex).max()) ** 2, lw=1.4, label=name)
    ax.set_xlabel(r"$\rho/\lambda$")
    ax.set_ylabel("intensidad normalizada")
    ax.set_title("Forma del punto focal (normalizada)")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 2.0)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for name, (Ex, _) in results.items():
        ax.plot(s, np.abs(Ex) / np.abs(Ex_ref), lw=1.4, label=name)
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.set_xlabel(r"$\rho/\lambda$")
    ax.set_ylabel(r"$|E_x^{\rm modelo}| \; / \; |E_x^{\rm ref}|$")
    ax.set_title("Cociente contra la referencia exacta")
    ax.set_ylim(0.0, 2.0)
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(s, np.abs(Ez_ref) ** 2, "k", lw=3.0, alpha=0.35, label="referencia (Franz)")
    for name, (_, Ez) in results.items():
        ax.plot(s, np.abs(Ez) ** 2, lw=1.4, label=name)
    ax.set_xlabel(r"$\rho/\lambda$")
    ax.set_ylabel(r"$|E_z|^2$  (escala absoluta)")
    ax.set_title("Canal longitudinal (comprobación independiente)")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Dioptrio estigmático  n0={n0}, ni={ni}, z0={z0}, zi={zi} mm, "
        rf"$\lambda$={lam*1e6:.0f} nm,  NA$_i$={ni*geom_edge.sin_ai[0]:.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    path = OUTPUT / "reference_vs_model.png"
    fig.savefig(path, dpi=140)
    print(f"figura: {path}")
    return path


if __name__ == "__main__":
    main()
