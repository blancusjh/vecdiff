"""Spin-orbit coupling at a curved interface, made visible.

A circularly polarized vortex carries spin sigma = +/-1 and orbital charge l.
The tangent-plane operator conserves m - s, so each Cartesian focal component
inherits its own topological charge — in particular the longitudinal
component carries l + sigma.  Focusing an l = 1 vortex through the
high-index stigmatic diopter then behaves completely differently for the two
handednesses:

* **antiparallel** (sigma = -1): E_z carries charge 0 — the interface pours
  light *into* the vortex core, and the "dark" beam focuses to a bright
  longitudinal spot;
* **parallel** (sigma = +1): E_z carries charge 2 — every component stays
  charged and the core stays dark.

This spin-controlled bright/dark switch (well known for ideal aplanatic
lenses, e.g. Zhan, Adv. Opt. Photon. 1, 1 (2009)) here appears from the bare
Fresnel physics of a single refracting surface, with the phase maps showing
the charge bookkeeping explicitly.

Configuration: n0 = 1 -> ni = 2.4, z0 = -50, zi = +6, aperture 0.97 x
grazing (sin(theta_i) = 0.79), evaluated at the Debye plane.

Run from the repository root::

    python examples/wave_vortex_diopter.py

Writes ``examples/output/wave_vortex_diopter.png``.
"""

from pathlib import Path

import numpy as np

from references.legacy.vecdiff import CartesianSurface
import references.legacy.vecdiff.wave as vw

OUTPUT = Path(__file__).resolve().parent / "output"

INK = "#E8EDF2"
MUT = "#9DACBB"
BG = "#0B0F14"
EDGE = "#2A3540"

L_CHARGE = 1


def vortex_circular(sigma):
    """l = 1 vortex on a circular basis of handedness ``sigma``."""

    def pol(v, phi):
        ramp = np.exp(1j * L_CHARGE * phi) / np.sqrt(2.0)
        return ramp, 1j * sigma * ramp

    return pol


def main():
    oval = CartesianSurface(n0=1.0, ni=2.4, z0=-50.0, zi=6.0)
    a = 0.97 * oval.aperture_limit
    grid = vw.Grid.from_spacing(0.2, 320)
    ax1d = np.linspace(-1.6, 1.6, 321)

    cases = []
    for sigma, name in ((-1, "antiparallel  ($\\ell$=1, $\\sigma$=$-$1)"),
                        (+1, "parallel  ($\\ell$=1, $\\sigma$=$+$1)")):
        spec = vw.surface_spectrum(
            vw.oval_surface(oval), grid, n1=1.0, n2=2.4, incident="point",
            source_distance=50.0, polarization=vortex_circular(sigma),
            aperture=a, m_max=4, n_rho=900, n_phi=64)
        f = spec.field_on(ax1d, ax1d, z=oval.zi)
        E = np.stack([f.Ex, f.Ey, f.components[2]])
        I = np.sum(np.abs(E) ** 2, axis=0)
        c = I.shape[0] // 2
        print(f"sigma={sigma:+d}: center/total peak = {I[c, c] / I.max():.3f}, "
              f"Ez share = {float(np.sum(np.abs(E[2])**2) / np.sum(I)):.3f}")
        cases.append((name, sigma, I, E))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.6), facecolor=BG,
                             gridspec_kw=dict(wspace=0.24, hspace=0.30,
                                              left=0.07, right=0.97,
                                              top=0.86, bottom=0.07))
    ext = [ax1d[0], ax1d[-1], ax1d[0], ax1d[-1]]

    for row, (name, sigma, I, E) in enumerate(cases):
        Iz = np.abs(E[2]) ** 2
        ph = np.angle(E[2])
        # blank the phase where there is no field to carry it
        ph = np.where(Iz > 0.02 * Iz.max(), ph, np.nan)

        ax = axes[row, 0]
        ax.imshow(I / I.max(), extent=ext, origin="lower", cmap="magma",
                  interpolation="bilinear")
        ax.set_title(("total $|E|^2$" if row == 0 else "total $|E|^2$"),
                     color=INK, fontsize=10.5)
        ax.set_ylabel(name + "\n\n" + r"$y/\lambda$", color=INK, fontsize=10)

        ax = axes[row, 1]
        ax.imshow(Iz / Iz.max(), extent=ext, origin="lower", cmap="magma",
                  interpolation="bilinear")
        ax.set_title(r"$|E_z|^2$   (charge $\ell+\sigma$ = "
                     f"{L_CHARGE + sigma})", color=INK, fontsize=10.5)

        ax = axes[row, 2]
        ax.set_facecolor("#111111")
        ax.imshow(ph, extent=ext, origin="lower", cmap="twilight",
                  interpolation="nearest", vmin=-np.pi, vmax=np.pi)
        ax.set_title(r"arg $E_z$", color=INK, fontsize=10.5)

        for ax in axes[row]:
            ax.tick_params(colors=MUT, labelsize=8.5)
            for s in ax.spines.values():
                s.set_color(EDGE)
            ax.set_xlabel(r"$x/\lambda$", color=MUT, fontsize=9)

    fig.suptitle("Spin-orbit switch at a refracting surface: the vortex core "
                 "fills only when spin opposes charge\n"
                 r"($\ell$=1 vortex, circular basis, through $n_0$=1 $\to$ "
                 r"$n_i$=2.4 at $\sin\theta_i$=0.79 — conservation of $m-s$ "
                 "written in the phase maps)",
                 color=INK, fontsize=12, y=0.965)
    path = OUTPUT / "wave_vortex_diopter.png"
    fig.savefig(path, dpi=150, facecolor=BG)
    print(f"figure: {path}")


if __name__ == "__main__":
    main()
