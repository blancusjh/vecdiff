"""The raindrop word: a photonic nanojet from the closed-body composition.

This script studies the once-transmitted word of a dielectric ball lens,

    T_back . P(2R) . T_front,

and renders a *photonic-nanojet-like* concentration.  The first interface is
driven by one plane wave and uses the spectrally linear path.  The field at the
second interface is dense, so that step explicitly uses the one-local-ray
approximation.  This is therefore a model study, not a rigorous full-wave
solution of the closed sphere.  The phenomenon being modelled is the narrow,
wavelength-scale jet outside a dielectric sphere (Chen, Taflove & Backman,
Opt. Express 12, 1214 (2004)).

Ball: R = 8 lambda, n = 1.5, aperture 6 lambda (the rim beyond that is
clipped by total internal reflection at the exit face anyway).

Run from the repository root::

    python examples/wave_nanojet.py

Writes ``examples/output/wave_nanojet.png`` and prints the jet metrics.
"""

from pathlib import Path

import numpy as np

import vecdiff.wave as vw

OUTPUT = Path(__file__).resolve().parent / "output"

R, N_GLASS, APERTURE = 8.0, 1.5, 6.0

INK = "#E8EDF2"
MUT = "#9DACBB"
BG = "#0B0F14"


def compute():
    grid = vw.Grid.from_spacing(0.25, 256)
    pw = vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.0, polarization="x")
    front = vw.InterfaceOperator(vw.Sphere(radius=+R), n1=1.0, n2=N_GLASS,
                                 aperture=APERTURE, n_rho=600, n_phi=32,
                                 m_max=2)
    back = vw.InterfaceOperator(
        vw.Sphere(radius=-R), n1=N_GLASS, n2=1.0,
        aperture=APERTURE, n_rho=600, n_phi=32, m_max=2,
        # The internal field has a dense spectrum.  This fast second-surface
        # step is the explicitly selected single-ray/geometrical approximation.
        incidence_model="local_ray",
    )
    inside = vw.FreeSpace(2 * R)(front(pw))     # internal spectrum at z = 0
    out = back(inside)                          # external spectrum at z = 2R
    return inside, out


def intensity_map(spec, x, zs):
    comps, _, _ = spec.meridional(x, zs)
    return np.sum(np.abs(comps) ** 2, axis=0)


def main():
    inside, out = compute()

    # ---- metrics -----------------------------------------------------
    z_ext = np.linspace(0.1, 10.0, 100)
    I_axis = out.focus_scan(z_ext)
    zj = float(z_ext[np.argmax(I_axis)])
    xw = np.linspace(-2.0, 2.0, 321)
    fw = out.field_on(xw, np.array([0.0]), z=zj)
    Iw = np.sum(np.abs(np.stack([fw.Ex[0], fw.Ey[0], fw.components[2][0]])) ** 2,
                axis=0)
    on = np.flatnonzero(Iw >= 0.5 * Iw.max())
    waist = float(xw[on[-1]] - xw[on[0]])
    on = np.flatnonzero(I_axis >= 0.5 * I_axis.max())
    length = float(z_ext[on[-1]] - z_ext[on[0]])
    print(f"jet peak {zj:.2f} lambda beyond the shadow surface "
          f"(paraxial ball-lens focus {0.5 * R:.1f})")
    print(f"waist FWHM {waist:.2f} lambda; axial FWHM {length:.2f} lambda")

    # ---- meridional composite ---------------------------------------
    x = np.linspace(-8.5, 8.5, 341)
    z_in = np.linspace(0.0, 2 * R, 170)
    z_out = np.linspace(0.0, 10.0, 100)
    I_in = intensity_map(inside, x, z_in)
    I_out = intensity_map(out, x, z_out)

    XX, ZZ = np.meshgrid(x, z_in)
    ball = (XX ** 2 + (ZZ - R) ** 2) <= R ** 2
    I_in = np.where(ball, I_in, np.nan)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.8, 9.2), facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1],
                          wspace=0.22, hspace=0.34, left=0.07, right=0.97,
                          top=0.87, bottom=0.08)

    vmax = max(np.nanmax(I_in), I_out.max())
    floor = plt.get_cmap("magma")(0.0)
    ax = fig.add_subplot(gs[:, 0], facecolor=floor)
    ax.imshow(I_in, extent=[x[0], x[-1], z_in[0], z_in[-1]], origin="lower",
              cmap="magma", vmin=0, vmax=vmax, interpolation="bilinear")
    ax.imshow(I_out, extent=[x[0], x[-1], 2 * R, 2 * R + z_out[-1]],
              origin="lower", cmap="magma", vmin=0, vmax=vmax,
              interpolation="bilinear")
    ax.set_aspect("equal")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.5, 2 * R + z_out[-1])
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R * np.cos(th), R + R * np.sin(th), color="#57D0E5", lw=1.0,
            alpha=0.85)
    ax.annotate("plane wave", xy=(-8.0, -0.1), color=MUT, fontsize=9)
    ax.annotate(f"jet: waist {waist:.2f}$\\lambda$\nlength {length:.1f}$\\lambda$",
                xy=(0.9, 2 * R + zj), color=INK, fontsize=9.5,
                xytext=(3.6, 2 * R + zj + 1.6),
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax.set_xlabel(r"$x/\lambda$", color=MUT)
    ax.set_ylabel(r"$z/\lambda$", color=MUT)
    ax.set_title(r"$|E|^2$ through the ball and beyond"
                 "  —  the word $T_{back}\\,P(2R)\\,T_{front}$",
                 color=INK, fontsize=11, pad=10)
    ax.tick_params(colors=MUT)
    for s in ax.spines.values():
        s.set_color("#2A3540")

    ax = fig.add_subplot(gs[0, 1], facecolor=BG)
    ax.plot(z_ext + 2 * R, I_axis / I_axis.max(), color="#F0A468", lw=1.8)
    ax.axhline(0.5, color=MUT, lw=0.8, ls=":")
    ax.set_xlabel(r"$z/\lambda$", color=MUT)
    ax.set_title("on-axis intensity", color=INK, fontsize=10)
    ax.tick_params(colors=MUT)
    for s in ax.spines.values():
        s.set_color("#2A3540")
    ax.grid(alpha=0.15)

    ax = fig.add_subplot(gs[1, 1], facecolor=BG)
    ax.plot(xw, Iw / Iw.max(), color="#57D0E5", lw=1.8)
    ax.axhline(0.5, color=MUT, lw=0.8, ls=":")
    ax.set_xlabel(r"$x/\lambda$", color=MUT)
    ax.set_title(f"waist profile at the peak  (FWHM {waist:.2f}$\\lambda$)",
                 color=INK, fontsize=10)
    ax.tick_params(colors=MUT)
    for s in ax.spines.values():
        s.set_color("#2A3540")
    ax.grid(alpha=0.15)

    fig.suptitle("Local-ray model of a dielectric-ball nanojet "
                 f"(R = {R:.0f}$\\lambda$, n = {N_GLASS})  —  "
                 "spectral first surface, local-ray second surface",
                 color=INK, fontsize=13, y=0.97)
    path = OUTPUT / "wave_nanojet.png"
    fig.savefig(path, dpi=150, facecolor=BG)
    print(f"figure: {path}")


if __name__ == "__main__":
    main()
