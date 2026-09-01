"""A longitudinal light needle from a single refracting surface.

Annular, radially polarized illumination focused at high aperture produces a
sub-wavelength, many-wavelength-long needle of *longitudinally* polarized
light (Wang et al., Nature Photonics 2, 501 (2008), with an ideal aplanatic
lens).  The Quabis-inversion study (examples/wave_radial_diopter.py) showed
the longitudinal channel wins through a bare refracting surface too; this
script renders what that channel actually builds in space: the needle that a
*single stigmatic diopter* — no lens, no binary optics — writes along its
axis, against the ordinary focus of linear illumination.

Configuration: the far-object high-index oval (n0 = 1 -> ni = 2.4,
z0 = -50, zi = +6, vacuum wavelength = 1), aperture at 97% of the grazing
limit, sin(theta_i) = 0.79 at the rim.

Run from the repository root::

    python examples/wave_light_needle.py

Writes ``examples/output/wave_light_needle.png`` and prints needle metrics.
"""

from pathlib import Path

import numpy as np

from vecdiff import CartesianSurface
import vecdiff.wave as vw

OUTPUT = Path(__file__).resolve().parent / "output"

INK = "#E8EDF2"
MUT = "#9DACBB"
BG = "#0B0F14"
EDGE = "#2A3540"


def polarizer(kind, annulus=None):
    def base(v, phi):
        one = np.ones_like(np.asarray(v, dtype=float))
        if kind == "x":
            return one.astype(complex), 0.0 * one
        if kind == "radial":
            return np.cos(phi).astype(complex), np.sin(phi).astype(complex)
        raise ValueError(kind)

    if annulus is None:
        return base
    lo, hi = annulus

    def masked(v, phi):
        ex, ey = base(v, phi)
        m = ((v >= lo) & (v <= hi)).astype(float)
        return ex * m, ey * m

    return masked


def main():
    oval = CartesianSurface(n0=1.0, ni=2.4, z0=-50.0, zi=6.0)
    a = 0.97 * oval.aperture_limit
    geom = oval.ray_geometry(np.array([a]))
    print(f"aperture {a:.2f} lambda, sin(theta_i) = {geom.sin_ai[0]:.3f}")
    grid = vw.Grid.from_spacing(0.2, 320)

    def spectrum(pol):
        return vw.surface_spectrum(
            vw.oval_surface(oval), grid, n1=1.0, n2=2.4, incident="point",
            source_distance=50.0, polarization=pol, aperture=a,
            m_max=4, n_rho=900, n_phi=48)

    spec_needle = spectrum(polarizer("radial", annulus=(0.9, 1.0)))
    spec_linear = spectrum(polarizer("x"))

    x = np.linspace(-1.6, 1.6, 161)
    z = np.linspace(3.2, 10.0, 180)
    comps_n, _, _ = spec_needle.meridional(x, z)
    comps_l, _, _ = spec_linear.meridional(x, z)
    I_n = np.sum(np.abs(comps_n) ** 2, axis=0)
    Iz_n = np.abs(comps_n[2]) ** 2
    I_l = np.sum(np.abs(comps_l) ** 2, axis=0)

    # ---- metrics -----------------------------------------------------
    i0 = np.argmin(np.abs(x))
    axis_n = I_n[:, i0]
    axis_l = I_l[:, i0]
    on = np.flatnonzero(axis_n >= 0.5 * axis_n.max())
    needle_len = float(z[on[-1]] - z[on[0]])
    on = np.flatnonzero(axis_l >= 0.5 * axis_l.max())
    focus_depth = float(z[on[-1]] - z[on[0]])
    iz = np.argmax(axis_n)
    row = I_n[iz]
    on = np.flatnonzero(row >= 0.5 * row.max())
    needle_w = float(x[on[-1]] - x[on[0]])
    row = I_l[np.argmax(axis_l)]
    on = np.flatnonzero(row >= 0.5 * row.max())
    lin_w = float(x[on[-1]] - x[on[0]])
    ez_share = float(Iz_n.sum() / I_n.sum())
    print(f"needle: length {needle_len:.1f} lambda, width {needle_w:.2f} lambda"
          f" (aspect {needle_len / needle_w:.0f}:1), Ez share {ez_share:.2f}")
    print(f"linear focus: depth {focus_depth:.1f} lambda, width {lin_w:.2f}")

    # ---- figure ------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 5.6), facecolor=BG,
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25],
                                              wspace=0.28, left=0.05,
                                              right=0.98, top=0.82,
                                              bottom=0.12))

    panels = [
        (I_n, r"radial + annulus:  $|E|^2$"),
        (Iz_n, r"radial + annulus:  $|E_z|^2$ (the needle)"),
        (I_l, r"linear x:  $|E|^2$"),
    ]
    for ax, (I, title) in zip(axes[:3], panels):
        ax.imshow(I / I.max(), extent=[x[0], x[-1], z[0], z[-1]],
                  origin="lower", cmap="magma", aspect="auto",
                  interpolation="bilinear")
        ax.axhline(oval.zi, color="#57D0E5", lw=0.7, ls=":", alpha=0.8)
        ax.set_title(title, color=INK, fontsize=10.5, pad=8)
        ax.set_xlabel(r"$x/\lambda$", color=MUT)
        ax.tick_params(colors=MUT)
        for s in ax.spines.values():
            s.set_color(EDGE)
    axes[0].set_ylabel(r"$z/\lambda$", color=MUT)
    axes[1].annotate(f"length {needle_len:.1f}$\\lambda$\n"
                     f"width {needle_w:.2f}$\\lambda$",
                     xy=(0.60, 0.72), xycoords="axes fraction",
                     color=INK, fontsize=9.5)
    axes[2].annotate(f"depth {focus_depth:.1f}$\\lambda$\n"
                     f"width {lin_w:.2f}$\\lambda$",
                     xy=(0.62, 0.68), xycoords="axes fraction",
                     color=INK, fontsize=9.5)

    ax = axes[3]
    ax.set_facecolor(BG)
    ax.plot(z, axis_n / axis_n.max(), color="#57D0E5", lw=1.8,
            label="radial + annulus")
    ax.plot(z, axis_l / axis_l.max(), color="#F0A468", lw=1.8,
            label="linear x")
    ax.axhline(0.5, color=MUT, lw=0.8, ls=":")
    ax.axvline(oval.zi, color=MUT, lw=0.7, ls=":")
    ax.set_xlabel(r"$z/\lambda$", color=MUT)
    ax.set_title("on-axis intensity", color=INK, fontsize=10.5, pad=8)
    leg = ax.legend(fontsize=8.5, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.tick_params(colors=MUT)
    for s in ax.spines.values():
        s.set_color(EDGE)
    ax.grid(alpha=0.15)

    fig.suptitle("The light needle a single refracting surface writes "
                 r"($n_0$=1 $\to$ $n_i$=2.4, $\sin\theta_i$=0.79):  "
                 f"{needle_len / needle_w:.0f}:1 needle vs "
                 f"{focus_depth / lin_w:.0f}:1 focus",
                 color=INK, fontsize=12.5, y=0.95)
    path = OUTPUT / "wave_light_needle.png"
    fig.savefig(path, dpi=150, facecolor=BG)
    print(f"figure: {path}")


if __name__ == "__main__":
    main()
