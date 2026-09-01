"""A freeform astigmat, resolved by the general NUFFT surface transform.

Nothing in the operator is tied to symmetry: a genuinely non-axisymmetric
surface uses the same local Fresnel model through the type-3 NUFFT of the
surface currents.  This script focuses a plane wave through a toroidal
freeform with two different curvatures,

    sag(x, y) = -( x^2 / 2Rx + y^2 / 2Ry ),   Rx = 8, Ry = 5  (lambda),

from glass (n = 1.5) into air, and renders the textbook anatomy of
astigmatism as full vector fields: the tangential line focus, the circle of
least confusion, and the sagittal line focus — each at the position
elementary theory predicts (paraxially |R|/(n1 - n2): z = 10 and 16).

Run from the repository root::

    python examples/wave_freeform_astigmat.py

Writes ``examples/output/wave_freeform_astigmat.png``.
"""

from pathlib import Path

import numpy as np

import vecdiff.wave as vw

OUTPUT = Path(__file__).resolve().parent / "output"

INK = "#E8EDF2"
MUT = "#9DACBB"
BG = "#0B0F14"
EDGE = "#2A3540"

RX, RY = 8.0, 5.0
N1, N2 = 1.5, 1.0
Z_FX = RX / (N1 - N2)        # sagittal focus (x-fan), paraxial
Z_FY = RY / (N1 - N2)        # tangential focus (y-fan), paraxial


def main():
    surf = vw.Freeform2D(
        sag_fn=lambda x, y: -(x ** 2 / (2 * RX) + y ** 2 / (2 * RY)),
        radius=6.0)
    grid = vw.Grid.from_spacing(0.25, 256)
    pw = vw.plane_wave_spectrum(grid, wavelength=1.0, n=N1, polarization="x")
    out = vw.InterfaceOperator(surf, n1=N1, n2=N2, aperture=5.5,
                               n_free=260)(pw)

    # ---- the two meridional maps ------------------------------------
    t = np.linspace(-2.2, 2.2, 121)
    zs = np.linspace(4.0, 18.0, 170)
    I_xz, _, _ = out.meridional(t, zs)          # (x, z) plane at y = 0
    I_xz = np.sum(np.abs(I_xz) ** 2, axis=0)
    I_yz = np.empty_like(I_xz)
    for i, zz in enumerate(zs):                 # (y, z) plane at x = 0
        f = out.field_on(np.array([0.0]), t, z=zz)
        I_yz[i] = np.sum(np.abs(np.stack(
            [f.Ex[:, 0], f.Ey[:, 0], f.components[2][:, 0]])) ** 2, axis=0)

    # measured line-focus positions: minimum second-moment width along each
    # direction (robust where a half-max criterion is not)
    def width_profile(I):
        w = []
        for row in I:
            pdf = row / row.sum()
            m = float((t * pdf).sum())
            w.append(2.355 * np.sqrt(float(((t - m) ** 2 * pdf).sum())))
        return np.array(w)

    z_tan = float(zs[np.argmin(width_profile(I_yz))])
    z_sag = float(zs[np.argmin(width_profile(I_xz))])
    print(f"tangential line focus: z = {z_tan:.1f} (paraxial {Z_FY:.1f})")
    print(f"sagittal   line focus: z = {z_sag:.1f} (paraxial {Z_FX:.1f})")
    print(f"separation {z_sag - z_tan:.1f} vs paraxial {Z_FX - Z_FY:.1f}; "
          f"common focal shift ~{Z_FY - z_tan:.1f}")

    # ---- three transverse slices ------------------------------------
    z_mid = 0.5 * (z_tan + z_sag)
    slices = []
    for zz, label in ((z_tan, "tangential line"),
                      (z_mid, "least confusion"),
                      (z_sag, "sagittal line")):
        f = out.field_on(t, t, z=zz)
        I = np.sum(np.abs(np.stack(
            [f.Ex, f.Ey, f.components[2]])) ** 2, axis=0)
        slices.append((zz, label, I))

    # ---- figure ------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.2, 7.6), facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.34,
                          wspace=0.24, left=0.06, right=0.97, top=0.86,
                          bottom=0.08)

    for k, (I, xlabel, zline) in enumerate(
            [(I_xz, r"$x/\lambda$  (y = 0)", z_sag),
             (I_yz, r"$y/\lambda$  (x = 0)", z_tan)]):
        ax = fig.add_subplot(gs[0, k], facecolor=BG)
        ax.imshow(I / I.max(), extent=[t[0], t[-1], zs[0], zs[-1]],
                  origin="lower", cmap="magma", aspect="auto",
                  interpolation="bilinear")
        ax.axhline(zline, color="#57D0E5", lw=0.8, ls=":")
        ax.set_xlabel(xlabel, color=MUT)
        ax.set_ylabel(r"$z/\lambda$", color=MUT)
        ax.set_title("meridional " + (r"$x$–$z$" if k == 0 else r"$y$–$z$"),
                     color=INK, fontsize=10.5)
        ax.tick_params(colors=MUT)
        for s in ax.spines.values():
            s.set_color(EDGE)

    ax = fig.add_subplot(gs[0, 2], facecolor=BG)
    ax.plot(zs, width_profile(I_xz), color="#F0A468", lw=1.8,
            label="width in x")
    ax.plot(zs, width_profile(I_yz), color="#57D0E5", lw=1.8,
            label="width in y")
    for zline, c in ((z_sag, "#F0A468"), (z_tan, "#57D0E5")):
        ax.axvline(zline, color=c, lw=0.8, ls=":")
    ax.set_xlabel(r"$z/\lambda$", color=MUT)
    ax.set_ylabel(r"FWHM  [$\lambda$]", color=MUT)
    ax.set_title("the two line foci", color=INK, fontsize=10.5)
    leg = ax.legend(fontsize=8.5, frameon=False)
    for txt in leg.get_texts():
        txt.set_color(INK)
    ax.tick_params(colors=MUT)
    for s in ax.spines.values():
        s.set_color(EDGE)
    ax.grid(alpha=0.15)

    for k, (zz, label, I) in enumerate(slices):
        ax = fig.add_subplot(gs[1, k], facecolor=BG)
        ax.imshow(I / I.max(), extent=[t[0], t[-1], t[0], t[-1]],
                  origin="lower", cmap="magma", interpolation="bilinear")
        ax.set_title(f"{label}   z = {zz:.1f}$\\lambda$", color=INK,
                     fontsize=10)
        prow = I[I.shape[0] // 2] / I[I.shape[0] // 2].sum()
        mx = float((t * prow).sum())
        wx = 2.355 * np.sqrt(float(((t - mx) ** 2 * prow).sum()))
        pcol = I[:, I.shape[1] // 2] / I[:, I.shape[1] // 2].sum()
        my = float((t * pcol).sum())
        wy = 2.355 * np.sqrt(float(((t - my) ** 2 * pcol).sum()))
        ax.annotate(f"w$_x$ {wx:.2f}$\\lambda$\nw$_y$ {wy:.2f}$\\lambda$",
                    xy=(0.03, 0.80), xycoords="axes fraction", color=INK,
                    fontsize=8.5)
        ax.set_xlabel(r"$x/\lambda$", color=MUT, fontsize=9)
        if k == 0:
            ax.set_ylabel(r"$y/\lambda$", color=MUT, fontsize=9)
        ax.tick_params(colors=MUT, labelsize=8.5)
        for s in ax.spines.values():
            s.set_color(EDGE)

    fig.suptitle("Astigmatism of a freeform surface, by the general NUFFT "
                 "surface transform\n"
                 rf"sag = $-(x^2/2R_x + y^2/2R_y)$, $R_x$={RX:.0f}, "
                 rf"$R_y$={RY:.0f}, glass $\to$ air:  line foci separated "
                 rf"{z_sag - z_tan:.1f}$\lambda$ (prediction "
                 rf"{Z_FX - Z_FY:.1f}$\lambda$)",
                 color=INK, fontsize=12.5, y=0.97)
    path = OUTPUT / "wave_freeform_astigmat.png"
    fig.savefig(path, dpi=150, facecolor=BG)
    print(f"figure: {path}")


if __name__ == "__main__":
    main()
