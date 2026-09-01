"""Does the Quabis tight spot survive a real refracting surface?

Quabis et al. (Opt. Commun. 179, 1 (2000)) showed that at NA = 0.9 an
annular, radially polarized doughnut focused by an *ideal aplanatic lens*
produces a smaller focal spot than linear polarization, because the
longitudinal channel J0 is narrower than any transverse structure.  A single
refracting surface is not an ideal lens: its apodization is the Fresnel
``t_p, t_s`` of the surface itself, its aperture is capped by grazing
incidence, and its focus carries the diffraction focal shift.

vecdiff's notebook study (docs/notebooks/tighter_spot/03) answered this with
the *exact chain's channel weights* at the Debye plane: the inversion
survives the Fresnel apodization, crossing at ``sin(theta_max) ~ 0.78``.
Here the same question is put to the *general spectral operator* — the full
leading-order field, rim included, evaluated at the same geometric (Debye)
focal plane — as an independent confirmation with the machinery that also
handles surfaces the chain cannot.  Measured result: the glass diopter stays
conventional, the high-index far-object diopter inverts, and the crossing
falls at ``sin(theta_i) ~ 0.75``.

The illumination enters through :func:`vecdiff.wave.surface_spectrum`'s
analytic point source, with the polarization callable carrying the radial
unit vector and the annular mask on the surface radius (the object cone of a
far source is far too narrow to resolve on a Cartesian source grid — the
named-source path has no source grid at all).

Two stigmatic ovals (vacuum wavelength = 1 unit):

* the "glass" diopter  n0=1 -> ni=1.5,  z0=-10, zi=+6 — its grazing bound
  keeps ``sin(theta_i)`` below ~0.74: the Quabis regime is unreachable;
* the far-object high-index diopter n0=1 -> ni=2.4, z0=-50, zi=+6 — reaching
  ``sin(theta_i) ~ 0.79`` at 97% of the grazing radius.

Run from the repository root::

    python examples/wave_radial_diopter.py

Writes ``examples/output/wave_radial_diopter.png`` and prints the tables.
"""

from pathlib import Path

import numpy as np

from vecdiff import CartesianSurface
import vecdiff.wave as vw

OUTPUT = Path(__file__).resolve().parent / "output"


def polarizer(kind, annulus=None):
    """Polarization callable ``(v, phi) -> (Ex, Ey)`` with an optional annular
    mask on the normalized surface radius ``v``."""

    def base(v, phi):
        one = np.ones_like(np.asarray(v, dtype=float))
        if kind == "x":
            return one.astype(complex), 0.0 * one
        if kind == "circular":
            return one / np.sqrt(2) + 0j, 1j * one / np.sqrt(2)
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


def diopter_spectrum(oval, grid, pol, aperture):
    """Transmitted spectrum of the oval under an analytic point source."""
    return vw.surface_spectrum(
        vw.oval_surface(oval), grid, n1=float(oval.n0), n2=float(oval.ni),
        incident="point", source_distance=abs(float(oval.z0)),
        polarization=pol, aperture=aperture, m_max=4, n_rho=900, n_phi=48)


def focal_metrics(oval, pol, aperture, grid, half=2.0, npts=401):
    """Half-maximum spot area (lambda^2) and Ez energy share.

    Evaluated at the geometric (Debye) focal plane ``z = zi``, as in Quabis'
    and the notebooks' comparison — an annular pupil forms a long conical
    beam whose raw axial-intensity maximum sits near the surface, which is
    not the focus being compared.
    """
    out = diopter_spectrum(oval, grid, pol, aperture)
    ax = np.linspace(-half, half, npts)
    f = out.field_on(ax, ax, z=float(oval.zi))
    I = f.intensity
    cell = (ax[1] - ax[0]) ** 2
    area = float(np.sum(I >= 0.5 * I.max()) * cell)
    fr = f.component_fractions()
    return {"area": area, "ez": fr["z"], "z": float(oval.zi)}


ILLUMINATIONS = [
    ("linear x", polarizer("x")),
    ("circular", polarizer("circular")),
    ("radial", polarizer("radial")),
    ("radial + annulus", polarizer("radial", annulus=(0.9, 1.0))),
]


def main():
    grid = vw.Grid.from_spacing(0.2, 320)
    results = {}

    for name, ni, z0 in (("glass  (ni=1.5, z0=-10)", 1.5, -10.0),
                         ("far-object high-index (ni=2.4, z0=-50)", 2.4, -50.0)):
        oval = CartesianSurface(n0=1.0, ni=ni, z0=z0, zi=6.0)
        a = 0.97 * oval.aperture_limit
        geom = oval.ray_geometry(np.array([a]))
        sin_i = float(geom.sin_ai[0])
        print(f"\n=== {name}: aperture 0.97 x grazing, sin(theta_i) = {sin_i:.3f} ===")
        print(f"{'illumination':<18} {'area [lam^2]':>12} {'Ez share':>9} "
              f"{'focus z':>8}")
        rows = {}
        for label, pol in ILLUMINATIONS:
            m = focal_metrics(oval, pol, a, grid)
            rows[label] = m
            print(f"{label:<18} {m['area']:>12.3f} {m['ez']:>9.3f} "
                  f"{m['z']:>8.2f}")
        results[name] = (sin_i, rows)
        ratio = rows["radial + annulus"]["area"] / rows["linear x"]["area"]
        print(f"  -> area(radial+annulus) / area(linear) = {ratio:.3f}"
              f"   ({'INVERTED (tight spot wins)' if ratio < 1 else 'conventional order'})")

    # ---- locate the crossing on the high-index oval ------------------
    oval = CartesianSurface(n0=1.0, ni=2.4, z0=-50.0, zi=6.0)
    print("\n=== crossing sweep on the high-index oval ===")
    print(f"{'a/a_gr':>7} {'sin_i':>6} {'A_radial+ann':>13} {'A_linear':>9} "
          f"{'ratio':>7}")
    sweep = []
    for frac in (0.75, 0.85, 0.92, 0.97):
        a = frac * oval.aperture_limit
        geom = oval.ray_geometry(np.array([a]))
        sin_i = float(geom.sin_ai[0])
        m_rad = focal_metrics(oval, polarizer("radial", annulus=(0.9, 1.0)),
                              a, grid)
        m_lin = focal_metrics(oval, polarizer("x"), a, grid)
        sweep.append((sin_i, m_rad["area"], m_lin["area"]))
        print(f"{frac:>7.2f} {sin_i:>6.3f} {m_rad['area']:>13.3f} "
              f"{m_lin['area']:>9.3f} {m_rad['area'] / m_lin['area']:>7.3f}")

    ratios = np.array([r / l for _, r, l in sweep])
    sines = np.array([s for s, _, _ in sweep])
    if np.any(ratios < 1) and np.any(ratios > 1):
        i = int(np.flatnonzero(ratios > 1)[-1])
        cross = sines[i] + (ratios[i] - 1) / (ratios[i] - ratios[i + 1]) * (
            sines[i + 1] - sines[i])
        print(f"  -> inversion crossing at sin(theta_i) ~ {cross:.3f} "
              "(chain weights found ~0.78)")

    # ---- figure ------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    sin_is = [s for s, _, _ in sweep]
    ax.plot(sin_is, [r for _, r, _ in sweep], "o-", label="radial + annulus")
    ax.plot(sin_is, [l for _, _, l in sweep], "s-", label="linear x")
    ax.axvline(0.78, color="k", ls=":", lw=1,
               label="chain-weights crossing (notebooks)")
    ax.set_xlabel(r"$\sin\theta_i$ at the rim")
    ax.set_ylabel(r"half-max spot area  [$\lambda^2$]")
    ax.set_title("The Quabis inversion through a refracting diopter\n"
                 "(full spectral operator, best focus)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2D spots at the largest aperture, radial+annulus vs linear
    a = 0.97 * oval.aperture_limit
    for k, (label, pol) in enumerate([("radial + annulus",
                                       polarizer("radial", annulus=(0.9, 1.0))),
                                      ("linear x", polarizer("x"))]):
        out = diopter_spectrum(oval, grid, pol, a)
        axx = np.linspace(-1.2, 1.2, 241)
        I = out.field_on(axx, axx, z=float(oval.zi)).intensity
        ax = axes[1 + k]
        ax.imshow(I / I.max(), extent=[-1.2, 1.2, -1.2, 1.2], origin="lower",
                  cmap="hot")
        ax.contour(axx, axx, I / I.max(), levels=[0.5], colors="cyan",
                   linewidths=1.0)
        ax.set_title(f"{label}  (half-max contour)")
        ax.set_xlabel(r"$x/\lambda$")
        ax.set_ylabel(r"$y/\lambda$")

    fig.suptitle("Radial polarization focused by a stigmatic diopter "
                 r"($n_0$=1 $\to$ $n_i$=2.4, $z_0$=-50, $z_i$=6)", fontsize=11)
    fig.tight_layout()
    path = OUTPUT / "wave_radial_diopter.png"
    fig.savefig(path, dpi=140)
    print(f"\nfigure: {path}")


if __name__ == "__main__":
    main()
