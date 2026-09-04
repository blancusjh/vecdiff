"""How the general operator's error scales — the domain-of-validity measurement.

The wave engine is a leading-order theory in 1/kR.  This script *measures*
what that means against the Franz/Stratton-Chu integral (an exact Maxwell
field in the image medium) on the stigmatic oval, for both amplitude measures
of the return integral:

* ``flat``  — the bare surface transform (the original construction);
* ``franz`` — the Kirchhoff obliquity pair times the chart radiation factor,
  the stationary-phase content of the exact Franz radiation (the default).

Two sweeps:

* the size parameter at fixed geometry (wavelength 2, 1, 0.5, 0.25 in units
  where the oval is z0 = -30, zi = +20), recording focal-profile RMS error and
  the absolute peak-amplitude ratio;
* the aperture toward the grazing limit at fixed size, recording the same
  against image-side NA.

The headline finding: the error of the *flat* measure is controlled by NA,
not by size — the amplitude ratio is nearly flat in kR and grows from +6.5%
to +14.8% over the aperture sweep — while the *franz* measure removes the NA
dependence almost entirely and stays within about 1% in absolute amplitude.

Run from the repository root::

    python examples/wave_error_scaling.py

Writes ``examples/output/wave_error_scaling.png`` and prints the tables.
"""

from pathlib import Path

import numpy as np

from references.legacy.vecdiff import CartesianSurface
from references.legacy.vecdiff.reference import focal_field_reference
from references.legacy.vecdiff.wave.propagation import raised_cosine
import references.legacy.vecdiff.wave as vw

OUTPUT = Path(__file__).resolve().parent / "output"
EDGE = 0.25


def duel(oval, wavelength, aperture, *, n_r_sc=1200, n_phi_sc=128,
         n_rho=600, half_width=2.5, n_obs=41):
    """Run the exact solver once and the operator under both measures."""

    def pupil(r):
        return raised_cosine(r, aperture * (1 - EDGE), aperture)

    rho = np.linspace(0.0, half_width * wavelength, n_obs)
    obs = np.stack([rho, np.zeros_like(rho), np.full_like(rho, oval.zi)],
                   axis=-1)
    _, E_sc = focal_field_reference(oval, wavelength, pupil,
                                    aperture=aperture, observation=obs,
                                    n_r=n_r_sc, n_phi=n_phi_sc)
    Ex_sc, Ez_sc = E_sc[:, 0], E_sc[:, 2]

    n_grid = int(np.clip(2 ** np.ceil(np.log2(9 * aperture / wavelength)),
                         128, 1024))
    grid = vw.Grid.from_spacing(0.25 * wavelength, n_grid)
    src = vw.object_spectrum(oval, grid, wavelength=wavelength)

    pn = lambda v: np.abs(v) / np.abs(v).max()
    out = {"rho": rho, "Ex_sc": Ex_sc}
    for measure in ("flat", "franz"):
        op = vw.stigmatic_operator(oval, wavelength=wavelength,
                                   aperture=aperture, edge_softness=EDGE,
                                   n_rho=n_rho, n_phi=32, m_max=2,
                                   measure=measure)
        fld = op(src).field_on(rho, np.array([0.0]), z=float(oval.zi))
        Ex_w, Ez_w = fld.Ex[0], fld.components[2][0]
        r_sc = np.abs(Ez_sc).max() / np.abs(Ex_sc).max()
        r_w = np.abs(Ez_w).max() / np.abs(Ex_w).max()
        out[measure] = {
            "profile_rms": float(np.sqrt(np.mean((pn(Ex_sc) - pn(Ex_w)) ** 2))),
            "amp_ratio": float(np.abs(Ex_w).max() / np.abs(Ex_sc).max()),
            "ez_err": float(abs(r_w - r_sc) / r_sc),
            "Ex_w": Ex_w,
        }
    return out


def main():
    oval = CartesianSurface(n0=1.0, ni=1.5, z0=-30.0, zi=20.0)
    a0 = 0.85 * oval.aperture_limit

    print("=== size sweep (aperture fixed at 0.85 x grazing, NA_i = 0.64) ===")
    print(f"{'lambda':>7} {'kR_ap':>8} | {'rms flat':>9} {'rms franz':>10} | "
          f"{'amp flat':>9} {'amp franz':>10}")
    size_rows = []
    for lam in (2.0, 1.0, 0.5, 0.25):
        k_i = 2 * np.pi * oval.ni / lam
        m = duel(oval, lam, a0,
                 n_r_sc=int(900 / lam) + 300, n_rho=int(500 / lam) + 200)
        size_rows.append((lam, k_i * a0, m))
        print(f"{lam:>7.2f} {k_i * a0:>8.1f} | {m['flat']['profile_rms']:>9.4f} "
              f"{m['franz']['profile_rms']:>10.4f} | "
              f"{m['flat']['amp_ratio']:>9.4f} {m['franz']['amp_ratio']:>10.4f}")

    print("\n=== aperture sweep toward grazing (lambda = 1) ===")
    print(f"{'a/a_gr':>7} {'NA_i':>6} | {'rms flat':>9} {'rms franz':>10} | "
          f"{'amp flat':>9} {'amp franz':>10}")
    na_rows = []
    for frac in (0.5, 0.7, 0.85, 0.95, 0.99):
        a = frac * oval.aperture_limit
        geom = oval.ray_geometry(np.array([a]))
        na_i = float(oval.ni * geom.sin_ai[0])
        m = duel(oval, 1.0, a)
        na_rows.append((frac, na_i, m))
        print(f"{frac:>7.2f} {na_i:>6.3f} | {m['flat']['profile_rms']:>9.4f} "
              f"{m['franz']['profile_rms']:>10.4f} | "
              f"{m['flat']['amp_ratio']:>9.4f} {m['franz']['amp_ratio']:>10.4f}")

    # ---- figure ------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    kr = [row[1] for row in size_rows]
    ax = axes[0]
    for msr, style in (("flat", "o--"), ("franz", "o-")):
        ax.semilogx(kr, [abs(row[2][msr]["amp_ratio"] - 1) for row in size_rows],
                    style, label=f"|amp ratio - 1|, {msr}")
        ax.semilogx(kr, [row[2][msr]["profile_rms"] for row in size_rows],
                    style.replace("o", "s"), alpha=0.6,
                    label=f"profile RMS, {msr}")
    ax.set_xlabel(r"size parameter $k_i a$")
    ax.set_ylabel("error")
    ax.set_title("Error vs size (fixed NA):\nflat in $kR$ — the error lives in NA")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    na = [row[1] for row in na_rows]
    for msr, style in (("flat", "o--"), ("franz", "o-")):
        ax.plot(na, [row[2][msr]["amp_ratio"] for row in na_rows], style,
                label=f"amplitude ratio, {msr}")
    ax.axhline(1.0, color="k", lw=1)
    ax.set_xlabel("image-side NA")
    ax.set_ylabel("peak amplitude / exact")
    ax.set_title("Amplitude vs aperture:\nthe franz measure removes the NA trend")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    m = na_rows[-1][2]
    s = m["rho"]
    ax.plot(s, (np.abs(m["Ex_sc"]) / np.abs(m["Ex_sc"]).max()) ** 2, "k",
            lw=2.5, alpha=0.5, label="Stratton-Chu (exact)")
    for msr, style in (("flat", "--"), ("franz", "-")):
        v = np.abs(m[msr]["Ex_w"]) / np.abs(m[msr]["Ex_w"]).max()
        ax.plot(s, v ** 2, style, lw=1.3, label=f"operator, {msr}")
    ax.set_xlabel(r"$\rho/\lambda$")
    ax.set_ylabel("normalized $|E_x|^2$")
    ax.set_title(f"Focal profile at NA = {na_rows[-1][1]:.2f}")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 2)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Leading-order error of the spectral interface operator, "
                 "measured against the exact Maxwell reference", fontsize=11)
    fig.tight_layout()
    path = OUTPUT / "wave_error_scaling.png"
    fig.savefig(path, dpi=140)
    print(f"\nfigure: {path}")
    return size_rows, na_rows


if __name__ == "__main__":
    main()
