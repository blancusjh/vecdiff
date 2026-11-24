"""The double-Brewster axicon: a radially polarized longitudinal needle from one element.

A ring beam, circularly polarized with its vortex removed,
E = exp(-((rho - rho0)/sigma)²) (e_rho + i e_phi), meets the cone tip first.
Both faces are met at Brewster incidence and their rotations add, so the
element sends a converging conical wave of half-angle 2 delta with
T_p = 1 at both faces, while E_phi / E_rho drops by cos(delta)² per element.
The ring lies between the radii whose rays leave the exit face before
crossing the axis and where the two faces would meet, so the tip itself is
never lit.

Cases (633 nm, depth 300 µm on the axis):
  sqrt3        n = sqrt(3): flat exit face, NA 0.866
  n2           n = 2: exit cone of slope 0.182, NA 0.96
  n2+shells    four nested n = 2 Brewster spike shells before the n = 2 axicon
For each, the same geometry without diattenuation (t_s replaced by t_p) is
run as the reference, and an ideal radially polarized ring as the limit.

Run as a script it prints the table and writes the figures and results.json
to the directory given as argument (default: figures/ next to this file).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import devices as dv  # noqa: E402
import meridional_operator as mo  # noqa: E402

WAVELENGTH = 633e-9
K0 = 2 * np.pi / WAVELENGTH
DEPTH = 300e-6


def ring(grid, inner, outer, polarization):
    rho0, sigma = 0.5 * (inner + outer), (outer - inner) / 6
    a = np.exp(-(((grid.rho - rho0) / sigma) ** 2)).astype(complex)
    radial, azimuthal = polarization
    return mo.Field(radial * a, azimuthal * a, 1.0), rho0, sigma


def case(grid, n, shells, polarization=(1, 1j), diattenuation=True):
    surfaces, (inner, outer) = dv.brewster_axicon(0.0, n, DEPTH)
    field, rho0, sigma = ring(grid, inner, outer, polarization)
    if shells:
        # Nested spikes: for a ring far from the axis their inward shifts
        # (0.3 x 10 µm each) only move the ring, and no funnel needs the
        # millimetre set-back that would let the narrow ring diffract.
        stack, end = dv.shell_stack("A" * shells, 2.0, 10e-6, outer, 50e-6)
        offset = -(end + 100e-6)
        stack = [
            mo.Surface(s.z_vertex + offset, s.sag, s.slope, s.n1, s.n2, s.name)
            for s in stack
        ]
        surfaces = stack + surfaces
    t = time.perf_counter()
    out, z, _, diagnostics = mo.run(
        grid, field, surfaces, K0, record=lambda i, s: False, diattenuation=diattenuation
    )
    elapsed = time.perf_counter() - t
    return dict(
        n=n,
        surfaces=surfaces,
        incident=field,
        out=out,
        z_exit=z,
        ring=(inner, outer, rho0, sigma),
        elapsed=elapsed,
        diagnostics=diagnostics,
    )


def needle(grid, result, rho, dz):
    """|E_rho|², |E_phi|², |E_z|² on (dz, rho) behind the exit vertex plane."""
    maps = np.zeros((3, len(dz), len(rho)))
    cached = mo.spectra(grid, result["out"])
    for i, d in enumerate(dz):
        components = mo.components(grid, result["out"], K0, rho, d, cached)
        maps[:, i] = [np.abs(c) ** 2 for c in components]
    return maps


def fwhm(rho, intensity):
    """Full width at half maximum of a profile peaked on the axis, by interpolation."""
    half = intensity[0] / 2
    below = np.flatnonzero(intensity < half)
    if not len(below):
        return np.nan
    i = below[0]
    x = rho[i - 1] + (half - intensity[i - 1]) * (rho[i] - rho[i - 1]) / (
        intensity[i] - intensity[i - 1]
    )
    return 2 * x


def metrics(grid, result):
    """Needle position, length, widths and the azimuthal power fraction."""
    inner, outer, rho0, sigma = result["ring"]
    na = dv.brewster(result["n"])["na"]
    reach = outer * 2.5 / np.tan(np.arcsin(na))
    dz = np.linspace(0, reach, 1201)
    cached = mo.spectra(grid, result["out"])
    axis = np.array(
        [
            np.abs(mo.components(grid, result["out"], K0, [0.0], d, cached)[2][0]) ** 2
            for d in dz
        ]
    )
    peak = dz[np.argmax(axis)]
    rho = np.linspace(0, 3 * WAVELENGTH, 1201)
    e = mo.components(grid, result["out"], K0, rho, peak, cached)
    rho_i, phi_i, z_i = (np.abs(c) ** 2 for c in e)
    total = rho_i + phi_i + z_i
    width = fwhm(rho, total)
    inside = rho <= width / 2
    weight = rho[inside]
    longitudinal = np.trapezoid(z_i[inside] * weight, rho[inside]) / np.trapezoid(
        total[inside] * weight, rho[inside]
    )
    above = dz[axis >= axis.max() / 2]
    tm, te = mo.powers(grid, result["out"], K0)
    tm0, te0 = mo.powers(grid, result["incident"], K0)
    return dict(
        n=float(result["n"]),
        na=float(na),
        peak_dz_um=float(peak * 1e6),
        needle_length_um=float((above.max() - above.min()) * 1e6),
        fwhm_total_wavelengths=float(width / WAVELENGTH),
        fwhm_ez_wavelengths=float(fwhm(rho, z_i) / WAVELENGTH),
        longitudinal_share_in_fwhm=float(longitudinal),
        te_over_tm=float(te / tm) if tm > 0 else float("nan"),
        tm_transmitted=float(tm / tm0) if tm0 > 0 else float("nan"),
        elapsed_s=float(result["elapsed"]),
        axis=(dz, axis),
        profile=(rho, rho_i, phi_i, z_i),
    )


def main(out=None):
    out = Path(out) if out else HERE / "figures"
    out.mkdir(parents=True, exist_ok=True)
    grid = mo.HankelGrid(700e-6, 5000)
    runs = {}
    for label, n, shells in (("sqrt3", np.sqrt(3), 0), ("n2", 2.0, 0), ("n2+shells", 2.0, 4)):
        for kind, kwargs in (
            ("brewster", {}),
            ("no diattenuation", {"diattenuation": False}),
            ("radial input", {"polarization": (1, 0)}),
        ):
            result = case(grid, n, shells, **kwargs)
            m = metrics(grid, result)
            runs[(label, kind)] = (result, m)
            print(
                f"{label:10s} {kind:17s} NA {m['na']:.3f}  TE/TM {m['te_over_tm']:.4f}  "
                f"T_TM {m['tm_transmitted']:.4f}  FWHM |E|² {m['fwhm_total_wavelengths']:.3f} λ  "
                f"FWHM |Ez|² {m['fwhm_ez_wavelengths']:.3f} λ  Ez share {m['longitudinal_share_in_fwhm']:.3f}  "
                f"needle {m['needle_length_um']:.1f} µm  ({m['elapsed_s']:.1f} s)"
            )
    table = {
        f"{label} | {kind}": {k: v for k, v in m.items() if k not in ("axis", "profile")}
        for (label, kind), (_, m) in runs.items()
    }
    (out / "axicon_results.json").write_text(json.dumps(table, indent=1))
    np.savez(
        out / "axicon_profiles.npz",
        **{
            f"{label}|{kind}|{name}": array
            for (label, kind), (_, m) in runs.items()
            for name, array in zip(("rho", "rho2", "phi2", "z2"), m["profile"])
        },
        **{
            f"{label}|{kind}|axis_{name}": array
            for (label, kind), (_, m) in runs.items()
            for name, array in zip(("dz", "ez2"), m["axis"])
        },
    )
    return grid, runs


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
