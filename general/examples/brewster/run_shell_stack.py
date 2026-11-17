"""Nested Brewster shells that suppress E_phi of a collimated doughnut beam.

Input: E = (rho/w) exp(-rho²/w²) (e_rho + i e_phi), w = 40 µm, 633 nm, the
circularly polarized doughnut with its vortex removed. Each shell is two
parallel cones of slope n, 10 µm apart on the axis. With N surfaces the
azimuthal share of the output power is c^N / (1 + c^N), c = (2n/(1 + n²))²,
and the radial field passes with T_p = 1; the shells shift the rays radially,
spikes (A) inward and funnels (B) outward.

Cases: alternating ABABAB (n = 2), all spikes AAAAAA, alternating with 30 µm
between the faces, BK7 (n = 1.515) with 14 alternating shells, and the
baseline with hyperbolic tips of 1 to 20 µm. "match" is the overlap of the
output E_rho with the input E_rho propagated freely over the same length,
the ideal diag(1, c^(N/2)) element.

Run as a script it prints the table and writes shell_results.json to the
directory given as argument (default: figures/ next to this file).
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

WAVELENGTH, WAIST = 633e-9, 40e-6
K0 = 2 * np.pi / WAVELENGTH
AFTER = 1e-3


def doughnut(grid):
    a = ((grid.rho / WAIST) * np.exp(-((grid.rho / WAIST) ** 2))).astype(complex)
    return mo.Field(a / np.sqrt(2), 1j * a / np.sqrt(2), 1.0)


def match(grid, a, b):
    w = grid.w_rho
    return float(
        abs(np.sum(w * np.conj(a) * b)) ** 2
        / (np.sum(w * abs(a) ** 2) * np.sum(w * abs(b) ** 2))
    )


def stack(grid, kinds, n, separation, tip=0.0):
    surfaces, end = dv.shell_stack(kinds, n, separation, 250e-6, 50e-6, tip=tip)
    incident = doughnut(grid)
    t = time.perf_counter()
    out, z, kept, _ = mo.run(
        grid, incident, surfaces, K0, record=lambda i, s: s.n2 == 1.0
    )
    out = mo.propagate(grid, out, K0, AFTER)
    elapsed = time.perf_counter() - t
    ideal = mo.propagate(grid, incident, K0, z + AFTER)
    c = dv.brewster(n)["power_ratio"]
    tm0, te0 = mo.powers(grid, incident, K0)
    shells = []
    for k, (_, _, f) in enumerate(kept, start=1):
        tm, te = mo.powers(grid, f, K0)
        shells.append(
            dict(
                shell=k,
                tm=tm / tm0,
                azimuthal_share=te / (tm + te),
                closed_form=c ** (2 * k) / (1 + c ** (2 * k)),
            )
        )
    tm, te = mo.powers(grid, out, K0)
    return dict(
        kinds=kinds,
        n=n,
        separation_um=separation * 1e6,
        wall_normal_um=separation / np.sqrt(1 + n * n) * 1e6,
        tip_um=tip * 1e6,
        length_mm=(z + AFTER) * 1e3,
        tm=tm / tm0,
        azimuthal_share=te / (tm + te),
        closed_form=shells[-1]["closed_form"],
        match=match(grid, out.rho, ideal.rho),
        elapsed_s=elapsed,
        shells=shells,
    ), out, ideal


CASES = (
    ("ABABAB", 2.0, 10e-6, 0.0),
    ("AAAAAA", 2.0, 10e-6, 0.0),
    ("ABABAB", 2.0, 30e-6, 0.0),
    ("AB" * 7, 1.515, 10e-6, 0.0),
    ("ABABAB", 2.0, 10e-6, 1e-6),
    ("ABABAB", 2.0, 10e-6, 5e-6),
    ("ABABAB", 2.0, 10e-6, 20e-6),
)


def main(out=None):
    out = Path(out) if out else HERE / "figures"
    out.mkdir(parents=True, exist_ok=True)
    grid = mo.HankelGrid(700e-6, 5000)
    results, fields = [], {}
    for kinds, n, separation, tip in CASES:
        r, field, ideal = stack(grid, kinds, n, separation, tip)
        results.append(r)
        fields[f"{kinds}|{n}|{separation}|{tip}"] = (field, ideal)
        print(
            f"{kinds:14s} n {n:.3f}  wall {r['wall_normal_um']:5.2f} µm  tip {r['tip_um']:4.1f} µm  "
            f"E_phi share {100 * r['azimuthal_share']:.3f} % (closed form {100 * r['closed_form']:.3f} %)  "
            f"T_rho {100 * r['tm']:.2f} %  match {100 * r['match']:.1f} %  ({r['elapsed_s']:.1f} s)"
        )
    (out / "shell_results.json").write_text(json.dumps(results, indent=1))
    np.savez(
        out / "shell_fields.npz",
        rho=grid.rho,
        **{f"{k}|out_rho": f.rho for k, (f, _) in fields.items()},
        **{f"{k}|out_phi": f.phi for k, (f, _) in fields.items()},
        **{f"{k}|ideal_rho": i.rho for k, (_, i) in fields.items()},
    )
    return grid, results, fields


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
