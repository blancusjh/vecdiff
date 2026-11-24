"""The meridional operator against vecdiff's Propagator on one Brewster shell.

The shell is two parallel cones of slope n = 2 (Brewster incidence for an
axial beam), 10 µm apart on the axis, with hyperbolic tips of a = 20 µm so
that vecdiff's meridian realization carries it (its axial ray must meet no
caustic). Input: the circularly polarized doughnut with its vortex removed,
E = (rho/w) exp(-rho²/w²) (e_rho + i e_phi), w = 40 µm, 633 nm. Both
methods give E on a plane 1 mm behind the shell; the script prints the
powers of the radial (TM) and azimuthal (TE) parts and the overlap of the
two fields.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import meridional_operator as mo  # noqa: E402
import vecdiff as vd  # noqa: E402
from hyperbolic_cone import Doughnut, HyperbolicCone  # noqa: E402
from vecdiff._backend import to_numpy  # noqa: E402
from vecdiff.geometry.frames import Frame  # noqa: E402

WAVELENGTH, N, WAIST, TIP, SEPARATION = 633e-9, 2.0, 40e-6, 20e-6, 10e-6
START, AFTER = -50e-6, 1e-3


def library(radii):
    air, glass = vd.media.air, vd.media.Dielectric(N, "glass")
    shell = [
        vd.system.Refraction(
            HyperbolicCone(frame=Frame([0, 0, z]), cone_slope=N, tip=TIP), a, b
        )
        for z, a, b in ((0.0, air, glass), (SEPARATION, glass, air))
    ]
    beam = vd.fields.PlaneBeam(
        Doughnut(WAIST),
        vd.fields.polarization.polar(1, 1j),
        wavelength0=WAVELENGTH,
        region=air,
        frame=Frame([0, 0, START]),
    )
    t = time.perf_counter()
    solution = vd.propagation.Propagator(vd.system.OpticalSystem(*shell))(beam)
    elapsed = time.perf_counter() - t
    harmonics, _ = solution.spectrum.harmonics(Frame([0, 0, SEPARATION + AFTER]), radii)
    e_rho, e_phi, e_z = (to_numpy(v) for v in harmonics[0])
    return e_rho, e_phi, e_z, elapsed, solution


def operator(radii):
    k0 = 2 * np.pi / WAVELENGTH
    t = time.perf_counter()
    grid = mo.HankelGrid(700e-6, 5000)
    a = (grid.rho / WAIST) * np.exp(-((grid.rho / WAIST) ** 2))
    field = mo.Field(a.astype(complex), 1j * a, 1.0)
    field = mo.propagate(grid, field, k0, -START)
    surfaces = [
        mo.cone(0.0, N, 1.0, N, tip=TIP),
        mo.cone(SEPARATION, N, N, 1.0, tip=TIP),
    ]
    out, z, _, _ = mo.run(grid, field, surfaces, k0)
    e_rho, e_phi, e_z = mo.components(grid, out, k0, radii, AFTER)
    return e_rho, e_phi, e_z, time.perf_counter() - t


def main():
    radii = np.linspace(0, 200e-6, 2001)
    power = lambda f: np.trapezoid(np.abs(f) ** 2 * radii, radii)  # noqa: E731
    lib = library(radii)
    own = operator(radii)
    print(lib[4])
    incident = power((radii / WAIST) * np.exp(-((radii / WAIST) ** 2)))
    for name, (er, ef, ez, dt) in (("vecdiff Propagator", lib[:4]), ("operator", own)):
        tm, te = power(er) + power(ez), power(ef)
        print(
            f"{name:20s} {dt:6.1f} s   TM {tm / incident:.5f}   TE {te / incident:.5f}"
            f"   TE/TM {te / tm:.5f}"
        )
    for k, label in ((0, "E_rho"), (1, "E_phi")):
        a, b = lib[k], own[k]
        overlap = abs(np.sum(np.conj(a) * b * radii)) ** 2 / (
            np.sum(abs(a) ** 2 * radii) * np.sum(abs(b) ** 2 * radii)
        )
        print(f"overlap of {label}: {overlap:.5f}")
    print(f"closed form for a sharp cone: TE/TM = c^2 = {(2 * N / (1 + N * N)) ** 4:.5f}")
    return lib, own, radii


if __name__ == "__main__":
    main()
