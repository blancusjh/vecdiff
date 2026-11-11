"""A Gaussian beam through a plano-convex lens, flat side first: E, H and S at focus.

The lens is BK7-like glass, 3 mm thick, with a 25 mm radius on its exit face
(focal length about 50 mm). The beam is collimated, 2 mm waist, polarized
along x, at 532 nm. Lengths are in metres here; any unit works if the
wavelength is given in it.
"""

from time import perf_counter

import numpy as np
import vecdiff as vd


def lens():
    air = vd.media.HomogeneousRegion("air", vd.media.Dielectric(1.0))
    glass = vd.media.HomogeneousRegion("BK7-like", vd.media.Dielectric(1.5))
    behind = vd.media.HomogeneousRegion("air behind", vd.media.Dielectric(1.0))
    flat = vd.surfaces.Plane.z(0.0)
    curved = vd.surfaces.SphericalCap(-25e-3, vd.geometry.Frame([0, 0, 3e-3]))
    return air, vd.system.OpticalSystem(
        (
            vd.system.Refraction(flat, air, glass),
            vd.system.Refraction(curved, glass, behind),
        )
    )


def main():
    air, system = lens()
    beam = vd.fields.PlaneBeam(
        vd.fields.profiles.Gaussian(2e-3),
        vd.fields.polarization.x,
        wavelength0=532e-9,
        region=air,
        frame=vd.geometry.Frame([0, 0, -1e-3]),
    )
    start = perf_counter()
    solution = vd.propagation.Propagator(system)(beam)
    print(solution)
    print(f"carried in {perf_counter() - start:.1f} s")

    # The focus: the brightest point of the axis behind the lens.
    z = np.linspace(45e-3, 60e-3, 301)
    axis = vd.sampling.PointGrid(np.c_[0 * z, 0 * z, z])
    on_axis = solution.field(axis)
    focus = float(z[np.argmax(vd.observables.irradiance(on_axis, [0, 0, 1]))])
    print(f"focus at z = {focus * 1e3:.3f} mm")

    # E and H together on the focal plane, and the Poynting vector.
    start = perf_counter()
    field = solution.field(z=focus, window=(-20e-6, 20e-6, -20e-6, 20e-6))
    S = vd.observables.poynting(field)
    print(f"focal plane {field.grid.shape} in {perf_counter() - start:.1f} s")
    centre = tuple(n // 2 for n in field.grid.shape)
    print(f"E at focus: {np.round(field.electric.values[centre], 3)} V/m")
    print(f"H at focus: {np.round(field.magnetic[centre] * 1e3, 4)} mA/m")
    print(f"S at focus: {np.round(S[centre], 6)} W/m²")
    print(f"peak irradiance {field.irradiance().max():.4g} W/m²")


if __name__ == "__main__":
    main()
