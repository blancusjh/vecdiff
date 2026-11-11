"""A Gaussian beam focused by a paraboloidal mirror: E, H and S at its focus.

The mirror is an ideal paraboloid of focal length 5 mm (vertex radius
-10 mm), met by a collimated 1 mm waist beam behind a 3 mm stop; the light
comes back toward the source and focuses 5 mm in front of the vertex. At
633 nm, in metres. The mirror's surface is met along its normal; the path
after it travels against the axis, and its axis turns over with the light.
"""

from time import perf_counter

import numpy as np
import vecdiff as vd


def mirror(focal_length=5e-3):
    """The path: a stop, then the paraboloid as an ideal mirror."""
    air = vd.media.air
    paraboloid = vd.surfaces.EvenAsphere(-1 / (2 * focal_length), conic=-1.0)
    return vd.system.OpticalSystem(
        vd.system.Stop(3e-3, z=-1e-3),
        vd.system.Reflection(paraboloid, air),
    )


def main():
    focal_length = 5e-3
    beam = vd.fields.PlaneBeam(
        vd.fields.profiles.Gaussian(1e-3),
        vd.fields.polarization.x,
        wavelength0=633e-9,
        region=vd.media.air,
        frame=vd.geometry.Frame([0, 0, -2e-3]),
    )
    start = perf_counter()
    solution = vd.propagation.Propagator(mirror(focal_length))(beam)
    print(solution)
    print(f"carried in {perf_counter() - start:.1f} s")

    # The paraboloid is stigmatic: every carried ray passes the focus.
    points, directions = solution.legs[-1].footprint()
    focus = np.array([0, 0, -focal_length])
    offset = focus - points
    miss = offset - np.sum(offset * directions, -1)[:, None] * directions
    print(f"largest ray miss at the focus: {np.linalg.norm(miss, axis=-1).max():.1e} m")

    # E and H on the focal plane, where the reflected rays arrive; there the
    # path also holds the incident beam, which is summed with them.
    start = perf_counter()
    field = solution.field(z=-focal_length)
    S = vd.observables.poynting(field)
    print(
        f"focal plane {field.grid.shape}, +-{field.grid.u[-1] * 1e6:.1f} um, "
        f"in {perf_counter() - start:.1f} s"
    )
    reflected = solution.spectrum.evaluate(field.grid)
    # It travels back, against the plane's normal: its irradiance is negative.
    peak = np.unravel_index(np.argmax(-reflected.irradiance()), field.grid.shape)
    print(
        f"E of the reflected light at focus: {np.round(reflected.electric.values[peak], 2)} V/m"
    )
    print(f"H at focus: {np.round(reflected.magnetic[peak] * 1e3, 3)} mA/m")
    print(f"S of the reflected light: {np.round(reflected.poynting()[peak], 3)} W/m²")
    print(f"S of all the light there: {np.round(S[peak], 3)} W/m²")


if __name__ == "__main__":
    main()
