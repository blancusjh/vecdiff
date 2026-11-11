"""The ghost of a glass plate in a converging beam, beside the light it transmits.

A beam of NA 0.25 converges through a plate 100 wavelengths thick. The light
transmitted at once and its first ghost (reflected at the back, then at the
front, before it leaves) are two paths built from the same two surfaces:

    direct: Refraction(front), Refraction(back)
    ghost:  Refraction(front), Reflection(back), Reflection(front), Refraction(back)

Each focuses on the axis. Unfolded, the ghost crosses three thicknesses of
glass and is folded back by two: it focuses 2 d / n nearer than the direct
light, and r^2 = 0.2^2 weaker in amplitude at normal incidence. Lengths in
wavelengths (wavelength0 = 1).
"""

from time import perf_counter

import numpy as np
import vecdiff as vd


def paths(thickness=100.0):
    air, glass = vd.media.air, vd.media.Dielectric(1.5, "glass")
    front = vd.surfaces.Plane.z(0.0)
    back = vd.surfaces.Plane.z(thickness)
    Refraction, Reflection = vd.system.Refraction, vd.system.Reflection
    direct = vd.system.OpticalSystem(
        Refraction(front, air, glass), Refraction(back, glass, air)
    )
    ghost = vd.system.OpticalSystem(
        Refraction(front, air, glass),
        Reflection(back, glass, behind=air),
        Reflection(front, glass, behind=air),
        Refraction(back, glass, air),
    )
    return direct, ghost


def main():
    beam = vd.fields.FocusedBeam(
        lambda x, y: np.clip(1 - (x**2 + y**2) / 0.25**2, 0, None) ** 3,
        vd.fields.polarization.x,
        focal_length=1.0,
        max_sine=0.25,
        wavelength0=1.0,
        region=vd.media.air,
        frame=vd.geometry.Frame([0, 0, 600.0]),
        apodization="none",
    )
    z = np.linspace(450.0, 700.0, 251)
    axis = vd.sampling.PointGrid(np.c_[0 * z, 0 * z, z])
    for name, system in zip(("direct", "ghost"), paths()):
        start = perf_counter()
        solution = vd.propagation.Propagator(system)(beam)
        on_axis = solution.spectrum.evaluate(axis)
        irradiance = vd.observables.irradiance(on_axis, [0, 0, 1])
        peak = int(np.argmax(irradiance))
        print(
            f"{name:6s} focus at z = {z[peak]:.0f}, irradiance {irradiance[peak]:.3e} "
            f"W/m², E there {np.round(on_axis.electric.values[peak], 4)} V/m "
            f"({perf_counter() - start:.1f} s)"
        )
        print(solution)


if __name__ == "__main__":
    main()
