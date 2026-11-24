"""Wave transport of a sampled field without a carrier, onto placed targets.

alternatives.WavePropagator is the check on the method (propagation.Propagator):
it carries the field exactly to the surface and back, with a local or a
direction-resolved Fresnel treatment between.
"""

import numpy as np
import vecdiff as optics


def main():
    wavelength = 532e-9
    axis = (np.arange(48) - 24) * wavelength / 2
    plane = optics.sampling.CartesianGrid(axis, axis)
    air = optics.media.HomogeneousRegion("air", optics.media.Dielectric(1))
    glass = optics.media.HomogeneousRegion("glass", optics.media.Dielectric(1.5))
    x, y = plane.points[..., 0], plane.points[..., 1]
    envelope = np.exp(-(x * x + y * y) / (3 * wavelength) ** 2)
    incident = optics.fields.TransverseElectricField(
        plane, wavelength, np.stack((envelope, 0.3j * envelope), axis=-1)
    )
    system = optics.system.OpticalSystem(
        [
            optics.system.Refraction(
                optics.surfaces.SphericalCap(40 * wavelength), air, glass
            )
        ]
    )
    for treatment in ("local", "spectral"):
        solution = optics.alternatives.WavePropagator(system, transmission=treatment)(
            incident
        )
        field = solution.field(z=20 * wavelength)
        print(f"{treatment}: peak irradiance {field.irradiance().max():.6g} W/m²")
        # Another target is read from the same Solution, with no new transport.
        target = optics.surfaces.Plane.y(0).sample(
            u=axis, v=np.linspace(10, 30, 41) * wavelength
        )
        meridian = solution.electric(target)
        print(f"Meridional samples: {meridian.grid.shape}")


if __name__ == "__main__":
    main()
