"""Electric-field propagation through a system onto placed target surfaces."""

import numpy as np
import vecdiff_general as optics


def main():
    wavelength = 532e-9
    axis = (np.arange(48) - 24) * wavelength / 2
    plane = optics.CartesianGrid(axis, axis)
    air = optics.HomogeneousRegion("air", optics.Dielectric(1))
    glass = optics.HomogeneousRegion("glass", optics.Dielectric(1.5))
    x, y = plane.points[..., 0], plane.points[..., 1]
    envelope = np.exp(-(x * x + y * y) / (3 * wavelength) ** 2)
    incident = optics.TransverseElectricField(
        plane, wavelength, np.stack((envelope, 0.3j * envelope), axis=-1)
    )
    system = optics.OpticalSystem(
        [optics.DielectricInterface(optics.SphericalCap(40 * wavelength), air, glass)]
    )
    for treatment in ("local", "spectral"):
        propagate = optics.PropagationPipeline(system, transmission=treatment)
        field = propagate(incident, z=20 * wavelength)
        print(f"{treatment}: peak |E|² = {field.norm2().max():.6g} V²/m²")
        # An explicit placed surface uses the same operator and cached solve.
        target = optics.Plane.y(0).sample(
            u=axis, v=np.linspace(10, 30, 41) * wavelength
        )
        meridian = propagate(incident, target)
        print(f"Meridional samples: {meridian.grid.shape}")


if __name__ == "__main__":
    main()
