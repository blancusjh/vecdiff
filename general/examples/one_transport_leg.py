"""One transport leg by hand: the ray map, the carried field and its report.

vecdiff.propagation.Propagator composes legs; this script builds one leg of
vecdiff.transport directly. A collimated beam meets a glass cap of radius
R = 100 wavelengths (n = 1 -> 1.5); its paraxial focus lies n R / (n - 1)
= 300 wavelengths behind the vertex, inside the glass. Lengths in wavelengths.

1. transport.InterfaceMap: the plane-surface-plane map of the carrier's
   rays, here on the vertex plane: path increment, flux factor g, and where
   the refracted rays cross the axis.
2. The same map with its reference plane 400 wavelengths behind the vertex,
   beyond the focus: the ray tube has collapsed in both transverse
   directions on the way, the caustic index is 2, and g carries
   exp(-i pi kappa / 2) = -1.
3. transport.LatticeTransport carries a Gaussian beam across the cap on that
   plane. On the axis the carried field is t g E_incident: the Fresnel
   transmission t = 2 / (1 + n), times the area ratio sqrt(1 / J) = 3 of a
   tube that converged to the focus at 300 and spread again over 100, times
   -1 for the focus. The TransportReport says what the leg did not carry.
"""

import numpy as np
import vecdiff as vd

T = vd.transport


def main(count=200):
    """Build and run the leg; ``count`` is the lattice size per axis."""
    air, glass = vd.media.air, vd.media.Dielectric(1.5, "glass")
    radius, n = 100.0, 1.5
    cap = vd.surfaces.SphericalCap(radius)
    carrier = vd.fields.Carrier.collimated([0, 0, 1])

    # 1. The map on the vertex plane: rays labelled by their point on it.
    vertex = T.InterfaceMap(cap, carrier, 1.0, n)
    heights = np.array([[1.0, 0.0], [10.0, 0.0], [30.0, 0.0]])
    rays = vertex.forward(heights)
    s = vd.backend.to_numpy(rays.transmitted)
    q = vd.backend.to_numpy(rays.surface_point)
    crossing = q[:, 2] - q[:, 0] * s[:, 2] / s[:, 0]
    # Expect 300 near the axis (n R / (n - 1)), and nearer for higher rays:
    # spherical aberration of a single refracting sphere.
    for h, z, g in zip(heights[:, 0], crossing, vd.backend.to_numpy(rays.flux)):
        print(
            f"ray at h = {h:4.0f}: crosses the axis at z = {z:7.2f}, flux factor {g:.4f}"
        )

    # 2. The map on a plane beyond the focus: a caustic inside the leg.
    beyond = 400.0
    far = T.InterfaceMap(cap, carrier, 1.0, n, offset=beyond)
    ray = far.forward(np.array([[1.0, 0.0]]))
    kappa, g = int(ray.caustics[0]), complex(vd.backend.to_numpy(ray.flux)[0])
    # Expect kappa = 2 (a point focus) and g = -3: the tube is a third as wide
    # 100 wavelengths past a focus 300 wavelengths from the surface.
    print(f"plane at z = {beyond:.0f}: caustic index {kappa}, flux factor {g:.4f}")

    # 3. A leg on a lattice of that plane, carrying a Gaussian beam.
    axis = (np.arange(count) - count // 2) * (160.0 / count)
    plane = vd.sampling.CartesianGrid(axis, axis, vd.geometry.Frame([0, 0, beyond]))
    beam = vd.fields.PlaneBeam(
        vd.fields.profiles.Gaussian(20.0),
        vd.fields.polarization.x,
        wavelength0=1.0,
        region=air,
    )
    leg = T.LatticeTransport(plane, vd.system.Refraction(cap, air, glass), carrier, 1.0)
    incident = vd.fields.BeamSpectrum(beam, plane)
    carried, report = leg.carry(incident)
    # Expect nothing unsupported, folded or band-removed, caustics 2, and many
    # dark samples: the output lattice keeps the input spacing and extent, and
    # the rays' image on it is a third as wide as the beam.
    print(report)
    centre = tuple(np.argmin(abs(a)) for a in (leg.output_plane.v, leg.output_plane.u))
    origin = vd.sampling.PointGrid(np.array([[0.0, 0.0, beyond]]))
    reading = beam.evaluate(origin).electric.values[0, 0]
    ratio = complex(carried.values[centre][0] / reading)
    # Expect about -2.4 = t g = 0.8 x (-3), times exp(i k0 Delta S) = 1 here:
    # the optical path -400 + 1.5 x 400 = 200 wavelengths is a whole number.
    print(f"carried / incident E_x on the axis: {ratio:.3f} (t g = {0.8 * g.real:.3f})")
    outgoing, report = leg.transmit(incident)
    print(
        f"re-expanded in {outgoing.region.name} as {type(outgoing).__name__}, "
        f"caustic index {report.caustics}"
    )


if __name__ == "__main__":
    main()
