"""What vecdiff.observables measures on a field: energy flow, polarization,
momentum and spin, forces.

Every observable takes an ElectromagneticField (E and H on a grid), as a
Solution's ``field`` returns it; the momenta take an ElectricFieldJet (E and
its derivatives), here from a beam's plane waves. In metres, at 633 nm: E is
in V/m, S in W/m², power in W, spin in J s/m³. (In any other length unit
the ratios printed below are the same.)

1. A circularly polarized plane wave carries spin sigma w / omega along its
   direction, helicity sigma w / omega and canonical momentum n w / c.
2. A circularly polarized beam focused at NA 0.8: the power through the focal
   plane is the pupil's; |E|² / (2 Z0) integrated over the plane exceeds it
   by the mean of 1 / cos(theta) over the pupil, 1.25 (|E|² is not the
   irradiance at high NA); on the axis the field is circular (Stokes
   S3 = S0); and only the mean cos(theta) = 0.817 of the light's angular
   momentum is spin, the rest orbital.
3. The radiation force on a sphere, by the Maxwell stress tensor around it,
   against Wiscombe's Mie code.
"""

import numpy as np
import vecdiff as vd
from vecdiff.media import C0, EPSILON0, Z0
from vecdiff.references import MieSphere

obs = vd.observables
wavelength = 633e-9


def sphere_quadrature(radius, count):
    """Gauss-Legendre in cos(theta), uniform in phi: a closed, outward surface."""
    t, w = np.polynomial.legendre.leggauss(count)
    phi = 2 * np.pi * np.arange(2 * count) / (2 * count)
    sine = np.sqrt(1 - t**2)
    normals = np.stack(
        np.broadcast_arrays(
            sine[:, None] * np.cos(phi), sine[:, None] * np.sin(phi), t[:, None]
        ),
        axis=-1,
    )
    weights = radius**2 * w[:, None] * np.full(phi.shape, np.pi / count)
    return vd.sampling.SurfaceGrid(radius * normals, normals, weights)


def main(count=72):
    """Run the three checks; ``count`` sets the force quadrature."""
    air, glass = vd.media.air, vd.media.Dielectric(1.5, "glass")
    omega = 2 * np.pi * C0 / wavelength
    circular = vd.fields.polarization.cartesian(1 / np.sqrt(2), 1j / np.sqrt(2))

    # 1. A plane wave in glass, left circular (sigma = +1 for exp(-i omega t)).
    wave = vd.fields.PlaneBeam(
        vd.fields.profiles.Uniform(), circular, wavelength0=wavelength, region=glass
    ).plane_waves(1.0)
    points = vd.sampling.PointGrid(np.array([[0.0, 0.0, 0.0], [1e-7, 2e-7, 3e-7]]))
    jet = obs.plane_wave_jet(wave, points)
    field = jet.field
    w = obs.energy_density(field, glass)[0]
    spin = obs.spin_density(field, glass)[0]
    momentum = obs.canonical_momentum(jet, glass)[0]
    # Expect spin_z / (w / omega) = 1, helicity / (w / omega) = 1 and
    # p_z / (n w / c) = 1.
    print(
        f"plane wave: spin_z = {spin[2] * omega / w:.6f} w/omega, "
        f"helicity = {obs.helicity_density(field, glass)[0] * omega / w:.6f} w/omega, "
        f"p_z = {momentum[2] * C0 / (1.5 * w):.6f} n w/c"
    )

    # 2. A beam focused at NA 0.8 in air, circular polarization.
    na, f = 0.8, 2e-3
    beam = vd.fields.FocusedBeam(
        vd.fields.profiles.Uniform(na * f),
        circular,
        focal_length=f,
        max_sine=na,
        wavelength0=wavelength,
        region=air,
    )
    axis = np.linspace(-6, 6, 121) * wavelength
    plane = vd.sampling.CartesianGrid(axis, axis)
    focus = beam.evaluate(plane)
    pupil = np.pi * (na * f) ** 2 / (2 * Z0)  # |E_pupil| = 1 V/m over its disk
    # Expect about 0.99: the aplanatic beam carries the pupil's power, less
    # what lies outside the 12-wavelength square.
    print(
        f"focus: power through the plane / pupil power = {obs.power(focus) / pupil:.4f}"
    )
    weights = plane.weights
    electric = np.sum(focus.electric.norm2() * weights) / (2 * Z0)
    # Expect about 1.25 = 2 (1 - sqrt(1 - NA²)) / NA²: by Parseval each plane
    # wave contributes |a|² to the integral of |E|², |a|² cos(theta) to power;
    # the square misses 2 % of the light, so a little less here.
    print(
        f"focus: integral of |E|² / 2 Z0 over the power = {electric / obs.power(focus):.4f}"
    )
    centre = (60, 60)
    s0, s1, s2, s3 = obs.stokes(focus)[centre]
    # Expect S3 / S0 = 1: on the axis the field is transverse and circular.
    print(f"focus: Stokes on the axis S1/S0 = {s1 / s0:.1e}, S3/S0 = {s3 / s0:.6f}")
    w = obs.energy_density(focus, air)
    spin = obs.spin_density(focus, air)[..., 2]
    fraction = np.sum(spin * weights) * omega / np.sum(w * weights)
    # Expect about 0.817 = 2 (1 - (1 - NA²)^1.5) / (3 NA²), the mean cos(theta):
    # each wave carries spin w / omega along its own direction (a little less
    # here, for the light outside the square).
    print(f"focus: spin_z over the plane = {fraction:.4f} of w / omega")
    near = vd.sampling.PointGrid(
        np.array([[0.0, 0.0, 0.0], [0.3e-6, 0.0, 0.0], [0.0, 0.2e-6, 0.1e-6]])
    )
    jet = obs.plane_wave_jet(beam.plane_waves(1e-5), near)
    field = jet.field
    p_o, p_s = obs.canonical_momentum(jet, air), obs.spin_momentum(jet, air)
    P = obs.poynting(field)
    mismatch = np.linalg.norm(EPSILON0 * vd.media.MU0 * P - p_o - p_s)
    # Expect roundoff: eps mu P = p_o + p_s is an identity of Maxwell's
    # equations, here near the focus.
    print(
        f"focus: |eps mu P - p_o - p_s| / |eps mu P| = {mismatch / np.linalg.norm(P / C0**2):.1e}"
    )

    # 3. Radiation pressure on a sphere of m = 0.75, x = 10 (Wiscombe, MIEV0:
    # Q_pr = Q_ext - g Q_sca = 2.232265 - 2.001164 = 0.231101).
    radius = 10 / (2 * np.pi) * wavelength
    incident = vd.fields.PlaneBeam(
        vd.fields.profiles.Uniform(),
        vd.fields.polarization.x,
        wavelength0=wavelength,
        region=air,
    ).plane_waves(radius)
    sphere = MieSphere(
        incident,
        vd.volumes.SphereVolume([0, 0, 0], radius),
        vd.media.HomogeneousRegion("m = 0.75", vd.media.Dielectric(0.75)),
    )
    total = sphere.evaluate(sphere_quadrature(1.4 * radius, count))
    intensity = 0.5 * EPSILON0 * C0  # of the 1 V/m incident wave
    efficiency = obs.force(total, air) * C0 / (intensity * np.pi * radius**2)
    print(
        f"sphere: Q_pr = {efficiency[2]:.6f} (Wiscombe 0.231101), transverse {np.abs(efficiency[:2]).max():.1e}"
    )


if __name__ == "__main__":
    main()
