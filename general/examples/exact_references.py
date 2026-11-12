"""The exact solutions in vecdiff.references, and the identities they satisfy.

MieSphere, LayeredSphere and InfiniteCylinder solve Maxwell's equations for
one body under any propagating incident field, and split the field into
Debye terms: term 1 entered once, term p met p - 1 internal reflections,
term 0 is the external reflection and diffraction, ("after", P) the rest.
They are what the transport is checked against; the transport never calls
them. A glass sphere and cylinder (n = 1.5) of radius 10 wavelengths, in air.
Lengths in wavelengths.
"""

import numpy as np
import vecdiff as vd
from vecdiff.references import InfiniteCylinder, LayeredSphere, MieSphere


def region(name, n):
    return vd.media.HomogeneousRegion(name, vd.media.Dielectric(n))


def relative(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def main(radius=10.0):
    """Check the references on a body of ``radius`` wavelengths."""
    air, glass = region("air", 1.0), region("glass", 1.5)
    wave = vd.fields.PlaneBeam(
        vd.fields.profiles.Uniform(),
        vd.fields.polarization.x,
        wavelength0=1.0,
        region=air,
    ).plane_waves(radius)
    ball = vd.volumes.SphereVolume([0, 0, 0], radius)
    mie = MieSphere(wave, ball, glass)

    # 1. Efficiencies of a lossless sphere: scattering equals extinction (the
    # optical theorem), and both tend to 2 for a large sphere (the extinction
    # paradox): expect equal values near 2, here 2.18 with the ripple of
    # x = 2 pi R = 63.
    q = mie.efficiencies()
    print(
        f"Mie, x = {2 * np.pi * radius:.1f}: Q_sca = {q['scattering']:.6f}, Q_ext = {q['extinction']:.6f}"
    )

    # 2. The Debye terms add up to the field: inside, terms 1, 2, 3 and the
    # rest after 3 are the Lorenz-Mie interior field. Expect roundoff.
    rho = np.linspace(0, 0.9 * radius, 40)
    inside = vd.sampling.PointGrid(np.c_[rho, 0 * rho, 0.3 * radius + 0 * rho])
    total = mie.evaluate(inside, side="interior").electric.values
    terms = sum(
        mie.evaluate(inside, side="interior", term=t).electric.values
        for t in (1, 2, 3, ("after", 3))
    )
    print(
        f"Mie: |sum of Debye terms - total| / |total| inside = {relative(terms, total):.1e}"
    )

    # 3. A layered sphere whose core has the shell's index is one sphere.
    # Expect roundoff.
    core = region("core", 1.5)
    layered = LayeredSphere(
        wave, (ball, vd.volumes.SphereVolume([0, 0, 0], radius / 2)), (glass, core)
    )
    same = layered.evaluate(inside).electric.values
    print(
        f"LayeredSphere with an index-matched core vs Mie: {relative(same, total):.1e}"
    )

    # 4. The cylinder: tangential E and H are continuous across its surface
    # (both traces at the same points), for a beam focused onto it.
    beam = vd.fields.FocusedBeam(
        vd.fields.profiles.Uniform(0.3),
        vd.fields.polarization.x,
        focal_length=1.0,
        max_sine=0.3,
        wavelength0=1.0,
        region=air,
        frame=vd.geometry.Frame([0, 0, 0.5 * radius]),
    )
    rod = vd.volumes.CylinderVolume([0, 0, 0], [0, 1, 0], radius)
    cylinder = InfiniteCylinder(beam, rod, glass)
    angle = np.linspace(-0.6, 0.6, 25)
    rim = vd.sampling.PointGrid(
        np.c_[radius * np.sin(angle), 0 * angle, -radius * np.cos(angle)]
    )
    normal = rim.points / radius
    jump = []
    for name in ("electric", "magnetic"):
        a, b = (cylinder.evaluate(rim, side=side) for side in ("exterior", "interior"))
        a = a.electric.values if name == "electric" else a.magnetic
        b = b.electric.values if name == "electric" else b.magnetic
        tangential = lambda v: v - np.sum(v * normal, -1)[:, None] * normal  # noqa: E731
        jump.append(relative(tangential(a), tangential(b)))
    # Expect roundoff: both sides solve the same boundary conditions.
    print(
        f"InfiniteCylinder: tangential E, H jump across the surface = {jump[0]:.1e}, {jump[1]:.1e}"
    )

    # 5. The first entry of the cylinder, the wave the transport models,
    # against its whole interior field: their difference is the internally
    # reflected waves, of the order of the amplitude reflection |r| = 0.2 at
    # normal incidence. This is why the transport is checked against term 1.
    x = np.linspace(-2.0, 2.0, 41)
    focus = vd.sampling.PointGrid(np.c_[x, 0 * x, 0 * x])
    first = cylinder.evaluate(focus, side="interior", term=1).electric.values
    whole = cylinder.evaluate(focus, side="interior").electric.values
    print(
        f"InfiniteCylinder: |total - term 1| / |term 1| near the axis = {relative(whole, first):.3f}"
    )


if __name__ == "__main__":
    main()
