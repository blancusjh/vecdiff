"""Light in vecdiff.fields: beams from a profile and a polarization, checked
against what optics says they must do.

A beam is an exact source-free sum of propagating plane waves in one medium,
built from a scalar profile (fields.profiles) times a polarization state
(fields.polarization). Lengths are in wavelengths (wavelength0 = 1); E is in
V/m, so irradiance is in W/m² and power in W per squared length unit.

1. A collimated Gaussian beam (PlaneBeam) doubles its area at the Rayleigh
   distance z_R = pi w² / lambda: the axial irradiance halves there.
2. A uniform pupil focused at low NA (FocusedBeam) gives the Airy pattern,
   first zero at 0.61 lambda / NA; aplanatic apodization carries the pupil's
   power to the focal plane.
3. At high NA the polarization decides the focus: radial polarization puts a
   longitudinal E_z on the axis, azimuthal polarization leaves the axis dark.
4. A zoned pupil (Zones): a phase plate flipping the sign of the outer half
   of the pupil's area cancels the axial field at the focus.
"""

import numpy as np
import vecdiff as vd

F = vd.fields


def axial_irradiance(beam, z):
    """S . z on the axis at the heights z."""
    points = vd.sampling.PointGrid(np.c_[0 * z, 0 * z, z])
    return vd.observables.irradiance(beam.evaluate(points), [0, 0, 1])


def focal_line(beam, half, count):
    """E and H on the x axis of the focal plane."""
    x = np.linspace(-half, half, count)
    return x, beam.evaluate(vd.sampling.PointGrid(np.c_[x, 0 * x, 0 * x]))


def main(scale=1.0):
    """Run the four checks; ``scale`` < 1 coarsens the sampling."""
    air = vd.media.air

    # 1. Rayleigh distance of a Gaussian beam, waist 4 wavelengths.
    waist = 4.0
    gaussian = F.PlaneBeam(
        F.profiles.Gaussian(waist), F.polarization.x, wavelength0=1.0, region=air
    )
    rayleigh = np.pi * waist**2
    near, far = axial_irradiance(gaussian, np.array([0.0, rayleigh]))
    # Expect 0.5: paraxial optics; the beam is exact, the 4-wavelength waist
    # nearly paraxial.
    print(
        f"Gaussian: irradiance at z_R / at the waist = {far / near:.4f} (paraxial 0.5)"
    )

    # 2. Airy focus of a uniform pupil at NA 0.2 (pupil radius NA * f).
    na, f = 0.2, 1.0
    airy = F.FocusedBeam(
        F.profiles.Uniform(na * f),
        F.polarization.x,
        focal_length=f,
        max_sine=na,
        wavelength0=1.0,
        region=air,
    )
    x, line = focal_line(airy, 4.0, int(801 * scale) // 2 * 2 + 1)
    intensity = vd.observables.irradiance(line, [0, 0, 1])
    centre = len(x) // 2
    first = centre + np.argmax(np.diff(intensity[centre:]) > 0)
    # Expect 0.61 / NA = 3.05 wavelengths (scalar Airy; the vector focus at
    # NA 0.2 differs by about a percent along x).
    print(
        f"Airy: first zero at x = {x[first]:.3f} lambda (0.61 / NA = {0.61 / na:.3f})"
    )
    count = int(161 * scale)
    axis = np.linspace(-20.0, 20.0, count)
    plane = vd.sampling.CartesianGrid(axis, axis)
    pupil_power = np.pi * (na * f) ** 2 / (2 * vd.media.Z0)
    # Expect 0.975: the aplanatic beam carries the pupil's power to the focal
    # plane, and the 40-wavelength square holds all of it but the Airy rings
    # beyond, 2 / (pi v) at v = 2 pi NA 20 = 25.
    print(
        f"Airy: focal-plane power / pupil power = {airy.evaluate(plane).power() / pupil_power:.4f}"
    )

    # 3. Radial and azimuthal polarization at NA 0.9.
    for name, state in (
        ("radial", F.polarization.radial),
        ("azimuthal", F.polarization.azimuthal),
    ):
        beam = F.FocusedBeam(
            F.profiles.Uniform(0.9),
            state,
            focal_length=1.0,
            max_sine=0.9,
            wavelength0=1.0,
            region=air,
        )
        x, line = focal_line(beam, 1.5, int(301 * scale) // 2 * 2 + 1)
        E = line.electric.values
        peak = np.max(np.sum(abs(E) ** 2, -1))
        centre = E[len(x) // 2]
        on_axis = np.sum(abs(centre) ** 2)
        # Expect radial: the axis holds the focal maximum, and all of it is
        # E_z. Azimuthal: E vanishes on the axis by symmetry (roundoff), the
        # light lies on a ring around it.
        share = abs(centre[2]) ** 2 / on_axis if name == "radial" else None
        print(
            f"{name}: |E|² on axis / focal maximum = {on_axis / peak:.2e}"
            + ("" if share is None else f", of which E_z {share:.3f}")
        )

    # 4. A binary phase plate: the outer half of the pupil's area flipped.
    a = 0.2
    plate = F.profiles.Zones((a / np.sqrt(2), a), (1, -1, 0))
    for name, profile in (
        ("open pupil", F.profiles.Uniform(a)),
        ("phase plate", plate),
    ):
        beam = F.FocusedBeam(
            profile,
            F.polarization.x,
            focal_length=1.0,
            max_sine=a,
            wavelength0=1.0,
            region=air,
            apodization="none",
        )
        field = beam.evaluate(vd.sampling.PointGrid(np.zeros((1, 3))))
        # Expect the plate to leave 2.5e-3 of the open pupil's axial field: with
        # apodization "none" the transverse spectrum is the pupil field, and
        # its two zones have equal area and opposite sign; only the vector
        # weight (1 + cos theta) / 2 of E_x survives, NA² / 16 of it.
        print(
            f"{name}: |E_x| at the focus = {abs(field.electric.values[0, 0]):.3e} V/m"
        )


if __name__ == "__main__":
    main()
