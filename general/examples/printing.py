"""What vecdiff objects print: systems as prescription tables, a Solution, a beam.

Four paths print as lens prescriptions (a plano-convex lens behind a stop, a
Cassegrain, the ghost of a plate, a fold mirror), then a converging beam is
carried through a glass ball's front face and its Solution, beam, field and
grid print. In Jupyter the system and the Solution display as HTML tables;
``system.table()`` gives the rows as data. Lengths are in the units of the
wavelength (metres for the lens, wavelengths elsewhere).
"""

import numpy as np
import vecdiff as vd

S, Y, Frame = vd.surfaces, vd.system, vd.geometry.Frame
air, glass = vd.media.air, vd.media.Dielectric(1.5, "glass")


def lens():
    """The plano-convex lens of plano_convex_lens.py, in metres."""
    bk7 = vd.media.Dielectric(1.5, "BK7-like")
    return Y.OpticalSystem(
        Y.Stop(4e-3, z=-0.5e-3),
        Y.Refraction(S.Plane.z(0.0), air, bk7),
        Y.Refraction(S.SphericalCap(-25e-3, Frame([0, 0, 3e-3])), bk7, air),
    )


def cassegrain():
    """A paraboloid primary and a hyperboloid secondary behind a stop."""
    primary = S.EvenAsphere(-0.01, conic=-1.0)
    secondary = S.EvenAsphere(-1 / 24, conic=-1.96, frame=Frame([0, 0, -40.0]))
    return Y.OpticalSystem(
        Y.Stop(36.0, z=-10.0),
        Y.Reflection(primary, air),
        Y.Reflection(secondary, air),
    )


def ghost(thickness=100.0):
    """The first ghost of a plate: the same two faces, met twice each."""
    front, back = S.Plane.z(0.0), S.Plane.z(thickness)
    return Y.OpticalSystem(
        Y.Refraction(front, air, glass),
        Y.Reflection(back, glass, behind=air),
        Y.Reflection(front, glass, behind=air),
        Y.Refraction(back, glass, air),
    )


def fold():
    """A plane mirror at 45 degrees, folding light from +z to +y."""
    c = 1 / np.sqrt(2)
    tilted = Frame([0, 0, 0], np.array([[1, 0, 0], [0, c, c], [0, -c, c]]).T)
    return Y.OpticalSystem(Y.Reflection(S.Plane(tilted), air))


def main():
    for system in (lens(), cassegrain(), ghost(), fold()):
        print(system, end="\n\n")

    beam = vd.fields.FocusedBeam(
        vd.fields.profiles.Gaussian(0.3),
        vd.fields.polarization.x,
        focal_length=1.0,
        max_sine=0.4,
        wavelength0=1.0,
        region=air,
        frame=Frame([0, 0, 15.0]),
    )
    face = Y.OpticalSystem(
        Y.Refraction(S.SphericalCap(10.0, Frame([0, 0, -10.0])), air, glass)
    )
    solution = vd.propagation.Propagator(face)(beam)
    print(repr(beam), beam, sep="\n", end="\n\n")
    print(repr(solution), solution, sep="\n", end="\n\n")
    axis = np.linspace(-2.0, 2.0, 41)
    grid = vd.sampling.CartesianGrid(axis, axis, Frame([0, 0, 5.0]))
    field = solution.field(grid)
    print(repr(grid), grid, sep="\n", end="\n\n")
    print(repr(field), field, sep="\n")


if __name__ == "__main__":
    main()
