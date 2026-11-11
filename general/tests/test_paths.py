"""Light through real sequential paths: stops, mirrors, folds, ghosts, a closed ball.

Every case runs through ``Propagator(system)(light)`` and is checked against an
exact solution: a plane mirror's image, the per-plane-wave boundary-value
problem of a planar stack (solved here from the continuity of tangential E
and H, or vanishing tangential E on a mirror, without a Fresnel formula),
the Debye terms of ``references.MieSphere``, or the same beam with an
equivalent hard pupil. Where the method is leading order its error falls as
the system grows; the sizes measured are in each docstring.
"""

import numpy as np
import pytest
import vecdiff as o
from numpy.testing import assert_allclose
from vecdiff.media.medium import Z0

host = o.backend.to_numpy
F, S, P, Y = o.fields, o.surfaces, o.propagation, o.system
Frame = o.geometry.Frame
AIR = o.media.HomogeneousRegion("air", o.media.air)
GLASS = o.media.Dielectric(1.5, "glass")


def relative(got, exact):
    """Relative misfit on the points where the exact field is bright."""
    got, exact = host(got), host(exact)
    size = np.linalg.norm(exact, axis=-1)
    bright = size > 0.2 * size.max()
    return np.linalg.norm(got[bright] - exact[bright]) / np.linalg.norm(exact[bright])


def lattice(half, step=0.4):
    count = int(2 * half / step) // 2 * 2
    axis = (np.arange(count) - count // 2) * step
    return o.sampling.CartesianGrid(axis, axis)


def rings(radius, z, count=30, azimuths=6):
    r, phi = np.meshgrid(
        np.linspace(0, radius, count), 2 * np.pi * np.arange(azimuths) / azimuths
    )
    return o.sampling.PointGrid(
        np.c_[(r * np.cos(phi)).ravel(), (r * np.sin(phi)).ravel(), np.full(r.size, z)]
    )


def smooth_pupil(sine):
    return lambda x, y: np.clip(1 - (x**2 + y**2) / sine**2, 0, None) ** 3


# --- The exact response of a planar path, plane wave by plane wave ---------


def tangents(normal):
    a = np.cross(normal, [1.0, 0, 0])
    if np.linalg.norm(a) < 0.5:
        a = np.cross(normal, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    return a, np.cross(normal, a)


def wave_basis(q, normal):
    """Two vectors transverse to each wavevector (bilinearly, for complex q)."""
    a = np.cross(normal, q)
    small = np.linalg.norm(a, axis=-1) < 1e-9 * np.linalg.norm(q, axis=-1)
    a[small] = tangents(normal)[0]
    return a, np.cross(q, a)


def boundary(k, e, n_i, n_t, normal, kind):
    """The wave a plane through the origin sends on, for waves e exp(i k . r).

    ``kind`` "transmitted" or "reflected": tangential E and k x E continuous;
    "mirror": tangential E vanishes. Rows of waves; returns (k, e).
    """
    k = np.asarray(k, complex)
    along = np.sign(np.real(k @ normal))[:, None] * normal
    k0 = np.sqrt(np.real(np.sum(k * k, -1)))[:, None] / n_i
    kn = np.sum(k * along, -1, keepdims=True)
    tangential = k - kn * along
    k_r = tangential - kn * along
    k_t = (
        tangential
        + np.sqrt((n_t * k0) ** 2 - np.sum(tangential**2, -1, keepdims=True) + 0j)
        * along
    )
    t1, t2 = tangents(normal)

    def tangential_part(v):
        return np.stack((v @ t1, v @ t2), -1)

    a_r, b_r = wave_basis(k_r, along[0])
    if kind == "mirror":
        matrix = np.stack((tangential_part(a_r), tangential_part(b_r)), -1)
        x = np.linalg.solve(matrix, -tangential_part(e)[..., None])[..., 0]
        return k_r, x[:, :1] * a_r + x[:, 1:2] * b_r
    a_t, b_t = wave_basis(k_t, along[0])
    columns = [
        sign
        * np.concatenate((tangential_part(v), tangential_part(np.cross(q, v) / k0)), -1)
        for q, v, sign in ((k_r, a_r, 1), (k_r, b_r, 1), (k_t, a_t, -1), (k_t, b_t, -1))
    ]
    rhs = -np.concatenate(
        (tangential_part(e), tangential_part(np.cross(k, e) / k0)), -1
    )
    x = np.linalg.solve(np.stack(columns, -1), rhs[..., None])[..., 0]
    if kind == "reflected":
        return k_r, x[:, :1] * a_r + x[:, 1:2] * b_r
    return k_t, x[:, 2:3] * a_t + x[:, 3:4] * b_t


def planar_path(beam, steps, radius):
    """The beam's plane waves after planar steps (z of the plane, n_i, n_t, kind)."""
    waves = beam.plane_waves(radius)
    k, e = np.asarray(host(waves.wavevectors), complex), host(waves.amplitudes)
    origin = host(waves.origin)
    for z, n_i, n_t, kind in steps:
        shift = np.array([0.0, 0.0, z]) - origin
        k_out, a = boundary(
            k,
            e * np.exp(1j * (k @ shift))[:, None],
            n_i,
            n_t,
            np.array([0, 0, 1.0]),
            kind,
        )
        k, e = k_out, a * np.exp(-1j * (k_out @ shift))[:, None]
    return k, e, origin


def plane_wave_sum(waves, grid):
    k, e, origin = waves
    phase = np.exp(1j * (host(grid.points) - origin) @ k.T)
    return phase @ e, phase @ np.cross(k, e) / (2 * np.pi * Z0)


# --- Stops ------------------------------------------------------------------


def stopped_and_pupil(a, face, realization):
    """A Gaussian of waist a through a stop of radius a on the face's vertex
    plane, and the same beam with the hard pupil Gaussian x Uniform(a)."""
    surface = S.Plane() if face == "flat" else S.SphericalCap(6 * a)
    depth = 4 * a if face == "flat" else 14.4 * a

    def beam(profile):
        return F.PlaneBeam(profile, F.polarization.x, wavelength0=1.0, region=AIR)

    options = {} if realization == "meridian" else {"lattice": lattice(3 * a)}
    face = Y.Refraction(surface, o.media.air, GLASS)
    gaussian = F.profiles.Gaussian(a)
    stopped = P.Propagator(Y.OpticalSystem(Y.Stop(a), face), **options)(beam(gaussian))
    pupil = P.Propagator(face, **options)(beam(gaussian * F.profiles.Uniform(a)))
    points = rings(1.2 * a, depth)
    error = relative(
        stopped.spectrum.electric(points).values, pupil.spectrum.electric(points).values
    )
    # The incident share outside the hole, on the stop's plane.
    x = np.linspace(-6 * a, 6 * a, 801)
    r2 = np.add.outer(x**2, x**2)
    intensity = np.exp(-2 * r2 / a**2)
    blocked = np.sqrt(intensity[r2 >= a**2].sum() / intensity.sum())
    return error, stopped.reports[0].unsupported_relative_amplitude, blocked


@pytest.mark.parametrize(
    "face, realization, sizes, bound",
    [
        ("flat", "meridian", (10.0, 20.0, 40.0), 4e-4),
        ("lens", "meridian", (10.0, 20.0, 40.0), 2e-3),
        ("flat", "lattice", (10.0, 20.0), 0.02),
        ("lens", "lattice", (10.0, 20.0), 5e-3),
    ],
)
def test_a_stopped_beam_is_the_beam_with_the_equivalent_hard_pupil(
    face, realization, sizes, bound
):
    """The stop cuts the field on its screen and the cut diffracts as a pupil's.

    Measured (radius 10, 20, 40 lambda): flat face 0.27, 0.049, 0.019 % on the
    meridian, 2.1, 1.2, 0.66 % on the lattice (its staircase of the edge);
    lens 0.30, 0.16, 0.10 % and 0.64, 0.23, 0.065 %. The share the stop
    blocks is reported as unsupported.
    """
    errors, unsupported, blocked = zip(
        *(stopped_and_pupil(a, face, realization) for a in sizes)
    )
    assert all(b < 0.8 * a for a, b in zip(errors, errors[1:]))
    assert errors[-1] < bound
    tolerance = 1e-3 if realization == "meridian" else 0.012
    assert all(abs(u / b - 1) < tolerance for u, b in zip(unsupported, blocked))


def test_a_stop_needs_a_positive_hole_and_one_placement():
    with pytest.raises(ValueError, match="positive radius"):
        Y.Stop(0.0)
    with pytest.raises(ValueError, match="not both"):
        Y.Stop(1.0, Frame(), z=2.0)
    stop = Y.Stop(2.0, z=3.0)
    assert_allclose(host(stop.frame.origin), [0, 0, 3.0]) and stop.aperture == 2.0


# --- Plane mirrors: normal incidence, the fold, the retro path --------------


def image_of(beam, points, normal, point):
    """E and H reflected by an ideal plane mirror: -M E(M r) and M H(M r)."""
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    mirror = np.eye(3) - 2 * np.outer(normal, normal)
    p = host(points.points)
    image = p - 2 * ((p - point) @ normal)[:, None] * normal
    field = beam.evaluate(o.sampling.PointGrid(image))
    return -host(field.electric.values) @ mirror.T, host(field.magnetic) @ mirror.T


@pytest.mark.parametrize("realization", ["meridian", "lattice"])
def test_a_collimated_beam_returns_from_a_plane_mirror_as_its_image(realization):
    """Exact to roundoff: every ray returns on itself with g = 1."""
    beam = F.PlaneBeam(
        F.profiles.Gaussian(4.0),
        F.polarization.x,
        wavelength0=1.0,
        region=o.media.air,
        frame=Frame([0, 0, -10.0]),
    )
    options = {} if realization == "meridian" else {"lattice": lattice(25.6)}
    solution = P.Propagator(Y.Reflection(S.Plane(), o.media.air), **options)(beam)
    points = rings(6.0, -12.0)
    E, H = image_of(beam, points, [0, 0, 1], np.zeros(3))
    field = solution.spectrum.evaluate(points)
    assert relative(field.electric.values, E) < 1e-11
    assert relative(field.magnetic, H) < 1e-11
    # In front of the mirror the path's light is the incident and the reflected.
    total = solution.field(points)
    both = E + host(beam.evaluate(points).electric.values)
    assert relative(total.electric.values, both) < 1e-11
    behind = solution.field(rings(3.0, 2.0, 4, 2))
    assert np.all(host(behind.electric.values) == 0) and behind.electric.valid.all()


def test_a_converging_beam_reflected_by_a_plane_mirror_tends_to_its_image():
    """The leading order's error: 1.7, 0.88, 0.44 % at foci 20, 40, 80 lambda
    behind the mirror (NA 0.3)."""
    errors = []
    for size in (20.0, 40.0, 80.0):
        beam = F.FocusedBeam(
            smooth_pupil(0.3),
            F.polarization.x,
            focal_length=1.0,
            max_sine=0.3,
            wavelength0=1.0,
            region=o.media.air,
            frame=Frame([0, 0, size]),
            apodization="none",
        )
        solution = P.Propagator(Y.Reflection(S.Plane(), o.media.air))(beam)
        points = rings(0.4 * size, -0.5 * size)
        E, _ = image_of(beam, points, [0, 0, 1], np.zeros(3))
        errors.append(relative(solution.spectrum.electric(points).values, E))
    assert all(b < 0.6 * a for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 6e-3


def fold(waist, polarization):
    """A collimated beam along z folded to +y by a mirror at 45 degrees."""
    c = 1 / np.sqrt(2)
    tilted = Frame([0, 0, 0.0], np.array([[1, 0, 0], [0, c, c], [0, -c, c]]).T)
    beam = F.PlaneBeam(
        F.profiles.Gaussian(waist),
        polarization,
        wavelength0=1.0,
        region=o.media.air,
        frame=Frame([0, 0, -2.5 * waist]),
    )
    system = Y.OpticalSystem(Y.Reflection(S.Plane(tilted), o.media.air))
    solution = P.Propagator(system, lattice=lattice(7.5 * waist, 0.35))(beam)
    y = np.linspace(1.2, 7.5, 6) * waist
    x = np.linspace(-0.5, 0.5, 5) * waist
    X, Yy = np.meshgrid(x, y)
    points = o.sampling.PointGrid(np.c_[X.ravel(), Yy.ravel(), 0.2 * X.ravel()])
    E, H = image_of(beam, points, host(tilted.normal), np.zeros(3))
    field = solution.spectrum.evaluate(points)
    return relative(field.electric.values, E), relative(field.magnetic, H), solution


@pytest.mark.parametrize("polarization", ["x", "y"])
def test_a_fold_mirror_at_45_degrees_sends_the_beam_along_its_image(polarization):
    """A tilted element, on the lattice: the reflected beam travels along +y.

    The method carries E transverse to the carrier ray; the beam's own E along
    it (of order lambda / waist) is what it misses at 45 degrees: measured
    4.6, 2.5, 1.3 % (x polarization) for waists 4, 8, 16 lambda, E and H
    alike.
    """
    errors = []
    for waist in (4.0, 8.0, 16.0):
        e, h, solution = fold(waist, getattr(F.polarization, polarization))
        assert h < 1.5 * e
        errors.append(e)
    assert all(b < 0.6 * a for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 0.015
    normal = host(solution.legs[0].map.surface.frame.normal)
    assert_allclose(normal, [0, -1 / np.sqrt(2), 1 / np.sqrt(2)], atol=1e-15)


def planar_case(size, kind, realization):
    """A plate 10 size thick in a beam converging 60 size beyond its front:
    ``kind`` "retro" returns from a mirror on its back, "ghost" is the
    plate's first double reflection, both against their plane-wave sums."""
    d = 10.0 * size
    beam = F.FocusedBeam(
        smooth_pupil(0.25),
        F.polarization.x,
        focal_length=1.0,
        max_sine=0.25,
        wavelength0=1.0,
        region=o.media.air,
        frame=Frame([0, 0, 60.0 * size]),
        apodization="none",
    )
    front, back = S.Plane(Frame([0, 0, 0.0])), S.Plane(Frame([0, 0, d]))
    air = o.media.air
    if kind == "retro":
        path = (
            Y.Refraction(front, air, GLASS),
            Y.Reflection(back, GLASS),
            Y.Refraction(front, GLASS, air),
        )
        steps = [(0, 1, 1.5, "transmitted"), (d, 1.5, 1, "mirror")]
        steps += [(0, 1.5, 1, "transmitted")]
        depth = -15.0 * size
    else:
        path = (
            Y.Refraction(front, air, GLASS),
            Y.Reflection(back, GLASS, behind=air),
            Y.Reflection(front, GLASS, behind=air),
            Y.Refraction(back, GLASS, air),
        )
        steps = [(0, 1, 1.5, "transmitted"), (d, 1.5, 1, "reflected")]
        steps += [(0, 1.5, 1, "reflected"), (d, 1.5, 1, "transmitted")]
        depth = d + 15.0 * size
    options = {} if realization == "meridian" else {"lattice": lattice(36 * size)}
    solution = P.Propagator(Y.OpticalSystem(*path), **options)(beam)
    points = rings(11.0 * size, depth, 12, 4)
    E, H = plane_wave_sum(planar_path(beam, steps, 200 * size), points)
    field = solution.spectrum.evaluate(points)
    return relative(field.electric.values, E), relative(field.magnetic, H), solution


@pytest.mark.parametrize(
    "kind, realization, sizes, bound",
    [
        ("retro", "meridian", (1, 2, 4), 7e-4),
        ("retro", "lattice", (1, 2), 1.3e-3),
        ("ghost", "meridian", (1, 2, 4), 3.5e-3),
        ("ghost", "lattice", (1, 2), 7e-3),
    ],
)
def test_a_plate_returns_and_ghosts_its_beam_as_its_plane_wave_sum(
    kind, realization, sizes, bound
):
    """t12 m t21 (retro, mirror on the back) and t12 r21 r21 t21 (ghost) with
    their phases, plane wave by plane wave. The surfaces met from behind are
    met as their reverses, and the axis turns over with the light.

    Measured (sizes 1, 2, 4), lattice and meridian alike: retro 0.22, 0.11,
    0.052 %; ghost 1.1, 0.54, 0.27 %.
    """
    errors = []
    for size in sizes:
        e, h, solution = planar_case(size, kind, realization)
        assert abs(h - e) < 0.05 * e
        errors.append(e)
    assert all(b < 0.6 * a for a, b in zip(errors, errors[1:]))
    assert errors[-1] < bound
    regions = [s.region.name for s in solution.spectra]
    if kind == "ghost":
        assert regions == ["air", "glass", "glass", "glass", "air 2"]
        assert [s.direction for s in solution.spectra[1:4]] == [1, -1, -1]
    else:
        assert regions == ["air", "glass", "glass", "air"]


def test_lattice_and_meridian_carry_the_ghost_alike():
    """Two realizations of the same flips: 1.3e-5 apart on the ghost."""
    a = planar_case(1, "ghost", "meridian")[2]
    b = planar_case(1, "ghost", "lattice")[2]
    points = rings(11.0, 25.0, 12, 4)
    assert (
        relative(b.spectrum.electric(points).values, a.spectrum.electric(points).values)
        < 3e-5
    )


# --- A closed ball: its external reflection and one internal reflection ------


def mie(beam, radius, points, term):
    from vecdiff.references import MieSphere

    sphere = o.volumes.SphereVolume([0, 0, 0], radius)
    glass = o.media.HomogeneousRegion("glass", GLASS)
    exact = MieSphere(beam, sphere, glass).evaluate(points, term=term, side="exterior")
    return exact.electric.values


def external_reflection(radius, realization):
    beam = F.PlaneBeam(
        F.profiles.Gaussian(radius / 5),
        F.polarization.x,
        wavelength0=1.0,
        region=AIR,
        frame=Frame([0, 0, -1.25 * radius]),
    )
    ball = S.Sphere(radius, Frame([0, 0, -radius]))
    options = {"fans": ((30.0, 40.0),)}
    if realization == "lattice":
        options = {"lattice": lattice(0.62 * radius)}
    solution = P.Propagator(
        Y.Reflection(ball, o.media.air, behind=GLASS),
        planes=(-radius / 4,),
        **options,
    )(beam)
    points = rings(0.8 * radius, -1.25 * radius - 2, 40)
    exact = mie(beam, radius, points, 0)
    return relative(solution.spectrum.electric(points).values, exact)


def internal_reflection(radius):
    """In through the ball's front, back from its far side, out through its
    front: observed outside, in front of the ball. A stop bounds the fan."""
    beam = F.FocusedBeam(
        smooth_pupil(0.25),
        F.polarization.x,
        focal_length=1.0,
        max_sine=0.25,
        wavelength0=1.0,
        region=AIR,
        frame=Frame([0, 0, 1.5 * radius]),
        apodization="none",
    )
    ball = S.Sphere(radius, Frame([0, 0, -radius]))
    air = o.media.air
    system = Y.OpticalSystem(
        Y.Stop(0.8 * radius, z=-1.05 * radius),
        Y.Refraction(ball, air, GLASS),
        Y.Reflection(ball, GLASS, behind=air),
        Y.Refraction(ball, GLASS, air),
    )
    half = np.tan(np.arcsin(0.25)) * 2.6 * radius + 3
    solution = P.Propagator(system, lattice=lattice(half))(beam)
    points = rings(0.9 * radius, -1.6 * radius, 40)
    exact = mie(beam, radius, points, 2)
    error = relative(solution.spectrum.electric(points).values, exact)
    return error, (solution, system, beam)


@pytest.mark.parametrize("realization", ["meridian", "lattice"])
def test_a_ball_reflects_its_debye_term_zero_through_the_propagator(realization):
    """Measured 29, 14, 7.2, 3.6 % (R = 10, 20, 40, 80 lambda) on both."""
    sizes = (10.0, 20.0, 40.0, 80.0) if realization == "meridian" else (10.0, 20.0)
    errors = [external_reflection(r, realization) for r in sizes]
    assert all(b < 0.55 * a for a, b in zip(errors, errors[1:]))
    assert errors[-1] < (0.04 if realization == "meridian" else 0.16)


def test_a_ball_reflects_once_inside_as_its_exterior_debye_term_two():
    """Refraction in, reflection from the far side, refraction out of one closed
    Sphere, which the path meets as itself, its far side and its reverse.

    Measured 12.5, 4.8, 2.1, 0.90 % (R = 10, 20, 40, 80 lambda).
    """
    errors, cases = zip(*(internal_reflection(r) for r in (10.0, 20.0, 40.0)))
    assert all(b < 0.6 * a for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 0.03
    solution, system, beam = cases[0]
    views = [type(e.surface).__name__ for e in solution.elements]
    radii = [getattr(e.surface, "radius", None) for e in solution.elements]
    assert views == ["Plane", "Sphere", "Sphere", "Sphere"]
    assert radii[1:] == [10.0, -10.0, -10.0]
    origins = [host(e.surface.frame.origin)[2] for e in solution.elements[1:]]
    assert_allclose(origins, [-10.0, 10.0, -10.0])
    # Inside, the light focuses before the far side: not on the meridian.
    with pytest.raises(ValueError, match="focuses between"):
        P.Propagator(system)(beam)


# --- A two-mirror system of revolution --------------------------------------


def cassegrain(scale):
    """A paraboloid (focus 50 scale in front) and a hyperboloid 40 scale in front
    of it, imaging that focus on 20 scale behind the paraboloid."""
    primary = S.EvenAsphere(-0.01 / scale, conic=-1.0)
    secondary = S.EvenAsphere(
        -1 / (24 * scale), conic=-1.96, frame=Frame([0, 0, -40.0 * scale])
    )
    air = o.media.air
    system = Y.OpticalSystem(
        Y.Stop(36.0 * scale, z=-10.0 * scale),
        Y.Reflection(primary, air),
        Y.Reflection(secondary, air),
    )
    beam = F.PlaneBeam(
        F.profiles.Gaussian(6.0 * scale),
        F.polarization.x,
        wavelength0=1.0,
        region=air,
        frame=Frame([0, 0, -60.0 * scale]),
    )
    return system, beam, np.array([0, 0, 20.0 * scale])


def test_a_cassegrain_images_a_collimated_beam_on_its_focus():
    """Stigmatic to roundoff; the meridian and the lattice carry one field to the
    focus (1e-8 apart), and the ideal mirrors carry its power there."""
    system, beam, focus = cassegrain(2.0)
    meridian = P.Propagator(system)(beam)
    points, directions = (host(a) for a in meridian.legs[-1].footprint())
    offset = focus - points
    along = np.sum(offset * directions, -1)
    miss = np.linalg.norm(offset - along[:, None] * directions, axis=-1)
    assert miss.max() < 1e-11 * 120 and (along > 0).all()
    assert [type(e.surface).__name__ for e in meridian.elements] == [
        "Plane",
        "EvenAsphere",
        "EvenAsphere",
    ]
    assert meridian.elements[2].surface.curvature > 0  # met from behind
    on_lattice = P.Propagator(system, lattice=lattice(80.0))(beam)
    x = np.linspace(-3, 3, 13)
    near = o.sampling.PointGrid(np.c_[x, 0.3 * x, np.full(13, focus[2])])
    assert (
        relative(
            on_lattice.spectrum.electric(near).values,
            meridian.spectrum.electric(near).values,
        )
        < 1e-6
    )
    # The ideal mirrors carry the beam's power to the focal plane, to the
    # leading order: the deficit falls as the square of the amplitude error.
    deficits = [power_deficit(scale) for scale in (1.0, 2.0)]
    assert deficits[1] < 0.3 * deficits[0] and deficits[1] < 0.012


def flux(field):
    """Power through a plane grid, in W: the irradiance summed over its cells."""
    irradiance = host(o.observables.irradiance(field))
    return float(irradiance.sum() * np.prod(field.grid.spacing))


def power_deficit(scale):
    """1 - power at the Cassegrain's focal plane / power of the beam.

    Measured 4.1, 1.1, 0.27 % at scales 1, 2, 4.
    """
    system, beam, focus = cassegrain(scale)
    solution = P.Propagator(system)(beam)
    side = np.linspace(-40 * scale, 40 * scale, int(160 * scale) + 1)
    incident = beam.evaluate(S.Plane.z(-60.0 * scale).sample(u=side, v=side))
    focal = solution.spectrum.evaluate(S.Plane.z(focus[2]).sample(u=side, v=side))
    return 1 - flux(focal) / flux(incident)
