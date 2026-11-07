"""Opt-in slow regression: the manuscript's spheres, eikonal transport vs Mie.

Run with VECDIFF_SLOW=1. A circular-input beam converges toward A = 1.5 R
(u_max = 0.4, Appendix C apodization) and enters a sphere n = 1 -> 1.5; the
eikonal pipeline's field on the 5 lambda0 focal disk at B = R / 1.5 is compared
with the sphere's exact first-entry (Debye p = 1) interior field, and with the
full field. The lattice leg runs at 30-50 lambda0, the axisymmetric leg at
1000 lambda0, with the non-aplanatic sphere (A = 2 R, B = 0.8 R) too.
"""

import functools
import os

import numpy as np
import pytest
import vecdiff_general as o
from scipy.integrate import simpson
from vecdiff_general.propagation.references import MieSphere

pytestmark = pytest.mark.skipif(
    os.environ.get("VECDIFF_SLOW") != "1", reason="slow regression: VECDIFF_SLOW=1"
)
host = o.backend.to_numpy
AIR = o.media.HomogeneousRegion("air", o.media.Dielectric(1.0))
GLASS = o.media.HomogeneousRegion("glass", o.media.Dielectric(1.5))
U_MAX = 0.4


def first_entry_error(radius, pupil):
    """Relative L2 error in percent on the focal disk, and the leg's report."""
    focal = 1.0
    profile = (
        o.fields.profiles.Uniform(U_MAX * focal)
        if pupil == "hard"
        else lambda x, y: (
            np.clip(1 - (x**2 + y**2) / (U_MAX * focal) ** 2, 0, None) ** 3
        )
    )
    beam = o.fields.FocusedBeam(
        profile,
        o.fields.polarization.cartesian(1 / np.sqrt(2), 1j / np.sqrt(2)),
        focal_length=focal,
        max_sine=U_MAX,
        wavelength0=1.0,
        region=AIR,
        frame=o.geometry.Frame([0, 0, 1.5 * radius]),
        apodization="none",
    )
    z0 = -radius - 1.0
    half = 1.35 * (1.5 * radius - z0) * np.tan(np.arcsin(U_MAX)) + 12.0
    count = int(np.ceil(2 * half / 0.25))
    count += count % 2
    axis = (np.arange(count) - count // 2) * 0.25
    plane = o.sampling.CartesianGrid(axis, axis, o.geometry.Frame([0, 0, z0]))
    sampled = beam.evaluate(plane, method="nufft").electric.values[..., :2]
    incident = o.fields.TransverseElectricField(plane, 1.0, sampled)
    sphere = o.surfaces.SphericalCap(radius, o.geometry.Frame([0, 0, -radius]))
    pipeline = o.propagation.PropagationPipeline(
        (o.system.DielectricInterface(sphere, AIR, GLASS),), transport="eikonal"
    )
    solution = pipeline.solve(
        o.fields.angular_spectrum(incident, AIR, evanescent="discard"),
        carrier=beam.carrier,
    )
    disk = np.arange(-5, 5.0625, 0.125)
    x, y = np.meshgrid(disk, disk)
    keep = np.hypot(x, y) <= 5
    points = o.sampling.PointGrid(
        np.c_[x[keep], y[keep], np.full(keep.sum(), radius / 1.5)]
    )
    field = host(solution.evaluate(points, method="nufft").electric.values)
    mie = MieSphere(beam, o.volumes.SphereVolume([0, 0, 0], radius), GLASS)
    exact = host(mie.evaluate(points, term=1, side="interior").electric.values)
    error = 100 * np.linalg.norm(field - exact) / np.linalg.norm(exact)
    return error, solution.interactions[0]


@pytest.mark.parametrize(
    "radius, pupil, expected, unsupported",
    [
        (30.0, "smooth", 0.3102, 1e-4),
        (50.0, "smooth", 0.1755, 1e-4),
        (50.0, "hard", 3.567, 0.05),
    ],
)
def test_aplanatic_sphere_first_entry_matches_the_archived_method(
    radius, pupil, expected, unsupported
):
    """The archived prototype gives 0.3102, 0.1755 and 3.64-3.69 % at these radii.

    Smooth pupils agree to four digits. The hard pupil depends on how the
    sharp-edged beam is represented and on the reference fan, as in the archive.
    """
    error, report = first_entry_error(radius, pupil)
    assert abs(error - expected) < 0.01 * expected
    assert report.unsupported_relative_amplitude < unsupported


# The axisymmetric leg at R = 1000 lambda0: the manuscript's Table 1 sizes.
GEOMETRIES = {
    "aplanatic": (1.5, 1 / 1.5, 0.4),
    "nonaplanatic": (2.0, 0.8, 0.3483228172),
}


def manuscript_beam(radius, geometry, pupil):
    """Circular input toward A (Appendix C), hard or (1 - (u / u_max)²)³ pupil."""
    focus, _, umax = GEOMETRIES[geometry]
    profile = (
        o.fields.profiles.Uniform(umax)
        if pupil == "hard"
        else lambda x, y: np.clip(1 - (x**2 + y**2) / umax**2, 0, None) ** 3
    )
    return o.fields.FocusedBeam(
        profile,
        o.fields.polarization.cartesian(1 / np.sqrt(2), 1j / np.sqrt(2)),
        focal_length=1.0,
        max_sine=umax,
        wavelength0=1.0,
        region=AIR,
        frame=o.geometry.Frame([0, 0, focus * radius]),
        apodization="none",
    )


def rim_offset(radius, geometry):
    """Depth below the vertex where the marginal ray meets the sphere."""
    focus, _, umax = GEOMETRIES[geometry]
    start = np.array([(focus + 1) * np.tan(np.arcsin(umax)), -1.0])
    direction = np.array([-start[0], focus + 1])
    direction /= np.linalg.norm(direction)
    b, c = start @ direction, start @ start - 1
    return (start[1] + (-b - np.sqrt(b * b - c)) * direction[1] + 1) * radius


@functools.lru_cache(maxsize=None)
def mie_focal_line(radius, geometry, pupil):
    """First-entry and full Mie fields on 81 radii of the focal plane."""
    _, image, _ = GEOMETRIES[geometry]
    rho = np.linspace(0, 5, 81)
    points = o.sampling.PointGrid(np.c_[rho, 0 * rho, np.full(81, image * radius)])
    mie = MieSphere(
        manuscript_beam(radius, geometry, pupil),
        o.volumes.SphereVolume([0, 0, 0], radius),
        GLASS,
    )
    return (
        rho,
        points,
        {
            "first": host(
                mie.evaluate(points, term=1, side="interior").electric.values
            ),
            "full": host(mie.evaluate(points, side="interior").electric.values),
        },
    )


@pytest.mark.parametrize(
    "geometry, pupil, plane, fan, expected",
    [
        (
            "aplanatic",
            "hard",
            "vertex",
            (50.0, 56.0),
            {"first": 0.1418, "full": 0.3865},
        ),
        ("aplanatic", "smooth", "vertex", None, {"first": 0.008515}),
        ("aplanatic", "hard", "rim", (50.0, 56.0), {"first": 0.01524}),
        (
            "nonaplanatic",
            "hard",
            "vertex",
            (55.0, 63.0),
            {"first": 4.036, "full": 4.068},
        ),
    ],
)
def test_axisymmetric_leg_meets_the_manuscript_spheres_at_a_thousand_wavelengths(
    geometry, pupil, plane, fan, expected
):
    """Raw complex-field L2 error in percent on the 5 lambda0 focal disk (Eq. 8.2).

    The archived axisymmetric code gives 0.1418 / 0.3865 (hard), 0.008515
    (smooth) and 0.01524 (rim plane) for the aplanatic sphere; the manuscript's
    Table 1, 0.1421 / 0.3874. A SphericalCap is a graph and ends at its
    equator; the fans close before it. The archived code carried rays on to
    80 degrees of incidence, into the ball beyond the equator: that changes
    nothing for the aplanatic sphere but gives 4.115 / 4.143 for the
    non-aplanatic one (Table 1: 4.127 / 4.155). With the fan cut where the cap
    ends, the archived code and this leg agree to 1e-5 (4.0422 / 4.0742).
    """
    radius = 1000.0
    offset = 0.0 if plane == "vertex" else rim_offset(radius, geometry)
    pipeline = o.propagation.PropagationPipeline(
        (
            o.system.DielectricInterface(
                o.surfaces.SphericalCap(radius, o.geometry.Frame([0, 0, -radius])),
                AIR,
                GLASS,
            ),
        ),
        transport="eikonal",
        reference_planes=(offset,),
        reference_fans=(fan,),
    )
    rho, points, references = mie_focal_line(radius, geometry, pupil)
    solution = pipeline.solve(manuscript_beam(radius, geometry, pupil))
    field = host(solution.evaluate(points).electric.values)
    assert solution.interactions[0].azimuthal_orders == (1,)

    def norm(E):
        return simpson(rho * np.sum(abs(E) ** 2, -1), x=rho)

    for term, value in expected.items():
        reference = references[term]
        error = 100 * np.sqrt(norm(field - reference) / norm(reference))
        assert abs(error - value) < 2e-3 * value


# Two concentric surfaces at R1 = 1000 lambda0 (the handoff's shell cases):
# indices outside / shell / core, A / R1, R2 / R1, u_max.
SHELLS = {
    "aplanatic pair": (1.0, 1.5, 2.0, 1.5, 0.5, 0.4),
    "meniscus": (1.0, 1.5, 1.0, 0.75, 0.85, 0.65),
}


def marginal_ray(geometry, u):
    """Unit-sphere meridional trace: surface points, and the axis crossing after both."""
    n0, n1, n2, focus, inner, _ = SHELLS[geometry]

    def refract(d, normal, ratio):
        c = -normal @ d
        return ratio * d + (ratio * c - np.sqrt(1 - ratio**2 * (1 - c**2))) * normal

    d = np.array([-np.sin(np.arcsin(u)), np.cos(np.arcsin(u))])
    p = np.array([0.0, focus]) - 5 * d
    b = p @ d
    q1 = p + (-b - np.sqrt(b * b - (p @ p - 1))) * d
    d1 = refract(d, q1, n0 / n1)
    b = q1 @ d1
    q2 = q1 + (-b - np.sqrt(b * b - (q1 @ q1 - inner**2))) * d1
    d2 = refract(d1, q2 / inner, n1 / n2)
    return q1, q2, q2[1] - q2[0] * d2[1] / d2[0]


def shell_beam(radius, geometry, pupil):
    _, _, _, focus, _, umax = SHELLS[geometry]
    profile = (
        o.fields.profiles.Uniform(umax)
        if pupil == "hard"
        else lambda x, y: np.clip(1 - (x**2 + y**2) / umax**2, 0, None) ** 3
    )
    return o.fields.FocusedBeam(
        profile,
        o.fields.polarization.cartesian(1 / np.sqrt(2), 1j / np.sqrt(2)),
        focal_length=1.0,
        max_sine=umax,
        wavelength0=1.0,
        region=AIR,
        frame=o.geometry.Frame([0, 0, focus * radius]),
        apodization="none",
    )


@functools.lru_cache(maxsize=None)
def layered_focal_line(radius, geometry, pupil):
    """First entry through both surfaces on 81 radii of the image plane."""
    from vecdiff_general.propagation.references import LayeredSphere

    n0, n1, n2, _, inner, _ = SHELLS[geometry]
    shell = o.media.HomogeneousRegion("shell", o.media.Dielectric(n1))
    core = o.media.HomogeneousRegion("core", o.media.Dielectric(n2))
    image = marginal_ray(geometry, 1e-5)[2] * radius
    rho = np.linspace(0, 5, 81)
    points = o.sampling.PointGrid(np.c_[rho, 0 * rho, np.full(81, image)])
    sphere = LayeredSphere(
        shell_beam(radius, geometry, pupil),
        tuple(o.volumes.SphereVolume([0, 0, 0], r * radius) for r in (1, inner)),
        (shell, core),
    )
    reference = host(sphere.evaluate(points, term=1, side="interior").electric.values)
    return rho, points, (shell, core), reference


@pytest.mark.parametrize(
    "geometry, pupil, planes, fans, expected",
    [
        ("aplanatic pair", "smooth", "vertex", (None, None), 0.01914),
        ("aplanatic pair", "smooth", "rim", (None, None), 0.02049),
        ("meniscus", "smooth", "vertex", (None, (34.0, 38.0)), 0.001924),
        ("meniscus", "smooth", "rim", (None, (34.0, 38.0)), 0.001038),
        ("meniscus", "hard", "vertex", (None, (34.0, 38.0)), 0.02670),
    ],
)
def test_two_surfaces_meet_the_layered_sphere_at_a_thousand_wavelengths(
    geometry, pupil, planes, fans, expected
):
    """First-entry error in percent on the 5 lambda0 image disk, both surfaces carried.

    The archived two-surface code gives 0.0191 / 0.0205 (aplanatic pair,
    smooth pupil, vertex / rim-depth planes), 0.00192 / 0.00104 (meniscus,
    smooth) and 0.0260 (meniscus, hard, vertex). The meniscus's second fan
    stops before total reflection, as there. With a hard pupil the aplanatic
    pair's marginal ray meets the inner cap at its equator and its tails go on
    beyond, which a graph cap does not carry (the archive: 0.883 / 0.0795).
    """
    radius = 1000.0
    _, _, _, _, inner, umax = SHELLS[geometry]
    rho, points, (shell, core), reference = layered_focal_line(radius, geometry, pupil)
    caps = [
        o.surfaces.SphericalCap(r * radius, o.geometry.Frame([0, 0, -r * radius]))
        for r in (1, inner)
    ]
    if planes == "vertex":
        offsets = (0.0, 0.0)
    else:
        q1, q2, _ = marginal_ray(geometry, umax)
        offsets = ((q1[1] + 1) * radius, (q2[1] + inner) * radius)
    pipeline = o.propagation.PropagationPipeline(
        (
            o.system.DielectricInterface(caps[0], AIR, shell),
            o.system.DielectricInterface(caps[1], shell, core),
        ),
        transport="eikonal",
        reference_planes=offsets,
        reference_fans=fans,
    )
    solution = pipeline.solve(shell_beam(radius, geometry, pupil))
    field = host(solution.evaluate(points).electric.values)
    assert [r.azimuthal_orders for r in solution.interactions] == [(1,), (1,)]

    def norm(E):
        return simpson(rho * np.sum(abs(E) ** 2, -1), x=rho)

    error = 100 * np.sqrt(norm(field - reference) / norm(reference))
    assert abs(error - expected) < 2e-3 * expected


# The closed Sphere at R = 1000 lambda0, with the archive's fans: incidence 72 to
# 80 degrees, and for a beam converging inside the sphere (the meniscus) a radial
# roll-off from the ray at sin(theta) = 0.845 to the archive's 3 R scan limit.
def radial_fan(start, end):
    def window(rays):
        radius = np.hypot(host(rays.start)[:, 0], host(rays.start)[:, 1])
        fraction = np.clip((radius - start) / (end - start), 0, 1)
        return 0.5 * (1 + np.cos(np.pi * fraction))

    return window


def test_closed_sphere_meets_the_archived_non_aplanatic_sphere():
    """Table 1: 4.127 / 4.155; the archive 4.115 / 4.143 (its rays reach 80 degrees)."""
    radius = 1000.0
    rho, points, references = mie_focal_line(radius, "nonaplanatic", "hard")
    pipeline = o.propagation.PropagationPipeline(
        (
            o.system.DielectricInterface(
                o.surfaces.Sphere(radius, o.geometry.Frame([0, 0, -radius])), AIR, GLASS
            ),
        ),
        transport="eikonal",
        reference_fans=((72.0, 80.0),),
    )
    solution = pipeline.solve(manuscript_beam(radius, "nonaplanatic", "hard"))
    field = host(solution.evaluate(points).electric.values)

    def norm(E):
        return simpson(rho * np.sum(abs(E) ** 2, -1), x=rho)

    for term, value in (("first", 4.1145), ("full", 4.1428)):
        reference = references[term]
        error = 100 * np.sqrt(norm(field - reference) / norm(reference))
        assert abs(error - value) < 2e-3 * value


@pytest.mark.parametrize(
    "geometry, planes, expected",
    [
        ("aplanatic pair", "vertex", 0.8832),
        ("aplanatic pair", "rim", 0.07967),
        ("meniscus", "vertex", 0.02603),
    ],
)
def test_closed_spheres_meet_the_archived_hard_pupil_shells(geometry, planes, expected):
    """The archive gives 0.883 / 0.0795 (aplanatic pair) and 0.0260 (meniscus, vertex)."""
    radius = 1000.0
    n0, n1, n2, focus, inner, umax = SHELLS[geometry]
    rho, points, (shell, core), reference = layered_focal_line(radius, geometry, "hard")
    spheres = [
        o.surfaces.Sphere(r * radius, o.geometry.Frame([0, 0, -r * radius]))
        for r in (1, inner)
    ]
    if planes == "vertex":
        offsets = (0.0, 0.0)
    else:
        q1, q2, _ = marginal_ray(geometry, umax)
        offsets = ((q1[1] + 1) * radius, (q2[1] + inner) * radius)
    if geometry == "meniscus":
        depth = focus * radius + radius - offsets[0]
        taper = 0.845 * depth / np.sqrt(1 - 0.845**2)
        fans = (radial_fan(taper, 3 * radius), (34.0, 38.0))
    else:
        fans = ((72.0, 80.0), (72.0, 80.0))
    pipeline = o.propagation.PropagationPipeline(
        (
            o.system.DielectricInterface(spheres[0], AIR, shell),
            o.system.DielectricInterface(spheres[1], shell, core),
        ),
        transport="eikonal",
        reference_planes=offsets,
        reference_fans=fans,
    )
    solution = pipeline.solve(shell_beam(radius, geometry, "hard"))
    field = host(solution.evaluate(points).electric.values)

    def norm(E):
        return simpson(rho * np.sum(abs(E) ** 2, -1), x=rho)

    error = 100 * np.sqrt(norm(field - reference) / norm(reference))
    assert abs(error - expected) < 2e-3 * expected


def archive_fan(umax, start, end):
    """The archive's lattice fan: sin(gamma) rolled off from start to end u_max,
    incidence from 78 to 85 degrees, and transmitted rays with s_t,z > 0.05."""
    from vecdiff_general.propagation.eikonal import ray_weights

    def window(rays):
        s = host(rays.incident)
        t = np.clip(
            (np.hypot(s[:, 0], s[:, 1]) - start * umax) / ((end - start) * umax), 0, 1
        )
        weight = 0.5 * (1 + np.cos(np.pi * t)) * host(ray_weights((78.0, 85.0), rays))
        return weight * (host(rays.transmitted)[:, 2] > 0.05)

    return window


def archive_lattice(beam, radius, count):
    axis = (np.arange(count) - count // 2) / 3
    plane = o.sampling.CartesianGrid(axis, axis, o.geometry.Frame([0, 0, -radius]))
    sampled = beam.evaluate(plane, method="nufft").electric.values[..., :2]
    return o.fields.angular_spectrum(
        o.fields.TransverseElectricField(plane, 1.0, sampled), AIR, evanescent="discard"
    )


@pytest.mark.parametrize(
    "radius, count, sampled, expected",
    [
        (100.0, 1186, True, (0.1930, 0.05519)),
        (300.0, 3556, False, (0.06420, 0.01834)),
    ],
)
def test_cylinder_meets_the_handoff(radius, count, sampled, expected):
    """The archive's cylinder rows: 0.193 / 0.0642 % (hard) and 0.0552 / 0.0183 %
    (smooth) at R = 100 / 300.

    One leg serves both pupils. At 100 the beam is sampled on the lattice; at
    300 it is read at the preimages (lattice=) and the streamed leg carries a
    3556² lattice. R = 1000 (11852²: 0.01918 / 0.00550 % against the archive's
    0.0192 / 0.0055) is left out: its two references alone take a quarter of
    an hour.
    """
    from vecdiff_general.propagation.references import InfiniteCylinder

    umax = 0.25
    axis = (np.arange(count) - count // 2) / 3
    plane = o.sampling.CartesianGrid(axis, axis, o.geometry.Frame([0, 0, -radius]))
    surface = o.surfaces.CylindricalCap(radius, o.geometry.Frame([0, 0, -radius]))
    pipeline = o.propagation.PropagationPipeline(
        (o.system.DielectricInterface(surface, AIR, GLASS),),
        transport="eikonal",
        reference_fans=(archive_fan(umax, 1.7, 1.9),),
        lattice=None if sampled else plane,
    )
    carrier = o.fields.Carrier.converging([0, 0, 1.5 * radius])
    xs = np.linspace(-3.5, 3.5, 140)
    x, y = np.meshgrid(xs, xs, indexing="ij")
    disk = np.hypot(x, y) <= 3.5
    points = o.sampling.PointGrid(
        np.c_[x[disk], y[disk], np.full(disk.sum(), radius / 1.5)]
    )
    for pupil, value in zip(("hard", "smooth"), expected):
        beam = o.fields.FocusedBeam(
            o.fields.profiles.Uniform(umax)
            if pupil == "hard"
            else lambda x, y: np.clip(1 - (x**2 + y**2) / umax**2, 0, None) ** 3,
            o.fields.polarization.cartesian(1 / np.sqrt(2), 1j / np.sqrt(2)),
            focal_length=1.0,
            max_sine=umax,
            wavelength0=1.0,
            region=AIR,
            frame=o.geometry.Frame([0, 0, 1.5 * radius]),
            apodization="none",
        )
        source = archive_lattice(beam, radius, count) if sampled else beam
        solution = pipeline.solve(source, carrier=carrier)
        field = host(solution.evaluate(points).electric.values)
        del solution, source
        pipeline.last_solution = None
        cylinder = InfiniteCylinder(
            beam, o.volumes.CylinderVolume([0, 0, 0], [0, 1, 0], radius), GLASS
        )
        exact = host(cylinder.evaluate(points, term=1, side="interior").electric.values)
        error = 100 * np.linalg.norm(field - exact) / np.linalg.norm(exact)
        assert abs(error - value) < 2e-3 * value


@pytest.mark.parametrize(
    "radius, count, sampled, expected",
    [(30.0, 678, True, 0.2499), (300.0, 6778, False, 8.004e-5)],
)
def test_oval_lattice_and_meridian_agree_as_the_archives_codes_do(
    radius, count, sampled, expected
):
    """The archive's oval (virtual object A = 2.5 R behind the vertex, focus 1.8 R).

    The archive's general and axisymmetric codes differ by 0.2424 % at R = 30
    and 0.115 % at 300 (its 6778² lattice); vecdiff's lattice and meridian
    legs by 0.2499 % and 8.0e-5 %. All of vecdiff's difference is the two fans':
    the meridian with the lattice's fan differs from its own by the same
    amount, and that falls as the hard pupil's shadow fades (0.0017 % at 100,
    where the archive's codes still differ by 0.056 %). At 30 the beam is
    sampled on the lattice; at 300 it is read at the preimages (lattice=) by
    the streamed leg.
    """
    umax = 0.4
    beam = manuscript_beam(radius, "aplanatic", "hard")
    ovoid = o.surfaces.CartesianOvoid.from_conjugates(
        1.0, 2.5 * radius, 1.5, 1.8 * radius, frame=o.geometry.Frame([0, 0, -radius])
    )
    system = (o.system.DielectricInterface(ovoid, AIR, GLASS),)
    # The archive's radial taper for the axisymmetric code: from the 72 to the
    # 80 degree ray, or its 3 R scan limit.
    table = np.linspace(0, 3 * radius, 30001)
    cosine = host(
        o.propagation.InterfaceMap(ovoid, beam.carrier, 1.0, 1.5)
        .forward(np.c_[table, 0 * table])
        .cos_incident
    )
    incidence = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    start = table[np.nanargmax(np.where(incidence <= 72, table, np.nan))]
    end = table[np.nanargmax(np.where(incidence <= 80, table, np.nan))]
    radial = radial_fan(start, end)
    xs = np.linspace(-3.5, 3.5, 140)
    x, y = np.meshgrid(xs, xs, indexing="ij")
    disk = np.hypot(x, y) <= 3.5
    points = o.sampling.PointGrid(
        np.c_[x[disk], y[disk], np.full(disk.sum(), 0.8 * radius)]
    )
    meridian = o.propagation.PropagationPipeline(
        system, transport="eikonal", reference_fans=(radial,)
    ).solve(beam)
    reference = host(meridian.evaluate(points).electric.values)
    axis = (np.arange(count) - count // 2) / 3
    plane = o.sampling.CartesianGrid(axis, axis, o.geometry.Frame([0, 0, -radius]))
    lattice = o.propagation.PropagationPipeline(
        system,
        transport="eikonal",
        reference_fans=(archive_fan(umax, 1.45, 1.6),),
        lattice=None if sampled else plane,
    ).solve(
        archive_lattice(beam, radius, count) if sampled else beam,
        carrier=beam.carrier,
    )
    field = host(lattice.evaluate(points).electric.values)
    difference = 100 * np.linalg.norm(field - reference) / np.linalg.norm(reference)
    assert abs(difference - expected) < 5e-3 * expected
