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
