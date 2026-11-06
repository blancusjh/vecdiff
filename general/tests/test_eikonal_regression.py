"""Opt-in slow regression: the aplanatic sphere of the manuscript, eikonal transport vs Mie.

Run with VECDIFF_SLOW=1. A circular-input beam converges toward A = 1.5 R
(u_max = 0.4, Appendix C apodization) and enters a sphere n = 1 -> 1.5; the
eikonal pipeline's field on the 5 lambda0 focal disk at B = R / 1.5 is compared
with the sphere's exact first-entry (Debye p = 1) interior field.
"""

import os

import numpy as np
import pytest
import vecdiff_general as o
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
