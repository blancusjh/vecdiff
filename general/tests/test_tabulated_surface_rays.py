"""A four-surface aplanatic lens of tabulated Cartesian ovals, object at a finite distance.

The lens is Table 1 of Silva-Lora and Torres, "Aplanatism in stigmatic
optical systems" (J. Opt. Soc. Am. A, 2020), at one tenth of its size: object
100 mm in front, image 90 mm behind, magnification -0.602. Its faces are
splined from the ovals' sags, whose height is smooth only to roundoff: a
Newton step of the ray-surface crossing can then stay a few ulps above its
bound, and such rays were marked lost, one here and there, which broke the
legs behind the first.
"""

import numpy as np
import vecdiff as vd
from scipy.interpolate import make_interp_spline

MM = 1e-4
VERTICES = (0.0, 15.0, 25.0, 35.0)
INDICES = (1.0, 1.517122, 1.670591, 1.851280, 1.0)
CURVATURES = (0.025478, 0.001985, 0.017027, -0.013144)  # the table's O_k, 1/mm


def conjugates():
    """Object and image of each face, from the table's paraxial curvatures."""
    d = [-100.0]
    for vertex, curvature, before, after in zip(
        VERTICES, CURVATURES, INDICES, INDICES[1:]
    ):
        power = curvature * (after - before) + before / (d[-1] - vertex)
        d.append(vertex + after / power)
    return d


def lens():
    d = conjugates()
    media = [vd.media.Dielectric(n, f"n {n:.3f}") for n in INDICES]
    r = np.linspace(0, 21.5, 1721)
    elements = [vd.system.Stop(20 * MM, z=-MM)]
    for k, vertex in enumerate(VERTICES):
        oval = vd.surfaces.CartesianOval.from_conjugates(
            INDICES[k], d[k] - vertex, INDICES[k + 1], d[k + 1] - vertex
        )
        slope = make_interp_spline(r * MM, oval.gradient(np.c_[r, 0 * r])[:, 0], k=5)
        face = vd.surfaces.AxisymmetricSurface(
            make_interp_spline(r * MM, oval.design.sag(r) * MM, k=5),
            slope,
            vd.geometry.Frame([0, 0, vertex * MM]),
            slope.derivative(),
        )
        elements.append(vd.system.Refraction(face, media[k], media[k + 1]))
    return vd.system.OpticalSystem(*elements), d[-1] * MM


def test_point_source_is_imaged_at_the_conjugate_of_the_aplanatic_lens():
    system, image = lens()
    assert abs(image - 90 * MM) < 1e-3 * MM
    source = vd.fields.FocusedBeam(
        vd.fields.profiles.Uniform(25 * MM),
        vd.fields.polarization.x,
        wavelength0=532e-9,
        region=system.elements[0].incident,
        focal_length=100 * MM,
        max_sine=0.25,
        frame=vd.geometry.Frame([0, 0, -100 * MM]),
    )
    solution = vd.propagation.Propagator(system)(
        source, carrier=vd.fields.Carrier.diverging([0, 0, -100 * MM])
    )

    assert all(report.caustics == 0 for report in solution.reports)
    assert all(np.isfinite(report.newton_residual) for report in solution.reports)
    z = image + np.linspace(-40e-6, 40e-6, 41)
    axis = solution.field(vd.sampling.PointGrid(np.c_[0 * z, 0 * z, z]))
    brightest = z[np.argmax(np.sum(np.abs(axis.electric.values) ** 2, axis=-1))]
    assert abs(brightest - image) < 2.1e-6

    w = 4e-6
    focal = solution.field(z=image, window=(-w, w, -w, w), shape=(81, 81))
    peak = np.abs(focal.electric.values).max(axis=(0, 1))
    # The ideal aplanatic lens of this aperture gives 6.2e-3 and 0.1205: the
    # four uncoated faces pass p better than s, which nearly cancels the
    # cross-polarized field (a ray sum with their Fresnel coefficients gives
    # 9.0e-4 and 0.1229 at full size).
    assert 7e-4 < peak[1] / peak[0] < 1.1e-3
    assert abs(peak[2] / peak[0] - 0.1229) < 2e-3
