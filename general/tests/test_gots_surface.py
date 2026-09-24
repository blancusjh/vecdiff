"""GOTS geometry checked against optical path and independent Snell refraction."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from vecdiff_general import CartesianOval, Frame, gots_parameters
from vecdiff_general.interfaces.directions import transmitted_direction


@pytest.mark.parametrize("no, ni", [(1.0, 1.5), (1.5, 1.0)])
@pytest.mark.parametrize("length", [1e-6, 1e-3, 1.0])
def test_gots_optical_path_graph_and_snell(no, ni, length):
    z0, zi = -10 * length, 6 * length
    surface = CartesianOval.from_conjugates(no, z0, ni, zi)
    rho = np.linspace(0, 0.3, 31) * length
    points = surface.parametric_points(rho, 0.37)
    xy = points[..., :2]
    object_vector = points - [0, 0, z0]
    image_vector = [0, 0, zi] - points
    l0 = np.linalg.norm(object_vector, axis=-1)
    li = np.linalg.norm(image_vector, axis=-1)

    assert_allclose(no * l0 + ni * li, ni * zi - no * z0, rtol=3e-15)
    assert_allclose(surface.points(xy), points, atol=3e-14 * length)
    assert_allclose(surface.rho_from_r(surface.r(rho)), rho, atol=3e-14 * length)

    refracted = transmitted_direction(
        object_vector / l0[:, None], surface.normals(xy), no, ni
    )
    assert_allclose(refracted, image_vector / li[:, None], atol=2e-14)


def test_gots_coefficients_and_units():
    # Direct expressions from thesis D.5–D.8, independent of normalized code.
    no, z0, ni, zi = 1.0, -0.01, 1.5, 0.006
    expected = (
        (ni**2 * zi - no**2 * z0) ** 2
        / (ni * no * (ni * zi - no * z0) * (ni * z0 - no * zi)),
        (ni * z0 - no * zi) / (z0 * zi * (ni - no)),
        (ni - no) * (ni + no) ** 2 / (4 * ni * no * z0 * zi * (ni * zi - no * z0)),
        (ni + no)
        * (ni**2 * zi - no**2 * z0)
        / (2 * ni * no * z0 * zi * (ni * zi - no * z0)),
    )
    assert_allclose(gots_parameters(no, z0, ni, zi), expected, rtol=1e-15)
    scaled = gots_parameters(no, 2 * z0, ni, 2 * zi)
    assert_allclose(scaled, np.asarray(expected) / [1, 2, 8, 4])


def test_parametric_placement_and_vertex():
    frame = Frame([1, 2, 3], np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    surface = CartesianOval.from_conjugates(1, -10, 1.5, 6, frame=frame)
    rho = np.linspace(0, 0.5, 5)[:, None]
    phi = np.linspace(0, 2 * np.pi, 9)[None, :]
    local = CartesianOval.from_conjugates(1, -10, 1.5, 6)

    assert_allclose(
        surface.parametric_points(rho, phi),
        frame.origin + local.parametric_points(rho, phi) @ frame.basis.T,
    )
    assert_allclose(surface.parametric_points(0, 0), frame.origin)
    assert_allclose(surface.gradient(np.zeros((1, 2))), 0)


@pytest.mark.parametrize("rho", [-1, np.nan, np.inf, 100])
def test_invalid_parameters_are_rejected(rho):
    surface = CartesianOval.from_conjugates(1, -10, 1.5, 6)

    with pytest.raises(ValueError):
        surface.z(rho)


@pytest.mark.parametrize("design", [(1, -10, 1, 6), (0, -10, 1.5, 6), (1, 10, 1.5, 6)])
def test_invalid_design_fails_at_construction(design):
    with pytest.raises(ValueError):
        CartesianOval.from_conjugates(*design)


@pytest.mark.parametrize("no, ni", [(1.5, 1.0), (1.0, 1.5)])
def test_collimated_gots_path_snell_curvature_and_finite_limit(no, ni):
    zi = 80e-6
    surface = CartesianOval.from_conjugates(no, -np.inf, ni, zi)
    radius = np.linspace(0, 0.15 * zi, 31)
    xy = np.stack((radius, np.zeros_like(radius)), axis=-1)
    points = surface.points(xy)
    image_vector = [0, 0, zi] - points
    distance = np.linalg.norm(image_vector, axis=-1)
    curvature = ni / (zi * (ni - no))

    assert_allclose(no * points[:, 2] + ni * distance, ni * zi, rtol=3e-15)
    assert_allclose(surface.gots, [-((no / ni) ** 2), curvature, 0, 0])
    assert_allclose(surface.z(surface.rho_from_r(radius)), points[:, 2], atol=1e-19)
    assert_allclose(
        transmitted_direction([0, 0, 1], surface.normals(xy), no, ni),
        image_vector / distance[:, None],
        atol=2e-14,
    )

    step = 1e-6 * zi
    assert_allclose(surface.gradient([[step, 0]])[0, 0] / step, curvature, rtol=1e-10)
    finite = CartesianOval.from_conjugates(no, -1e7 * zi, ni, zi)
    assert_allclose(
        finite.z(surface.rho_from_r(radius)), points[:, 2], rtol=3e-7, atol=1e-19
    )

    with pytest.raises(ValueError, match="no finite object"):
        surface.design.object_point


def test_collimated_oval_matches_spherical_exit_vertex():
    from vecdiff_general import SphericalCap

    radius = -80 * 532e-9
    no, ni = 1.5, 1.0
    zi = ni * radius / (ni - no)
    oval = CartesianOval.from_conjugates(no, -np.inf, ni, zi)
    sphere = SphericalCap(radius)
    xy = np.array([[abs(radius) * 1e-6, 0]])

    assert_allclose(oval.gradient(xy), sphere.gradient(xy), rtol=1e-10)


@pytest.mark.parametrize("z0, zi", [(np.inf, 1), (-np.inf, np.inf), (np.nan, 1)])
def test_unsupported_infinite_conjugates(z0, zi):
    with pytest.raises(ValueError):
        CartesianOval.from_conjugates(1.5, z0, 1, zi)


def test_collimated_graph_fold_is_rejected():
    oval = CartesianOval.from_conjugates(1, -np.inf, 1.5, 1)

    with pytest.raises(ValueError, match="fold"):
        oval.height(np.array([[1, 0]]))
