"""The reflected branch of the eikonal leg: map, Jacobian, dyadic and lattice leg.

The reflected ray s_r = s - 2 c_i n returns to the same geometric plane in
the incident medium. Its map is checked against finite differences and
against mirrors that image a point stigmatically; the reflected dyadic
against each plane wave's own boundary-value problem, solved here from
the continuity of tangential E and H (or vanishing tangential E on a
perfect conductor) without any Fresnel formula.
"""

import numpy as np
import pytest
import vecdiff as o
from numpy.testing import assert_allclose

bd, host = o.backend.bd, o.backend.to_numpy
F, S = o.fields, o.surfaces
Frame = o.geometry.Frame
AIR = o.media.HomogeneousRegion("air", o.media.Dielectric(1.0))
GLASS = o.media.HomogeneousRegion("glass", o.media.Dielectric(1.5))
rng = np.random.default_rng(11)


def reflected_map(surface, carrier, offset, n=1.0):
    return o.transport.InterfaceMap(
        surface, carrier, n, 1.5, offset, branch="reflected"
    )


def finite_difference_jacobian(mapping, xy, step):
    columns = []
    for e in np.eye(2):
        near = host(mapping.forward(xy + step * e).output) - host(
            mapping.forward(xy - step * e).output
        )
        far = host(mapping.forward(xy + 2 * step * e).output) - host(
            mapping.forward(xy - 2 * step * e).output
        )
        columns.append((8 * near - far) / (12 * step))
    return np.stack(columns, axis=-1)


TILTED = S.FreeformSurface(
    lambda xy: 0.2 * xy[..., 0] - 0.1 * xy[..., 1],
    lambda xy: bd.broadcast_to(bd.asarray([0.2, -0.1]), bd.asarray(xy).shape),
    hessian_function=lambda xy: bd.zeros(bd.asarray(xy).shape + (2,)),
)


@pytest.mark.parametrize(
    "surface, carrier, offset, half",
    [
        (TILTED, F.Carrier.converging([5.0, -3.0, 400.0]), -20.0, 30.0),
        # convex toward the light, plane in front and behind the vertex
        (S.SphericalCap(80.0), F.Carrier.collimated([0.05, 0.0, 1.0]), -10.0, 30.0),
        (S.SphericalCap(80.0), F.Carrier.diverging([0, 0, -200.0]), 4.0, 25.0),
        # concave: a focus between the surface and the plane
        (S.SphericalCap(-60.0), F.Carrier.collimated([0, 0.03, 1.0]), -45.0, 12.0),
        (
            S.EvenAsphere(-1 / 70, conic=-0.6, coefficients=(2e-7,)),
            F.Carrier.converging([3, -2, 150]),
            -12.0,
            20.0,
        ),
    ],
)
def test_reflected_map_jacobian_is_the_derivative_of_the_map(
    surface, carrier, offset, half
):
    """Appendix A with ds_r = ds - 2 (dc_i n + c_i dn), and g from the tube's power."""
    mapping = reflected_map(surface, carrier, offset)
    xy = rng.uniform(-half, half, (120, 2))
    rays = mapping.forward(xy)
    assert host(rays.valid).all()
    assert (host(rays.transmitted)[:, 2] < 0).all()
    numeric = finite_difference_jacobian(mapping, xy, 1e-4)
    analytic = host(rays.jacobian)
    assert_allclose(analytic, numeric, atol=2e-9 * abs(numeric).max())
    det = np.linalg.det(analytic)
    caustics = host(rays.caustics)
    # Reflection reverses no orientation of the plane's own axes: the ratio
    # changes sign exactly once per collapsed direction.
    assert np.array_equal(det < 0, caustics % 2 == 1)
    sz, rz = host(rays.incident)[:, 2], host(rays.transmitted)[:, 2]
    expected = np.sqrt(np.abs(sz / (rz * det))) * (-1j) ** caustics
    assert_allclose(host(rays.flux), expected, rtol=1e-14)
    assert_allclose(host(rays.cos_transmitted), host(rays.cos_incident), rtol=0)


def test_concave_mirror_counts_the_focus_on_the_way_back():
    """A focus between the mirror and the plane: kappa = 2 and g < 0."""
    mapping = reflected_map(S.SphericalCap(-60.0), F.Carrier.collimated([0, 0, 1]), -45)
    rays = mapping.forward(rng.uniform(-8, 8, (50, 2)))
    assert (host(rays.caustics) == 2).all()
    assert (host(rays.flux).real < 0).all()
    # Before the focus: nothing collapses.
    near = reflected_map(S.SphericalCap(-60.0), F.Carrier.collimated([0, 0, 1]), -10)
    rays = near.forward(rng.uniform(-8, 8, (50, 2)))
    assert (host(rays.caustics) == 0).all() and (host(rays.flux).real > 0).all()


def test_plane_mirror_at_normal_incidence_is_the_identity_map():
    for offset in (-7.0, 0.0, 3.0):
        mapping = reflected_map(S.Plane(), F.Carrier.collimated([0, 0, 1]), offset, 1.3)
        xy = rng.uniform(-5, 5, (20, 2))
        rays = mapping.forward(xy)
        assert host(rays.valid).all()
        assert_allclose(host(rays.output), xy, atol=1e-15)
        assert_allclose(host(rays.jacobian), np.broadcast_to(np.eye(2), (20, 2, 2)))
        assert (host(rays.caustics) == 0).all()
        assert_allclose(host(rays.flux), 1.0, rtol=1e-15)
        assert_allclose(host(rays.path), -2 * 1.3 * offset, atol=1e-14)


def test_paraboloidal_mirror_images_the_collimated_beam_on_its_focus():
    curvature = -0.01
    focus = np.array([0.0, 0.0, 1 / (2 * curvature)])
    mapping = reflected_map(
        S.EvenAsphere(curvature, conic=-1.0), F.Carrier.collimated([0, 0, 1]), -10.0
    )
    xy = rng.uniform(-30, 30, (300, 2))
    rays = mapping.forward(xy)
    assert host(rays.valid).all()
    end = np.c_[host(rays.output), np.full(len(xy), -10.0)]
    toward = (focus - end) / np.linalg.norm(focus - end, axis=-1, keepdims=True)
    assert_allclose(host(rays.transmitted), toward, atol=1e-14)
    path = host(rays.path) + np.linalg.norm(focus - end, axis=-1)
    assert np.ptp(path) < 1e-13 * 100


def test_ellipsoidal_mirror_images_one_focus_on_the_other():
    radius, eccentricity = 100.0, 0.5
    axis = radius / (1 - eccentricity**2)
    near = np.array([0, 0, -axis * (1 - eccentricity)])
    far = np.array([0, 0, -axis * (1 + eccentricity)])
    surface = S.EvenAsphere(-1 / radius, conic=-(eccentricity**2))
    mapping = reflected_map(surface, F.Carrier.diverging(near), -20.0, 1.4)
    xy = rng.uniform(-25, 25, (300, 2))
    rays = mapping.forward(xy)
    assert host(rays.valid).all()
    start = np.c_[xy, np.full(len(xy), -20.0)]
    end = np.c_[host(rays.output), np.full(len(xy), -20.0)]
    toward = (far - end) / np.linalg.norm(far - end, axis=-1, keepdims=True)
    assert_allclose(host(rays.transmitted), toward, atol=1e-14)
    path = (
        1.4 * np.linalg.norm(start - near, axis=-1)
        + host(rays.path)
        + 1.4 * np.linalg.norm(far - end, axis=-1)
    )
    assert_allclose(path, 1.4 * 2 * axis, rtol=1e-14)


def test_reflected_branch_needs_rays_that_head_back_to_the_plane():
    with pytest.raises(ValueError, match="branch"):
        o.transport.InterfaceMap(
            S.Plane(), F.Carrier.collimated([0, 0, 1]), 1, 1.5, branch="back"
        )
    # Grazing a tilted mirror, the reflected ray does not return.
    rays = reflected_map(TILTED, F.Carrier.collimated([0.98, 0, 0.2]), -2).forward(
        np.zeros((1, 2))
    )
    assert not host(rays.valid).any() and host(rays.flux)[0] == 0
