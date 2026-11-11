"""A surface seen from its other side, and a closed surface from its far side.

The reverse of a surface is the same set of points with the opposite
orientation, in the frame (u, -v, -n): light that meets a surface against
its normal meets its reverse along the normal. Its graph, gradient, Hessian,
domain and implicit form are checked against finite differences and against
the surface itself, point by point in space.
"""

import numpy as np
import pytest
import vecdiff as o
from numpy.testing import assert_allclose

bd, host = o.backend.bd, o.backend.to_numpy
S, Frame = o.surfaces, o.geometry.Frame
rng = np.random.default_rng(5)


def turned(angle=0.3):
    c, s = np.cos(angle), np.sin(angle)
    tilt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    spin = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Frame([0.4, -0.2, 1.1], tilt @ spin)


def freeform(frame):
    return S.FreeformSurface(
        lambda p: 0.05 * p[..., 0] ** 2 - 0.02 * p[..., 0] * p[..., 1] ** 2,
        lambda p: np.stack(
            (0.1 * p[..., 0] - 0.02 * p[..., 1] ** 2, -0.04 * p[..., 0] * p[..., 1]),
            -1,
        ),
        frame=frame,
        hessian_function=lambda p: np.stack(
            (
                np.stack((0.1 + 0 * p[..., 0], -0.04 * p[..., 1]), -1),
                np.stack((-0.04 * p[..., 1], -0.04 * p[..., 0]), -1),
            ),
            -2,
        ),
    )


SURFACES = {
    "plane": S.Plane(turned()),
    "convex cap": S.SphericalCap(4.0, turned()),
    "concave cap": S.SphericalCap(-6.0, turned()),
    "sphere": S.Sphere(3.0, turned()),
    "concave sphere": S.Sphere(-5.0, turned()),
    "asphere": S.EvenAsphere(0.2, -0.6, (1e-3, -2e-5), turned()),
    "surface of revolution": S.AxisymmetricSurface(
        lambda r: 0.1 * r**2 + 0.01 * r**4,
        lambda r: 0.2 * r + 0.04 * r**3,
        turned(),
        lambda r: 0.2 + 0.12 * r**2,
    ),
    "freeform": freeform(turned()),
    "cylinder": S.Cylinder(3.5, turned()),
    "half-cylinder": S.CylindricalCap(-4.0, turned()),
    "Cartesian ovoid": S.CartesianOvoid(-np.inf, 6.0, 1.0, 1.5, turned()),
}


def charts(surface, count=40):
    """Points of the surface's domain, away from its rim."""
    xy = rng.uniform(-1.2, 1.2, (4 * count, 2))
    return xy[host(surface.domain(xy))][:count]


def derivative(function, xy, step=1e-5):
    """Fourth-order central differences of function(xy) along x and y (last axis)."""
    columns = []
    for e in np.eye(2):
        near = host(function(xy + step * e)) - host(function(xy - step * e))
        far = host(function(xy + 2 * step * e)) - host(function(xy - 2 * step * e))
        columns.append((8 * near - far) / (12 * step))
    return np.stack(columns, axis=-1)


@pytest.mark.parametrize("name", list(SURFACES))
def test_reverse_is_the_same_surface_with_the_opposite_normal(name):
    surface = SURFACES[name]
    view = surface.reversed()
    assert view.reversed().frame is surface.frame or np.allclose(
        host(view.reversed().frame.basis), host(surface.frame.basis), atol=1e-15
    )
    assert_allclose(host(view.frame.normal), -host(surface.frame.normal), atol=1e-15)
    xy = charts(view)
    mirrored = xy * [1, -1]
    assert host(surface.domain(mirrored)).all()
    assert_allclose(host(view.points(xy)), host(surface.points(mirrored)), atol=1e-13)
    assert_allclose(
        host(view.normals(xy)), -host(surface.normals(mirrored)), atol=1e-13
    )


@pytest.mark.parametrize("name", list(SURFACES))
def test_reverse_declares_the_derivatives_of_its_own_graph(name):
    view = SURFACES[name].reversed()
    xy = charts(view)
    assert_allclose(host(view.gradient(xy)), derivative(view.height, xy), atol=1e-9)
    assert_allclose(host(view.hessian(xy)), derivative(view.gradient, xy), atol=1e-8)


@pytest.mark.parametrize("name", list(SURFACES))
def test_specialized_reverse_agrees_with_the_generic_view(name):
    surface = SURFACES[name]
    view, generic = surface.reversed(), o.surfaces.ReversedSurface.of(surface)
    xy = charts(view)
    for method in ("height", "gradient", "hessian"):
        assert_allclose(
            host(getattr(view, method)(xy)),
            host(getattr(generic, method)(xy)),
            rtol=1e-12,
            atol=1e-12,
        )
    if isinstance(surface, o.surfaces.ImplicitSurface):
        points = rng.normal(size=(30, 3))
        for a, b in zip(view.implicit(points), generic.implicit(points)):
            assert_allclose(host(a), host(b), rtol=1e-11, atol=1e-11)


CLOSED = [n for n, s in SURFACES.items() if isinstance(s, o.surfaces.ImplicitSurface)]


@pytest.mark.parametrize("name", CLOSED)
def test_reversed_implicit_form_is_minus_the_surface_seen_from_behind(name):
    """F' vanishes on the reversed graph, grows along its normal, and its gradient
    and Hessian are the derivatives of F'."""
    view = o.surfaces.ReversedSurface.of(SURFACES[name])
    xy = charts(view)
    on = np.c_[xy, host(view.height(xy))]
    value, gradient, _ = (host(a) for a in view.implicit(on))
    assert np.abs(value).max() < 1e-12
    graph = np.c_[-host(view.gradient(xy)), np.ones(len(xy))]
    cosine = np.sum(gradient * graph, -1) / np.linalg.norm(gradient, axis=-1)
    assert_allclose(cosine, np.linalg.norm(graph, axis=-1), rtol=1e-10)
    points = 0.5 * rng.normal(size=(25, 3))
    _, gradient, hessian = (host(a) for a in view.implicit(points))
    step = 1e-5
    for axis in range(3):
        e = np.eye(3)[axis] * step
        up, down = view.implicit(points + e), view.implicit(points - e)
        assert_allclose(
            gradient[:, axis], host(up[0] - down[0]) / (2 * step), atol=1e-8
        )
        assert_allclose(
            hessian[:, :, axis], host(up[1] - down[1]) / (2 * step), atol=2e-7
        )


def global_line(surface, local_point, local_direction):
    frame = surface.frame
    return (
        host(frame.origin) + local_point @ host(frame.basis).T,
        local_direction @ host(frame.basis).T,
    )


def local(surface, point, direction):
    basis = host(surface.frame.basis)
    return (point - host(surface.frame.origin)) @ basis, direction @ basis


@pytest.mark.parametrize("name", CLOSED)
def test_the_reverse_enters_where_the_surface_is_left_and_leaves_where_it_is_entered(
    name,
):
    surface = SURFACES[name]
    view = surface.reversed()
    start = np.c_[rng.uniform(-0.4, 0.4, (20, 2)), np.full(20, -0.5)]
    direction = np.c_[rng.uniform(-0.2, 0.2, (20, 2)), np.ones(20)]
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    point, toward = global_line(surface, start, direction)
    p, s = local(view, point, toward)
    for leaving in (False, True):
        a = host(surface.crossing(start, direction, leaving=leaving))
        b = host(view.crossing(p, s, leaving=not leaving))
        assert np.isfinite(a).all()
        assert_allclose(b, a, rtol=1e-10, atol=1e-10)
        # The crossing is where the line meets the surface.
        value, gradient, _ = surface.implicit(start + a[:, None] * direction)
        assert np.abs(host(value)).max() < 1e-9
        slope = np.sum(host(gradient) * direction, -1)
        assert ((slope < 0) if leaving else (slope > 0)).all()


@pytest.mark.parametrize("kind", [S.Sphere, S.Cylinder])
@pytest.mark.parametrize("radius", [3.0, -2.0])
def test_far_side_is_the_closed_surface_entered_where_light_leaves_it(kind, radius):
    surface = kind(radius, turned())
    far = surface.far_side()
    assert_allclose(host(far.frame.basis), host(surface.frame.basis), atol=0)
    points = rng.normal(size=(30, 3))
    world = host(surface.frame.origin) + points @ host(surface.frame.basis).T
    there, _ = local(far, world, np.zeros(3))
    assert_allclose(
        host(far.implicit(there)[0]), -host(surface.implicit(points)[0]), atol=1e-12
    )
    start = np.c_[rng.uniform(-0.4, 0.4, (20, 2)), np.full(20, -0.5)]
    direction = np.c_[rng.uniform(-0.2, 0.2, (20, 2)), np.ones(20)]
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    point, toward = global_line(surface, start, direction)
    p, s = local(far, point, toward)
    assert_allclose(
        host(far.crossing(p, s)),
        host(surface.crossing(start, direction, leaving=True)),
        rtol=1e-12,
        atol=1e-12,
    )
