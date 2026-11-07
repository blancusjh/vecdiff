"""The closed sphere: an implicit surface that rays meet beyond its cap's equator."""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose

bd, host = o.backend.bd, o.backend.to_numpy
F, S, P = o.fields, o.surfaces, o.propagation
Frame = o.geometry.Frame


@pytest.mark.parametrize("radius", [7.0, -4.0])
def test_sphere_gradient_and_hessian_are_complex_step_derivatives(radius):
    sphere = S.Sphere(radius)
    points = np.random.default_rng(1).normal(size=(30, 3)) * abs(radius)
    value, gradient, hessian = (host(a) for a in sphere.implicit(points))
    h = 1e-20
    for axis in range(3):
        step = np.zeros(3, complex)
        step[axis] = 1j * h
        shifted = points + step
        offset = shifted - [0, 0, radius]
        f = (radius**2 - np.sum(offset**2, -1)) / (2 * radius)
        assert_allclose(gradient[:, axis], f.imag / h, rtol=1e-14, atol=1e-14)
        g = -offset / radius
        assert_allclose(hessian[:, :, axis], g.imag / h, atol=1e-14)
    # The gradient is the graph normal of the cap where both exist.
    xy = points[:, :2] / (3 * np.max(np.abs(points[:, :2]))) * abs(radius)
    on_cap = np.c_[xy, host(sphere.height(xy))]
    _, normal, _ = sphere.implicit(on_cap)
    graph = np.c_[-host(sphere.gradient(xy)), np.ones(len(xy))]
    graph /= np.linalg.norm(graph, axis=-1, keepdims=True)
    assert_allclose(host(normal), graph, atol=1e-13)


def test_sphere_maps_like_its_cap_where_the_cap_suffices():
    """Rays meeting the front cap: identical RayMaps, to roundoff."""
    radius = 50.0
    carrier = F.Carrier.converging([0, 0, 75.0])
    xy = np.random.default_rng(2).uniform(-30, 30, (200, 2))
    for offset in (0.0, 20.0):
        cap, ball = (
            P.InterfaceMap(
                kind(radius, Frame([0, 0, -radius])), carrier, 1.0, 1.5, offset
            )
            for kind in (S.SphericalCap, S.Sphere)
        )
        a, b = cap.forward(xy), ball.forward(xy)
        assert host(a.valid).all() and host(b.valid).all()
        for name in ("output", "path", "flux", "jacobian", "transmitted"):
            x, y = host(getattr(a, name)), host(getattr(b, name))
            assert_allclose(y, x, rtol=1e-11, atol=1e-11 * np.abs(x).max())


def test_far_side_rays_enter_the_closed_sphere_with_an_analytic_jacobian():
    """Beyond the equator the cap misses; the sphere's map matches complex steps."""
    radius, focus = 40.0, 60.0
    carrier = F.Carrier.converging([0, 0, focus])
    cap, ball = (
        P.InterfaceMap(kind(radius, Frame([0, 0, -radius])), carrier, 1.0, 1.5)
        for kind in (S.SphericalCap, S.Sphere)
    )
    rho = np.linspace(70.0, 85.0, 12)  # 60 to 80 degrees of incidence, far side
    labels = np.c_[rho, 0 * rho]
    assert not host(cap.forward(labels).valid).any()
    rays = ball.forward(labels)
    assert host(rays.valid).all() and (host(rays.surface_height) > radius).all()
    # Independent complex-step trace through the whole sphere (centre at z = 0).
    h = 1e-20
    r = rho + 1j * h
    p = np.stack([r, np.full_like(r, -radius)])
    s = np.stack([-r, np.full_like(r, focus + radius)])
    s = s / np.sqrt(s[0] ** 2 + s[1] ** 2)
    b = p[0] * s[0] + p[1] * s[1]
    c = p[0] ** 2 + p[1] ** 2 - radius**2
    q = p + c / (-b + np.sqrt(b * b - c)) * s
    n = -q / radius
    ci = s[0] * n[0] + s[1] * n[1]
    t = (s - ci * n) / 1.5
    st = t + np.sqrt(1 - (t[0] ** 2 + t[1] ** 2)) * n
    image = q[0] + (-radius - q[1]) / st[1] * st[0]
    assert_allclose(host(rays.output)[:, 0], image.real, rtol=1e-12)
    assert_allclose(host(rays.jacobian)[:, 0, 0], image.imag / h, rtol=1e-9)
    assert_allclose(host(rays.jacobian)[:, 1, 1], image.real / rho, rtol=1e-12)
    # A ray that misses the ball is unsupported, as a grazing one is.
    assert not host(ball.forward([[200.0, 0.0]]).valid).any()


def test_sphere_carries_the_hard_pupils_tail_on_the_meridian():
    """The meridian fan of a sphere runs on past the cap's equator, to grazing."""
    radius = 30.0
    air = o.media.HomogeneousRegion("air", o.media.Dielectric(1.0))
    glass = o.media.HomogeneousRegion("glass", o.media.Dielectric(1.5))
    beam = F.FocusedBeam(
        F.profiles.Uniform(0.4),
        F.polarization.cartesian(1, 1j),
        focal_length=1.0,
        max_sine=0.4,
        wavelength0=1.0,
        region=air,
        frame=Frame([0, 0, 1.5 * radius]),
        apodization="none",
    )
    edges = {}
    for kind in (S.SphericalCap, S.Sphere):
        surface = kind(radius, Frame([0, 0, -radius]))
        pipeline = P.PropagationPipeline(
            (o.system.DielectricInterface(surface, air, glass),),
            transport="eikonal",
            reference_fans=((72.0, 80.0),),
        )
        solution = pipeline.solve(beam)
        edges[kind] = pipeline._operators[0].edge
        report = solution.interactions[0]
        assert report.azimuthal_orders == (1,)
    # The cap ends at its equator (input radius 1.66 R); the sphere at 80 degrees.
    assert edges[S.SphericalCap] < 1.67 * radius < 2.1 * radius < edges[S.Sphere]
