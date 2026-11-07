"""Circular cylinders and Cartesian ovals with virtual conjugates, through eikonal transport."""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose
from vecdiff_general.propagation.eikonal import ray_weights

bd, host = o.backend.bd, o.backend.to_numpy
F, S, P = o.fields, o.surfaces, o.propagation
Frame = o.geometry.Frame
AIR = o.media.HomogeneousRegion("air", o.media.Dielectric(1.0))
GLASS = o.media.HomogeneousRegion("glass", o.media.Dielectric(1.5))
rng = np.random.default_rng(9)


@pytest.mark.parametrize("radius", [6.0, -4.0])
def test_cylinder_graph_and_implicit_forms_agree_with_their_derivatives(radius):
    cap, closed = S.CylindricalCap(radius), S.Cylinder(radius)
    xy = rng.uniform(-0.6, 0.6, (40, 2)) * abs(radius)
    step = 1e-5
    numeric = np.stack(
        [
            (host(cap.gradient(xy + step * e)) - host(cap.gradient(xy - step * e)))
            / (2 * step)
            for e in np.eye(2)
        ],
        axis=-1,
    )
    assert_allclose(host(cap.hessian(xy)), numeric, atol=1e-8)
    points = np.c_[xy, host(cap.height(xy))]
    value, normal, hessian = (host(a) for a in closed.implicit(points))
    assert_allclose(value, 0, atol=1e-13 * abs(radius))
    graph = np.c_[-host(cap.gradient(xy)), np.ones(len(xy))]
    assert_allclose(
        normal, graph / np.linalg.norm(graph, axis=-1, keepdims=True), atol=1e-13
    )
    h = 1e-20
    for axis in range(3):
        shifted = points + 1j * h * np.eye(3)[axis]
        g = (
            np.stack((-shifted[:, 0], 0 * shifted[:, 1], radius - shifted[:, 2]), -1)
            / radius
        )
        assert_allclose(hessian[:, :, axis], g.imag / h, atol=1e-14)
    assert not host(cap.domain([[abs(radius) * 1.01, 0.0]])).any()


def test_closed_cylinder_maps_like_its_half_and_on_beyond():
    radius = 40.0
    carrier = F.Carrier.converging([3.0, -2.0, 60.0])
    maps = [
        P.InterfaceMap(kind(radius, Frame([0, 0, -radius])), carrier, 1.0, 1.5)
        for kind in (S.CylindricalCap, S.Cylinder)
    ]
    xy = rng.uniform(-25, 25, (150, 2))
    a, b = (m.forward(xy) for m in maps)
    assert host(a.valid).all() and host(b.valid).all()
    for name in ("output", "path", "flux", "jacobian"):
        x = host(getattr(a, name))
        assert_allclose(
            host(getattr(b, name)), x, rtol=1e-11, atol=1e-11 * abs(x).max()
        )
    far = np.array([[75.0, 5.0], [80.0, -3.0]])  # beyond the half-cylinder's edge
    assert not host(maps[0].forward(far).valid).any()
    rays = maps[1].forward(far)
    assert host(rays.valid).all() and (host(rays.surface_height) > radius).all()


def cylinder_error(radius, pupil):
    """The archive's cylinder (A = 1.5 R, u_max = 0.25) on its lattice and fan, vs Mie."""
    from vecdiff_general.propagation.references import InfiniteCylinder

    umax = 0.25
    profile = (
        F.profiles.Uniform(umax)
        if pupil == "hard"
        else lambda x, y: np.clip(1 - (x**2 + y**2) / umax**2, 0, None) ** 3
    )
    beam = F.FocusedBeam(
        profile,
        F.polarization.cartesian(1 / np.sqrt(2), 1j / np.sqrt(2)),
        focal_length=1.0,
        max_sine=umax,
        wavelength0=1.0,
        region=AIR,
        frame=Frame([0, 0, 1.5 * radius]),
        apodization="none",
    )

    def fan(rays):
        s = host(rays.incident)
        sine = np.hypot(s[:, 0], s[:, 1])
        t = np.clip((sine - 1.7 * umax) / (0.2 * umax), 0, 1)
        weight = 0.5 * (1 + np.cos(np.pi * t)) * host(ray_weights((78.0, 85.0), rays))
        return weight * (host(rays.transmitted)[:, 2] > 0.05)

    count = int(round(11.86 * radius)) + int(round(11.86 * radius)) % 2
    axis = (np.arange(count) - count // 2) / 3
    plane = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, -radius]))
    sampled = beam.evaluate(plane, method="nufft").electric.values[..., :2]
    incident = F.angular_spectrum(
        F.TransverseElectricField(plane, 1.0, sampled), AIR, evanescent="discard"
    )
    surface = S.CylindricalCap(radius, Frame([0, 0, -radius]))
    pipeline = P.PropagationPipeline(
        (o.system.DielectricInterface(surface, AIR, GLASS),),
        transport="eikonal",
        reference_fans=(fan,),
    )
    solution = pipeline.solve(incident, carrier=beam.carrier)
    xs = np.linspace(-3.5, 3.5, 140)
    x, y = np.meshgrid(xs, xs, indexing="ij")
    disk = np.hypot(x, y) <= 3.5
    points = o.sampling.PointGrid(
        np.c_[x[disk], y[disk], np.full(disk.sum(), radius / 1.5)]
    )
    field = host(solution.evaluate(points).electric.values)
    cylinder = InfiniteCylinder(
        beam, o.volumes.CylinderVolume([0, 0, 0], [0, 1, 0], radius), GLASS
    )
    exact = host(cylinder.evaluate(points, term=1, side="interior").electric.values)
    return 100 * np.linalg.norm(field - exact) / np.linalg.norm(exact)


@pytest.mark.parametrize("pupil, expected", [("hard", 5.56765), ("smooth", 1.14523)])
def test_cylinder_on_the_lattice_meets_the_archive_at_ten_wavelengths(pupil, expected):
    """The archive's general code gives 5.56765 % (hard) and 1.14523 % (smooth).

    The hard pupil needs the re-expansion band: modes within 0.1 % of the light
    circle took E_z to 10 times its error.
    """
    pytest.importorskip("finufft")
    assert abs(cylinder_error(10.0, pupil) - expected) < 1e-4 * expected


def test_virtual_conjugate_oval_is_stigmatic_and_its_ovoid_closes_it():
    """The archive's oval: rays toward A = 2.5 R behind the vertex refract to 1.8 R."""
    radius = 20.0
    oval = S.CartesianOval.from_conjugates(1.0, 2.5 * radius, 1.5, 1.8 * radius)
    ovoid = S.CartesianOvoid.from_conjugates(1.0, 2.5 * radius, 1.5, 1.8 * radius)
    xy = rng.uniform(-15, 15, (60, 2))
    step = 1e-5
    numeric = np.stack(
        [
            (host(oval.gradient(xy + step * e)) - host(oval.gradient(xy - step * e)))
            / (2 * step)
            for e in np.eye(2)
        ],
        axis=-1,
    )
    assert_allclose(host(oval.hessian(xy)), numeric, atol=1e-7)
    carrier = F.Carrier.converging([0, 0, 2.5 * radius])
    for surface in (oval, ovoid):
        rays = P.InterfaceMap(surface, carrier, 1.0, 1.5).forward(xy)
        assert host(rays.valid).all()
        q = host(rays.surface_point)
        toward = np.array([0, 0, 1.8 * radius]) - q
        toward /= np.linalg.norm(toward, axis=-1, keepdims=True)
        assert_allclose(host(rays.transmitted), toward, atol=1e-11)
    # The ovoid's rays run on past the graph's widest point onto its far side.
    far = P.InterfaceMap(ovoid, carrier, 1.0, 1.5).forward([[60.0, 0.0]])
    assert host(far.valid).all() and host(far.surface_height)[0] > 1.1 * radius
    # Real conjugates keep their GOTS form and their results.
    real = S.CartesianOval.from_conjugates(1.0, -6.0, 1.5, 4.0)
    assert np.isfinite(real.gots).all() and real.design.real
    with pytest.raises(NotImplementedError, match="real"):
        oval.gots


def test_oval_on_the_lattice_agrees_with_its_meridian():
    """The archive's oval at R = 10: lattice and meridian legs on one sampled beam."""
    pytest.importorskip("finufft")
    radius, umax = 10.0, 0.4
    beam = F.FocusedBeam(
        lambda x, y: np.clip(1 - (x**2 + y**2) / umax**2, 0, None) ** 3,
        F.polarization.cartesian(1, 1j),
        focal_length=1.0,
        max_sine=umax,
        wavelength0=1.0,
        region=AIR,
        frame=Frame([0, 0, 1.5 * radius]),
        apodization="none",
    )
    ovoid = S.CartesianOvoid.from_conjugates(
        1.0, 2.5 * radius, 1.5, 1.8 * radius, frame=Frame([0, 0, -radius])
    )
    system = (o.system.DielectricInterface(ovoid, AIR, GLASS),)
    axis = (np.arange(240) - 120) / 3
    plane = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, -radius]))
    sampled = beam.evaluate(plane, method="nufft").electric.values[..., :2]
    incident = F.angular_spectrum(
        F.TransverseElectricField(plane, 1.0, sampled), AIR, evanescent="discard"
    )
    xs = np.linspace(-2, 2, 9)
    x, y = np.meshgrid(xs, xs)
    points = o.sampling.PointGrid(
        np.c_[x.ravel(), y.ravel(), np.full(81, 0.8 * radius)]
    )

    def fan(rays):
        """The archive's lattice fan for the oval: sin(gamma) to 1.6 u_max."""
        s = host(rays.incident)
        t = np.clip((np.hypot(s[:, 0], s[:, 1]) - 1.45 * umax) / (0.15 * umax), 0, 1)
        weight = 0.5 * (1 + np.cos(np.pi * t)) * host(ray_weights((78.0, 85.0), rays))
        return weight * (host(rays.transmitted)[:, 2] > 0.05)

    fields = []
    for kwargs in ({}, dict(axisymmetric=True)):
        pipeline = P.PropagationPipeline(
            system, transport="eikonal", reference_fans=(fan,), **kwargs
        )
        solution = pipeline.solve(incident, carrier=beam.carrier)
        fields.append(host(solution.evaluate(points).electric.values))
    assert np.linalg.norm(fields[0] - fields[1]) < 3e-3 * np.linalg.norm(fields[1])
