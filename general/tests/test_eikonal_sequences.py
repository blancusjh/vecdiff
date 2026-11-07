"""Eikonal transport through a sequence of surfaces: the refracted ray family."""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose
from scipy.integrate import simpson
from vecdiff_general.propagation._axisymmetric import AxialTransport, BeamField
from vecdiff_general.propagation.eikonal import RefractedRays

bd, host = o.backend.bd, o.backend.to_numpy
F, S, P = o.fields, o.surfaces, o.propagation
Frame = o.geometry.Frame
AIR = o.media.HomogeneousRegion("air", o.media.Dielectric(1.0))
GLASS = o.media.HomogeneousRegion("glass", o.media.Dielectric(1.5))
GLASS2 = o.media.HomogeneousRegion("glass, second region", o.media.Dielectric(1.5))
CORE = o.media.HomogeneousRegion("core", o.media.Dielectric(2.0))
CIRCULAR = F.polarization.cartesian(1 / np.sqrt(2), 1j / np.sqrt(2))


def shells(outer, inner, core=CORE):
    """Concentric caps about the origin, vertices at -R: air | glass | core."""
    caps = [S.SphericalCap(R, Frame([0, 0, -R])) for R in (outer, inner)]
    return (
        o.system.DielectricInterface(caps[0], AIR, GLASS),
        o.system.DielectricInterface(caps[1], GLASS, core),
    )


def beam(radius, pupil="smooth", umax=0.4):
    profile = {
        "smooth": lambda x, y: np.clip(1 - (x**2 + y**2) / umax**2, 0, None) ** 3,
        "gaussian": F.profiles.Gaussian(0.6 * umax),
    }[pupil]
    return F.FocusedBeam(
        profile,
        CIRCULAR,
        focal_length=1.0,
        max_sine=umax,
        wavelength0=1.0,
        region=AIR,
        frame=Frame([0, 0, 1.5 * radius]),
        apodization="none",
    )


def chain(rho, outer, inner, focus, plane):
    """Complex-step-safe meridian trace: Pi1- (z = -outer) to the plane z = plane."""
    p = np.stack([rho, np.full_like(rho, -outer)])
    s = np.stack([-rho, np.full_like(rho, focus + outer)])
    s = s / np.sqrt(s[0] ** 2 + s[1] ** 2)
    b = p[0] * s[0] + p[1] * s[1]
    c = p[0] ** 2 + p[1] ** 2 - outer**2
    leg = c / (-b + np.sqrt(b * b - c))
    q = p + leg * s
    n = -q / outer
    ci = s[0] * n[0] + s[1] * n[1]
    t = (s - ci * n) / 1.5
    st = t + np.sqrt(1 - (t[0] ** 2 + t[1] ** 2)) * n
    move = (plane - q[1]) / st[1]
    return q[0] + move * st[0], st


def test_refracted_family_derivatives_match_complex_steps():
    """d P / d label and d s / d label on the next plane, through two spheres."""
    outer, inner = 40.0, 20.0
    first, second = shells(outer, inner)
    carrier = F.Carrier.converging([0, 0, 1.5 * outer])
    leg = P.InterfaceMap(first.surface, carrier, 1.0, 1.5)
    family = RefractedRays(leg)
    rho = np.linspace(0.5, 25, 30)
    frame = second.surface.frame
    p, s, dp, ds, alive = (
        host(a) for a in family.rays(np.c_[rho, 0 * rho], frame, 0.0)
    )
    assert alive.all()
    h = 1e-20
    start, direction = chain(rho + 1j * h, outer, inner, 1.5 * outer, -inner)
    assert_allclose(p[:, 0], start.real, rtol=1e-13)
    assert_allclose(dp[:, 0, 0], start.imag / h, rtol=1e-11)
    assert_allclose(s[:, [0, 2]], direction.real.T, atol=1e-14)
    assert_allclose(ds[:, 0, 0], direction[0].imag / h, rtol=1e-10, atol=1e-15)
    assert_allclose(ds[:, 2, 0], direction[1].imag / h, rtol=1e-10, atol=1e-15)
    # Off the meridian the azimuthal derivatives are the meridian's over rho.
    assert_allclose(dp[:, 1, 1], p[:, 0] / rho, rtol=1e-12)
    assert_allclose(ds[:, 1, 1], s[:, 0] / rho, rtol=1e-11)


def finite_difference(mapping, labels, step):
    columns = []
    for e in np.eye(2):
        near = host(mapping.forward(labels + step * e).output) - host(
            mapping.forward(labels - step * e).output
        )
        far = host(mapping.forward(labels + 2 * step * e).output) - host(
            mapping.forward(labels - 2 * step * e).output
        )
        columns.append((8 * near - far) / (12 * step))
    return np.stack(columns, axis=-1)


def test_second_map_jacobian_without_symmetry_matches_finite_differences():
    """An asphere then a decentred sphere, a tilted collimated carrier."""
    first = S.EvenAsphere(1 / 60, conic=-0.5, coefficients=(1e-7,))
    second = S.SphericalCap(-45.0, Frame([1.5, -1.0, 12.0]))
    leg = P.InterfaceMap(first, F.Carrier.collimated([0.05, -0.03, 1.0]), 1.0, 1.6)
    mapping = P.InterfaceMap(second, RefractedRays(leg), 1.6, 1.2, -2.0)
    labels = np.random.default_rng(3).uniform(-10, 10, (80, 2))
    rays = mapping.forward(labels)
    assert host(rays.valid).all()
    numeric = finite_difference(mapping, labels, 1e-4)
    assert_allclose(host(rays.jacobian), numeric, atol=2e-9 * abs(numeric).max())
    det = np.linalg.det(host(rays.jacobian)) / np.linalg.det(host(rays.start_jacobian))
    sz, ct = host(rays.incident)[:, 2], host(rays.cos_transmitted)
    ci, tz = host(rays.cos_incident), host(rays.transmitted)[:, 2]
    assert_allclose(host(rays.flux), np.sqrt(sz * ct / (ci * tz * det)), rtol=1e-13)


def test_index_matched_second_leg_returns_the_single_surface_field():
    """The meridian leg itself across glass | glass: the handoff gives 1.8e-5.

    The inner sphere (0.8 R) meets the whole geometric beam before its equator.
    """
    outer = 30.0
    first, _ = shells(outer, 24.0)
    matched = o.system.DielectricInterface(
        S.SphericalCap(24.0, Frame([0, 0, -24.0])), GLASS, GLASS2
    )
    source = beam(outer)
    chart = o.sampling.PolarGrid(np.linspace(0, outer, 5), np.arange(4) * np.pi / 2)
    leg1 = AxialTransport(first, source.carrier, source.frame, 1.0, scale=40.0)
    field1, _ = leg1.transmit(BeamField(source, chart))
    leg2 = AxialTransport(
        matched, RefractedRays(leg1.map), source.frame, 1.0, scale=leg1.edge
    )
    field2, report = leg2.transmit(field1)
    rho = np.linspace(0, 5, 41)
    points = o.sampling.PointGrid(np.c_[rho, 0 * rho, np.full(41, outer / 1.5)])
    one, two = (host(f.evaluate(points).electric.values) for f in (field1, field2))
    assert report.azimuthal_orders == (1,)
    assert report.unsupported_relative_amplitude < 3e-4  # field beyond the fan
    assert np.linalg.norm(two - one) < 1e-4 * np.linalg.norm(one)


def lattice_spectrum(source, radius, spacing):
    """The beam sampled on a lattice one wavelength before the vertex."""
    z0 = -radius - 1.0
    half = 1.35 * (1.5 * radius - z0) * np.tan(np.arcsin(0.4)) + 8.0
    count = int(np.ceil(2 * half / spacing))
    count += count % 2
    axis = (np.arange(count) - count // 2) * spacing
    plane = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, z0]))
    sampled = source.evaluate(plane, method="nufft").electric.values[..., :2]
    return F.angular_spectrum(
        F.TransverseElectricField(plane, 1.0, sampled), AIR, evanescent="discard"
    )


def test_two_surfaces_on_the_lattice_and_the_meridian_agree():
    """One sampled beam through glass and an n = 2 core, both legs, and the beam itself."""
    pytest.importorskip("finufft")
    outer = 8.0
    system = shells(outer, 6.4)
    source = beam(outer)
    # NA 1.6 in the core: 0.25 lambda0 resolves it.
    spectrum = lattice_spectrum(source, outer, spacing=0.25)
    axis = np.linspace(-2, 2, 9)
    x, y = np.meshgrid(axis, axis)
    points = o.sampling.PointGrid(np.c_[x.ravel(), y.ravel(), np.full(x.size, 3.0)])
    fields = {}
    for name, kwargs, incident in (
        ("lattice", {}, spectrum),
        ("meridian", dict(axisymmetric=True), spectrum),
        ("beam", {}, source),
        ("beam on the lattice", dict(lattice=spectrum.grid), source),
    ):
        pipeline = P.PropagationPipeline(system, transport="eikonal", **kwargs)
        solution = pipeline.solve(incident, carrier=source.carrier)
        fields[name] = host(solution.evaluate(points).electric.values)
        assert len(solution.interactions) == 2
    scale = np.linalg.norm(fields["beam"])
    assert np.linalg.norm(fields["meridian"] - fields["beam"]) < 1e-6 * scale
    assert np.linalg.norm(fields["lattice"] - fields["meridian"]) < 1e-3 * scale
    # Read at the preimages, the beam crosses both legs as its samples do.
    difference = np.linalg.norm(fields["beam on the lattice"] - fields["lattice"])
    assert difference < 1e-6 * scale


def test_aplanatic_pair_meets_the_layered_sphere_at_thirty_wavelengths():
    """n 1, 1.5, 2 with R2 = R1 / 2: both surfaces aplanatic, NA 1.6 in the core.

    The archived two-surface code, with its second fan cut where the inner cap
    ends (53.13 degrees, the beam's marginal ray), gives 1.0798 %.
    """
    from vecdiff_general.propagation.references import LayeredSphere

    outer = 30.0
    source = beam(outer)
    pipeline = P.PropagationPipeline(shells(outer, outer / 2), transport="eikonal")
    rho = np.linspace(0, 5, 81)
    focus = 0.375 * outer
    points = o.sampling.PointGrid(np.c_[rho, 0 * rho, np.full(81, focus)])
    field = host(pipeline.solve(source).evaluate(points).electric.values)
    sphere = LayeredSphere(
        source,
        (
            o.volumes.SphereVolume([0, 0, 0], outer),
            o.volumes.SphereVolume([0, 0, 0], outer / 2),
        ),
        (GLASS, CORE),
    )
    exact = host(sphere.evaluate(points, term=1, side="interior").electric.values)

    def norm(E):
        return simpson(rho * np.sum(abs(E) ** 2, -1), x=rho)

    error = 100 * np.sqrt(norm(field - exact) / norm(exact))
    assert abs(error - 1.0797) < 2e-3
    reports = pipeline.last_solution.interactions
    assert all(r.azimuthal_orders == (1,) for r in reports)
