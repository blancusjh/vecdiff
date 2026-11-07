"""Axisymmetric eikonal transport: the meridian leg and its Hankel spectra."""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose
from scipy.integrate import simpson
from vecdiff_general.propagation._axisymmetric import AxialTransport, BeamField
from vecdiff_general.propagation.eikonal import carried
from vecdiff_general.propagation.references import MieSphere

bd, host = o.backend.bd, o.backend.to_numpy
F, S, P = o.fields, o.surfaces, o.propagation
Frame = o.geometry.Frame
AIR = o.media.HomogeneousRegion("air", o.media.Dielectric(1.0))
AIR2 = o.media.HomogeneousRegion("air, second region", o.media.Dielectric(1.0))
GLASS = o.media.HomogeneousRegion("glass", o.media.Dielectric(1.5))
CIRCULAR = F.polarization.cartesian(1 / np.sqrt(2), 1j / np.sqrt(2))
rng = np.random.default_rng(17)


def converging(radius, pupil="smooth", polarization=CIRCULAR, umax=0.4):
    """The manuscript's beam toward A = 1.5 R (Appendix C apodization)."""
    profile = {
        "hard": F.profiles.Uniform(umax),
        "smooth": lambda x, y: np.clip(1 - (x**2 + y**2) / umax**2, 0, None) ** 3,
        "gaussian": F.profiles.Gaussian(0.6 * umax),
    }[pupil]
    return F.FocusedBeam(
        profile,
        polarization,
        focal_length=1.0,
        max_sine=umax,
        wavelength0=1.0,
        region=AIR,
        frame=Frame([0, 0, 1.5 * radius]),
        apodization="none",
    )


def sphere(radius, inside=GLASS):
    cap = S.SphericalCap(radius, Frame([0, 0, -radius]))
    return (o.system.DielectricInterface(cap, AIR, inside),)


def chart(beam, radius):
    plane = Frame([0, 0, -radius])
    return o.sampling.PolarGrid(
        np.linspace(0, radius, 5), np.arange(4) * np.pi / 2, plane
    )


def test_meridian_leg_is_the_lattice_map_turned_about_the_axis():
    """Phi, g, Delta S and the dyadic off the meridian are the meridian's, rotated."""
    radius = 40.0
    (interface,) = sphere(radius)
    carrier = F.Carrier.converging([0, 0, 60.0])
    leg = AxialTransport(interface, carrier, Frame(), 1.0, scale=30.0)
    nodes = np.linspace(0, len(leg.radius) - 1, 25).astype(int)
    rho, image = leg.preimage[nodes], leg.radius[nodes]
    phi = rng.uniform(0, 2 * np.pi, len(nodes))
    turn = np.stack((np.cos(phi), np.sin(phi)), -1)
    rays = leg.map.forward(rho[:, None] * turn)
    assert host(rays.valid).all()
    assert_allclose(host(rays.output), image[:, None] * turn, atol=1e-11 * radius)
    meridian = leg.rays.subset(bd.asarray(nodes))
    # The last node lies at the cap's equator, where its slope diverges.
    for inner, tolerance in ((slice(None), 1e-8), (slice(-1), 1e-12)):
        for name in ("flux", "path"):
            assert_allclose(
                host(getattr(rays, name))[inner],
                host(getattr(meridian, name))[inner],
                rtol=tolerance,
                atol=tolerance,
            )
    # The dyadic on a random incident field, by the lattice leg's TE axis n x s_i.
    E = rng.normal(size=(len(nodes), 3)) + 1j * rng.normal(size=(len(nodes), 3))
    s = np.cross(host(rays.normal), host(rays.incident))
    s /= np.linalg.norm(s, axis=-1, keepdims=True)
    k0 = 2 * np.pi
    lattice = (host(rays.flux) * np.exp(1j * k0 * host(rays.path)))[:, None] * host(
        carried(bd.asarray(E), rays, bd.asarray(s), 1.0, 1.5)
    )
    c, sn = np.cos(phi), np.sin(phi)
    cylindrical = np.stack(
        (c * E[:, 0] + sn * E[:, 1], -sn * E[:, 0] + c * E[:, 1], E[:, 2]), -1
    )
    radial, azimuthal = np.einsum(
        "nij,nj->ni", host(leg._transfer)[nodes], cylindrical
    ).T
    expected = np.stack((c * radial - sn * azimuthal, sn * radial + c * azimuthal), -1)
    assert_allclose(lattice[:, :2], expected, atol=1e-11 * abs(expected).max())
    # The slope d rho' / d rho of the inverse map against fourth-order differences.
    h = 1e-4
    out = [host(leg._meridian(rho[1:-1] + j * h).output)[:, 0] for j in (-2, -1, 1, 2)]
    numeric = (out[0] - 8 * out[1] + 8 * out[2] - out[3]) / (12 * h)
    assert_allclose(host(meridian.jacobian)[1:-1, 0, 0], numeric, rtol=1e-8)
    assert leg.residual < 1e-11 * leg.image


def test_index_matched_meridian_leg_is_the_identity():
    """Glass replaced by air: the Hankel re-expansion returns the beam itself."""
    radius = 25.0
    (interface,) = sphere(radius, AIR2)
    # A narrow Gaussian pupil: the field on the plane ends well inside the cap.
    beam = F.FocusedBeam(
        F.profiles.Gaussian(0.1),
        CIRCULAR,
        focal_length=1.0,
        max_sine=0.8,
        wavelength0=1.0,
        region=AIR,
        frame=Frame([0, 0, 1.5 * radius]),
    )
    leg = AxialTransport(interface, beam.carrier, beam.frame, 1.0, scale=20.0)
    field, report = leg.transmit(BeamField(beam, chart(beam, radius)))
    # The unsupported amplitude is sqrt(1 - carried / total): roundoff leaves 1e-8.
    assert (
        report.azimuthal_orders == (1,) and report.unsupported_relative_amplitude < 1e-7
    )
    xy = rng.uniform(-4, 4, (40, 2))
    points = o.sampling.PointGrid(np.c_[xy, rng.uniform(5, 40, 40)])
    exact = beam.evaluate(points)
    for method in ("direct", "auto"):
        approximate = field.evaluate(points, method=method)
        for a, b in (
            (approximate.electric.values, exact.electric.values),
            (approximate.magnetic, exact.magnetic),
        ):
            assert_allclose(host(a), host(b), atol=1e-9 * abs(host(b)).max())


def lattice_spectrum(beam, radius, spacing=0.3, margin=1.35):
    """The beam sampled on a lattice one wavelength before the vertex."""
    z0 = -radius - 1.0
    half = margin * (1.5 * radius - z0) * np.tan(np.arcsin(0.4)) + 8.0
    count = int(np.ceil(2 * half / spacing))
    count += count % 2
    axis = (np.arange(count) - count // 2) * spacing
    plane = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, z0]))
    sampled = beam.evaluate(plane, method="nufft").electric.values[..., :2]
    return F.angular_spectrum(
        F.TransverseElectricField(plane, 1.0, sampled), AIR, evanescent="discard"
    )


def focal_points(radius, extent=3.0, count=13):
    axis = np.linspace(-extent, extent, count)
    x, y = np.meshgrid(axis, axis)
    return o.sampling.PointGrid(
        np.c_[x.ravel(), y.ravel(), np.full(x.size, radius / 1.5)]
    )


def test_meridian_leg_matches_the_lattice_leg_on_the_same_input():
    """One sampled beam through both legs, and the beam itself through the meridian.

    The meridian reads the lattice on rings; the lattice leg resamples the
    carried field on its output lattice, its own error (2e-4 here, 2.5e-5 at
    R = 12 on a 0.3 lattice).
    """
    pytest.importorskip("finufft")
    radius = 8.0
    beam = converging(radius, "smooth")
    spectrum = lattice_spectrum(beam, radius, spacing=0.35)
    points = focal_points(radius)
    fields = {}
    for name, kwargs, source in (
        ("lattice", {}, spectrum),
        ("meridian", dict(axisymmetric=True), spectrum),
        ("beam", {}, beam),
    ):
        pipeline = P.PropagationPipeline(sphere(radius), transport="eikonal", **kwargs)
        solution = pipeline.solve(source, carrier=beam.carrier)
        fields[name] = host(solution.evaluate(points).electric.values)
        orders = solution.interactions[0].azimuthal_orders
        # Rings through the square lattice also meet its faint orders 1 + 4j.
        assert orders == () if name == "lattice" else 1 in orders
        assert name != "beam" or orders == (1,)
    scale = np.linalg.norm(fields["beam"])
    assert np.linalg.norm(fields["meridian"] - fields["beam"]) < 2e-6 * scale
    assert np.linalg.norm(fields["lattice"] - fields["meridian"]) < 6e-4 * scale


def mie_error(radius, pupil):
    beam = converging(radius, pupil)
    rho = np.linspace(0, 5, 81)
    points = o.sampling.PointGrid(np.c_[rho, 0 * rho, np.full(rho.size, radius / 1.5)])
    pipeline = P.PropagationPipeline(sphere(radius), transport="eikonal")
    field = host(pipeline.solve(beam).evaluate(points).electric.values)
    mie = MieSphere(beam, o.volumes.SphereVolume([0, 0, 0], radius), GLASS)
    exact = host(mie.evaluate(points, term=1, side="interior").electric.values)

    def norm(E):
        return simpson(rho * np.sum(abs(E) ** 2, -1), x=rho)

    return 100 * np.sqrt(norm(field - exact) / norm(exact)), pipeline.last_solution


@pytest.mark.parametrize("pupil, expected", [("smooth", 0.31022), ("hard", 10.3243)])
def test_beam_through_the_aplanatic_sphere_meets_the_archived_method(pupil, expected):
    """R = 30 lambda0 against Mie first entry, as the archived axisymmetric code.

    The prototype gives 0.31017 % (smooth) and, with its fan cut where the
    cap ends, 10.3243 % (hard); the lattice leg gives 0.31022 % (smooth).
    """
    error, solution = mie_error(30.0, pupil)
    assert abs(error - expected) < 2e-4 * expected
    assert solution.interactions[0].azimuthal_orders == (1,)


def test_linear_input_is_the_mean_of_the_circular_ones_and_radial_is_order_zero():
    radius = 15.0
    points = focal_points(radius, 2.0, 7)
    values = {}
    for name, state in (
        ("x", F.polarization.x),
        ("+", F.polarization.cartesian(1, 1j)),
        ("-", F.polarization.cartesian(1, -1j)),
        ("radial", F.polarization.radial),
    ):
        pipeline = P.PropagationPipeline(sphere(radius), transport="eikonal")
        solution = pipeline.solve(converging(radius, "gaussian", state))
        values[name] = host(solution.evaluate(points).electric.values)
        values[name, "orders"] = solution.interactions[0].azimuthal_orders
    assert values["x", "orders"] == (-1, 1) and values["radial", "orders"] == (0,)
    mean = (values["+"] + values["-"]) / 2
    assert_allclose(values["x"], mean, atol=1e-12 * abs(mean).max())
    on_axis = values["radial"][len(values["radial"]) // 2]
    assert abs(on_axis[:2]).max() < 1e-12 * abs(on_axis[2])


def test_transmitted_field_obeys_maxwell_in_the_glass():
    """Z0 H = curl E / (i k0) and div E = 0 by central differences."""
    radius = 15.0
    pipeline = P.PropagationPipeline(sphere(radius), transport="eikonal")
    solution = pipeline.solve(converging(radius, "gaussian", F.polarization.x))
    centre = np.array([[0.7, -0.4, radius / 1.5 + 0.3]])
    h, steps = 2e-3, (-2, -1, 1, 2)
    shifts = [centre + s * h * np.eye(3)[i] for i in range(3) for s in steps]
    sampled = solution.evaluate(o.sampling.PointGrid(np.concatenate([centre] + shifts)))
    E, H = host(sampled.electric.values), host(sampled.magnetic)
    # Fourth-order differences dE/dx_i.
    d = [
        (E[1 + 4 * i] - 8 * E[2 + 4 * i] + 8 * E[3 + 4 * i] - E[4 + 4 * i]) / (12 * h)
        for i in range(3)
    ]
    curl = np.array([d[1][2] - d[2][1], d[2][0] - d[0][2], d[0][1] - d[1][0]])
    Z0 = o.media.medium.Z0
    assert_allclose(Z0 * H[0], curl / (2j * np.pi), atol=1e-6 * abs(Z0 * H[0]).max())
    assert abs(d[0][0] + d[1][1] + d[2][2]) < 1e-6 * 2 * np.pi * abs(E[0]).max()


def test_pipeline_chooses_and_guards_the_meridian_leg():
    radius = 10.0
    beam = converging(radius, "gaussian")
    system = sphere(radius)
    pipeline = P.PropagationPipeline(system, transport="eikonal")
    target = focal_points(radius, 1.0, 3)
    field = pipeline(beam, target)
    assert field.valid.all() and pipeline.last_solution.interactions[
        0
    ].azimuthal_orders == (1,)
    leg = pipeline._operators[0]
    again = pipeline(beam, focal_points(radius, 0.5, 3))
    assert pipeline._operators[0] is leg and again.valid.all()
    # An index-matched plane behind the sphere is bookkeeping: the same field.
    glass = o.media.HomogeneousRegion("glass, second region", o.media.Dielectric(1.5))
    behind = o.system.DielectricInterface(S.Plane(Frame([0, 0, 2.0])), GLASS, glass)
    longer = P.PropagationPipeline(system + (behind,), transport="eikonal")
    assert_allclose(host(longer(beam, target).values), host(field.values), atol=1e-14)
    with pytest.raises(ValueError, match="CartesianGrid"):
        P.PropagationPipeline(system, transport="eikonal", axisymmetric=False).solve(
            beam
        )
    with pytest.raises(TypeError, match="eikonal"):
        P.PropagationPipeline(system).solve(beam)
    with pytest.raises(ValueError, match="axisymmetric"):
        P.PropagationPipeline(system, axisymmetric=True)
    with pytest.raises(ValueError, match="one reference fan"):
        P.PropagationPipeline(
            system, transport="eikonal", reference_fans=((0, 50), None)
        )
    with pytest.raises(ValueError, match="incidence"):
        P.PropagationPipeline(system, transport="eikonal", reference_fans=((50, 20),))
    cap = system[0].surface
    freeform = S.FreeformSurface(cap.height, cap.gradient, cap.frame, cap.hessian)
    with pytest.raises(ValueError, match="surface of revolution"):
        P.PropagationPipeline(
            (o.system.DielectricInterface(freeform, AIR, GLASS),), transport="eikonal"
        ).solve(beam)
    shifted = S.SphericalCap(radius, Frame([0.5, 0, -radius]))
    with pytest.raises(ValueError, match="centred"):
        P.PropagationPipeline(
            (o.system.DielectricInterface(shifted, AIR, GLASS),), transport="eikonal"
        ).solve(beam)
    with pytest.raises(ValueError, match="carrier is not symmetric"):
        pipeline.solve(beam, carrier=F.Carrier.converging([1.0, 0, 15.0]))
    # A window closes the fan before the cap's equator.
    windowed = P.PropagationPipeline(
        system, transport="eikonal", reference_fans=((20, 30),)
    )
    windowed.solve(beam)
    edge = np.degrees(
        np.arccos(
            host(
                windowed._operators[0]
                ._meridian([windowed._operators[0].edge])
                .cos_incident
            )
        )
    )
    assert_allclose(edge, 30, atol=1e-6)
