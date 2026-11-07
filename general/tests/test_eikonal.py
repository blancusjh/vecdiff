"""Eikonal transport: surface Hessians, the plane-surface-plane map and its leg."""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose

bd, host = o.backend.bd, o.backend.to_numpy
F, S, P = o.fields, o.surfaces, o.propagation
Frame = o.geometry.Frame
AIR = o.media.HomogeneousRegion("air", o.media.Dielectric(1.0))
AIR2 = o.media.HomogeneousRegion("air, second region", o.media.Dielectric(1.0))
GLASS = o.media.HomogeneousRegion("glass", o.media.Dielectric(1.5))
rng = np.random.default_rng(5)


SURFACES = {
    "convex cap": (S.SphericalCap(5.0), 2.0),
    "concave cap": (S.SphericalCap(-3.0), 1.5),
    "asphere": (S.EvenAsphere(0.2, conic=-0.6, coefficients=(1e-3, -2e-5)), 1.5),
    "surface of revolution": (
        S.AxisymmetricSurface(
            lambda r: 0.1 * r**2 + 0.01 * r**4,
            lambda r: 0.2 * r + 0.04 * r**3,
            curvature_function=lambda r: 0.2 + 0.12 * r**2,
        ),
        1.5,
    ),
    "oval, collimated object": (
        S.CartesianOval.from_conjugates(1.0, -np.inf, 1.5, 3.0),
        0.8,
    ),
    "oval, finite object": (S.CartesianOval.from_conjugates(1.0, -6.0, 1.5, 4.0), 0.8),
    "plane": (S.Plane(), 1.0),
}


@pytest.mark.parametrize("name", list(SURFACES))
def test_surface_hessians_are_the_derivatives_of_their_gradients(name):
    surface, half = SURFACES[name]
    xy = rng.uniform(-half, half, (40, 2)) / np.sqrt(2)
    xy[0] = 0
    step = 1e-5
    numeric = np.stack(
        [
            (
                host(surface.gradient(xy + step * e))
                - host(surface.gradient(xy - step * e))
            )
            / (2 * step)
            for e in np.eye(2)
        ],
        axis=-1,
    )
    analytic = host(surface.hessian(xy))
    assert_allclose(analytic, numeric, atol=1e-9 * abs(analytic).max() + 1e-12)
    assert_allclose(analytic, np.swapaxes(analytic, -1, -2), atol=1e-15)
    assert host(surface.domain(xy)).all()


def test_surfaces_declare_their_graph_domain_and_hessian():
    assert host(S.SphericalCap(2.0).domain([[0.0, 1.9], [0.0, 2.1]])).tolist() == [
        True,
        False,
    ]
    freeform = S.FreeformSurface(lambda xy: xy[..., 0] ** 2, lambda xy: 0 * xy)
    with pytest.raises(NotImplementedError, match="Hessian"):
        freeform.hessian(bd.zeros((1, 2)))
    with pytest.raises(NotImplementedError, match="Hessian"):
        P.InterfaceMap(freeform, F.Carrier.collimated([0, 0, 1]), 1.0, 1.5)


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


@pytest.mark.parametrize(
    "surface, carrier, indices, offset, half",
    [
        # sphere at its vertex plane and at a plane 12 behind the vertex
        (
            S.SphericalCap(100.0, Frame([0, 0, -100.0])),
            F.Carrier.converging([0, 0, 150.0]),
            (1.0, 1.5),
            0.0,
            40.0,
        ),
        (
            S.SphericalCap(100.0, Frame([0, 0, -100.0])),
            F.Carrier.converging([0, 0, 150.0]),
            (1.0, 1.5),
            12.0,
            40.0,
        ),
        (
            S.EvenAsphere(
                1 / 80, conic=-0.5, coefficients=(1e-8,), frame=Frame([0, 0, 10])
            ),
            F.Carrier.diverging([3, -2, -50]),
            (1.0, 1.6),
            0.0,
            15.0,
        ),
        (
            S.SphericalCap(-60.0),
            F.Carrier.collimated([0.1, 0.05, 1.0]),
            (1.5, 1.0),
            -5.0,
            20.0,
        ),
    ],
)
def test_analytic_map_jacobian_matches_finite_differences(
    surface, carrier, indices, offset, half
):
    """Appendix A: D Phi from the surface Hessian and the carrier's direction Jacobian."""
    mapping = P.InterfaceMap(surface, carrier, *indices, offset)
    xy = rng.uniform(-half, half, (120, 2))
    rays = mapping.forward(xy)
    valid = host(rays.valid)
    assert valid.all()
    numeric = finite_difference_jacobian(mapping, xy, 1e-4)
    analytic = host(rays.jacobian)
    assert_allclose(analytic, numeric, atol=2e-9 * abs(numeric).max())
    det = np.linalg.det(analytic)
    sz, ct = host(rays.incident)[:, 2], host(rays.cos_transmitted)
    ci, tz = host(rays.cos_incident), host(rays.transmitted)[:, 2]
    assert_allclose(host(rays.flux), np.sqrt(sz * ct / (ci * tz * det)), rtol=1e-14)


def test_aplanatic_sphere_map_is_stigmatic():
    """Rays toward A = 1.5 R refract through B = R / 1.5 with a constant optical path."""
    radius = 100.0
    surface = S.SphericalCap(radius, Frame([0, 0, -radius]))
    mapping = P.InterfaceMap(
        surface, F.Carrier.converging([0, 0, 1.5 * radius]), 1, 1.5
    )
    xy = rng.uniform(-35, 35, (300, 2))
    rays = mapping.forward(xy)
    assert host(rays.valid).all()
    image = np.array([0, 0, radius + radius / 1.5])  # B in vertex coordinates
    start = np.c_[xy, np.zeros(len(xy))]
    end = np.c_[host(rays.output), np.zeros(len(xy))]
    toward = image - end
    toward /= np.linalg.norm(toward, axis=-1, keepdims=True)
    assert_allclose(host(rays.transmitted), toward, atol=1e-14)
    path = (
        np.linalg.norm([0, 0, 2.5 * radius] - start, axis=-1)
        - host(rays.path)
        - 1.5 * np.linalg.norm(image - end, axis=-1)
    )
    assert abs(path).max() < 1e-12 * radius


def plane_wave(grid, kx, ky, amplitude, region):
    x, y = np.meshgrid(host(grid.u), host(grid.v))
    values = np.asarray(amplitude) * np.exp(1j * (kx * x + ky * y))[..., None]
    return F.angular_spectrum(
        F.TransverseElectricField(grid, 1.0, values), region, evanescent="discard"
    )


@pytest.mark.parametrize("vertex", [0.0, 1.3])
@pytest.mark.parametrize("offset", [0.0, -3.0, 2.5])
def test_carried_plane_wave_is_exact_fresnel_transmission(vertex, offset):
    """On a plane interface a plane wave along its carrier is transmitted exactly,
    from reference planes in front of, at, or behind the interface."""
    n, step = 64, 0.4
    axis = (np.arange(n) - n // 2) * step
    frequency = 2 * np.pi * np.fft.fftfreq(n, step)
    kx, ky = frequency[5], frequency[3]
    direction = np.array([kx, ky, np.sqrt((2 * np.pi) ** 2 - kx**2 - ky**2)])
    surface = S.Plane(Frame([0, 0, vertex]))
    grid = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, vertex + offset]))
    incident = plane_wave(grid, kx, ky, [1.0, 0.5j], AIR)
    leg = P.EikonalTransport(
        grid,
        o.system.DielectricInterface(surface, AIR, GLASS),
        F.Carrier.collimated(direction),
        1.0,
    )
    carried, _ = leg.carry(incident, method="direct")
    exact = P.scatter_plane(
        incident.expansion, o.system.PlanarInterface(surface.frame, AIR, GLASS)
    ).transmitted.evaluate(carried.grid, method="direct")
    lit = host(leg.lit).reshape(carried.grid.shape)
    assert lit.sum() > 0.95 * n * n
    error = abs(host(carried.values) - host(exact.electric.values)[..., :2])
    assert error[lit].max() < 1e-13


def test_index_matched_leg_is_the_identity_on_a_curved_surface():
    n, step = 96, 0.4
    axis = (np.arange(n) - n // 2) * step
    x, y = np.meshgrid(axis, axis)
    grid = o.sampling.CartesianGrid(axis, axis)
    values = np.stack(
        [
            np.exp(-(x**2 + y**2) / 6**2) * np.exp(0.3j * x),
            0.4j * np.exp(-((x - 2) ** 2 + y**2) / 5**2),
        ],
        axis=-1,
    )
    incident = F.angular_spectrum(
        F.TransverseElectricField(grid, 1.0, values), AIR, evanescent="discard"
    )
    leg = P.EikonalTransport(
        grid,
        o.system.DielectricInterface(S.SphericalCap(40.0), AIR, AIR2),
        F.Carrier.converging([0, 0, 80.0]),
        1.0,
    )
    outgoing, report = leg.transmit(incident, method="direct")
    before = host(incident.propagate(0.0).electric.values)
    after = host(outgoing.propagate(0.0).electric.values)
    assert_allclose(after, before, atol=1e-13 * abs(before).max())
    assert report.unsupported_relative_amplitude == 0 and report.dark_samples == 0


def test_carrier_fresnel_matches_spectral_fresnel_for_a_plane_wave_and_is_invisible():
    n, step = 32, 0.4
    axis = (np.arange(n) - n // 2) * step
    frequency = 2 * np.pi * np.fft.fftfreq(n, step)
    kx, ky = frequency[3], frequency[-2]
    direction = np.array([kx, ky, np.sqrt((2 * np.pi) ** 2 - kx**2 - ky**2)])
    grid = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, -1.0]))
    incident = plane_wave(grid, kx, ky, [0.7, -0.4j], AIR)
    system = (o.system.DielectricInterface(S.Plane(), AIR, GLASS),)
    target = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, 2.0]))
    carrier = F.Carrier.collimated(direction)
    spectral = P.PropagationPipeline(system)(incident, target)
    carried = P.PropagationPipeline(system, transmission="carrier")(
        incident, target, carrier=carrier
    )
    assert_allclose(
        host(carried.values),
        host(spectral.values),
        atol=1e-10 * abs(host(spectral.values)).max(),
    )
    # Index-matched: the transmitted trace is the incident trace, E_par included.
    axis = (np.arange(64) - 32) * 0.4
    flat = o.sampling.CartesianGrid(axis, axis)
    surface = S.SphericalCap(30.0)
    interface = o.system.DielectricInterface(surface, AIR, AIR2)
    transport = P.SurfaceTransport(flat, surface, AIR, 1.0)
    gaussian = np.exp(-(axis[None] ** 2 + axis[:, None] ** 2) / 4)
    beam = F.angular_spectrum(
        F.TransverseElectricField(flat, 1.0, np.stack((gaussian, 0.3j * gaussian), -1)),
        AIR,
        evanescent="discard",
    )
    response = P.CarrierFresnel(F.Carrier.converging([0, 0, 50.0])).transmit(
        beam, interface, transport
    )
    trace = transport.forward(beam).electric.values
    assert_allclose(
        host(response.field.electric.values), host(trace), atol=1e-14 * abs(trace).max()
    )


def test_pipeline_checks_the_eikonal_options():
    system = (o.system.DielectricInterface(S.SphericalCap(30.0), AIR, GLASS),)
    with pytest.raises(ValueError, match="transport"):
        P.PropagationPipeline(system, transport="ray")
    with pytest.raises(ValueError, match="carrier"):
        P.PropagationPipeline(system, transport="eikonal", transmission="spectral")
    with pytest.raises(ValueError, match="reference_planes"):
        P.PropagationPipeline(system, reference_planes=(0.0,))
    with pytest.raises(ValueError, match="one reference-plane offset"):
        P.PropagationPipeline(system, transport="eikonal", reference_planes=(0, 1))
    axis = (np.arange(16) - 8) * 0.5
    grid = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, -1.0]))
    incident = plane_wave(grid, 0.0, 0.0, [1.0, 0.0], AIR)
    with pytest.raises(ValueError, match="carrier="):
        P.PropagationPipeline(system, transport="eikonal").solve(incident)
    two = system + (
        o.system.DielectricInterface(
            S.SphericalCap(-20.0, Frame([0, 0, 5.0])), GLASS, AIR
        ),
    )
    with pytest.raises(ValueError, match="one refracting surface"):
        P.PropagationPipeline(two, transmission="carrier").solve(
            incident, carrier=F.Carrier.collimated([0, 0, 1])
        )


def test_output_lattice_holds_the_refracted_image_and_windows_weight_the_rays():
    """Refraction toward a closer image spreads P' beyond the input footprint."""
    radius, n = 20.0, 96
    axis = (np.arange(n) - n // 2) * 0.5
    surface = S.SphericalCap(radius, Frame([0, 0, -radius]))
    grid = o.sampling.CartesianGrid(axis, axis, surface.frame)
    interface = o.system.DielectricInterface(surface, AIR, GLASS)
    carrier = F.Carrier.converging([0, 0, 1.5 * radius])
    leg = P.EikonalTransport(grid, interface, carrier, 1.0)
    out = leg.output_plane
    assert out.shape[0] > n and out.shape[0] % 2 == 0
    assert_allclose(host(out.u[1] - out.u[0]), 0.5)
    image = host(leg.map.forward(np.c_[host(grid.u), 0 * host(grid.u)]).output)
    assert abs(image[:, 0]).max() <= host(out.u[-1])
    x, y = np.meshgrid(axis, axis)
    values = np.stack([np.exp(-(x**2 + y**2) / 8**2), 0 * x], axis=-1)
    incident = F.angular_spectrum(
        F.TransverseElectricField(grid, 1.0, values), AIR, evanescent="discard"
    )
    full, _ = leg.carry(incident)
    half = P.EikonalTransport(
        grid, interface, carrier, 1.0, window=lambda rays: 0.5 + 0 * rays.cos_incident
    )
    assert_allclose(host(half.carry(incident)[0].values), 0.5 * host(full.values))
    rolled = P.EikonalTransport(grid, interface, carrier, 1.0, window=(10.0, 30.0))
    incidence = np.degrees(np.arccos(host(rolled.rays.cos_incident)))
    weight = host(rolled._window(rolled.rays))
    assert_allclose(weight[incidence <= 10], 1)
    assert_allclose(weight[incidence >= 30], 0, atol=1e-15)
    assert ((weight > 0) & (weight < 1)).any()
    with pytest.raises(ValueError, match="window"):
        P.EikonalTransport(grid, interface, carrier, 1.0, window=(30.0, 10.0))
