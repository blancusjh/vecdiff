"""The intermediate-surface transport of a sphere's field (experiments.intermediate)."""

import numpy as np

import vecdiff_general as o
from vecdiff_general.propagation.experiments.intermediate import (
    ClosedTransport,
    IntermediateSphere,
    PlaneMarch,
    sphere_samples,
)
from vecdiff_general.propagation.experiments.orders import OrderInteraction

GLASS = o.HomogeneousRegion("glass", o.Dielectric(1.5, "glass"))
AIR = o.HomogeneousRegion("air", o.Dielectric(1.0))


def test_closed_transport_recovers_waves_from_where_they_leave():
    radius, center = 1.5, np.array([0.0, 0.0, 1.5])
    points, normals, areas = sphere_samples(radius, 1 / 6)
    transport = ClosedTransport(900, GLASS, 1.0, points + center, normals, areas, center)
    rng = np.random.default_rng(1)
    coefficients = rng.normal(size=(900, 2)) + 1j * rng.normal(size=(900, 2))
    amplitudes = np.einsum("mi,mij->mj", coefficients, transport.basis)
    fit, report = transport.inverse(transport.matrix @ amplitudes, tolerance=1e-12)
    assert report.relative_residual < 1e-5
    inner = center + rng.normal(size=(200, 3)) * 0.4
    wanted = transport.expansion(amplitudes).evaluate(o.PointGrid(inner)).electric.values
    found = fit.evaluate(o.PointGrid(inner)).electric.values
    assert np.linalg.norm(found - wanted) / np.linalg.norm(wanted) < 1e-4


def _dipole(points, center, k):
    r = points - center
    d = np.linalg.norm(r, axis=-1)
    n = r / d[:, None]
    p = np.array([1.0, 0.0, 0.0])
    far = k**2 * np.cross(np.cross(n, p), n) / d[:, None]
    near = (3 * n * (n @ p)[:, None] - p) * (1 / d**3 - 1j * k / d**2)[:, None]
    return (far + near) * np.exp(1j * k * d)[:, None]


def test_march_carries_an_outgoing_wave_off_the_equatorial_plane():
    # The field of a dipole at the centre, given on the band around the sphere
    # (going up above the equator, down below it), marched through the planes.
    radius, step, k = 2.0, 1 / 8, 2 * np.pi
    center = np.array([0.0, 0.0, radius])
    z = np.arange(-6, 10 + 1e-9, step)
    march = PlaneMarch(z, 20.0, step, center, radius, 1.0)
    band = _dipole(march.shell_points, center, k)
    rising = (march.shell_points[:, 2] >= center[2])[:, None]
    rows = march.march(band * rising, 1) + march.march(band * ~rising, -1)
    zz, xx = np.meshgrid(z, march.x, indexing="ij")
    points = np.stack((xx, 0 * xx, zz), -1).reshape(-1, 3)
    far = (np.linalg.norm(points - center, axis=-1) > radius + 0.5) & (
        abs(points[:, 2] - center[2]) > 2
    ) & (abs(points[:, 0]) < 8)
    exact = _dipole(points[far], center, k)
    error = np.linalg.norm(rows.reshape(-1, 3)[far] - exact) / np.linalg.norm(exact)
    # Near-horizontal waves split between the two marches limit the accuracy
    # (about 5% for R = 5 on a 48-wavelength lattice, 17% in this small case).
    assert error < 0.2


def test_index_matched_sphere_leaves_the_beam_unchanged():
    # With the same index inside, every wave crosses the surface unchanged: the
    # shadow (-incident) of encounter 0 and the exit of pass 1 must cancel, no
    # wave is reflected, and the interior field is the beam itself.
    radius, waist, step = 1.5, 1.0, 1 / 8
    center = np.array([0.0, 0.0, radius])
    axis = np.arange(-40, 41) / 4
    kx, ky = np.meshgrid(2 * np.pi * np.fft.fftfreq(len(axis), 0.25), 2 * np.pi * np.fft.fftfreq(len(axis), 0.25))
    keep = kx**2 + ky**2 < (0.9 * 2 * np.pi) ** 2
    kz = np.sqrt((2 * np.pi) ** 2 - kx[keep] ** 2 - ky[keep] ** 2)
    k = np.stack((kx[keep], ky[keep], kz), -1)
    spectrum = np.pi * waist**2 * np.exp(-(kx[keep] ** 2 + ky[keep] ** 2) * waist**2 / 4) / (len(axis) * 0.25) ** 2
    amplitudes = np.stack((spectrum, 0 * spectrum, -k[:, 0] * spectrum / k[:, 2]), -1)
    beam = o.PlaneWaveExpansion(k, amplitudes, center, 1.0, AIR)
    matched = o.HomogeneousRegion("matched", o.Dielectric(1.0, "matched"))
    sphere = IntermediateSphere(beam, center, radius, matched, AIR, OrderInteraction(0), directions=1500)
    z = np.arange(-3, 7 + 1e-9, step)
    march = PlaneMarch(z, 16.0, step, center, radius, 1.0)
    solution = sphere.run(2, march, terms=2)
    scattered = sum(
        march.march(up, 1) + march.march(down, -1) for up, down in solution.emissions.values()
    )
    zz, xx = np.meshgrid(z, march.x, indexing="ij")
    points = np.stack((xx, 0 * xx, zz), -1).reshape(-1, 3)
    outside = (np.linalg.norm(points - center, axis=-1) > radius) & (abs(points[:, 0]) < 5)
    incident = beam.evaluate(o.PointGrid(points[outside])).electric.values
    assert np.linalg.norm(scattered.reshape(-1, 3)[outside]) / np.linalg.norm(incident) < 0.05
    inner = points[np.linalg.norm(points - center, axis=-1) < radius - 0.3]
    wanted = beam.evaluate(o.PointGrid(inner)).electric.values
    found = solution.internal[0].evaluate(o.PointGrid(inner)).electric.values
    assert np.linalg.norm(found - wanted) / np.linalg.norm(wanted) < 0.02
    assert np.linalg.norm(solution.internal[1].amplitudes) < 1e-6 * np.linalg.norm(
        solution.internal[0].amplitudes
    )
