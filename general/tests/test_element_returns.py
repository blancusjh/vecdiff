"""Exact Debye terms of a sphere and the operational internal-return iteration."""

import numpy as np
import vecdiff_general as o
from vecdiff_general.propagation.experiments.orders import OrderInteraction
from vecdiff_general.propagation.experiments.returns import (
    ElementReturns,
    reversed_surface,
)
from vecdiff_general.propagation.high_frequency import _chart_surface
from vecdiff_general.propagation.operators import SurfaceTransport
from vecdiff_general.propagation.references.debye import (
    SphereDebyeSeries,
    sphere_debye_coefficients,
)
from vecdiff_general.propagation.references.mie import MieSphere

AIR = o.HomogeneousRegion("air", o.Dielectric(1.0))
GLASS = o.HomogeneousRegion("glass", o.Dielectric(1.5, "glass"))


def _points(center, low, high, count, seed):
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    return center + direction * rng.uniform(low, high, count)[:, None]


def test_debye_coefficients_sum_to_mie():
    radius = 3.0
    source = o.PlaneWaveExpansion([[0, 0, 2 * np.pi]], [[1, 0, 0]], [0, 0, 0], 1, AIR)
    mie = MieSphere(source, o.SphereVolume([0, 0, radius], radius), GLASS)
    c = sphere_debye_coefficients(2 * np.pi * radius, 1.5, 1.0, np.arange(1, 21))
    for key, scattered, interior in (("TE", mie.b, mie.c), ("TM", mie.a, mie.d)):
        q = c[key]
        series = q["T21"] * q["T12"] / (1 - q["R11"])
        np.testing.assert_allclose(
            0.5 * (1 - q["R22"] - series), scattered[:20], atol=1e-13
        )
        np.testing.assert_allclose(q["mie"], interior[:20], rtol=1e-12)


def test_debye_terms_close_on_mie_fields():
    radius, center = 3.0, np.array([0.0, 0.0, 3.0])
    d = np.array([0.2, -0.1, 1.0])
    d /= np.linalg.norm(d)
    a = np.cross(d, [0, 1, 0])
    a /= np.linalg.norm(a)
    source = o.PlaneWaveExpansion(
        [2 * np.pi * d], [a + 0.3j * np.cross(d, a)], [0.1, 0, 0.2], 1, AIR
    )
    mie = MieSphere(source, o.SphereVolume(center, radius), GLASS)
    series = SphereDebyeSeries(source, center, radius, GLASS, order=mie.order)
    inside = _points(center, 0.05 * radius, 0.95 * radius, 30, 1)
    outside = _points(center, 1.1 * radius, 3 * radius, 30, 2)
    E, H = series.fields(inside, "interior", [1, 2, 3, ("after", 3)])
    exact = mie.evaluate(o.PointGrid(inside))
    np.testing.assert_allclose(E.sum(0), exact.electric.values, atol=1e-12)
    np.testing.assert_allclose(H.sum(0), exact.magnetic, atol=1e-12 / 377)
    E, _ = series.fields(outside, "exterior", [0, 1, 2, ("after", 2)])
    scattered = (
        mie.evaluate(o.PointGrid(outside)).electric.values
        - source.evaluate(o.PointGrid(outside)).electric.values
    )
    assert np.abs(E.sum(0) - scattered).max() < 1e-9 * np.abs(scattered).max()


def test_reversed_chart_inverts_backward_modes():
    radius, axis = 3.0, np.arange(-18, 19) / 4
    surface = reversed_surface(o.SphericalCap(radius))
    plane = o.CartesianGrid(axis, axis, surface.frame)
    q = SurfaceTransport(plane, _chart_surface(surface, 0.85 * radius), AIR, 1.0)
    assert np.allclose(q.grid.normals[18, 18], [0, 0, -1])
    period = len(axis) / 4
    k = 2 * np.pi * np.array([2 / period, 1 / period, 0])
    k[2] = -np.sqrt((2 * np.pi) ** 2 - k[0] ** 2 - k[1] ** 2)
    polarization = np.cross(k, [0, 1, 0])
    wave = o.PlaneWaveExpansion(
        [k], [polarization / np.linalg.norm(polarization)], [0, 0, 0], 1, AIR
    )
    spectrum, report = q.inverse(wave.evaluate(q.grid))
    assert report.relative_residual < 1e-8
    probe = o.PointGrid(np.array([[0.3, 0.2, -1.0], [0.0, 0.0, -2.0]]))
    np.testing.assert_allclose(
        spectrum.evaluate(probe).electric.values,
        wave.evaluate(probe).electric.values,
        atol=1e-8,
    )


def test_matched_element_returns_nothing_and_passes_the_beam():
    matched = o.HomogeneousRegion("matched", o.Dielectric(1.0, "matched"))
    radius, axis = 5.0, np.arange(-30, 31) / 4
    front = o.SphericalCap(radius)
    back = o.SphericalCap(-radius, o.Frame([0, 0, 2 * radius]))
    plane = o.CartesianGrid(axis, axis)
    flat = SurfaceTransport(plane, o.Plane(), AIR, 1.0, max_sine=0.95)
    x, y = plane.points[..., 0], plane.points[..., 1]
    electric = np.zeros(plane.shape + (3,), complex)
    electric[..., 0] = np.exp(-(x * x + y * y) / 1.2**2)
    trace = o.ElectromagneticField(
        o.ElectricField(flat.grid, 1.0, electric), np.zeros_like(electric)
    )
    source, _ = flat.inverse(trace)
    element = ElementReturns(
        front,
        back,
        matched,
        AIR,
        axis,
        1.0,
        (4.5, 4.5),
        OrderInteraction(3),
        max_sine=0.95,
    )
    solution = element.run(source, 2)
    # The beam tail beyond the entrance edge is not carried; it is reported.
    assert solution.outside < 1e-4
    probe = o.PointGrid(np.array([[0.0, 0.0, 11.0], [0.4, -0.3, 12.0]]))
    np.testing.assert_allclose(
        solution.exits[1].evaluate(probe).electric.values,
        source.evaluate(probe).electric.values,
        atol=1e-4,
    )
    assert np.abs(solution.exits[0].expansion.amplitudes).max() == 0
    assert np.abs(solution.internal[1].expansion.amplitudes).max() == 0


def _lattice_beam(waist, half, samples):
    """Band-limited Gaussian (E_x at z = 0) as a plane-wave sum on a lattice."""
    axis = np.arange(-half * samples, half * samples + 1) / samples
    plane = o.CartesianGrid(axis, axis)
    flat = SurfaceTransport(plane, o.Plane(), AIR, 1.0, max_sine=0.95)
    x, y = plane.points[..., 0], plane.points[..., 1]
    electric = np.zeros(plane.shape + (3,), complex)
    electric[..., 0] = np.exp(-(x * x + y * y) / waist**2)
    trace = o.ElectromagneticField(
        o.ElectricField(flat.grid, 1.0, electric), np.zeros_like(electric)
    )
    return flat.inverse(trace)[0].expansion


def test_axial_beam_matches_plane_wave_debye_sum():
    from vecdiff_general.propagation.references.axial_beam import AxialBeamSphere

    radius, waist = 2.0, 1.6  # its spectrum is below 1e-10 at the 0.95 band edge
    center = np.array([0, 0, radius])
    source = _lattice_beam(waist, 12, 4)
    lattice = SphereDebyeSeries(source, center, radius, GLASS)
    axial = AxialBeamSphere(waist, 0.0, radius, radius, 1.5)
    inside = _points(center, 0, 0.95 * radius, 40, 1)
    outside = _points(center, 1.05 * radius, 3 * radius, 40, 2)
    for points, side, terms in (
        (inside, "interior", [1, 2, ("after", 2)]),
        (outside, "exterior", [0, 1, 2, ("after", 2)]),
    ):
        want = lattice.fields(points, side, terms, workers=1)[0]
        got = axial.fields(points, side, terms)[0]
        for w, g in zip(want, got):
            assert np.linalg.norm(g - w) <= 1e-7 * np.linalg.norm(w)
    incident = source.evaluate(o.PointGrid(outside)).electric.values
    got = axial.fields(outside, "exterior", ["incident"])[0][0]
    assert np.linalg.norm(got - incident) <= 1e-7 * np.linalg.norm(incident)


def test_axial_beam_tends_to_plane_wave_mie():
    from vecdiff_general.propagation.references.axial_beam import AxialBeamSphere

    radius, waist = 2.0, 200.0
    source = o.PlaneWaveExpansion([[0, 0, 2 * np.pi]], [[1, 0, 0]], [0, 0, 0], 1, AIR)
    mie = MieSphere(source, o.SphereVolume([0, 0, radius], radius), GLASS)
    # Every point is within 3R of the centre: k 3R ~ 38 multipoles suffice.
    axial = AxialBeamSphere(waist, radius, radius, radius, 1.5, order=80)
    center = np.array([0, 0, radius])
    # Includes points on and just off the surface.
    inside = _points(center, 0, radius, 40, 3)
    outside = _points(center, radius * (1 + 1e-9), 3 * radius, 40, 4)
    want = mie.evaluate(o.PointGrid(np.concatenate((inside, outside)))).electric.values
    got = np.concatenate(
        (
            axial.fields(inside, "interior", ["mie"])[0][0],
            axial.fields(outside, "exterior", ["incident", "mie"])[0].sum(0),
        )
    )
    # The beam amplitude varies by (r / w)^2 ~ 4e-4 across the sphere.
    assert np.linalg.norm(got - want) <= 1e-3 * np.linalg.norm(want)
