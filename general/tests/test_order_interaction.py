"""Order-N per-plane-wave interaction inside the production forward chain."""

import numpy as np
import vecdiff_general as o
from numpy.testing import assert_allclose
from vecdiff_general.propagation.experiments.curvature import TaylorInterfaceOperator
from vecdiff_general.propagation.experiments.orders import OrderInteraction
from vecdiff_general.propagation.high_frequency import propagate_high_frequency

PROBES = o.PointGrid([[0.2, 0.1, 5.0], [0.0, 0.0, 7.0], [-0.8, 0.5, 9.0]])


def _single_sphere(count=32):
    air = o.HomogeneousRegion("air", o.Dielectric(1.0))
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    interface = o.DielectricInterface(o.SphericalCap(10), air, glass)
    axis = (np.arange(count) - count // 2) * 8 / count
    grid = o.CartesianGrid(axis, axis)
    source = o.PlaneWaveExpansion([[0, 0, 2 * np.pi]], [[1, 0, 0]], [0, 0, 0], 1, air)
    return interface, grid, source


def test_order_zero_equals_production_fresnel_for_a_plane_wave():
    interface, grid, source = _single_sphere()
    system = o.OpticalSystem((interface,), (3,), (3,))
    production = propagate_high_frequency(source, system, grid, max_sine=0.85)
    ordered = propagate_high_frequency(
        source, system, grid, max_sine=0.85, interaction=OrderInteraction(0)
    )
    assert_allclose(
        ordered.evaluate(PROBES).electric.values,
        production.evaluate(PROBES).electric.values,
        atol=1e-10,
    )


def test_order_three_equals_the_direct_local_recurrence():
    interface, grid, source = _single_sphere()
    reference = TaylorInterfaceOperator(
        source,
        interface,
        grid,
        height=lambda x, y: 1 - (1 - x * x - y * y) ** 0.5,
        length_scale=10,
        order=3,
        aperture=3,
        max_sine=0.85,
    )
    expected, _ = reference.spectrum(3)
    ordered = propagate_high_frequency(
        source,
        o.OpticalSystem((interface,), (3,), (3,)),
        grid,
        max_sine=0.85,
        interaction=OrderInteraction(3),
    )
    assert_allclose(
        ordered.evaluate(PROBES).electric.values,
        expected.evaluate(PROBES).electric.values,
        atol=2e-8,
    )
    # Orders genuinely differ here, so the comparison is not vacuous.
    zeroth, _ = reference.spectrum(0)
    assert (
        np.abs(
            expected.evaluate(PROBES).electric.values
            - zeroth.evaluate(PROBES).electric.values
        ).max()
        > 1e-4
    )


def test_matched_biconvex_element_is_invisible_at_every_order():
    air = o.HomogeneousRegion("air", o.Dielectric(1.0))
    inside = o.HomogeneousRegion("matched", o.Dielectric(1.0, "matched"))
    system = o.OpticalSystem(
        (
            o.DielectricInterface(o.SphericalCap(12.0), air, inside),
            o.DielectricInterface(
                o.SphericalCap(-12.0, o.Frame([0, 0, 2.0])), inside, air
            ),
        ),
        apertures=(3.0, 3.0),
    )
    axis = (np.arange(64) - 32) * 16 / 64
    source = o.PlaneWaveExpansion([[0, 0, 2 * np.pi]], [[1, 0, 0]], [0, 0, 0], 1, air)
    solution = propagate_high_frequency(
        source, system, o.CartesianGrid(axis, axis), interaction=OrderInteraction(3)
    )
    probe = o.PointGrid([[0.0, 0.0, 8.0], [1.5, -1.0, 9.0]])
    field = solution.evaluate(probe, backend="direct").electric.values
    exact = np.zeros_like(field)
    exact[:, 0] = np.exp(2j * np.pi * probe.points[:, 2])
    # Curved charts use the iterative Q^-1 (tolerance 1e-9 by default).
    assert np.abs(field - exact).max() < 1e-8


def test_plano_convex_lens_runs_with_outside_light():
    air = o.HomogeneousRegion("air", o.Dielectric(1.0))
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    system = o.OpticalSystem(
        (
            o.DielectricInterface(o.Plane(), air, glass),
            o.DielectricInterface(
                o.SphericalCap(-12.0, o.Frame([0, 0, 2.0])), glass, air
            ),
        ),
        apertures=(3.0, 3.0),
    )
    axis = (np.arange(64) - 32) * 16 / 64
    source = o.PlaneWaveExpansion([[0, 0, 2 * np.pi]], [[1, 0, 0]], [0, 0, 0], 1, air)
    grid = o.CartesianGrid(axis, axis)
    interaction = OrderInteraction(3)
    three = propagate_high_frequency(source, system, grid, interaction=interaction)
    zero = propagate_high_frequency(
        source, system, grid, interaction=OrderInteraction(0)
    )
    assert len(three.exterior_spectra[0]) == 1
    probe = o.PointGrid([[0.0, 0.0, 20.0]])
    a = three.evaluate(probe, backend="direct").electric.values
    b = zero.evaluate(probe, backend="direct").electric.values
    change = np.linalg.norm(a - b) / np.linalg.norm(b)
    assert 1e-6 < change < 0.05
    assert all(limit > 0 for limit in interaction.limits.values())


def test_complete_curl_boundary_jumps_fall_with_order():
    from vecdiff_general.observables.maxwell import interface_l2_residuals
    from vecdiff_general.sampling import SurfaceGrid

    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    air = o.HomogeneousRegion("air", o.Dielectric(1.0))
    interface = o.DielectricInterface(o.SphericalCap(-20.0), glass, air)
    axis = (np.arange(48) - 24) * 16 / 48
    grid = o.CartesianGrid(axis, axis)
    source = o.PlaneWaveExpansion([[0, 0, 3 * np.pi]], [[1, 0, 0]], [0, 0, 0], 1, glass)
    system = o.OpticalSystem((interface,), (5.0,), (5.0,))
    jumps = {}
    for order in (0, 3):
        solution = propagate_high_frequency(
            source,
            system,
            grid,
            interaction=OrderInteraction(order, complete_curl=True),
        )
        encounter = solution.encounters[0][0]
        q = solution.transports[0][0]
        ids = np.flatnonzero((np.linalg.norm(q.xy, axis=-1) <= 5.0).ravel())
        cut = SurfaceGrid(
            q.grid.points.reshape(-1, 3)[ids],
            q.grid.normals.reshape(-1, 3)[ids],
            q.grid.weights.reshape(-1)[ids],
        )

        def restrict(field, ids=ids, cut=cut):
            return o.ElectromagneticField(
                o.ElectricField(cut, 1, field.electric.values.reshape(-1, 3)[ids]),
                field.magnetic.reshape(-1, 3)[ids],
            )

        inc, ref = restrict(encounter.incident), restrict(encounter.reflected)
        minus = o.ElectromagneticField(
            o.ElectricField(cut, 1, inc.electric.values + ref.electric.values),
            inc.magnetic + ref.magnetic,
        )
        jumps[order] = interface_l2_residuals(
            minus, restrict(encounter.transmitted), inc, cut, n_minus=1.5, n_plus=1.0
        )["tangential_H"]
    assert jumps[0] > 1e-4
    assert jumps[3] < jumps[0] / 50


def test_beam_corrections_do_not_radiate_from_dark_chart():
    """Summed over a beam's plane waves, the corrections stay where the light is.

    A hard cut at the convergent limit (no taper, no window) leaves the
    corrections of some waves without their cancelling partners where the
    beam is dark; the transmitted trace then stops being an outgoing field.
    """
    from vecdiff_general.propagation.experiments.returns import ElementReturns
    from vecdiff_general.propagation.operators import SurfaceTransport

    air = o.HomogeneousRegion("air", o.Dielectric(1.0))
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5, "glass"))
    radius, axis = 10.0, np.arange(-45, 46) / 3
    plane = o.CartesianGrid(axis, axis)
    flat = SurfaceTransport(plane, o.Plane(), air, 1.0, max_sine=0.95)
    x, y = plane.points[..., 0], plane.points[..., 1]
    electric = np.zeros(plane.shape + (3,), complex)
    electric[..., 0] = np.exp(-(x * x + y * y) / 1.8**2)
    trace = o.ElectromagneticField(
        o.ElectricField(flat.grid, 1.0, electric), np.zeros_like(electric)
    )
    source, _ = flat.inverse(trace)
    residual = {}
    for label, interaction in (
        ("zero", OrderInteraction(0)),
        ("hard", OrderInteraction(3, taper=0.0, window=None)),
        ("default", OrderInteraction(3)),
    ):
        element = ElementReturns(
            o.SphericalCap(radius),
            o.SphericalCap(-radius, o.Frame([0, 0, 2 * radius])),
            glass,
            air,
            axis,
            1.0,
            (8.5, 8.5),
            interaction,
            max_sine=0.95,
        )
        q_out, q_in = element.entrance
        encounter = element._encounter(
            "entrance", q_out, source, element.clear_front, element.clear_front
        )
        residual[label] = q_in.inverse(encounter.transmitted)[1].relative_residual
    assert residual["hard"] > 30 * residual["zero"]
    assert residual["default"] < 3 * residual["zero"]
