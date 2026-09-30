"""Explicit Q^-1 (SurfaceTransport.radiate) against an exact curved-interface field."""

import numpy as np
import vecdiff_general as o

WAVELENGTH = 1.0


def test_explicit_inverse_recovers_a_curved_interface_field_from_both_traces():
    """Exact air-to-glass sphere field: traces on Sigma -> radiated spectrum.

    The trace is apodized by a smooth erf edge well outside the compared region;
    the remaining difference is the edge's own diffraction.
    """
    from scipy.special import erfc
    from vecdiff_general.propagation.references.debye import sphere_single_interface

    radius = 300 / (2 * np.pi)
    half = 0.45 * radius
    count = 2 * int(np.ceil(half / 0.5))
    count += (-count) % 16
    axis = (np.arange(count) - count // 2) * (2 * half / count)
    grid = o.CartesianGrid(axis, axis)
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    transport = o.SurfaceTransport(grid, o.SphericalCap(radius), glass, WAVELENGTH)
    points = transport.grid.points
    exact = sphere_single_interface(points, WAVELENGTH, radius, 1.0, 1.5)["transmitted"]
    edge = 0.5 * erfc(
        (np.hypot(points[..., 0], points[..., 1]) - 0.36 * radius) / (0.025 * radius)
    )
    trace = o.ElectromagneticField(
        o.ElectricField(transport.grid, WAVELENGTH, exact[0] * edge[..., None]),
        exact[1] * edge[..., None],
    )
    spectrum = transport.radiate(trace)
    probe = np.linspace(-0.15, 0.15, 9) * radius
    X, Y = np.meshgrid(probe, probe)
    height = radius - np.sqrt(radius**2 - 2 * (0.15 * radius) ** 2) + 2.0
    targets = np.stack((X, Y, np.full_like(X, height)), -1)
    ours = spectrum.expansion.evaluate(o.PointGrid(targets)).electric.values
    reference = sphere_single_interface(targets, WAVELENGTH, radius, 1.0, 1.5)[
        "transmitted"
    ][0]
    assert np.linalg.norm(ours - reference) <= 1e-4 * np.linalg.norm(reference)
