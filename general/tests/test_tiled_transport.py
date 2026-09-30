"""Tiled Q, Q^H and explicit Q^-1 against the global operators and exact fields."""

import numpy as np
import pytest
import vecdiff_general as o

pytest.importorskip("finufft")

WAVELENGTH = 1.0


def lattice(count=256, spacing=0.5):
    axis = (np.arange(count) - count // 2) * spacing
    return o.CartesianGrid(axis, axis)


def beams(grid, k, components=6):
    """Gaussian beams on lattice-compatible tilts, vanishing at the cell edges."""
    X, Y = np.meshgrid(grid.u, grid.v)
    length = len(grid.u) * grid.spacing[0]

    def snap(sine):
        return 2 * np.pi / length * np.round(k * sine * length / (2 * np.pi))

    field = np.zeros(X.shape + (components,), complex)
    for x0, y0, sx, sy in (
        (0, 0, 0.1, 0.0),
        (12, -8, -0.2, 0.1),
        (-10, 14, 0.05, -0.3),
    ):
        beam = np.exp(
            -((X - x0) ** 2 + (Y - y0) ** 2) / 9.0**2
            + 1j * (snap(sx) * X + snap(sy) * Y)
        )
        field[..., 0] += beam
        field[..., components - 2] += 0.5 * beam
    return field


SURFACES = {
    "offset plane": o.Plane(o.Frame([0, 0, 6.0])),
    "tilted plane": o.FreeformSurface(
        lambda xy: 0.06 * xy[..., 0] - 0.04 * xy[..., 1],
        lambda xy: np.broadcast_to([0.06, -0.04], xy.shape),
    ),
    "convex sphere": o.SphericalCap(400.0),
    "concave sphere": o.SphericalCap(-400.0),
}


@pytest.mark.parametrize("name", SURFACES)
def test_tiled_forward_and_adjoint_match_the_global_operators(name):
    grid = lattice()
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    exact = o.SurfaceTransport(grid, SURFACES[name], glass, WAVELENGTH)
    tiled = o.TiledSurfaceTransport(
        grid, SURFACES[name], glass, WAVELENGTH, tile=32, tolerance=1e-8
    )
    field = beams(grid, exact.k)
    coefficients = (
        np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(field, axes=(0, 1)), axes=(0, 1)), axes=(0, 1)
        )
        * exact.active[..., None]
    )
    plane = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(coefficients, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )
    on_surface = exact._synthesize(coefficients)
    forward = tiled.forward_field(plane)
    assert np.linalg.norm(forward - on_surface) <= 1e-7 * np.linalg.norm(on_surface)

    adjoint = exact._adjoint(on_surface) * exact.size * exact.active[..., None]
    tiled_adjoint = tiled.adjoint_spectrum(on_surface)
    assert np.linalg.norm(tiled_adjoint - adjoint) <= 1e-7 * np.linalg.norm(adjoint)


def test_explicit_inverse_recovers_a_curved_interface_field_from_both_traces():
    # The tiled Q^H equals the global one (test above); small cells use the latter.
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
    tiled = o.SurfaceTransport(grid, o.SphericalCap(radius), glass, WAVELENGTH)
    points = tiled.grid.points
    exact = sphere_single_interface(points, WAVELENGTH, radius, 1.0, 1.5)["transmitted"]
    edge = 0.5 * erfc(
        (np.hypot(points[..., 0], points[..., 1]) - 0.36 * radius) / (0.025 * radius)
    )
    trace = o.ElectromagneticField(
        o.ElectricField(tiled.grid, WAVELENGTH, exact[0] * edge[..., None]),
        exact[1] * edge[..., None],
    )
    spectrum = tiled.radiate(trace)
    probe = np.linspace(-0.15, 0.15, 9) * radius
    X, Y = np.meshgrid(probe, probe)
    height = radius - np.sqrt(radius**2 - 2 * (0.15 * radius) ** 2) + 2.0
    targets = np.stack((X, Y, np.full_like(X, height)), -1)
    ours = spectrum.expansion.evaluate(o.PointGrid(targets)).electric.values
    reference = sphere_single_interface(targets, WAVELENGTH, radius, 1.0, 1.5)[
        "transmitted"
    ][0]
    assert np.linalg.norm(ours - reference) <= 1e-4 * np.linalg.norm(reference)
