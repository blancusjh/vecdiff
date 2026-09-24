"""Finite plano-convex lens assembly: sphere table, entrance quadrature, guards."""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose
from scipy.special import j0
from vecdiff_general.media.medium import Z0
from vecdiff_general.propagation.experiments.polar_lens import (
    PolarTaylorLens,
    SphericalExitTable,
    canonical_values,
    convergent_incidence,
    entrance_modes,
    series_ratio,
)
from vecdiff_general.propagation.experiments.taylor_interface import interface_expansion


def _direct(
    points_xy, direction, amplitude, radius, thickness, wavelength, branch, order
):
    """Plain recurrence at physical exit points; no rotation, no table."""
    expansion = interface_expansion(
        lambda u, v: -1 + (1 - u * u - v * v) ** 0.5,
        points_xy,
        n_incident=1.5,
        n_transmitted=1.0,
        direction=direction,
        amplitude=amplitude,
        order=order,
        length_scale=radius,
    )
    trace = expansion.trace(branch, wavelength, order=order)
    # Chart vertex at z=0; the lens vertex is at z=thickness.
    k = 2 * np.pi * 1.5 / wavelength
    shift = np.exp(1j * k * direction[2] * thickness)
    return shift * trace.electric.values, shift * Z0 * trace.magnetic


def test_sphere_table_matches_direct_recurrence_at_random_pairs():
    rng = np.random.default_rng(3)
    radius, wavelength = 30.0, 1.0
    table = SphericalExitTable(radius, wavelength, 1.5, 1.0, np.radians(38.5), order=3)
    assert table.interpolation_tail < 1e-13
    for _ in range(4):
        sine, azimuth = rng.uniform(0, 0.25), rng.uniform(0, 2 * np.pi)
        d = np.array(
            [sine * np.cos(azimuth), sine * np.sin(azimuth), np.sqrt(1 - sine**2)]
        )
        a = rng.normal(size=3) + 1j * rng.normal(size=3)
        a -= d * (a @ d)
        xy = rng.uniform(-8, 8, (4, 2))
        z = -radius + np.sqrt(radius**2 - np.sum(xy**2, -1))
        normals = np.column_stack((xy, z + radius)) / radius
        phase = np.exp(1j * 2 * np.pi * 1.5 * (np.column_stack((xy, z)) @ d))[:, None]
        for branch in ("reflected", "transmitted"):
            local = table.apply(d[None], a[None], normals, branch)
            for n in range(4):
                e, h = _direct(xy, d, a, radius, 0.0, wavelength, branch, n)
                assert_allclose(phase * local[:, n, 0], e, atol=1e-11 * abs(a).max())
                assert_allclose(phase * local[:, n, 1], h, atol=1e-11 * abs(a).max())


def test_entrance_quadrature_reproduces_band_limited_disk_on_axis():
    aperture, wavelength, sine = 4.0, 1.0, 0.3
    vectors, amplitudes, retained = entrance_modes(
        aperture, wavelength, 1.0, 1.5, sine, 40, 16
    )
    kmax = 2 * np.pi * 1.5 * sine / wavelength
    # Sum of all modes at the origin: t (1 - J0(a kmax)), exactly.
    assert_allclose(amplitudes.sum(0)[0], 0.8 * (1 - j0(aperture * kmax)), rtol=1e-12)
    assert abs(amplitudes.sum(0)[1]) < 1e-15
    assert 0 < retained < 1
    assert_allclose(
        np.linalg.norm(vectors, axis=-1), 2 * np.pi * 1.5 / wavelength, rtol=1e-14
    )
    assert np.all(vectors[:, 2] > 0)


def _system(radius, aperture, thickness):
    air, glass = [o.HomogeneousRegion(str(n), o.Dielectric(n)) for n in (1.0, 1.5)]
    return o.OpticalSystem(
        (
            o.DielectricInterface(o.Plane(), air, glass),
            o.DielectricInterface(
                o.SphericalCap(-radius, o.Frame([0, 0, thickness])), glass, air
            ),
        ),
        apertures=(aperture, aperture),
    )


def test_default_band_stops_where_the_local_series_stops_converging():
    radius, aperture, thickness = 10.0, 4.0, 2.0
    grid = o.CartesianGrid(np.linspace(-16, 16, 32), np.linspace(-16, 16, 32))
    lens = PolarTaylorLens(_system(radius, aperture, thickness), grid, 1.0)
    limit = convergent_incidence(radius, 1.0, 1.5, 1.0, 3)
    assert_allclose(lens.series_limit, limit)
    assert_allclose(lens.max_incidence, limit, rtol=1e-12)
    assert lens.table.series_ratio(lens.max_incidence) <= 0.5
    # Beyond the limit the retained terms grow: the band must not extend there.
    beyond = canonical_values(radius, 1.0, 1.5, 1.0, [limit + np.radians(3)], 3)
    assert series_ratio(np.moveaxis(beyond[2], -1, 0), 2 * np.pi * radius)[0] > 0.5
    with pytest.raises(ValueError, match="converges"):
        PolarTaylorLens(
            _system(radius, aperture, thickness),
            grid,
            1.0,
            internal_sine=1.05 * lens.internal_sine,
        )


def test_lens_trace_equals_direct_mode_sum():
    radius, aperture, thickness, wavelength = 10.0, 4.0, 2.0, 1.0
    axis = (np.arange(32) - 16) * 8 * aperture / 32
    grid = o.CartesianGrid(axis, axis)
    lens = PolarTaylorLens(
        _system(radius, aperture, thickness),
        grid,
        wavelength,
        radial_modes=3,
        azimuthal_modes=4,
        branches=("transmitted", "reflected"),
    )
    vectors, amplitudes, _ = entrance_modes(
        aperture, wavelength, 1.0, 1.5, lens.internal_sine, 3, 4
    )
    points = lens.transport.grid.points
    picks = [(16, 16), (18, 14), (14, 18)]
    xy = np.array([points[i, j, :2] for i, j in picks])
    k = 2 * np.pi * 1.5 / wavelength
    for branch in ("transmitted", "reflected"):
        for n in (0, 3):
            e_sum = np.zeros((len(xy), 3), complex)
            h_sum = np.zeros_like(e_sum)
            for vector, amplitude in zip(vectors, amplitudes):
                e, h = _direct(
                    xy, vector / k, amplitude, radius, thickness, wavelength, branch, n
                )
                e_sum += e
                h_sum += h
            trace = lens.traces[branch][n]
            got_e = np.array([trace.electric.values[i, j] for i, j in picks])
            got_h = np.array([Z0 * trace.magnetic[i, j] for i, j in picks])
            assert_allclose(got_e, e_sum, atol=1e-12 * abs(e_sum).max())
            assert_allclose(got_h, h_sum, atol=1e-12 * abs(h_sum).max())
