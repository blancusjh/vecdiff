"""Physical controls for the opt-in local flux-factor investigation."""

from dataclasses import replace

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose
from vecdiff_general.propagation.experiments.flux import normal_flux, refine_flux
from vecdiff_general.propagation.operators import SurfaceTransport, fresnel_transmission
from vecdiff_general.propagation.references.benchmarks import lens_case
from vecdiff_general.propagation.references.diagnostics import maxwell_residuals


def scaled(spectrum, factor):
    return replace(
        spectrum,
        expansion=replace(
            spectrum.expansion, amplitudes=spectrum.expansion.amplitudes * factor
        ),
    )


@pytest.mark.parametrize("wavelength", [1.0, 532e-9])
@pytest.mark.parametrize("curved", [False, True])
def test_exact_wave_and_manufactured_amplitude_error(wavelength, curved):
    region = o.HomogeneousRegion("medium", o.Dielectric(1.3))
    axis = (np.arange(16) - 8) * wavelength / 4
    plane = o.CartesianGrid(axis, axis)
    surface = o.SphericalCap(20 * wavelength) if curved else o.Plane()
    transport = SurfaceTransport(plane, surface, region, wavelength)
    ku = 2 * np.pi / (4 * wavelength)
    k = 2 * np.pi * region.material.n / wavelength
    kz = np.sqrt(k * k - ku * ku)
    source = o.PlaneWaveExpansion(
        [[ku, 0, kz]], [[kz / k, 0.3j, -ku / k]], [0, 0, 0], wavelength, region
    )
    target = source.evaluate(transport.grid)
    exact, _ = transport.inverse(target, tolerance=1e-12)
    actual = list(
        refine_flux(transport, scaled(exact, 0.8), normal_flux(target), iterations=2)
    )
    assert actual[-1].power_error < 1e-12
    field = transport.forward(actual[-1].spectrum)
    assert_allclose(field.electric.values, target.electric.values, atol=2e-9)
    assert_allclose(field.magnetic, target.magnetic, atol=2e-11)
    unchanged = list(refine_flux(transport, exact, normal_flux(target), iterations=1))
    assert_allclose(
        unchanged[-1].spectrum.expansion.amplitudes,
        exact.expansion.amplitudes,
        atol=2e-9,
    )


def test_plane_fresnel_target_includes_signed_reflected_power():
    air = o.HomogeneousRegion("air", o.Dielectric(1))
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    axis = (np.arange(16) - 8) / 4
    transport = SurfaceTransport(o.CartesianGrid(axis, axis), o.Plane(), glass, 1)
    direction = np.array([0.25, 0, np.sqrt(1 - 0.25**2)])
    source = o.PlaneWaveExpansion(
        [2 * np.pi * direction], [[0, 1, 0]], [0, 0, 0], 1, air
    )
    encounter = fresnel_transmission(
        source.evaluate(transport.grid),
        o.DielectricInterface(o.Plane(), air, glass),
        np.broadcast_to(direction, transport.grid.points.shape),
    )
    target = normal_flux(encounter.incident) + normal_flux(encounter.reflected)
    assert np.all(normal_flux(encounter.reflected) < 0)
    assert_allclose(target, normal_flux(encounter.transmitted), rtol=1e-14)
    spectrum, _ = transport.inverse(encounter.transmitted)
    result = list(refine_flux(transport, spectrum, target, iterations=2))
    assert result[-1].flux_rms < 1e-12
    assert_allclose(
        result[-1].spectrum.expansion.amplitudes,
        spectrum.expansion.amplitudes,
        atol=1e-12,
    )


def test_local_correction_reduces_flux_error_and_preserves_maxwell_field():
    source, _, volume, system = lens_case(2, wavelength0=1)
    axis = (np.arange(64) - 32) * 3 * volume.rim_radius / 64
    solution = o.propagate_high_frequency(source, system, o.CartesianGrid(axis, axis))
    transport = solution.transports[-1][1]
    encounter = solution.encounters[0][-1]
    target = normal_flux(encounter.incident) + normal_flux(encounter.reflected)
    steps = list(refine_flux(transport, solution.transmitted[0], target))
    assert steps[-1].flux_rms < 0.7 * steps[0].flux_rms
    assert steps[-1].power_error < 1e-12
    residuals = maxwell_residuals(
        steps[-1].spectrum.evaluate, [[0.1, 0.1, 1.0]], wavelength0=1, index=1
    )
    assert max(residuals.values()) < 2e-6


def test_flux_does_not_determine_phase_and_signed_targets_are_rejected():
    region = o.HomogeneousRegion("air", o.Dielectric(1))
    transport = SurfaceTransport(
        o.CartesianGrid(np.arange(8) / 4, np.arange(8) / 4), o.Plane(), region, 1
    )
    source = o.PlaneWaveExpansion(
        [[0, 0, 2 * np.pi]], [[1, 0, 0]], [0, 0, 0], 1, region
    )
    target_field = source.evaluate(transport.grid)
    exact, _ = transport.inverse(target_field)
    target = normal_flux(target_field)
    wrong_phase = scaled(exact, 1j)
    step = list(refine_flux(transport, wrong_phase, target, iterations=1))[-1]
    assert step.flux_rms < 1e-12
    assert_allclose(
        step.spectrum.expansion.amplitudes, wrong_phase.expansion.amplitudes, atol=1e-12
    )
    with pytest.raises(ValueError, match="nonnegative"):
        list(refine_flux(transport, exact, -target))
