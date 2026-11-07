"""Physical checks for prescriptions, coherent sources, and multi-surface transport."""

import numpy as np
import pytest
import vecdiff_general as optics
from numpy.testing import assert_allclose
from vecdiff_general.io import load_system, save_system
from vecdiff_general.observables import detector_image, energy_density
from vecdiff_general.observables.polarization import stokes

WAVELENGTH = 532e-9


def slab_system(apertures=(5e-6, 8e-6)):
    incident = optics.media.HomogeneousRegion("entrance", optics.media.Dielectric(1.0))
    glass = optics.media.HomogeneousRegion("glass", optics.media.Dielectric(1.5))
    exit_region = optics.media.HomogeneousRegion("exit", optics.media.Dielectric(1.0))
    front = optics.surfaces.Plane()
    back = optics.surfaces.Plane(optics.geometry.Frame(origin=[0, 0, 10e-6]))
    interfaces = (
        optics.system.DielectricInterface(front, incident, glass),
        optics.system.DielectricInterface(back, glass, exit_region),
    )
    return optics.system.OpticalSystem(interfaces, apertures)


def plane_wave(region, direction=(0, 0, 1), amplitude=(1, 0, 0)):
    return optics.fields.PlaneWaveExpansion(
        2 * np.pi * region.material.n / WAVELENGTH * np.array([direction]),
        np.array([amplitude]),
        np.zeros(3),
        WAVELENGTH,
        region,
    )


def test_spatial_material_assignment_including_explicit_lens_rim():
    system = slab_system()
    front, back = (interface.surface for interface in system.interfaces)
    volume = optics.volumes.VolumeBetweenSurfaces(front, back, 5e-6)
    body = optics.media.MaterialRegion("lens", volume, optics.media.Dielectric(1.5))
    matter = optics.media.MaterialDistribution(optics.media.Dielectric(1.0), (body,))
    points = np.array([[0, 0, 5e-6], [6e-6, 0, 5e-6], [0, 0, -1e-6]])
    assert_allclose(matter.refractive_index(points), [1.5, 1.0, 1.0])
    assert matter.refractive_index(points[0]) == 1.5


def test_prescription_round_trip_preserves_coherent_complex_solution(tmp_path):
    system = slab_system(apertures=(None, None))
    path = tmp_path / "slab.json"
    save_system(system, path)
    restored = load_system(path)
    source = plane_wave(system.interfaces[0].minus)
    first = optics.propagation.PropagationPipeline(system, transport="wave").solve(
        sample(source)
    )
    second = optics.propagation.PropagationPipeline(restored, transport="wave").solve(
        sample(source)
    )
    grid = optics.sampling.PointGrid([[0, 0, 11e-6], [0.2e-6, 0.1e-6, 12e-6]])
    assert_allclose(
        first.evaluate(grid).electric.values,
        second.evaluate(grid).electric.values,
    )
    assert restored.apertures == system.apertures


def test_operational_transport_matches_two_planar_fresnel_encounters():
    system = slab_system(apertures=(None, None))
    source = plane_wave(system.interfaces[0].minus, amplitude=(1, 0.3j, 0))
    axis = (np.arange(24) - 12) * WAVELENGTH
    quadrature = optics.sampling.CartesianGrid(axis, axis)
    result = optics.propagation.PropagationPipeline(system, transport="wave").solve(
        sample(source, quadrature)
    )
    expected_phase = np.exp(2j * np.pi / WAVELENGTH * 1.5 * 10e-6)
    expected = 0.8 * 1.2 * expected_phase * np.array([1, 0.3j, 0])
    second = result.interactions[1].field
    assert_allclose(
        second.electric.values,
        np.broadcast_to(expected, second.electric.grid.points.shape),
        atol=2e-10,
    )


def test_modal_transport_preserves_interference_and_linearity():
    system = slab_system(apertures=(None, None))
    region = system.interfaces[0].minus
    first = plane_wave(region, amplitude=(1, 0.2j, 0))
    second = plane_wave(
        region, direction=(0.25, 0, np.sqrt(1 - 0.25**2)), amplitude=(0, 0.7j, 0)
    )
    combined = optics.fields.PlaneWaveExpansion(
        np.concatenate((first.wavevectors, second.wavevectors)),
        np.concatenate((first.amplitudes, second.amplitudes)),
        first.origin,
        WAVELENGTH,
        region,
    )
    axis = (np.arange(24) - 12) * WAVELENGTH
    grid = optics.sampling.CartesianGrid(axis, axis)
    observation = optics.sampling.PointGrid([[0, 0, 20e-6], [0.5e-6, 0.2e-6, 20e-6]])
    values = [
        optics.propagation.PropagationPipeline(system, transport="wave")
        .solve(sample(s, grid))
        .evaluate(observation)
        .electric.values
        for s in (first, second, combined)
    ]
    assert_allclose(values[2], values[0] + values[1], atol=1e-12)


def test_energy_polarization_and_coherent_detector_interference():
    region = optics.media.HomogeneousRegion("air", optics.media.Dielectric(1.0))
    grid = optics.sampling.PlaneGrid(
        np.linspace(-1e-6, 1e-6, 8), np.linspace(-1e-6, 1e-6, 8)
    )
    field = plane_wave(region, amplitude=(1, 1j, 0)).evaluate(grid)
    assert_allclose(
        stokes(field.electric),
        np.broadcast_to([2, 0, 0, 2], grid.shape + (4,)),
        atol=1e-14,
    )
    assert_allclose(
        energy_density(field, region.material), region.material.epsilon, rtol=1e-12
    )
    coherent = detector_image([field, field], [1, -1], coherent=True)
    incoherent = detector_image([field, field], [1, -1], coherent=False)
    assert_allclose(coherent, 0, atol=1e-30)
    assert_allclose(incoherent, 2 * field.poynting()[..., 2])


def test_sequential_plane_assembly_preserves_single_pass_phase():
    system = slab_system(apertures=(None, None))
    source = plane_wave(system.interfaces[0].minus)
    axis = (np.arange(24) - 12) * WAVELENGTH
    solution = optics.propagation.PropagationPipeline(system, transport="wave").solve(
        sample(source, optics.sampling.CartesianGrid(axis, axis))
    )
    observation = optics.sampling.PointGrid([[0, 0, 12e-6]])
    actual = solution.evaluate(observation).electric.values[0]
    expected = np.array([0.96, 0, 0]) * np.exp(
        2j * np.pi / WAVELENGTH * (1.5 * 10e-6 + 2e-6)
    )
    assert_allclose(actual, expected, atol=1e-13)
    assert len(solution.interactions) == 2


def test_infinite_plane_assembly_rejects_finite_aperture_prescription():
    system = slab_system()
    with pytest.raises(ValueError, match="finite material edges"):
        optics.propagation.PropagationPipeline(system, transport="wave")


def sample(source, grid=None):
    if grid is None:
        axis = (np.arange(24) - 12) * WAVELENGTH
        grid = optics.sampling.CartesianGrid(axis, axis)
    return optics.fields.angular_spectrum(source.evaluate(grid).electric, source.region)
