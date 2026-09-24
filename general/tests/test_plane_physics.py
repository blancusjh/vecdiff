import numpy as np
import pytest
from numpy.testing import assert_allclose
from vecdiff_general import (
    Z0,
    Dielectric,
    ElectricField,
    Frame,
    HomogeneousRegion,
    PlanarInterface,
    PlaneGrid,
    PlaneWaveExpansion,
    PointGrid,
    TransverseElectricField,
    angular_spectrum,
    boundary_residuals,
    scatter_plane,
)

WAVELENGTH = 532e-9


def region(n, name="incident"):
    return HomogeneousRegion(name, Dielectric(n))


def wave(
    n=1.0,
    angle=0.0,
    polarization="p",
    *,
    origin=(0, 0, 0),
    basis=np.eye(3),
    name="incident",
):
    direction = np.array([np.sin(angle), 0.0, np.cos(angle)]) @ basis.T
    p = np.array([np.cos(angle), 0.0, -np.sin(angle)]) @ basis.T
    s = np.array([0.0, 1.0, 0.0]) @ basis.T
    electric = {"p": p, "s": s, "elliptic": p + 0.4j * s}[polarization]
    return PlaneWaveExpansion(
        [2 * np.pi * n / WAVELENGTH * direction],
        [electric],
        origin,
        WAVELENGTH,
        region(n, name),
    )


def plane(frame=None):
    return PlaneGrid(
        np.arange(8) * WAVELENGTH / 3,
        np.arange(8) * WAVELENGTH / 3,
        frame if frame is not None else Frame(),
    )


@pytest.mark.parametrize(
    "n1,n2,angle",
    [
        (1.0, 1.5, 0.0),
        (1.0, 1.5, 0.6),
        (1.5, 1.0, 0.4),
        (1.5, 1.0, 1.0),
        (1.0, 1.0, 0.6),
    ],
)
@pytest.mark.parametrize("polarization", ["p", "s", "elliptic"])
def test_boundary_conditions_and_signed_power(n1, n2, angle, polarization):
    incident = wave(n1, angle, polarization)
    interface = PlanarInterface(Frame(), incident.region, region(n2, "transmitted"))
    solution = scatter_plane(incident, interface)
    grid = plane()
    assert max(boundary_residuals(solution, grid).values()) < 3e-12
    incoming = solution.incident.evaluate(grid).power()
    reflected = solution.reflected.evaluate(grid).power()
    transmitted = solution.transmitted.evaluate(grid).power()
    assert reflected <= 1e-14 * incoming
    assert_allclose(
        incoming + reflected, transmitted, rtol=2e-12, atol=2e-12 * incoming
    )


@pytest.mark.parametrize("angle", [0.0, 0.4, 0.9])
def test_independent_tangential_admittance_reference(angle):
    incident = wave(1.0, angle, "elliptic")
    solution = scatter_plane(
        incident, PlanarInterface(Frame(), incident.region, region(1.5, "glass"))
    )
    q = np.sin(angle)
    a, b = np.cos(angle), np.sqrt(1.5**2 - q * q)
    rs = (a - b) / (a + b)
    rp_tangent = (1 / a - 1.5**2 / b) / (1 / a + 1.5**2 / b)
    reflected = np.array([rp_tangent * a, 0.4j * rs, rp_tangent * q])
    assert_allclose(solution.reflected.amplitudes[0], reflected, rtol=1e-12, atol=1e-14)


def test_brewster_cancels_reflected_p():
    incident = wave(1.0, np.arctan(1.5), "p")
    solution = scatter_plane(
        incident, PlanarInterface(Frame(), incident.region, region(1.5, "glass"))
    )
    assert np.linalg.norm(solution.reflected.amplitudes) < 2e-15


def test_tir_phase_and_transmitted_evanescent_decay():
    angle = 1.0
    incident = wave(1.5, angle, "s")
    solution = scatter_plane(
        incident, PlanarInterface(Frame(), incident.region, region(1.0, "air"))
    )
    alpha = np.sqrt((1.5 * np.sin(angle)) ** 2 - 1.0)
    analytic_r = (1.5 * np.cos(angle) - 1j * alpha) / (1.5 * np.cos(angle) + 1j * alpha)
    assert_allclose(solution.reflected.amplitudes[0, 1], analytic_r, atol=1e-14)
    grid = PointGrid([[0, 0, 0], [0, 0, WAVELENGTH]])
    evaluated = solution.evaluate(grid, side="plus")
    assert_allclose(
        evaluated.electric.Ey[1] / evaluated.electric.Ey[0],
        np.exp(-2 * np.pi * alpha),
        rtol=1e-13,
    )
    assert np.max(abs(evaluated.poynting()[:, 2])) < 1e-17
    with pytest.raises(ValueError, match="half-space"):
        solution.evaluate(PointGrid([[0, 0, -WAVELENGTH]]), side="plus")


def test_reversed_incidence_uses_spatial_sides():
    air, glass = region(1.0, "air"), region(1.5, "glass")
    incoming = PlaneWaveExpansion(
        [[0, 0, -2 * np.pi * 1.5 / WAVELENGTH]],
        [[1, 0, 0]],
        [0, 0, 0],
        WAVELENGTH,
        glass,
    )
    solution = scatter_plane(incoming, PlanarInterface(Frame(), air, glass))
    assert solution.incident_side == "plus"
    assert_allclose(solution.transmitted.amplitudes, [[1.2, 0, 0]], atol=1e-14)
    assert max(boundary_residuals(solution, plane()).values()) < 1e-13


def test_rotation_and_translation_covariance():
    basis = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    shift = np.array([2.0, -3.0, 5.0]) * WAVELENGTH
    original = wave(angle=0.4, polarization="elliptic")
    moved = wave(angle=0.4, polarization="elliptic", origin=shift, basis=basis)
    reference = scatter_plane(
        original, PlanarInterface(Frame(), original.region, region(1.5, "glass"))
    )
    transformed = scatter_plane(
        moved, PlanarInterface(Frame(shift, basis), moved.region, region(1.5, "glass"))
    )
    points = np.array([[0.1, 0.2, 0.4], [-0.3, 0.1, 0.7]]) * WAVELENGTH
    expected = reference.evaluate(PointGrid(points), side="plus")
    actual = transformed.evaluate(PointGrid(points @ basis.T + shift), side="plus")
    assert_allclose(
        actual.electric.values, expected.electric.values @ basis.T, atol=2e-14
    )
    assert_allclose(actual.magnetic, expected.magnetic @ basis.T, atol=2e-16)


@pytest.mark.parametrize(
    "basis", [np.eye(3), np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])]
)
def test_fft_completion_and_asm_on_embedded_plane(basis):
    grid = PlaneGrid(
        np.arange(8) * WAVELENGTH / 2,
        np.arange(8) * WAVELENGTH / 2,
        Frame(np.array([0.0, 0.0, 50.0]), basis),
    )
    ku = 2 * np.pi / (8 * WAVELENGTH / 2)
    phase = np.exp(1j * ku * grid.u)[None, :] * np.ones((8, 1))
    values = np.stack((phase, 0.3j * phase), axis=-1)
    spectrum = angular_spectrum(
        TransverseElectricField(grid, WAVELENGTH, values), region(1.0)
    )
    distance = 0.23 * WAVELENGTH
    actual = spectrum.propagate(distance)
    kz = np.sqrt((2 * np.pi / WAVELENGTH) ** 2 - ku * ku)
    vector = np.array([1.0, 0.3j, -ku / kz]) @ basis.T
    expected = phase[..., None] * vector * np.exp(1j * kz * distance)
    assert_allclose(actual.electric.values, expected, atol=3e-14)
    k = np.array([ku, 0.0, kz]) @ basis.T
    assert_allclose(
        actual.magnetic,
        np.cross(k, expected) / (2 * np.pi / WAVELENGTH * Z0),
        atol=1e-15,
    )
    assert_allclose(
        actual.grid.points, grid.points + distance * basis[:, 2], atol=1e-15
    )


@pytest.mark.parametrize("direction", [-1, 1])
def test_evanescent_asm_stays_local_at_large_global_origin(direction):
    grid = PlaneGrid(
        np.arange(8) * WAVELENGTH / 10,
        np.arange(8) * WAVELENGTH / 10,
        Frame([0, 0, 50.0]),
    )
    mode = np.broadcast_to(np.exp(2j * np.pi * np.arange(8) / 8)[None, :], grid.shape)
    values = np.stack((np.zeros_like(mode), mode), axis=-1)
    spectrum = angular_spectrum(
        TransverseElectricField(grid, WAVELENGTH, values),
        region(1.0),
        direction=direction,
    )
    distance = direction * 0.2 * WAVELENGTH
    field = spectrum.propagate(distance)
    alpha = np.sqrt(
        (2 * np.pi / (8 * WAVELENGTH / 10)) ** 2 - (2 * np.pi / WAVELENGTH) ** 2
    )
    assert_allclose(
        field.electric.Ey, mode * np.exp(-alpha * abs(distance)), atol=1e-14
    )


def test_complete_input_is_validated_without_projection():
    grid = plane()
    values = np.broadcast_to([1.0, 0.0, 1.0], grid.shape + (3,))
    with pytest.raises(ValueError, match="not transverse"):
        angular_spectrum(ElectricField(grid, WAVELENGTH, values), region(1.0))


def test_real_grazing_mode_is_rejected_and_filtering_is_reported():
    grid = PlaneGrid(np.arange(8) * WAVELENGTH / 2, np.arange(8) * WAVELENGTH / 2)
    mode = np.broadcast_to((-1.0) ** np.arange(8)[None, :], grid.shape)
    values = np.stack((mode, np.zeros_like(mode)), axis=-1)
    with pytest.raises(ValueError, match="grazing"):
        angular_spectrum(TransverseElectricField(grid, WAVELENGTH, values), region(1.0))
    grid = PlaneGrid(np.arange(8) * WAVELENGTH / 10, np.arange(8) * WAVELENGTH / 10)
    values = np.stack((np.zeros_like(mode), mode), axis=-1)
    filtered = angular_spectrum(
        TransverseElectricField(grid, WAVELENGTH, values),
        region(1.0),
        evanescent="discard",
    )
    assert_allclose(filtered.evanescent_removed_rms, 1.0, atol=1e-15)
    # Empty modes must not trigger overflowing backward evanescent phase factors.
    assert_allclose(filtered.propagate(-1.0).electric.values, 0, atol=0)


def test_identical_electric_traces_do_not_determine_flux():
    forward = wave()
    k = 2 * np.pi / WAVELENGTH
    standing = PlaneWaveExpansion(
        [[0, 0, k], [0, 0, -k]],
        [[0.5, 0, 0], [0.5, 0, 0]],
        [0, 0, 0],
        WAVELENGTH,
        forward.region,
    )
    a, b = forward.evaluate(plane()), standing.evaluate(plane())
    assert_allclose(a.electric.values, b.electric.values, atol=1e-15)
    assert_allclose(a.poynting()[..., 2], 1 / (2 * Z0), atol=1e-17)
    assert_allclose(b.poynting(), 0, atol=1e-17)


def test_coherent_superposition_at_interface():
    first, second = wave(angle=0.2), wave(angle=0.5, polarization="s")
    interface = PlanarInterface(Frame(), first.region, region(1.5, "glass"))
    combined = PlaneWaveExpansion(
        np.concatenate((first.wavevectors, second.wavevectors)),
        np.concatenate((first.amplitudes, 0.7j * second.amplitudes)),
        [0, 0, 0],
        WAVELENGTH,
        first.region,
    )
    grid = PointGrid(
        [[0.2 * WAVELENGTH, 0, WAVELENGTH], [0, WAVELENGTH, 2 * WAVELENGTH]]
    )
    expected = (
        scatter_plane(first, interface).transmitted.electric(grid).values
        + 0.7j * scatter_plane(second, interface).transmitted.electric(grid).values
    )
    assert_allclose(
        scatter_plane(combined, interface).transmitted.electric(grid).values,
        expected,
        atol=1e-14,
    )


def test_coordinate_contract_and_no_material_on_field():
    positions = np.array([[1.0, 2.0, 3.0]])
    grid = PointGrid(positions)
    electric = ElectricField(grid, WAVELENGTH, [[1.0, 0.0, 0.0]])
    positions[:] = 0
    assert_allclose(grid.points, [[1, 2, 3]])
    assert not hasattr(electric, "medium") and not hasattr(electric, "sampling")
    with pytest.raises(ValueError):
        electric.values[0, 0] = 2
    with pytest.raises(TypeError, match="Grid"):
        ElectricField(object(), WAVELENGTH, [[1, 0, 0]])
