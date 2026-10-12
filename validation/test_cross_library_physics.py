"""Independent physical checks. Neither library imports this validation code."""

from pathlib import Path

import numpy as np
import pytest
import vecdiff_general as general
import vecdiff_stigmatic as stigmatic
from numpy.testing import assert_allclose

WAVELENGTH = 532e-9


def focused_solution():
    system = stigmatic.StigmaticSystem(
        -0.010, 0.006, stigmatic.Dielectric(1.0), stigmatic.Dielectric(1.5)
    )
    rays = system.rays(0.002, radial_order=24, azimuthal_order=64)
    source = stigmatic.cartesian_source(rays, WAVELENGTH, polarization=(1.0, 0.3j))
    return stigmatic.DebyeFocus(stigmatic.transfer(rays, source))


def maxwell_errors(evaluate, points, index, step):
    """Independent central differences of evaluated fields, not modal identities."""
    e_derivatives, h_derivatives = [], []
    for axis in np.eye(3):
        plus = evaluate(points + step * axis)
        minus = evaluate(points - step * axis)
        e_derivatives.append(
            (plus.electric.values - minus.electric.values) / (2 * step)
        )
        h_derivatives.append((plus.magnetic - minus.magnetic) / (2 * step))

    def curl(derivatives):
        dx, dy, dz = derivatives
        return np.stack(
            (dy[:, 2] - dz[:, 1], dz[:, 0] - dx[:, 2], dx[:, 1] - dy[:, 0]), axis=-1
        )

    def divergence(derivatives):
        return sum(derivatives[i][:, i] for i in range(3))

    field = evaluate(points)
    electric, magnetic = field.electric.values, field.magnetic
    k0 = 2 * np.pi / WAVELENGTH
    # omega mu0 = k0 Z0; omega epsilon = k0 n^2 / Z0.
    e_scale, h_scale = np.linalg.norm(electric), np.linalg.norm(magnetic)
    return np.array(
        [
            np.linalg.norm(curl(e_derivatives) - 1j * k0 * general.Z0 * magnetic)
            / (k0 * index * e_scale),
            np.linalg.norm(
                curl(h_derivatives) + 1j * k0 * index**2 / general.Z0 * electric
            )
            / (k0 * index * h_scale),
            np.linalg.norm(divergence(e_derivatives)) / (k0 * index * e_scale),
            np.linalg.norm(divergence(h_derivatives)) / (k0 * index * h_scale),
        ]
    )


@pytest.mark.parametrize("engine", ["general", "stigmatic"])
def test_real_space_maxwell_equations_converge(engine):
    if engine == "stigmatic":
        solution = focused_solution()
        points = solution.origin + WAVELENGTH * np.array(
            [[0.2, 0.1, 0.15], [-0.3, 0.2, -0.1]]
        )

        def evaluate(p):
            return solution.evaluate(stigmatic.PointGrid(p))

        index = 1.5
    else:
        region = general.HomogeneousRegion("glass", general.Dielectric(1.5))
        directions = np.array([[0.6, 0, 0.8], [0, 0.8, 0.6]])
        electric = np.array([[0.8, 0.2j, -0.6], [0.3j, 0.6, -0.8]])
        solution = general.PlaneWaveExpansion(
            2 * np.pi * 1.5 / WAVELENGTH * directions,
            electric,
            np.zeros(3),
            WAVELENGTH,
            region,
        )
        points = WAVELENGTH * np.array([[0.2, 0.1, 0.15], [-0.3, 0.2, -0.1]])

        def evaluate(p):
            return solution.evaluate(general.PointGrid(p))

        index = 1.5
    coarse = maxwell_errors(evaluate, points, index, WAVELENGTH * 2e-4)
    fine = maxwell_errors(evaluate, points, index, WAVELENGTH * 1e-4)
    assert np.max(fine) < 3e-7
    assert np.linalg.norm(fine) < 0.3 * np.linalg.norm(coarse)


def test_independent_angular_synthesis_matches_all_complex_fields():
    focus = focused_solution()
    context = general.HomogeneousRegion("image", general.Dielectric(1.5))
    general_solution = general.PlaneWaveExpansion(
        focus.wavevectors, focus.amplitudes, focus.origin, WAVELENGTH, context
    )
    points = focus.origin + WAVELENGTH * np.array(
        [[0.2, 0.3, -0.5], [-0.7, 0.1, 0], [0.1, -0.2, 0.5]]
    )
    a = focus.evaluate(stigmatic.PointGrid(points))
    b = general_solution.evaluate(general.PointGrid(points))
    assert_allclose(a.electric.values, b.electric.values, rtol=1e-12, atol=1e-10)
    assert_allclose(a.magnetic, b.magnetic, rtol=1e-12, atol=1e-12)
    # This checks independent evaluation of the same modes, not surface physics.


def test_legacy_geometry_transfer_and_green_asymptotic_reference():
    fixture = Path(__file__).parent / "reference_data/legacy_stigmatic.npz"
    with np.load(fixture) as data:
        system = stigmatic.StigmaticSystem(
            -0.010, 0.006, stigmatic.Dielectric(1.0), stigmatic.Dielectric(1.5)
        )
        rays = system.rays(
            float(data["aperture_m"]), radial_order=48, azimuthal_order=96
        )
        result = stigmatic.transfer(
            rays, stigmatic.cartesian_source(rays, float(data["wavelength0_m"]))
        )
        assert_allclose(rays.radius[:, 0], data["radius_m"], rtol=1e-13)
        assert_allclose(
            rays.surface_grid.points[:, 0, 2], data["sag_m"], atol=2e-15, rtol=1e-10
        )
        assert_allclose(
            rays.geometric_amplitude[:, 0], data["geometric_amplitude"], rtol=2e-12
        )
        assert_allclose(result.tp[:, 0], data["tp"], rtol=2e-12)
        assert_allclose(result.ts[:, 0], data["ts"], rtol=2e-12)
        actual = (
            stigmatic.DebyeFocus(result)
            .electric(stigmatic.PointGrid(data["points_m"]))
            .values
        )
        reference = data["green_electric"]
        # Predeclared model-error budget, based on the audit's 8.1e-5 comparison.
        # Absolute complex vector comparison: no fitted gain, phase or Ez sign.
        error = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
        assert error < 1e-4


def test_general_physical_surface_radiation_preserves_legacy_stigmatic_case():
    """Independent prescribed-current comparison, isolated from operational ASM."""
    fixture = Path(__file__).parent / "reference_data/legacy_stigmatic.npz"
    with np.load(fixture) as data:
        incident_region = general.HomogeneousRegion("object", general.Dielectric(1))
        image_region = general.HomogeneousRegion("image", general.Dielectric(1.5))
        surface = general.CartesianOval(-0.010, 0.006, 1, 1.5)
        object_point = np.array([0, 0, -0.010])

        def optical_path(points):
            return np.linalg.norm(points - object_point, axis=-1) - 0.010

        def path_gradient(points):
            displacement = points - object_point
            return displacement / np.linalg.norm(displacement, axis=-1)[..., None]

        def envelope(points):
            displacement = points - object_point
            distance = np.linalg.norm(displacement, axis=-1)
            electric = np.zeros_like(points)
            electric[..., 0] = 1
            electric[..., 2] = -displacement[..., 0] / displacement[..., 2]
            return electric * (0.010 / distance)[..., None]

        source = general.EikonalField(
            optical_path, path_gradient, envelope, WAVELENGTH, incident_region
        )
        quadrature = general.sample_surface(
            surface, float(data["aperture_m"]), radial_order=32, azimuthal_order=96
        )
        # This fixture records a prescribed-current comparison, not the primary
        # operational ASM model. Construct its local Fresnel trace explicitly.
        from vecdiff_general.comparisons import SurfaceRadiation
        from vecdiff_general.propagation._directions import transmitted_direction
        from vecdiff_general.propagation._fresnel import transmit

        incident_direction = source.directions(quadrature.points)
        outgoing = transmitted_direction(incident_direction, quadrature.normals, 1, 1.5)
        electric = transmit(
            source.evaluate(quadrature).electric,
            incident_direction,
            outgoing,
            quadrature.normals,
            1,
            1.5,
        )
        trace = general.ElectromagneticField(
            electric, 1.5 / general.Z0 * np.cross(outgoing, electric.values)
        )
        comparison = SurfaceRadiation.from_field(trace, image_region)
        actual = comparison.evaluate(
            general.PointGrid(data["points_m"])
        ).electric.values
        reference = data["green_electric"]
        relative_error = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
        assert relative_error < 1e-9
