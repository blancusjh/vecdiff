"""Surface operators preserve support and compute only the requested branch."""

import ast
from pathlib import Path

import numpy as np
import pytest
import vecdiff_stigmatic as optics
from numpy.testing import assert_allclose
from vecdiff_stigmatic.fourier import frequency_grid
from vecdiff_stigmatic.interfaces import fresnel
from vecdiff_stigmatic.interfaces.directions import transmitted_direction


def test_fourier_depends_on_mathematics_only():
    source = Path(optics.__file__).parent / "fourier"
    external_modules = {"__future__", "numpy", "scipy", "finufft"}

    for path in source.glob("*.py"):
        syntax = ast.parse(path.read_text())

        for node in ast.walk(syntax):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    assert node.level == 1, (
                        f"Fourier dependency leaves its package: {path}"
                    )
                else:
                    assert node.module.split(".")[0] in external_modules

            elif isinstance(node, ast.Import):
                assert all(
                    name.name.split(".")[0] in external_modules for name in node.names
                )


def test_frequency_coordinates_need_only_shape_and_spacing():
    frequency_u, frequency_v = frequency_grid((3, 4), (0.5, 2.0))

    assert_allclose(frequency_u[0], [0, np.pi, -2 * np.pi, -np.pi])
    assert_allclose(frequency_v[:, 0], [0, np.pi / 3, -np.pi / 3])


@pytest.mark.parametrize("branch,amplitude", [("transmit", 0.8), ("reflect", -0.2)])
def test_normal_incidence_operator_preserves_surface_without_computing_other_branch(
    monkeypatch, branch, amplitude
):
    def unwanted_coefficients(*args, **kwargs):
        raise AssertionError("the unrequested branch was evaluated")

    unrequested = (
        "reflection_coefficients"
        if branch == "transmit"
        else "transmission_coefficients"
    )
    monkeypatch.setattr(fresnel, unrequested, unwanted_coefficients)

    normal = np.array([0.6, 0, 0.8])
    polarization = np.array([0.8, 0.2j, -0.6])
    grid = optics.SurfaceGrid([[1e-6, 0, 2e-6]], [normal], [1e-12])
    field = optics.ElectricField(grid, 532e-9, [polarization])

    operation = getattr(fresnel, branch)
    result = operation(field, normal, normal, normal, 1.0, 1.5)

    assert result.grid is grid
    assert result.wavelength0 == field.wavelength0
    assert_allclose(result.values, amplitude * field.values, atol=1e-14)
    assert_allclose(field.values, [polarization])


def test_transmission_operator_retains_complex_tir_direction_and_boundary_trace():
    incident_direction = np.array([np.sin(1.0), 0, np.cos(1.0)])
    normal = np.array([0, 0, 1])
    transmitted = transmitted_direction(incident_direction, normal, 1.5, 1.0)

    assert transmitted[2].imag > 0
    assert_allclose(np.sum(transmitted * transmitted), 1)

    grid = optics.SurfaceGrid([[0, 0, 0]], [normal], [1e-12])
    electric = optics.ElectricField(grid, 532e-9, [[np.cos(1.0), 0, -np.sin(1.0)]])

    after = fresnel.transmit(
        electric, incident_direction, transmitted, normal, 1.5, 1.0
    )
    reflected = fresnel.reflect(
        electric, incident_direction, transmitted, normal, 1.5, 1.0
    )

    assert_allclose(
        after.values[..., :2], (electric.values + reflected.values)[..., :2], atol=1e-14
    )
    assert_allclose(after.Ez, 1.5**2 * (electric.Ez + reflected.Ez), atol=1e-14)
    assert_allclose(np.sum(after.values * transmitted, axis=-1), 0, atol=1e-14)
