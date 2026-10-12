"""Independent field checks shared by numerical references and the fast method."""

import numpy as np

from vecdiff_general.media.medium import Z0
from vecdiff_general.observables.diagnostics import interface_residuals
from vecdiff_general.sampling import PointGrid


def maxwell_residuals(
    evaluate, points, *, wavelength0, index, step=None, reference_electric=1.0
):
    """Fourth-order Cartesian finite differences, away from interfaces.

    Each 13-point stencil MUST stay in one homogeneous source-free region.
    E/H are normalized by E_ref and n E_ref/Z0; derivatives by k or k².
    Repeat with step/2: derivative truncation error is distinct from model error.
    Transversality means div E=div H=0 here, not a pointwise E dot ray test.
    """
    if wavelength0 <= 0 or index <= 0 or reference_electric <= 0:
        raise ValueError("wavelength, index and reference amplitude must be positive")
    k = 2 * np.pi * index / wavelength0
    if step is None:
        step = wavelength0 / index / 80
    if not np.isfinite(step) or step <= 0:
        raise ValueError("finite difference step must be finite and positive")
    points = np.asarray(points, float).reshape(-1, 3)
    shifts = [np.zeros(3)]
    for axis in np.eye(3):
        shifts.extend(factor * step * axis for factor in (-2, -1, 1, 2))
    grid = PointGrid(points[None, ...] + np.asarray(shifts)[:, None, :])
    field = evaluate(grid)
    E = field.electric.values / reference_electric
    H = field.magnetic * Z0 / (index * reference_electric)

    def derivatives(values):
        gradients, laplacian = [], np.zeros_like(values[0])
        for axis in range(3):
            a, b, c, d = values[1 + 4 * axis : 5 + 4 * axis]
            gradients.append((a - 8 * b + 8 * c - d) / (12 * step * k))
            laplacian += (-a + 16 * b - 30 * values[0] + 16 * c - d) / (
                12 * step**2 * k**2
            )
        g = np.stack(gradients, -2)
        divergence = np.trace(g, axis1=-2, axis2=-1)
        curl = np.stack(
            (g[:, 1, 2] - g[:, 2, 1], g[:, 2, 0] - g[:, 0, 2], g[:, 0, 1] - g[:, 1, 0]),
            -1,
        )
        return divergence, curl, laplacian

    divE, curlE, lapE = derivatives(E)
    divH, curlH, lapH = derivatives(H)

    def norm(values):
        return float(np.max(np.linalg.norm(values, axis=-1)))

    return {
        "div_E": float(np.max(abs(divE))),
        "div_H": float(np.max(abs(divH))),
        "faraday": norm(curlE - 1j * H[0]),
        "ampere": norm(curlH + 1j * E[0]),
        "vector_wave_E": norm(lapE + E[0]),
        "vector_wave_H": norm(lapH + H[0]),
    }


def boundary_residuals(solution, grid, *, reference_electric=1.0):
    """Identical-point interior/exterior limits on independent surface nodes."""
    return interface_residuals(
        solution.evaluate(grid, side="exterior"),
        solution.evaluate(grid, side="interior"),
        grid.normals,
        solution.exterior.material,
        solution.interior.material,
        reference_electric=reference_electric,
        reference_index=solution.exterior.material.n,
    )


def closed_flux_residual(solution, enclosing_grid):
    """Net outward power / incoming incident power through a CLOSED surface.

    The enclosing quadrature must lie entirely in the exterior. A lossless
    solution has zero net flux; refine this quadrature independently.
    """
    field = solution.evaluate(enclosing_grid, side="exterior")
    incident = solution.source.evaluate(enclosing_grid)
    incoming_density = np.sum(incident.poynting() * enclosing_grid.normals, -1)
    incoming = np.sum(np.maximum(-incoming_density, 0) * enclosing_grid.weights)
    if incoming <= 0:
        raise ValueError("closed flux check needs positive incoming incident power")
    return float(field.power() / incoming)
