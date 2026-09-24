"""Physical diagnostics evaluated on the reconstructed total fields."""

import numpy as np

from ..media.medium import Z0


def boundary_residuals(solution, grid):
    """Maximum of each boundary residual on a grid on the actual interface.

    Normalization uses the largest incident |E| at these points, with material
    factors for H, D, B. Zero illumination has no defined relative residual.
    """
    frame = solution.interface.frame
    distance = (grid.points - frame.origin) @ frame.normal
    scale = max(1.0, float(np.max(np.abs(grid.points))))

    if np.max(np.abs(distance)) > 32 * np.finfo(float).eps * scale:
        raise ValueError("boundary diagnostics require points on the interface")

    minus = solution.evaluate(grid, side="minus")
    plus = solution.evaluate(grid, side="plus")
    e1, e2 = minus.electric.values, plus.electric.values
    h1, h2 = minus.magnetic, plus.magnetic
    n1, n2 = solution.interface.minus.material.n, solution.interface.plus.material.n
    incident = solution.incident.evaluate(grid).electric.values
    e_scale = np.max(np.linalg.norm(incident, axis=-1))

    if e_scale == 0:
        raise ValueError(
            "relative boundary residual needs nonzero incident illumination"
        )

    ni = solution.incident.region.material.n
    h_scale = ni * e_scale / Z0
    normal = frame.normal

    return {
        "tangential_E": float(
            np.max(np.linalg.norm(np.cross(normal, e2 - e1), axis=-1)) / e_scale
        ),
        "tangential_H": float(
            np.max(np.linalg.norm(np.cross(normal, h2 - h1), axis=-1)) / h_scale
        ),
        "normal_D": float(
            np.max(np.abs((n2 * n2 * e2 - n1 * n1 * e1) @ normal)) / (ni * ni * e_scale)
        ),
        "normal_B": float(np.max(np.abs((h2 - h1) @ normal)) / h_scale),
    }


def interface_residuals(
    minus,
    plus,
    normals,
    minus_material,
    plus_material,
    *,
    reference_electric,
    reference_index,
):
    """Four Maxwell jump residuals for explicitly supplied one-sided traces.

    Use reconstructed, converged one-sided limits to assess a radiation model.
    Passing the prescribed local Fresnel traces only tests the local law. The
    caller supplies a fixed incident E scale so a dark point cannot change the
    normalization silently. E residuals are scaled by E_ref; H by n_ref E_ref/Z0;
    D by epsilon0 n_ref² E_ref; B by mu0 n_ref E_ref/Z0.
    """

    if not np.isfinite(reference_electric) or reference_electric <= 0:
        raise ValueError("reference electric amplitude must be finite and positive")

    if not np.isfinite(reference_index) or reference_index <= 0:
        raise ValueError("reference index must be finite and positive")

    if minus.electric.values.shape != plus.electric.values.shape:
        raise ValueError("one-sided traces must have matching shapes")

    if minus.electric.wavelength0 != plus.electric.wavelength0:
        raise ValueError("one-sided traces must have the same vacuum wavelength")

    normals = np.broadcast_to(
        np.asarray(normals, dtype=float), minus.electric.values.shape
    )

    if not np.allclose(np.linalg.norm(normals, axis=-1), 1, rtol=0, atol=1e-12):
        raise ValueError("boundary normals must be unit vectors")

    electric_jump = plus.electric.values - minus.electric.values
    magnetic_jump = plus.magnetic - minus.magnetic
    displacement_jump = (
        plus_material.n**2 * plus.electric.values
        - minus_material.n**2 * minus.electric.values
    )
    magnetic_scale = reference_index * reference_electric / Z0

    return {
        "tangential_E": float(
            np.max(np.linalg.norm(np.cross(normals, electric_jump), axis=-1))
            / reference_electric
        ),
        "tangential_H": float(
            np.max(np.linalg.norm(np.cross(normals, magnetic_jump), axis=-1))
            / magnetic_scale
        ),
        "normal_D": float(
            np.max(np.abs(np.sum(normals * displacement_jump, axis=-1)))
            / (reference_index**2 * reference_electric)
        ),
        "normal_B": float(
            np.max(np.abs(np.sum(normals * magnetic_jump, axis=-1))) / magnetic_scale
        ),
    }
