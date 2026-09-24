"""Electromagnetic observables with explicit SI units and integration context."""

import numpy as np

from ..fields import ElectricField, ElectromagneticField
from ..media.medium import MU0


def poynting(field: ElectromagneticField):
    """Time-averaged energy-flow density in W/m²."""

    return 0.5 * np.real(np.cross(field.electric.values, field.magnetic.conj()))


def power(field: ElectromagneticField):
    """Signed power in watts through the field grid's oriented area measure."""

    if not hasattr(field.grid, "weights") or not hasattr(field.grid, "normals"):
        raise ValueError("power requires an oriented surface quadrature")

    normal_flux = np.sum(poynting(field) * field.grid.normals, axis=-1)

    return float(np.sum(normal_flux * field.grid.weights))


def energy_density(field: ElectromagneticField, material):
    """Time-averaged J/m³ for the initial nondispersive, lossless material model."""

    return (
        material.epsilon * field.electric.norm2()
        + MU0 * np.sum(np.abs(field.magnetic) ** 2, axis=-1)
    ) / 4


def coherent_sum(fields, amplitudes=None):
    """Sum fields before forming quadratic observables; preserve interference."""
    fields = tuple(fields)

    if not fields:
        raise ValueError("at least one field is required")

    coefficients = (
        np.ones(len(fields), dtype=complex)
        if amplitudes is None
        else np.asarray(amplitudes, dtype=complex)
    )

    if coefficients.shape != (len(fields),) or not np.isfinite(coefficients).all():
        raise ValueError("provide one finite complex amplitude per field")

    first = fields[0]

    for field in fields[1:]:
        if (
            field.electric.wavelength0 != first.electric.wavelength0
            or not np.array_equal(field.grid.points, first.grid.points)
        ):
            raise ValueError(
                "coherent fields must share wavelength and observation positions"
            )

    electric = sum(
        coefficient * field.electric.values
        for coefficient, field in zip(coefficients, fields)
    )
    magnetic = sum(
        coefficient * field.magnetic for coefficient, field in zip(coefficients, fields)
    )

    return ElectromagneticField(
        ElectricField(first.grid, first.electric.wavelength0, electric), magnetic
    )
