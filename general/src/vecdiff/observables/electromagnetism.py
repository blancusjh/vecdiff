"""Electromagnetic observables with explicit SI units and integration context."""

from vecdiff._backend import bd
from vecdiff.fields import ElectricField, ElectromagneticField
from vecdiff.media.medium import MU0


def poynting(field: ElectromagneticField):
    """Time-averaged energy-flow density in W/m²."""

    return 0.5 * bd.real(bd.cross(field.electric.values, field.magnetic.conj()))


def irradiance(field: ElectromagneticField, normal=None):
    """Time-averaged power per unit area through a surface, in W/m²: S . n.

    This is what a detector on that surface measures, and what "intensity"
    means for a beam; |E|² (ElectricField.norm2) is not, at high NA or off
    normal incidence. ``normal`` is a unit vector (3,) or one per point; by
    default the normals of the field's grid (a plane or a surface grid).
    """
    if normal is None:
        if not hasattr(field.grid, "normals"):
            raise ValueError(
                "irradiance needs the surface's normal: pass normal=, or "
                "evaluate on a plane or surface grid"
            )
        normal = field.grid.normals
    normal = bd.asarray(normal, dtype=float)
    if not bd.allclose(bd.linalg.norm(normal, axis=-1), 1, rtol=0, atol=1e-12):
        raise ValueError("normal must be a unit vector")
    return bd.sum(poynting(field) * normal, axis=-1)


def power(field: ElectromagneticField):
    """Signed power in watts through the field grid's oriented area measure."""

    if not hasattr(field.grid, "weights") or not hasattr(field.grid, "normals"):
        raise ValueError("power requires an oriented surface quadrature")

    normal_flux = bd.sum(poynting(field) * field.grid.normals, axis=-1)

    return float(bd.sum(normal_flux * field.grid.weights))


def energy_density(field: ElectromagneticField, material):
    """Time-averaged J/m³ for the initial nondispersive, lossless material model."""

    return (
        material.epsilon * field.electric.norm2()
        + MU0 * bd.sum(bd.abs(field.magnetic) ** 2, axis=-1)
    ) / 4


def coherent_sum(fields, amplitudes=None):
    """Sum fields before forming quadratic observables; preserve interference."""
    fields = tuple(fields)

    if not fields:
        raise ValueError("at least one field is required")

    coefficients = (
        bd.ones(len(fields), dtype=complex)
        if amplitudes is None
        else bd.asarray(amplitudes, dtype=complex)
    )

    if coefficients.shape != (len(fields),) or not bd.isfinite(coefficients).all():
        raise ValueError("provide one finite complex amplitude per field")

    first = fields[0]

    for field in fields[1:]:
        if (
            field.electric.wavelength0 != first.electric.wavelength0
            or not bd.array_equal(field.grid.points, first.grid.points)
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
