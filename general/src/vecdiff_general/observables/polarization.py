"""Polarization of a declared two-axis projection of the electric phasor."""

from vecdiff_general._backend import bd, divide


def jones_components(field, basis=None):
    """Project global Cartesian E onto two orthonormal real basis columns."""
    basis = bd.eye(3)[:, :2] if basis is None else bd.asarray(basis, dtype=float)

    if basis.shape != (3, 2) or not bd.allclose(
        basis.T @ basis, bd.eye(2), rtol=0, atol=1e-12
    ):
        raise ValueError(
            "polarization basis must have two orthonormal Cartesian columns"
        )

    return field.values @ basis


def stokes(field, basis=None):
    """Return S0,S1,S2,S3 in electric-amplitude-squared units.

    S3 = 2 Im(E1* E2). For exp(-i omega t), positive S3 means rotation from
    basis axis 1 toward axis 2 as time increases. A high-NA field's projection
    does not describe its entire three-dimensional polarization.
    """
    return stokes_components(jones_components(field, basis))


def ellipse_parameters(field, basis=None):
    """Return orientation, signed ellipticity (radians), and projected amplitude."""
    parameters = stokes(field, basis)
    intensity = parameters[..., 0]
    orientation = 0.5 * bd.arctan2(parameters[..., 2], parameters[..., 1])
    ratio = divide(parameters[..., 3], intensity, intensity > 0)
    ellipticity = 0.5 * bd.arcsin(bd.clip(ratio, -1, 1))

    return orientation, ellipticity, bd.sqrt(intensity)


def stokes_components(components):
    """Stokes values from (..., 2) complex components; same S3 convention as stokes."""
    first, second = bd.asarray(components)[..., 0], bd.asarray(components)[..., 1]
    correlation = first.conj() * second

    return bd.stack(
        (
            abs(first) ** 2 + abs(second) ** 2,
            abs(first) ** 2 - abs(second) ** 2,
            2 * correlation.real,
            2 * correlation.imag,
        ),
        axis=-1,
    )


def ensemble_weights(fields, weights=None):
    """Validate mutually incoherent responses on the same monochromatic grid."""
    if not fields:
        raise ValueError("an ensemble needs at least one field")

    weights = (
        bd.ones(len(fields)) if weights is None else bd.asarray(weights, dtype=float)
    )

    if (
        weights.shape != (len(fields),)
        or not bd.isfinite(weights).all()
        or bd.any(weights < 0)
        or not bd.any(weights > 0)
    ):
        raise ValueError("provide finite nonnegative weights with positive total")

    first = fields[0]

    for field in fields[1:]:
        if field.wavelength0 != first.wavelength0 or not bd.array_equal(
            field.grid.points, first.grid.points
        ):
            raise ValueError("ensemble fields must share wavelength and coordinates")

    return weights


def ensemble_stokes(fields, weights=None, basis=None):
    """Add Stokes values of mutually incoherent electric fields, never their E."""
    fields = tuple(fields)
    weights = ensemble_weights(fields, weights)

    return sum(weight * stokes(field, basis) for field, weight in zip(fields, weights))
