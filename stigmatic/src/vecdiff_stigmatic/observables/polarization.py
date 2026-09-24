"""Polarization of a declared two-axis projection of the electric phasor."""

import numpy as np


def jones_components(field, basis=None):
    """Project global Cartesian E onto two orthonormal real basis columns."""
    basis = np.eye(3)[:, :2] if basis is None else np.asarray(basis, dtype=float)

    if basis.shape != (3, 2) or not np.allclose(
        basis.T @ basis, np.eye(2), rtol=0, atol=1e-12
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
    components = jones_components(field, basis)
    first, second = components[..., 0], components[..., 1]
    correlation = first.conj() * second

    return np.stack(
        (
            np.abs(first) ** 2 + np.abs(second) ** 2,
            np.abs(first) ** 2 - np.abs(second) ** 2,
            2 * correlation.real,
            2 * correlation.imag,
        ),
        axis=-1,
    )


def ellipse_parameters(field, basis=None):
    """Return orientation, signed ellipticity (radians), and projected amplitude."""
    parameters = stokes(field, basis)
    intensity = parameters[..., 0]
    orientation = 0.5 * np.arctan2(parameters[..., 2], parameters[..., 1])
    ratio = np.divide(
        parameters[..., 3], intensity, out=np.zeros_like(intensity), where=intensity > 0
    )
    ellipticity = 0.5 * np.arcsin(np.clip(ratio, -1, 1))

    return orientation, ellipticity, np.sqrt(intensity)
