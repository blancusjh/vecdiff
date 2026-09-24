"""Specular reflection and Snell directions at an oriented dielectric surface.

The normal points from the incident material into the transmitted material.
A transmitted direction may be complex under total internal reflection.
"""

import numpy as np

from .._arrays import positive


def unit_direction(vector):
    """Normalize real Cartesian directions, including arrays of directions."""
    vector = np.asarray(vector)

    if np.iscomplexobj(vector) and np.any(vector.imag != 0):
        raise ValueError("a geometrical direction or normal must be real")

    vector = np.asarray(vector.real, dtype=float)

    if vector.shape[-1:] != (3,) or not np.isfinite(vector).all():
        raise ValueError("directions must be finite Cartesian vectors")

    length = np.linalg.norm(vector, axis=-1)

    if np.any(length < 1e-12):
        raise ValueError("cannot normalize a near-zero direction")

    return vector / length[..., None]


def direction_from_wavevector(wavevector, wavenumber):
    """Recover a propagating incident direction without discarding evanescence."""
    wavevector = np.asarray(wavevector, dtype=complex)
    wavenumber = positive(wavenumber, "wavenumber")

    if wavevector.shape[-1:] != (3,) or not np.isfinite(wavevector).all():
        raise ValueError("wavevectors must be finite Cartesian vectors")

    if np.any(np.abs(wavevector.imag) > 1e-12 * wavenumber):
        raise ValueError(
            "incident evanescent waves require a different boundary operator"
        )

    direction = wavevector.real / wavenumber

    if not np.allclose(np.linalg.norm(direction, axis=-1), 1, rtol=1e-11, atol=1e-12):
        raise ValueError("incident wavevectors violate dispersion")

    return direction


def reflected_direction(incident_direction, normal):
    """Specular reflected direction; no field or Fresnel coefficient is needed."""
    incident_direction = unit_direction(incident_direction)
    normal = unit_direction(normal)

    cos_incidence = np.sum(incident_direction * normal, axis=-1)

    return incident_direction - 2 * cos_incidence[..., None] * normal


def transmitted_direction(
    incident_direction, normal, incident_index, transmitted_index
):
    """Snell direction with the outgoing/decaying transmitted branch.

    The result has bilinear norm u dot u = 1. Under TIR its Hermitian norm
    differs from one; normalizing by that norm would change the wavevector.
    """
    incident_direction = unit_direction(incident_direction)
    normal = unit_direction(normal)

    incident_index = positive(incident_index, "incident index")
    transmitted_index = positive(transmitted_index, "transmitted index")

    cos_incidence = np.sum(incident_direction * normal, axis=-1)

    if np.any(cos_incidence <= 1e-12):
        raise ValueError(
            "incident directions must approach the interface away from grazing"
        )

    tangent = incident_direction - cos_incidence[..., None] * normal
    index_ratio = incident_index / transmitted_index

    cos_transmission = np.sqrt(
        (1 - index_ratio**2 * np.sum(tangent * tangent, axis=-1)).astype(complex)
    )

    if np.any(np.abs(cos_transmission) <= 1e-12):
        raise ValueError("exact critical incidence requires a limiting representation")

    return index_ratio * tangent + cos_transmission[..., None] * normal
