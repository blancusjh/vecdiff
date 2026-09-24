"""Explicit illumination models on the entrance reference sphere."""

from __future__ import annotations

import numpy as np

from .electromagnetic_field import ElectricField


def cartesian_source(rays, wavelength0, *, polarization=(1.0, 0.0), amplitude=1.0):
    """Prescribe global Ex/Ey on G, then complete along the incident ray.

    The source is a geometric reference-sphere model, not an exact spherical
    Maxwell point source. Polarization and amplitude may vary over the grid.
    """
    shape = rays.entrance_grid.shape
    transverse = np.broadcast_to(np.asarray(polarization, complex), shape + (2,)).copy()
    transverse *= np.broadcast_to(np.asarray(amplitude, complex), shape)[..., None]
    direction = rays.incident_direction
    longitudinal = -np.sum(direction[..., :2] * transverse, axis=-1) / direction[..., 2]
    values = np.concatenate((transverse, longitudinal[..., None]), axis=-1)

    return ElectricField(rays.entrance_grid, wavelength0, values)
