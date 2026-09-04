"""Independent Lorenz-Mie reference, optional miepython 3.3.0 dependency.

Incident E=x exp(ik_env z), time exp(-i omega t). Upstream H is normalized
to the incident magnetic amplitude; multiply by n_env for vecdiff's Z0 H.
No production Fresnel or radiation routine is called by this reference.
"""
import numpy as np


def fields(points, radius, *, wavelength=1., sphere_index=1.5, environment_index=1., n_pole=0):
    from miepython.field import eh_near_cartesian
    p = np.asarray(points)
    e, h = eh_near_cartesian(wavelength, 2*radius, sphere_index, environment_index,
                             p[..., 0], p[..., 1], p[..., 2], n_pole=n_pole)
    return np.moveaxis(e, 0, -1), environment_index*np.moveaxis(h, 0, -1)
