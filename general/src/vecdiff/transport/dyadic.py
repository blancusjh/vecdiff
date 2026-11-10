"""The continued Fresnel dyadic of a carried field, and the weights of a fan of rays."""

import numpy as np

from vecdiff._backend import bd


def carried(incident, rays, s, n_incident, n_transmitted):
    """t_s E_s s + t_p E_p p_t + t_p E_par s_t for unit TE axes s, rows of vectors.

    The continued dyadic of Eq. 4.3 without the factor g exp(i k0 Delta S).
    It does not depend on the sign of s.
    """
    p_i, p_t = bd.cross(s, rays.incident), bd.cross(s, rays.transmitted)
    c_i, c_t = rays.cos_incident, rays.cos_transmitted
    t_s = 2 * n_incident * c_i / (n_incident * c_i + n_transmitted * c_t)
    t_p = 2 * n_incident * c_i / (n_transmitted * c_i + n_incident * c_t)
    e_s = bd.sum(incident * s, axis=-1)
    e_p = bd.sum(incident * p_i, axis=-1)
    e_par = bd.sum(incident * rays.incident, axis=-1)

    return (
        (t_s * e_s)[:, None] * s
        + (t_p * e_p)[:, None] * p_t
        + (t_p * e_par)[:, None] * rays.transmitted
    )


def ray_weights(window, rays):
    """Ray weights: one, the window's callable, or a raised-cosine incidence roll-off."""
    cosine = rays.cos_incident
    if window is None:
        return bd.ones_like(cosine)
    if callable(window):
        return bd.asarray(window(rays), dtype=float)
    start, end = (np.radians(angle) for angle in window)
    fraction = bd.clip(
        (bd.arccos(bd.clip(cosine, -1, 1)) - start) / (end - start), 0, 1
    )
    return 0.5 * (1 + bd.cos(bd.pi * fraction))


def check_window(window):
    """None, a callable of the RayMap, or (start, end) incidence degrees as floats."""
    if window is None or callable(window):
        return window
    window = tuple(float(angle) for angle in window)
    if not (len(window) == 2 and 0 <= window[0] < window[1] <= 90):
        raise ValueError(
            "window is (start, end) incidence in degrees with "
            "0 <= start < end <= 90, or a callable of the RayMap"
        )
    return window
