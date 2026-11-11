"""The continued Fresnel dyadics of a carried field, and the weights of a fan of rays."""

import numpy as np

from vecdiff._backend import bd

BRANCHES = ("transmitted", "reflected", "mirror")


def fresnel(rays, n_incident, n_transmitted, branch="transmitted"):
    """The TE and TM coefficients (a_s, a_p) a branch carries, per ray.

    "transmitted": t_s, t_p of the boundary n_incident | n_transmitted;
    "reflected": its r_s, r_p, complex of unit modulus beyond the critical
    angle (the transmitted wave decays, Im c_t > 0); "mirror": the ideal
    mirror, r_s = -1 and r_p = 1. They multiply E_s along the TE axis s and
    E_p along p = s x s, with p_r = s x s_r for the reflected wave, so that
    tangential E and H = k x E / (k0 Z0) are continuous across the boundary
    (tangential E vanishes on the mirror). Real indices only.
    """
    c_i = rays.cos_incident
    if branch == "transmitted":
        c_t = rays.cos_transmitted
        t_s = 2 * n_incident * c_i / (n_incident * c_i + n_transmitted * c_t)
        t_p = 2 * n_incident * c_i / (n_transmitted * c_i + n_incident * c_t)
        return t_s, t_p
    if branch == "mirror":
        return -bd.ones_like(c_i), bd.ones_like(c_i)
    if branch != "reflected":
        raise ValueError(f"branch must be one of {BRANCHES}")
    eta = n_incident / n_transmitted
    c_t = bd.sqrt((1 - eta**2 * (1 - c_i**2)).astype(complex))
    r_s = (n_incident * c_i - n_transmitted * c_t) / (
        n_incident * c_i + n_transmitted * c_t
    )
    r_p = (n_transmitted * c_i - n_incident * c_t) / (
        n_transmitted * c_i + n_incident * c_t
    )
    return r_s, r_p


def carried(incident, rays, s, n_incident, n_transmitted, branch="transmitted"):
    """a_s E_s s + a_p E_p p_o + a_p E_par s_o for unit TE axes s, rows of vectors.

    The continued dyadic of Eq. 4.3 without the factor g exp(i k0 Delta S):
    s_o is the outgoing direction of the branch (``rays.transmitted``),
    p_o = s x s_o, and (a_s, a_p) its fresnel() coefficients. It does not
    depend on the sign of s.
    """
    p_i, p_t = bd.cross(s, rays.incident), bd.cross(s, rays.transmitted)
    t_s, t_p = fresnel(rays, n_incident, n_transmitted, branch)
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
