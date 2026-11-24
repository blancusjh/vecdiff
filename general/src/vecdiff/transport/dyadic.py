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
    (tangential E vanishes on the mirror). An absorbing medium beyond the
    surface (complex n_transmitted) reflects by the same coefficients with
    n_t c_t = sqrt(n_t^2 - n_i^2 sin^2 theta_i) on the branch Im >= 0 (the
    transmitted wave decays away from the boundary); its |r| < 1. The
    transmitted branch needs real indices: its rays enter the medium.
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
    if isinstance(n_transmitted, complex):
        return absorbing_reflection(c_i, n_incident, n_transmitted)
    eta = n_incident / n_transmitted
    c_t = bd.sqrt((1 - eta**2 * (1 - c_i**2)).astype(complex))
    r_s = (n_incident * c_i - n_transmitted * c_t) / (
        n_incident * c_i + n_transmitted * c_t
    )
    r_p = (n_transmitted * c_i - n_incident * c_t) / (
        n_transmitted * c_i + n_incident * c_t
    )
    return r_s, r_p


def absorbing_reflection(c_i, n_i, n_t):
    """r_s, r_p of the boundary between a transparent medium n_i and an
    absorbing one n_t (complex), for incidence cosines c_i.

    w = n_t c_t = sqrt(n_t^2 - n_i^2 (1 - c_i^2)) is the principal root:
    Im n_t^2 = 2 n' kappa > 0 puts the radicand in the upper half plane, so
    Re w > 0 and Im w > 0, the transmitted wave exp(i k0 w z) decaying into
    the medium. r_p = (n_t c_i - n_i c_t) / (n_t c_i + n_i c_t) is taken
    times n_t / n_t, without c_t itself.
    """
    w = bd.sqrt(n_t**2 - n_i**2 * (1 - c_i**2).astype(complex))
    r_s = (n_i * c_i - w) / (n_i * c_i + w)
    r_p = (n_t**2 * c_i - n_i * w) / (n_t**2 * c_i + n_i * w)
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


def encounter_indices(encounter):
    """(branch, n of the incident region, n beyond the surface) of an encounter.

    An ideal mirror has nothing beyond: its index is the incident one, unused.
    """
    branch = encounter.branch
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}")
    incident = encounter.incident
    if incident is None:
        raise ValueError("the encounter's incident medium is not known")
    from vecdiff.media.absorption import TRANSPORT, require_transparent

    require_transparent(incident, "carrying light along rays", TRANSPORT)
    n_incident = incident.material.n
    far = encounter.far
    if branch == "transmitted":
        require_transparent(far, "carrying light along rays", TRANSPORT)
    return branch, n_incident, n_incident if far is None else far.material.n


def ray_weights(window, rays, aperture=None):
    """Ray weights: one, the window's callable, or a raised-cosine incidence roll-off,
    times the encounter's aperture.

    ``aperture`` is the radius, in the surface's chart, of the hole of a stop
    or the edge of a face: rays that meet the surface beyond it weigh nothing.
    """
    cosine = rays.cos_incident
    if window is None:
        weight = bd.ones_like(cosine)
    elif callable(window):
        weight = bd.asarray(window(rays), dtype=float)
    else:
        start, end = (np.radians(angle) for angle in window)
        fraction = bd.clip(
            (bd.arccos(bd.clip(cosine, -1, 1)) - start) / (end - start), 0, 1
        )
        weight = 0.5 * (1 + bd.cos(bd.pi * fraction))
    if aperture is None:
        return weight
    radius = bd.linalg.norm(rays.surface_point[..., :2], axis=-1)
    return bd.where(radius < aperture, weight, 0.0)


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
