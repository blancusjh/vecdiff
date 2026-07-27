"""Pupil mappings for the transform from the image sphere G' to the focal plane.

The focal-plane step is a Debye reduction, not a Fraunhofer transform from a
plane, and that fixes both the integration variable and its weight.

Starting from the Debye integral over the exit sphere of radius ``|zi|``

    E(P) = e^{i k_i |zi|} (-i k_i |zi| / 2 pi) * int E_G'(s) e^{i k_i s . r_P} dOmega

and substituting the *sine* mapping ``u = |zi| sin(alpha_i)``, for which
``dOmega = u du dphi / (|zi|^2 cos(alpha_i))``, gives

    E(P) = e^{i k_i |zi|} (-i k_i / 2 pi |zi|)
           * int [E_G' / cos(alpha_i)] e^{i q u cos(phi - phi_P)} u du dphi

with ``q = k_i rho_P / |zi|``.  So the transform integrates over
``u = |zi| sin(alpha_i)`` and the integrand carries ``1 / cos(alpha_i)``.  That
``q`` is exactly the package's long-standing output scaling
``rho_P = lambda zi q / (2 pi ni)``, and it is the weighting the Debye-Wolf
benchmark notebook already uses.

The perspective mapping ``|zi| tan(alpha_i)`` with a ``cos(alpha_i)`` weight is
also available.  It is the correct *plane-to-sphere flux* relation -- the field
on the tangent plane really is ``cos(alpha_i)`` times the field on the sphere --
but it is not the right change of variable for this transform, and using it
costs about 14 percent in focal amplitude at high aperture.  Measured against
the Franz reference at NA_i = 0.91: the sine mapping reproduces the exact
Maxwell field to a relative RMS of 2e-8, the perspective mapping to 4e-2.

``identity`` reproduces the behaviour the package had before pupil mapping
existed: integrate over the raw pupil radius with no weight.  It is kept so
that tests pinning plumbing have a stable target, and is not a physical model.
"""

import numpy as np

#: Accepted pupil mappings.
MAPPINGS = ("sine", "tangent", "identity")

#: Which mapping pairs with which ``geometry`` mode when none is given.
_DEFAULT_FOR_GEOMETRY = {"full": "sine", "none": "identity"}


def resolve_mapping(mapping: str | None, geometry: str) -> str:
    """Return the mapping to use, defaulting from the ``geometry`` mode."""
    if mapping is None:
        try:
            return _DEFAULT_FOR_GEOMETRY[geometry]
        except KeyError:
            raise ValueError(f"unknown geometry mode {geometry!r}.") from None
    if mapping not in MAPPINGS:
        raise ValueError(f"mapping must be one of {MAPPINGS}; got {mapping!r}.")
    return mapping


def pupil_coordinate(geom, mapping: str) -> np.ndarray:
    """Return the transform variable ``u`` on the pupil grid."""
    zi = abs(geom.zi)
    if mapping == "identity":
        return geom.r
    if mapping == "sine":
        return zi * geom.sin_ai
    if mapping == "tangent":
        with np.errstate(divide="ignore", invalid="ignore"):
            return zi * np.divide(
                geom.sin_ai,
                geom.cos_ai,
                out=np.zeros_like(geom.r),
                where=geom.cos_ai > 0.0,
            )
    raise ValueError(f"mapping must be one of {MAPPINGS}; got {mapping!r}.")


def pupil_weight(geom, mapping: str) -> np.ndarray:
    """Return the Jacobian weight the integrand carries under ``mapping``."""
    if mapping == "identity":
        return np.ones_like(geom.r)
    if mapping == "sine":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.divide(
                1.0,
                geom.cos_ai,
                out=np.zeros_like(geom.r),
                where=geom.cos_ai > 0.0,
            )
    if mapping == "tangent":
        return geom.cos_ai
    raise ValueError(f"mapping must be one of {MAPPINGS}; got {mapping!r}.")


def pupil_transform(geom, mapping: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(u, weight)`` for ``mapping`` on the pupil grid of ``geom``."""
    return pupil_coordinate(geom, mapping), pupil_weight(geom, mapping)


def debye_prefactor(surface, wavelength: float) -> complex:
    """Return the absolute Debye prefactor of the focal-plane field.

    Together with the ``2 pi`` the azimuthal reduction produces, this turns the
    package's Hankel channels into an absolute amplitude:
    ``exp(i k_i |zi|) * (-i k_i / |zi|)``.  Verified against the Franz integral
    in amplitude and phase, not just in shape.
    """
    k_i = 2.0 * np.pi * surface.ni / float(wavelength)
    zi = abs(surface.zi)
    return np.exp(1j * k_i * zi) * (-1j * k_i / zi)


def radius_from_pupil_coordinate(surface, u, mapping: str, *, n_table: int = 20_001):
    """Invert ``u(r)`` on the usable aperture.

    Returns ``(r, valid)``.  ``u`` is strictly increasing in ``r`` for every
    mapping, so a monotone table suffices; the transform grids are much coarser
    than the table, and unlike ``rho_from_r`` this inversion never sits in an
    inner loop.
    """
    u = np.asarray(u, dtype=float)
    r_table = np.linspace(0.0, surface.aperture_limit, int(n_table))
    geom = surface.ray_geometry(r_table)
    u_table = pupil_coordinate(geom, mapping)

    keep = np.concatenate(([True], np.diff(u_table) > 0.0))
    u_table, r_table = u_table[keep], r_table[keep]

    valid = (u >= 0.0) & (u <= u_table[-1])
    r = np.interp(np.clip(u, 0.0, u_table[-1]), u_table, r_table)
    return r, valid


def pupil_extent(surface, mapping: str, r_max: float | None = None) -> float:
    """Return the largest ``u`` the aperture reaches."""
    limit = surface.aperture_limit if r_max is None else min(float(r_max), surface.aperture_limit)
    geom = surface.ray_geometry(np.array([limit]))
    return float(pupil_coordinate(geom, mapping)[0])
