"""Reference surfaces and ray geometry for the stigmatic diopter.

A field is the pair ``(r, E(r))``: the samples and the geometry they live on.
This module supplies the second half of that pair.

Two kinds of reference geometry appear in the stigmatic problem:

* the reference spheres ``G`` (centred on the object point ``A = (0, 0, z0)``,
  radius ``|z0|``) and ``G'`` (centred on the image point ``A' = (0, 0, zi)``,
  radius ``|zi|``), on which the field is transverse and carries two amplitudes;
* the tangent planes ``P`` and ``P'``, both touching the refracting surface at
  the vertex ``O``.

Everything downstream is labelled by ``r``, the *cylindrical* radius of the
point ``Q`` on the refracting surface.  That is the physical aperture
coordinate -- the stop sits on the diopter -- and it fixes the ray uniquely.
``RayGeometry`` turns an ``r`` grid into every derived quantity the transfer
operator and the reference solver need.
"""

from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------ #
#  Reference surfaces                                                  #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class TangentPlane:
    """A transverse plane at axial position ``z``."""

    z: float = 0.0

    def __str__(self) -> str:
        return f"TangentPlane(z={self.z:g})"


@dataclass(frozen=True)
class ReferenceSphere:
    """A sphere of radius ``radius`` centred at ``(0, 0, center_z)``."""

    center_z: float
    radius: float

    def __post_init__(self):
        if self.radius <= 0.0:
            raise ValueError("ReferenceSphere radius must be positive.")

    def __str__(self) -> str:
        return f"ReferenceSphere(center_z={self.center_z:g}, radius={self.radius:g})"


def incident_sphere(surface) -> ReferenceSphere:
    """Return ``G``, the incident reference sphere of ``surface``."""
    return ReferenceSphere(center_z=float(surface.z0), radius=abs(float(surface.z0)))


def image_sphere(surface) -> ReferenceSphere:
    """Return ``G'``, the image-side reference sphere of ``surface``."""
    return ReferenceSphere(center_z=float(surface.zi), radius=abs(float(surface.zi)))


# ------------------------------------------------------------------ #
#  Ray geometry                                                        #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class RayGeometry:
    """Per-ray geometry on the refracting surface, labelled by ``r``.

    Implements Eqs. (6)-(13), (47) and (81)-(89) of the stigmatic-refraction
    note.  Every array has the shape of the ``r`` grid it was built from.

    Angles
    ------
    ``alpha`` are the *meridional* angles measured from the axis: ``alpha_0``
    for the incident ray ``A -> Q`` and ``alpha_i`` for the transmitted ray
    ``Q -> A'``.  ``theta`` are the *incidence* angles measured from the local
    surface normal, the ones Snell's law and the Fresnel coefficients use.
    """

    r: np.ndarray
    rho: np.ndarray
    z: np.ndarray
    dz_dr: np.ndarray

    l0: np.ndarray
    li: np.ndarray

    sin_a0: np.ndarray
    cos_a0: np.ndarray
    sin_ai: np.ndarray
    cos_ai: np.ndarray

    cos_t0: np.ndarray
    cos_ti: np.ndarray

    normal_r: np.ndarray
    normal_z: np.ndarray

    A: np.ndarray
    valid: np.ndarray

    n0: float
    ni: float
    z0: float
    zi: float

    # -- derived pupil coordinates ---------------------------------- #

    @property
    def u_sine(self) -> np.ndarray:
        """Sine-mapped image-side pupil coordinate ``|zi| sin(alpha_i)``."""
        return abs(self.zi) * self.sin_ai

    @property
    def r_tan_img(self) -> np.ndarray:
        """Perspective image-side pupil coordinate ``|zi| tan(alpha_i)``."""
        return abs(self.zi) * self.sin_ai / self.cos_ai

    @property
    def r_tan_obj(self) -> np.ndarray:
        """Perspective object-side pupil coordinate ``|z0| tan(alpha_0)``, Eq. (85)."""
        return abs(self.z0) * self.sin_a0 / self.cos_a0

    @property
    def sin_t0(self) -> np.ndarray:
        return np.sqrt(np.clip(1.0 - self.cos_t0**2, 0.0, 1.0))

    @property
    def sin_ti(self) -> np.ndarray:
        return np.sqrt(np.clip(1.0 - self.cos_ti**2, 0.0, 1.0))

    @property
    def optical_path(self) -> np.ndarray:
        """``n0 l0 + ni li``; constant on a stigmatic surface, Eq. (15)."""
        return self.n0 * self.l0 + self.ni * self.li

    @property
    def area_element_per_r(self) -> np.ndarray:
        """``dS / (dr dphi)`` on the refracting surface."""
        return self.r * np.sqrt(1.0 + self.dz_dr**2)

    # -- local s/p frame -------------------------------------------- #

    def local_frame(self, phi):
        """Return the local Fresnel frame ``(s_hat, p0_hat, pi_hat)`` at azimuth ``phi``.

        Vectors are returned as ``(..., 3)`` Cartesian arrays broadcast against
        ``phi``.  On a surface of revolution ``s_hat`` is ``phi_hat`` and the
        ``p`` directions are the meridional unit vectors ``theta_hat_0`` and
        ``theta_hat_i`` of Eqs. (17) and (19); keeping them explicit is what
        lets the same operator serve non-revolution surfaces later.
        """
        phi = np.asarray(phi, dtype=float)
        cph, sph = np.cos(phi), np.sin(phi)
        zero = np.zeros_like(cph)

        r_hat = np.stack([cph, sph, zero], axis=-1)
        phi_hat = np.stack([-sph, cph, zero], axis=-1)
        z_hat = np.stack([zero, zero, np.ones_like(cph)], axis=-1)

        c0 = self.cos_a0[..., None]
        s0 = self.sin_a0[..., None]
        ci = self.cos_ai[..., None]
        si = self.sin_ai[..., None]

        # Eq. (17): theta_hat_0 = cos(a0) r_hat - sin(a0) z_hat
        p0_hat = c0 * r_hat - s0 * z_hat
        # Eq. (19): theta_hat_i = cos(ai) r_hat + sin(ai) z_hat
        pi_hat = ci * r_hat + si * z_hat
        return phi_hat, p0_hat, pi_hat

    def directions(self, phi):
        """Return the propagation directions ``(u0_hat, ui_hat)``, Eq. (10)."""
        phi = np.asarray(phi, dtype=float)
        cph, sph = np.cos(phi), np.sin(phi)
        zero = np.zeros_like(cph)
        r_hat = np.stack([cph, sph, zero], axis=-1)
        z_hat = np.stack([zero, zero, np.ones_like(cph)], axis=-1)

        s0 = self.sin_a0[..., None]
        c0 = self.cos_a0[..., None]
        si = self.sin_ai[..., None]
        ci = self.cos_ai[..., None]

        u0 = s0 * r_hat + c0 * z_hat
        ui = -si * r_hat + ci * z_hat
        return u0, ui

    def positions(self, phi):
        """Return the surface points ``Q`` as a ``(..., 3)`` Cartesian array."""
        phi = np.asarray(phi, dtype=float)
        r = self.r
        return np.stack(
            [r * np.cos(phi), r * np.sin(phi), np.broadcast_to(self.z, np.shape(r * phi))],
            axis=-1,
        )

    def normals(self, phi):
        """Return the unit normal at ``Q``, oriented towards the incident medium."""
        phi = np.asarray(phi, dtype=float)
        cph, sph = np.cos(phi), np.sin(phi)
        nr = self.normal_r
        nz = self.normal_z
        return np.stack([nr * cph, nr * sph, np.broadcast_to(nz, np.shape(nr * cph))], axis=-1)


def ray_geometry(surface, r) -> RayGeometry:
    """Build the :class:`RayGeometry` of ``surface`` on the pupil grid ``r``.

    ``r`` is the cylindrical radius of the surface point ``Q``.  Samples beyond
    the surface's usable aperture are flagged in ``valid`` and filled with
    benign values rather than NaNs, so downstream arithmetic stays finite.
    """
    r = np.asarray(r, dtype=float)
    if np.any(r < 0.0):
        raise ValueError("pupil radius r must be non-negative.")

    rho, in_domain = surface.rho_from_r(r, return_valid=True)
    return ray_geometry_from_rho(surface, rho, in_domain=in_domain)


def ray_geometry_from_rho(surface, rho, *, in_domain=None) -> RayGeometry:
    """Build the :class:`RayGeometry` from the surface parameter ``rho``.

    ``rho`` is the spherical radius from the vertex, the surface's own
    parametrisation.  Prefer :func:`ray_geometry`, which takes the physical
    aperture coordinate; this entry point exists for probes that sweep the
    surface directly.
    """
    rho = np.asarray(rho, dtype=float)

    n0 = float(surface.n0)
    ni = float(surface.ni)
    z0 = float(surface.z0)
    zi = float(surface.zi)

    valid = np.ones(rho.shape, dtype=bool) if in_domain is None else np.asarray(in_domain, dtype=bool)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = surface.z(rho)
        r = np.sqrt(np.maximum(rho**2 - z**2, 0.0))
        dz_drho = surface.dz(rho)
        # dr/drho = (rho - z dz/drho) / r, from r^2 = rho^2 - z^2
        dr_drho = np.divide(
            rho - z * dz_drho,
            np.where(r > 0.0, r, 1.0),
            out=np.ones_like(rho),
            where=r > 0.0,
        )
        dz_dr = np.divide(
            dz_drho,
            dr_drho,
            out=np.zeros_like(rho),
            where=np.abs(dr_drho) > 1e-15,
        )

    # Unit normal in the (r_hat, z_hat) meridional plane, oriented towards the
    # incident medium -- the side the object point A sits on, Eq. (13).  The
    # orientation is a global constant rather than a per-point test: any test
    # built from the incident ray degenerates exactly at grazing, which is the
    # one place the answer matters.
    norm = np.hypot(dz_dr, 1.0)
    orientation = 1.0 if z0 > 0.0 else -1.0
    normal_r = -orientation * dz_dr / norm
    normal_z = orientation * np.ones_like(dz_dr) / norm

    l0 = np.hypot(r, z - z0)
    li = np.hypot(r, zi - z)

    with np.errstate(divide="ignore", invalid="ignore"):
        sin_a0 = np.divide(r, l0, out=np.zeros_like(r), where=l0 > 0.0)
        cos_a0 = np.divide(z - z0, l0, out=np.ones_like(r), where=l0 > 0.0)
        sin_ai = np.divide(r, li, out=np.zeros_like(r), where=li > 0.0)
        cos_ai = np.divide(zi - z, li, out=np.ones_like(r), where=li > 0.0)

    # Eq. (8): physical propagation directions.
    u0_r, u0_z = sin_a0, cos_a0
    ui_r, ui_z = -sin_ai, cos_ai

    # Eq. (13): both cosines are positive on the refracting branch.  Where they
    # are not, the sample has run past grazing incidence onto the sheet of the
    # oval that ``A`` no longer illuminates; ``valid`` records that below.
    cos_t0 = -(u0_r * normal_r + u0_z * normal_z)
    cos_ti = -(ui_r * normal_r + ui_z * normal_z)
    refracting = (cos_t0 > 0.0) & (cos_ti > 0.0)
    cos_t0 = np.clip(cos_t0, 0.0, 1.0)
    cos_ti = np.clip(cos_ti, 0.0, 1.0)

    # Eq. (47): the geometric amplitude factor.
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.divide(
            abs(z0) * li,
            abs(zi) * l0,
            out=np.ones_like(r),
            where=l0 > 0.0,
        )

    valid = valid & np.isfinite(z) & np.isfinite(dz_dr) & refracting

    def _clean(a, fill):
        return np.where(valid, np.nan_to_num(a, nan=fill, posinf=fill, neginf=fill), fill)

    return RayGeometry(
        r=r,
        rho=_clean(rho, 0.0),
        z=_clean(z, 0.0),
        dz_dr=_clean(dz_dr, 0.0),
        l0=_clean(l0, abs(z0)),
        li=_clean(li, abs(zi)),
        sin_a0=_clean(sin_a0, 0.0),
        cos_a0=_clean(cos_a0, 1.0),
        sin_ai=_clean(sin_ai, 0.0),
        cos_ai=_clean(cos_ai, 1.0),
        cos_t0=_clean(cos_t0, 1.0),
        cos_ti=_clean(cos_ti, 1.0),
        normal_r=_clean(normal_r, 0.0),
        normal_z=_clean(normal_z, 1.0 if z0 > 0.0 else -1.0),
        A=_clean(A, 1.0),
        valid=valid,
        n0=n0,
        ni=ni,
        z0=z0,
        zi=zi,
    )


def grazing_radius(surface, n_probe: int = 200_001) -> float:
    """Return the largest usable pupil radius ``r``, where ``cos(theta_0) -> 0``.

    Beyond grazing incidence the closed-form oval branch continues, but on a
    sheet the object point no longer illuminates, so this is the physical
    aperture ceiling.  Falls back to the fold radius when grazing is never
    reached within the surface's domain.
    """
    r_fold = surface.r_fold
    probe = np.linspace(0.0, r_fold, int(n_probe))
    geom = ray_geometry(surface, probe)
    below = np.flatnonzero(~geom.valid)
    if below.size == 0:
        return float(r_fold)
    first = int(below[0])
    if first == 0:
        raise ValueError(
            f"({surface.n0}, {surface.ni}, z0={surface.z0}, zi={surface.zi}) is not a "
            "refracting configuration: the axial ray already fails cos(theta_i) > 0, so "
            "the object and image points lie on the same side of the surface. The "
            "physical convention is z0 < 0 < zi."
        )
    return float(probe[first - 1])
