"""The interface transfer operator, from the incident sphere G to the image sphere G'.

The stigmatic-refraction note derives this pointwise: a ray tube, a local
plane-wave Fresnel split at ``Q``, and flux conservation inside each medium.
Nothing in that derivation touches the azimuthal structure of the incident
field, so the operator below is valid for an arbitrary ``E_G(Q)`` -- any
polarization, any dependence on ``phi``.

What *is* tied to a surface of revolution is the identification of the Fresnel
``s`` direction with ``phi_hat`` and ``p`` with the meridional direction.  That
is why the operator is expressed here in the local ``(s, p0, pi)`` frame first
and only then specialised: :func:`sphere_transfer_eigenvalues` is the
revolution case, and the ``t_plus``/``t_minus`` pair the propagators consume is
a consequence of it rather than a definition.  A freeform surface changes the
frame provider, not the physics.

The three eigenvalues, Eqs. (62)-(63):

===============  ==============================================
radial           ``A * t_p * cos(alpha_i) / cos(alpha_0)``
azimuthal        ``A * t_s``
longitudinal     ``A * t_p * sin(alpha_i) / cos(alpha_0)``
===============  ==============================================

with the geometric amplitude factor of Eq. (47)

    ``A(Q) = |z0| * l_i(Q) / (|zi| * l_0(Q))`` .

``A`` is what conserves the mean Poynting flux between the two reference
spheres; dropping it, as the package used to, costs a factor of about two in
amplitude at the edge of a high-aperture pupil while leaving the axis exact.
"""

from dataclasses import dataclass

import numpy as np

from .fresnel import FresnelOvoid
from .geometry import RayGeometry

#: Accepted values of the ``geometry`` switch carried by the propagators.
GEOMETRY_MODES = ("full", "none")


def _check_mode(geometry: str) -> bool:
    if geometry not in GEOMETRY_MODES:
        raise ValueError(f"geometry must be one of {GEOMETRY_MODES}; got {geometry!r}.")
    return geometry == "full"


@dataclass(frozen=True)
class InterfaceOperator:
    """Pointwise transfer operator at the refracting surface.

    Diagonal in the local Fresnel frame by construction: ``t_p`` acts on the
    meridional (``p``) amplitude and ``t_s`` on the azimuthal (``s``) one, and
    ``A`` scales both.
    """

    geom: RayGeometry
    t_p: np.ndarray
    t_s: np.ndarray
    A: np.ndarray

    def apply_local(self, E_p0, E_s):
        """Map incident local amplitudes on ``G`` to their images on ``G'``, Eq. (46)."""
        return self.A * self.t_p * E_p0, self.A * self.t_s * E_s

    def eigenvalues(self):
        """Return ``(lam_r, lam_phi, lam_z)`` of Eqs. (62)-(63).

        These act on the *cylindrical transverse* components ``(E_r, E_phi)``
        of the incident field, with the longitudinal one reconstructed from
        ``E_r`` -- the reduced form the note uses once transversality on ``G``
        has removed the third degree of freedom.
        """
        g = self.geom
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_cos_a0 = np.divide(
                1.0,
                g.cos_a0,
                out=np.zeros_like(g.r),
                where=np.abs(g.cos_a0) > 0.0,
            )
        lam_r = self.A * self.t_p * g.cos_ai * inv_cos_a0
        lam_phi = self.A * self.t_s
        lam_z = self.A * self.t_p * g.sin_ai * inv_cos_a0
        zero = np.zeros_like(g.r)
        return (
            np.where(g.valid, lam_r, zero),
            np.where(g.valid, lam_phi, zero),
            np.where(g.valid, lam_z, zero),
        )

    def cartesian_weights(self):
        """Return ``(lam_plus, lam_minus, lam_z)`` for the Cartesian form, Eq. (67).

        The transverse operator is
        ``lam_plus * I + lam_minus * [[cos 2phi, sin 2phi], [sin 2phi, -cos 2phi]]``,
        which is the shape the propagators already implement.
        """
        lam_r, lam_phi, lam_z = self.eigenvalues()
        return 0.5 * (lam_r + lam_phi), 0.5 * (lam_r - lam_phi), lam_z


def interface_operator(surface, r, *, geometry: str = "full") -> InterfaceOperator:
    """Build the :class:`InterfaceOperator` of ``surface`` on the pupil grid ``r``.

    ``geometry="none"`` drops both the amplitude factor ``A`` and the
    meridional projection, recovering the bare ``t_p``/``t_s`` weighting the
    package used before this correction.  It exists so that tests which pin
    plumbing rather than physics keep a stable target; it is not a physical
    model.
    """
    geometric = _check_mode(geometry)
    geom = surface.ray_geometry(r)
    return interface_operator_from_geometry(surface, geom, geometry=geometry)


def interface_operator_from_geometry(surface, geom, *, geometry: str = "full") -> InterfaceOperator:
    """As :func:`interface_operator`, reusing an already-built geometry."""
    geometric = _check_mode(geometry)
    fresnel = FresnelOvoid(ovoid=surface)
    t_p, t_s = fresnel.coefficients_from_geometry(geom)

    if geometric:
        A = geom.A
        return InterfaceOperator(geom=geom, t_p=t_p, t_s=t_s, A=A)

    # Legacy weighting: no flux factor, no meridional projection.  Faking
    # cos_ai == cos_a0 is what removes the projection from ``eigenvalues``.
    flat = RayGeometry(
        r=geom.r,
        rho=geom.rho,
        z=geom.z,
        dz_dr=geom.dz_dr,
        l0=geom.l0,
        li=geom.li,
        sin_a0=geom.sin_a0,
        cos_a0=np.ones_like(geom.r),
        sin_ai=geom.sin_ai,
        cos_ai=np.ones_like(geom.r),
        cos_t0=geom.cos_t0,
        cos_ti=geom.cos_ti,
        normal_r=geom.normal_r,
        normal_z=geom.normal_z,
        A=np.ones_like(geom.r),
        valid=geom.valid,
        n0=geom.n0,
        ni=geom.ni,
        z0=geom.z0,
        zi=geom.zi,
    )
    return InterfaceOperator(geom=flat, t_p=t_p, t_s=t_s, A=np.ones_like(geom.r))


def sphere_transfer_eigenvalues(surface, r, *, geometry: str = "full"):
    """Return ``(lam_r, lam_phi, lam_z)`` on the pupil grid ``r``."""
    return interface_operator(surface, r, geometry=geometry).eigenvalues()


def channel_transmittance(surface, r):
    """Return ``(T_p, T_s)``, the power transmittances of Eq. (30).

    Provided next to the operator because Eq. (92) -- the note's own
    verification that the flux ratio between the two reference spheres equals
    exactly this -- is the test that pins ``A`` down.
    """
    return FresnelOvoid(ovoid=surface).transmittances(r)
