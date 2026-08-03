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
spheres.  It equals one on axis and falls to about one half at the edge of a
high-aperture pupil, so it is a pure high-aperture apodization.
"""

from dataclasses import dataclass

import numpy as np

from .fresnel import FresnelOvoid
from .geometry import RayGeometry

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


def interface_operator(surface, r) -> InterfaceOperator:
    """Build the :class:`InterfaceOperator` of ``surface`` on the pupil grid ``r``."""
    return interface_operator_from_geometry(surface, surface.ray_geometry(r))


def interface_operator_from_geometry(surface, geom) -> InterfaceOperator:
    """As :func:`interface_operator`, reusing an already-built geometry."""
    fresnel = FresnelOvoid(ovoid=surface)
    t_p, t_s = fresnel.coefficients_from_geometry(geom)
    return InterfaceOperator(geom=geom, t_p=t_p, t_s=t_s, A=geom.A)


def sphere_transfer_eigenvalues(surface, r):
    """Return ``(lam_r, lam_phi, lam_z)`` on the pupil grid ``r``."""
    return interface_operator(surface, r).eigenvalues()


def channel_transmittance(surface, r):
    """Return ``(T_p, T_s)``, the power transmittances of Eq. (30).

    Provided next to the operator because Eq. (92) -- the note's own
    verification that the flux ratio between the two reference spheres equals
    exactly this -- is the test that pins ``A`` down.
    """
    return FresnelOvoid(ovoid=surface).transmittances(r)


def focal_channel_weights(surface, r, *, mapping: str | None = None):
    """Return ``(w_p, w_s, w_z, u)`` ready to feed the focal Hankel channels.

    A convenience for code that drives the radial path by hand rather than
    through :func:`~vecdiff.propagation.propagate_to_focal_plane_through_diopter`.
    The three weights are the transfer eigenvalues times the pupil-mapping
    Jacobian, and ``u`` is the variable the transform integrates over.

    ``w_p`` and ``w_s`` slot straight into the ``t_plus = (w_p+w_s)/2`` and
    ``t_minus = (w_p-w_s)/2`` combinations the Cartesian form consumes.
    """
    from .pupil_mapping import pupil_transform, resolve_mapping

    r = np.asarray(r, dtype=float)
    mapping = resolve_mapping(mapping)
    operator = interface_operator(surface, r)
    lam_r, lam_phi, lam_z = operator.eigenvalues()
    u, weight = pupil_transform(operator.geom, mapping)
    return lam_r * weight, lam_phi * weight, lam_z * weight, u


def paraxial_channel_weights(surface, r):
    """Return the O(r^2) expansion of ``(w_p, w_s)`` from :func:`focal_channel_weights`.

    Useful as an analytic reference for aberration fits.  Expanding each factor
    of ``w_p = A t_p / cos(alpha_0)`` and ``w_s = A t_s / cos(alpha_i)``:

        w_p = gamma_0 + [ n0 xi + gamma_0 (A' + 1/(2 z0^2)) ] r^2
        w_s = gamma_0 + [ ni xi + gamma_0 (A' + 1/(2 zi^2)) ] r^2

    with ``gamma_0 = 2 n0/(n0+ni)``, ``xi`` the coefficient of
    :class:`~vecdiff.fresnel.FresnelOvoidParax`, and

        A' = -(O/2)(1/|zi| + 1/|z0|) + (1/2)(1/zi^2 - 1/z0^2)

    the quadratic coefficient of the geometric factor, where ``O`` is the
    vertex curvature parameter of the oval.  Note the factors multiply
    ``gamma_0``, not one, so the geometric slopes enter weighted by it.
    """
    r = np.asarray(r, dtype=float)
    n0, ni = float(surface.n0), float(surface.ni)
    z0, zi = float(surface.z0), float(surface.zi)

    gamma_0 = 2.0 * n0 / (n0 + ni)
    xi = n0 * (z0 - zi) ** 2 / (z0**2 * zi**2 * (n0**2 - ni**2))
    A_slope = -0.5 * surface.O * (1.0 / abs(zi) + 1.0 / abs(z0)) + 0.5 * (
        1.0 / zi**2 - 1.0 / z0**2
    )

    w_p = gamma_0 + (n0 * xi + gamma_0 * (A_slope + 0.5 / z0**2)) * r**2
    w_s = gamma_0 + (ni * xi + gamma_0 * (A_slope + 0.5 / zi**2)) * r**2
    return w_p, w_s
