"""Helmholtz-Kirchhoff scalar reference -- the arbiter for the scalar problem.

The scalar transmission problem through a dielectric interface is the
Helmholtz equation in each medium with ``U`` and ``dU/dn`` continuous on the
surface.  For a locally plane wave that continuity gives the transmission
coefficient ``2 n0 cos(theta_0) / (n0 cos(theta_0) + ni cos(theta_i))`` --
exactly the electromagnetic ``t_s``: the scalar problem is the TE problem.

This module evaluates the exact scalar field in the image half-space by
Green's second identity over the refracting surface,

    U(P) = integral over Sigma of [ G dU/dN - U dG/dN ] dS ,

with ``N`` the outward normal of the image half-space (the one oriented
towards the incident medium) and physical-optics boundary data on the
transmitted side: the incident spherical wave carried down from the reference
sphere ``G``, split by the scalar (s) Fresnel coefficient.  The same
hypothesis level as the vector Franz reference, and an exact Helmholtz
solution in the image medium for the same reason the Franz field is an exact
Maxwell one.
"""

import numpy as np

from ..fresnel import FresnelOvoid

π = np.pi


def kirchhoff_integral(P, Q, U, dU_dN, N_hat, weights, k):
    """Evaluate the Helmholtz-Kirchhoff integral at observation points ``P``.

    ``U`` and ``dU_dN`` are the boundary values and normal derivative on the
    surface samples ``Q`` with unit normals ``N_hat`` (outward normal of the
    image-side volume) and quadrature weights ``weights``.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    R_vec = P[:, None, :] - Q[None, :, :]
    R = np.sqrt(np.sum(R_vec * R_vec, axis=-1))
    R_hat = R_vec / R[..., None]

    G = np.exp(1j * k * R) / (4.0 * π * R)
    f = (1j * k - 1.0 / R) * G                       # dG/dR along R_hat (at P)

    # dG/dN at the source point: grad_Q G . N = -f (R_hat . N)
    RdotN = np.einsum("psi,si->ps", R_hat, N_hat)
    integrand = G * (dU_dN * weights)[None, :] + f * RdotN * (U * weights)[None, :]
    return np.sum(integrand, axis=1)


def kirchhoff_integral_chunked(P, Q, U, dU_dN, N_hat, weights, k, *, chunk=40_000,
                               progress=None):
    """Chunked :func:`kirchhoff_integral`, to bound peak memory."""
    P = np.asarray(P, dtype=float)
    total = np.zeros(P.shape[0], dtype=complex)
    n_src = np.asarray(Q).shape[0]
    for start in range(0, n_src, chunk):
        stop = min(start + chunk, n_src)
        total += kirchhoff_integral(
            P, Q[start:stop], U[start:stop], dU_dN[start:stop],
            N_hat[start:stop], weights[start:stop], k,
        )
        if progress is not None:
            progress(stop, n_src)
    return total


def scalar_focal_field_reference(
    surface,
    wavelength,
    U_G,
    *,
    aperture=None,
    observation,
    n_r=2001,
    n_phi=192,
    chunk=40_000,
    progress=None,
):
    """Exact scalar field of the refracted wave at the observation points.

    ``U_G`` is a callable giving the scalar amplitude on the incident
    reference sphere as a function of the pupil radius ``r`` -- the scalar
    analogue of the input the propagators take.  Boundary data on the
    transmitted side of the surface are physical optics with the scalar
    (s) Fresnel coefficient; the field radiated from them into the image
    half-space is an exact Helmholtz solution there.
    """
    if aperture is None:
        aperture = 0.999 * surface.aperture_limit

    k0 = 2.0 * π / wavelength
    k = surface.ni * k0

    nodes, gl_w = np.polynomial.legendre.leggauss(int(n_r))
    r = 0.5 * aperture * (nodes + 1.0)
    w_r = 0.5 * aperture * gl_w

    phi = np.arange(int(n_phi)) * (2.0 * π / int(n_phi))
    w_phi = 2.0 * π / int(n_phi)

    geom = surface.ray_geometry(r)
    PHI = phi[:, None]

    # Spherical-wave transfer from G down to the surface (Eq. 36, scalar).
    spread = abs(surface.z0) / geom.l0
    phase = np.exp(1j * surface.n0 * k0 * (geom.l0 - abs(surface.z0)))
    U_minus = np.asarray(U_G(r), dtype=complex) * spread * phase

    # Scalar transmission: continuity of U and dU/dn across Sigma gives t_s.
    _, t_s = FresnelOvoid(ovoid=surface).coefficients_from_geometry(geom)
    U_plus = t_s * U_minus

    # Locally plane transmitted wave along u_i: dU/dN = i k (u_i . N) U.
    # The geometry orients N towards the incident medium (Eq. 13:
    # cos(theta_i) = -u_i . N), which is the outward normal of the image
    # half-space, so dU/dN = -i k cos(theta_i) U.
    dU_dN = -1j * k * geom.cos_ti * U_plus

    U_row = np.broadcast_to(U_plus[None, :], (int(n_phi), r.size))
    dU_row = np.broadcast_to(dU_dN[None, :], (int(n_phi), r.size))

    Q = geom.positions(PHI)
    N_hat = geom.normals(PHI)
    dS = np.broadcast_to(
        geom.area_element_per_r[None, :] * (w_r[None, :] * w_phi), Q.shape[:-1]
    )

    return kirchhoff_integral_chunked(
        np.asarray(observation, dtype=float),
        Q.reshape(-1, 3),
        U_row.reshape(-1).astype(complex),
        dU_row.reshape(-1).astype(complex),
        N_hat.reshape(-1, 3),
        dS.reshape(-1),
        k,
        chunk=chunk,
        progress=progress,
    )
