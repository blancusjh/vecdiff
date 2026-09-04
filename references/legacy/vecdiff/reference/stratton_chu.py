"""Franz / Stratton-Chu vector diffraction integral -- the reference solver.

Given equivalent surface currents on the refracting surface, this evaluates the
electromagnetic field they radiate into the homogeneous image medium.  The
Franz representation is an *exact* Maxwell field there, so this module settles
questions the ray-optical chain cannot answer on its own: whether the geometric
amplitude factor ``A(Q)`` is right, whether the meridional projection belongs
in the transverse operator, and which pupil mapping the focal-plane transform
should use.

What it does **not** settle is the boundary condition itself: the currents come
from a provider (see :mod:`vecdiff.reference.currents`), and today's provider
makes the same local plane-wave Fresnel assumption the analytic derivation
does.  Swapping in a boundary-integral solve would close that gap.

Conventions
-----------
Time dependence ``exp(-i omega t)``, so outgoing waves carry ``exp(+i k R)``.
Impedance of free space is set to one, hence ``H = n (u x E)`` for a locally
plane wave and ``omega mu = k0``.  With ``A = mu int J G dS`` and
``F = eps int M G dS`` the field is

    E(P) = i omega mu int [ J G + (1/k^2) grad (J . grad G) ] dS
           - int grad G x M dS ,

with ``G = exp(i k R) / (4 pi R)``, ``R = P - Q`` and ``grad`` taken at ``P``.
Near-field terms in ``1/R`` are retained even though ``k R ~ 1e5`` here: they
cost almost nothing and remove any doubt about the result.
"""

import numpy as np

π = np.pi


def _green_terms(P, Q, k):
    """Return ``(G, f, f_prime, R_hat)`` for every ``(P, Q)`` pair.

    ``grad_P G = f * R_hat`` and ``grad grad G = f' R_hat R_hat +
    (f / R) (I - R_hat R_hat)``.
    """
    R_vec = P[:, None, :] - Q[None, :, :]
    R = np.sqrt(np.sum(R_vec * R_vec, axis=-1))
    R_hat = R_vec / R[..., None]

    G = np.exp(1j * k * R) / (4.0 * π * R)
    inv_R = 1.0 / R
    f = (1j * k - inv_R) * G
    f_prime = ((1j * k - inv_R) ** 2 + inv_R**2) * G
    return G, f, f_prime, R_hat, R


def franz_integral(P, Q, J, M, weights, k, n):
    """Evaluate the Franz integral at observation points ``P``.

    Parameters
    ----------
    P : (n_obs, 3) array
        Observation points, in the homogeneous image medium.
    Q : (n_src, 3) array
        Source points on the surface.
    J, M : (n_src, 3) complex arrays
        Equivalent electric and magnetic surface current densities.
    weights : (n_src,) array
        Quadrature weights carrying the surface element ``dS``.
    k : float
        Wave number in the image medium, ``n * k0``.
    n : float
        Refractive index of the image medium.

    Returns
    -------
    (n_obs, 3) complex array
        The electric field at ``P``.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    J = np.asarray(J, dtype=complex)
    M = np.asarray(M, dtype=complex)
    w = np.asarray(weights, dtype=float)

    G, f, f_prime, R_hat, R = _green_terms(P, Q, k)

    Jw = J * w[:, None]
    Mw = M * w[:, None]

    # i omega mu [ J G + (1/k^2) (grad grad G) . J ]
    RdotJ = np.einsum("psi,si->ps", R_hat, Jw)
    hess_J = (
        f_prime[..., None] * RdotJ[..., None] * R_hat
        + (f / R)[..., None] * (Jw[None, :, :] - RdotJ[..., None] * R_hat)
    )
    k0 = k / n
    electric = 1j * k0 * (G[..., None] * Jw[None, :, :] + hess_J / k**2)

    # - grad G x M
    magnetic = -np.cross(f[..., None] * R_hat, Mw[None, :, :])

    return np.sum(electric + magnetic, axis=1)


def franz_integral_chunked(P, Q, J, M, weights, k, n, *, chunk=20_000, progress=None):
    """Chunked :func:`franz_integral`, to bound peak memory."""
    P = np.asarray(P, dtype=float)
    total = np.zeros((P.shape[0], 3), dtype=complex)
    n_src = np.asarray(Q).shape[0]
    for start in range(0, n_src, chunk):
        stop = min(start + chunk, n_src)
        total += franz_integral(
            P, Q[start:stop], J[start:stop], M[start:stop], weights[start:stop], k, n
        )
        if progress is not None:
            progress(stop, n_src)
    return total


# ------------------------------------------------------------------ #
#  Driver for the stigmatic diopter                                    #
# ------------------------------------------------------------------ #

def focal_field_reference(
    surface,
    wavelength,
    Ex_G,
    Ey_G=None,
    *,
    aperture=None,
    observation=None,
    n_r=2001,
    n_phi=192,
    chunk=20_000,
    progress=None,
):
    """Reference focal field of the stigmatic diopter, by Franz integration.

    ``Ex_G``/``Ey_G`` are callables giving the transverse Cartesian components
    of the incident field on the reference sphere ``G``, as functions of the
    pupil radius ``r`` -- exactly the input ``vecdiff`` propagators take.

    ``observation`` is an ``(n_obs, 3)`` array of points; by default a radial
    cut along ``x`` in the plane ``z = zi``.
    """
    from .currents import PhysicalOpticsCurrents

    if aperture is None:
        aperture = 0.999 * surface.aperture_limit

    k0 = 2.0 * π / wavelength
    k = surface.ni * k0

    # Gauss-Legendre in r (the amplitude varies smoothly and the phase barely
    # varies at all on a stigmatic surface), uniform in phi (periodic).
    nodes, gl_w = np.polynomial.legendre.leggauss(int(n_r))
    r = 0.5 * aperture * (nodes + 1.0)
    w_r = 0.5 * aperture * gl_w

    phi = np.arange(int(n_phi)) * (2.0 * π / int(n_phi))
    w_phi = 2.0 * π / int(n_phi)

    geom = surface.ray_geometry(r)
    PHI = phi[:, None]

    source = PhysicalOpticsCurrents(surface, wavelength, Ex_G, Ey_G)
    J, M = source.currents(geom, PHI)

    Q = geom.positions(PHI)
    dS = np.broadcast_to(
        geom.area_element_per_r[None, :] * (w_r[None, :] * w_phi), Q.shape[:-1]
    )

    Q = Q.reshape(-1, 3)
    J = J.reshape(-1, 3)
    M = M.reshape(-1, 3)
    dS = dS.reshape(-1)

    if observation is None:
        observation = default_focal_cut(surface, wavelength)

    E = franz_integral_chunked(
        observation, Q, J, M, dS, k, surface.ni, chunk=chunk, progress=progress
    )
    return observation, E


def default_focal_cut(surface, wavelength, *, half_width_lambda=6.0, n_obs=161, axis="x"):
    """Return a radial cut of the focal plane ``z = zi``, spaced in wavelengths."""
    s = np.linspace(0.0, half_width_lambda, int(n_obs)) * wavelength
    zeros = np.zeros_like(s)
    zi = np.full_like(s, float(surface.zi))
    if axis == "x":
        return np.stack([s, zeros, zi], axis=-1)
    if axis == "y":
        return np.stack([zeros, s, zi], axis=-1)
    raise ValueError("axis must be 'x' or 'y'.")
