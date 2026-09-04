"""The interface as a spectral operator, in the meridional plane.

This is the direct route: carry the incident field to the interface, split it
there with Fresnel, and carry the transmitted field back to a tangent plane --
with no reference spheres, no ray tubes and no geometric amplitude factor.  The
only modelling choice is *where the incidence cosine comes from*, and that is
the question this module exists to settle.

Two answers are implemented.

``"local"``
    One direction per surface point, read off the phase of the field itself:
    if ``phi`` is the phase of the incident potential along the surface and
    ``s`` is arc length, then ``sin(theta_i) = (d phi / d s) / k1``, exactly,
    for a single plane wave.  Costs one transform.  It is the stationary-phase
    (single-ray) reading of the field and it is *only* defined when one ray
    reaches each point.

``"spectral"``
    One direction per plane wave, ``cos(theta_i) = k_hat(kappa) . n_hat(Q)``,
    summed over the whole angular spectrum.  Exact for a plane interface, at
    any illumination.  Costs a kappa-by-Q operator, whose factorisation is the
    subject of :func:`fresnel_operator_rank`.

Reduction to scalars
--------------------
The interface here is a *cylinder*: ``z = zeta(x)``, invariant along ``y``.
For such a surface the vector problem splits exactly, with no approximation,
into two scalar Helmholtz problems -- ``psi = E_y`` (TE, the ``s`` channel) and
``psi = H_y`` (TM, the ``p`` channel).  The rotation of the ``p`` frame from
``p_i`` to ``p_t`` is carried automatically by ``H_y`` staying along ``y``, so
nothing vectorial is lost.  Every statement below is therefore about the full
vector field, written in the representation that makes it two scalars.

Conventions
-----------
``exp(-i omega t)``; medium 1 occupies ``z < zeta(x)`` and medium 2 ``z >
zeta(x)``; ``n_hat`` points into medium 2, and the radiation integral uses
``N_hat = -n_hat``, the outward normal of the image half-space, to match
:mod:`vecdiff.reference.kirchhoff`.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import hankel1

π = np.pi

POLARIZATIONS = ("TE", "TM")
CHANNEL_MODELS = ("spectral", "local")


# ------------------------------------------------------------------ #
#  The interface profile                                              #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class Profile:
    """A cylindrical interface ``z = zeta(x)`` sampled on ``x``.

    ``n_x, n_z`` is the unit normal pointing into medium 2, and ``ds_dx`` the
    arc-length element, so a line integral over the profile is
    ``sum(f * ds_dx * dx)``.
    """

    x: np.ndarray
    z: np.ndarray
    dz_dx: np.ndarray
    n_x: np.ndarray
    n_z: np.ndarray
    ds_dx: np.ndarray

    @property
    def points(self) -> np.ndarray:
        """The profile as ``(N, 2)`` points in the meridional plane."""
        return np.stack([self.x, self.z], axis=-1)

    @property
    def outward_normals(self) -> np.ndarray:
        """``N_hat = -n_hat``: outward normal of the image half-space."""
        return np.stack([-self.n_x, -self.n_z], axis=-1)


def _profile(x, z, dz_dx) -> Profile:
    norm = np.hypot(dz_dx, 1.0)
    return Profile(
        x=x,
        z=z,
        dz_dx=dz_dx,
        n_x=-dz_dx / norm,
        n_z=np.ones_like(dz_dx) / norm,
        ds_dx=norm,
    )


def flat_profile(x) -> Profile:
    """The plane ``z = 0``, on which the spectral channel is *exact*."""
    x = np.asarray(x, dtype=float)
    zero = np.zeros_like(x)
    return _profile(x, zero, zero)


def profile_from_surface(surface, x) -> Profile:
    """The meridional profile of a :class:`~vecdiff.CartesianSurface`.

    The oval's defining condition ``n0 l0 + ni li = const`` is a statement
    about one ray in one plane, so extruding the profile along ``y`` gives a
    cylinder that is stigmatic for a line source -- the two-dimensional twin of
    the surface the rest of the package uses.
    """
    x = np.asarray(x, dtype=float)
    r = np.abs(x)
    geom = surface.ray_geometry(r)
    return _profile(x, geom.z, np.sign(x) * geom.dz_dr)


# ------------------------------------------------------------------ #
#  Fresnel, for the scalar potential of each polarization             #
# ------------------------------------------------------------------ #

def transmission(cos_i, n1, n2, polarization):
    """Transmission of the scalar potential across the tangent plane.

    ``TE`` carries ``psi = E_y`` and is the ordinary ``t_s``.  ``TM`` carries
    ``psi = H_y``; since ``|H| = n |E| / Z0`` its coefficient is ``(n2/n1)``
    times the electric ``t_p``, i.e. ``2 n2 c1 / (n2 c1 + n1 c2)``.
    """
    cos_i = np.asarray(cos_i, dtype=complex)
    cos_t = np.sqrt(1.0 - (n1 / n2) ** 2 * (1.0 - cos_i**2))
    if polarization == "TE":
        return 2.0 * n1 * cos_i / (n1 * cos_i + n2 * cos_t), cos_t
    if polarization == "TM":
        return 2.0 * n2 * cos_i / (n2 * cos_i + n1 * cos_t), cos_t
    raise ValueError(f"polarization must be one of {POLARIZATIONS}, got {polarization!r}")


# ------------------------------------------------------------------ #
#  The incident field, as an explicit angular spectrum                #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class PlaneWaveSet:
    """An incident field written as ``sum_j amp_j exp(i (kx_j x + kz_j z))``.

    Keeping the spectrum explicit is what makes the comparison rigorous: every
    member is an exact solution of Maxwell's equations in medium 1, so the
    superposition is exact too, and any error measured downstream belongs to
    the interface model rather than to the propagation.
    """

    kx: np.ndarray
    amp: np.ndarray
    k: float

    @property
    def kz(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.k**2 - self.kx**2, 0.0))

    @property
    def sin_theta(self) -> np.ndarray:
        return self.kx / self.k

    def phasor(self, x, z) -> np.ndarray:
        """``exp(i k.Q)`` for every sample ``Q`` and every plane wave, ``(N, J)``."""
        x = np.asarray(x, dtype=float)[:, None]
        z = np.asarray(z, dtype=float)[:, None]
        return np.exp(1j * (self.kx[None, :] * x + self.kz[None, :] * z))

    def field(self, x, z) -> np.ndarray:
        """The scalar potential of the superposition at ``(x, z)``."""
        return self.phasor(x, z) @ self.amp


def line_source(kx, k, x_s, z_s, *, amplitude=1.0) -> np.ndarray:
    """Spectral amplitudes of a line source at ``(x_s, z_s)``, for ``z > z_s``.

    The two-dimensional Weyl identity, ``(i/4) H0(k|r-r'|) = (i/4 pi) int dkx
    exp(i kx (x-x') + i kz |z-z'|) / kz``.  Band-limiting the integral to the
    sampled ``kx`` is the aperture stop, and is the only thing that makes the
    source finite-aperture rather than a full point source.
    """
    kx = np.asarray(kx, dtype=float)
    kz = np.sqrt(np.maximum(k**2 - kx**2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(kz > 0.0, 1.0 / np.maximum(kz, 1e-300), 0.0)
    return amplitude * weight * np.exp(-1j * (kx * x_s + kz * z_s))


# ------------------------------------------------------------------ #
#  The two channel models                                             #
# ------------------------------------------------------------------ #

def surface_field_spectral(pw: PlaneWaveSet, profile: Profile, n1, n2, polarization):
    """Transmitted potential and its normal derivative, one cosine per plane wave.

    ``cos(theta_i) = k_hat(kappa) . n_hat(Q)`` is evaluated for every pair, so
    the Fresnel split acts on each spectral component in the frame that
    component actually sees.  On a flat profile this is the rigorous answer.
    """
    phasor = pw.phasor(profile.x, profile.z)                      # (N, J)
    cos_i = (
        profile.n_x[:, None] * pw.kx[None, :]
        + profile.n_z[:, None] * pw.kz[None, :]
    ) / pw.k
    t, cos_t = transmission(cos_i, n1, n2, polarization)

    k2 = n2 / n1 * pw.k
    weighted = phasor * t * pw.amp[None, :]
    psi = weighted.sum(axis=1)
    dpsi_dN = (weighted * (-1j * k2 * cos_t)).sum(axis=1)
    return psi, dpsi_dN


def local_incidence_sine(psi, profile, k1):
    """``sin(theta_i)`` from the phase of the field along the surface.

    For a single plane wave the tangential derivative of the phase *is* the
    tangential wavevector, so this is exact -- and it is exactly the quantity
    that stops meaning anything when two waves overlap, because the phase of a
    sum has a gradient of its own that belongs to neither.
    """
    phase = np.unwrap(np.angle(psi))
    dphi_dx = np.gradient(phase, profile.x, edge_order=2)
    return dphi_dx / (profile.ds_dx * k1)


def surface_field_local(pw: PlaneWaveSet, profile: Profile, n1, n2, polarization):
    """Transmitted potential and normal derivative, one cosine per surface point.

    Also returns the local incidence sine, whose excursions beyond one are the
    diagnostic: they mark the points at which the field is not a single plane
    wave and the model has no direction to report.
    """
    psi_minus = pw.field(profile.x, profile.z)
    sin_i = local_incidence_sine(psi_minus, profile, pw.k)
    cos_i = np.sqrt(np.clip(1.0 - sin_i**2, 0.0, None))
    t, cos_t = transmission(cos_i, n1, n2, polarization)

    k2 = n2 / n1 * pw.k
    psi = t * psi_minus
    dpsi_dN = -1j * k2 * cos_t * psi
    return psi, dpsi_dN, sin_i


def surface_field(pw, profile, n1, n2, polarization, model):
    """Dispatch on ``model``, returning ``(psi, dpsi_dN)``."""
    if model == "spectral":
        return surface_field_spectral(pw, profile, n1, n2, polarization)
    if model == "local":
        return surface_field_local(pw, profile, n1, n2, polarization)[:2]
    raise ValueError(f"model must be one of {CHANNEL_MODELS}, got {model!r}")


# ------------------------------------------------------------------ #
#  Radiation into medium 2                                            #
# ------------------------------------------------------------------ #

def radiate(profile: Profile, psi, dpsi_dN, observation, k2, *, dx=None):
    """Green's second identity in two dimensions, ``G = (i/4) H0(k R)``.

    Whatever boundary data are handed in, the field this returns is an exact
    Helmholtz solution in medium 2 -- so it is a neutral carrier, and any
    difference between two models at the observation points is a difference in
    the surface field alone.
    """
    P = np.asarray(observation, dtype=float)
    Q = profile.points
    if dx is None:
        dx = float(np.mean(np.diff(profile.x)))
    ds = profile.ds_dx * dx
    N_hat = profile.outward_normals

    R_vec = P[:, None, :] - Q[None, :, :]
    R = np.sqrt(np.sum(R_vec * R_vec, axis=-1))
    R_hat = R_vec / R[..., None]

    G = 0.25j * hankel1(0, k2 * R)
    dG_dR = -0.25j * k2 * hankel1(1, k2 * R)
    RdotN = np.einsum("psi,si->ps", R_hat, N_hat)

    integrand = G * (dpsi_dN * ds)[None, :] + dG_dR * RdotN * (psi * ds)[None, :]
    return integrand.sum(axis=1)


# ------------------------------------------------------------------ #
#  Cost of the spectral channel                                       #
# ------------------------------------------------------------------ #

def fresnel_operator_rank(pw: PlaneWaveSet, profile: Profile, n1, n2, polarization,
                          *, tol=1e-10):
    """Numerical rank of the ``(Q, kappa)`` Fresnel matrix, and its spectrum.

    The spectral channel looks like an ``N x J`` operator, but it depends on the
    pair only through ``cos(theta_i) = k_hat . n_hat``, a bilinear form in two
    slowly varying two-vectors.  A smooth function of such a form is low rank,
    so the sum over ``kappa`` collapses to a handful of transforms rather than
    one per spectral sample.  This measures how many.
    """
    cos_i = (
        profile.n_x[:, None] * pw.kx[None, :]
        + profile.n_z[:, None] * pw.kz[None, :]
    ) / pw.k
    t, _ = transmission(cos_i, n1, n2, polarization)
    singular = np.linalg.svd(np.nan_to_num(t), compute_uv=False)
    rank = int(np.sum(singular > tol * singular[0]))
    return rank, singular
