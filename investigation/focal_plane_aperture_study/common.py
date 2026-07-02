"""Shared tools for the focal-plane aperture study.

Physical model
--------------
A monochromatic field crosses a single refracting Cartesian-oval surface
("diopter") that images the axial source point ``z0`` (object side, ``z0 < 0``)
perfectly onto ``zi`` (image side).  The transverse focal field produced by a
radially uniform, x-polarized pupil ``P(r)`` is (see ``vecdiff.propagation``)::

    E_x(q, phi) = A(q) - cos(2 phi) * B(q)      (co-polarized)
    E_y(q, phi) =      - sin(2 phi) * B(q)      (cross-polarized)

with the two radial kernels

    A(q) = 2 pi H0[ t_plus(r)  P(r) ](q)        t_plus  = (tp + ts) / 2
    B(q) = 2 pi H2[ t_minus(r) P(r) ](q)        t_minus = (tp - ts) / 2

``H0``/``H2`` are Hankel transforms of order 0/2.  The *scalar* limit used
throughout the study is ``t_minus = 0`` (same t_plus apodization, no
polarization mixing), so every scalar/vectorial difference is attributable to
``B`` alone.

Useful azimuthal integrals (energy bookkeeping):

    E_total = 2 pi Int (|A|^2 + |B|^2) q dq
    E_cross = pi  Int |B|^2 q dq                (energy in E_y)
    f_cross = E_cross / E_total
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import jv

from vecdiff.fresnel import FresnelOvoid

TAU = 2.0 * np.pi

# Baseline geometry shared by the runnable examples of the repository.
N0 = 1.0
NI = 1.5
Z0 = -10.0
ZI = 6.0
LAM = 532e-6  # mm


def output_dir(script_file: str | Path) -> Path:
    """Return and create ``output/<script_stem>/`` next to the study scripts."""
    path = Path(script_file).resolve().parent / "output" / Path(script_file).stem
    path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------ #
#  Fresnel coefficients                                                #
# ------------------------------------------------------------------ #

def exact_coeffs(r, *, n0=N0, ni=NI, z0=Z0, zi=ZI):
    """Sanitized exact Fresnel coefficients ``(t_plus, t_minus)`` on ``r``.

    Non-finite samples (the r=0 indeterminacy of the oval parametrization)
    are repaired by interpolation from finite neighbours.
    """
    r = np.asarray(r, dtype=float)
    fresnel = FresnelOvoid(n0=n0, ni=ni, z0=z0, zi=zi)
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = np.asarray(fresnel.tp(r), dtype=float)
        ts = np.asarray(fresnel.ts(r), dtype=float)
    for t in (tp, ts):
        bad = ~np.isfinite(t)
        if bad.any():
            good = ~bad
            t[bad] = np.interp(r[bad], r[good], t[good])
    return 0.5 * (tp + ts), 0.5 * (tp - ts)


def oval_max_radius(*, n0=N0, ni=NI, z0=Z0, zi=ZI, r_probe=None, n_probe=4000):
    """Largest pupil radius on which the Cartesian oval remains real-valued."""
    if r_probe is None:
        r_probe = 3.0 * max(abs(z0), abs(zi))
    surface = FresnelOvoid(n0=n0, ni=ni, z0=z0, zi=zi).ovoid
    r = np.linspace(1e-6, r_probe, n_probe)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(surface.z(r)) & np.isfinite(surface.r(r))
    if ok.all():
        return float(r[-1])
    return float(r[np.argmin(ok) - 1])


def grazing_radius(*, n0=N0, ni=NI, z0=Z0, zi=ZI, n_probe=6000):
    """Pupil radius where incidence becomes grazing (``cos_i`` -> 0).

    The Cartesian oval is a closed surface: beyond this radius the ``rho``
    parametrization walks along the back face, which is not a physical pupil
    for first-surface transmission.  Sweeps should stay below this limit.
    """
    fresnel = FresnelOvoid(n0=n0, ni=ni, z0=z0, zi=zi)
    r_max = oval_max_radius(n0=n0, ni=ni, z0=z0, zi=zi)
    rho = np.linspace(1e-4, 0.999 * r_max, n_probe)
    with np.errstate(all="ignore"):
        cos_i, _ = fresnel._cosines(rho)
    return float(rho[np.nanargmin(cos_i)])


def geometry_angles(rho, *, n0=N0, ni=NI, z0=Z0, zi=ZI):
    """True ray angles (deg) at pupil parameter ``rho``.

    Returns ``(alpha_obj, theta_inc, theta_img)``: object-side ray angle from
    the source, incidence angle on the surface, and image-side convergence
    angle, all computed with the cylindrical radius ``r_cyl(rho)`` of the
    oval (not with ``rho`` itself).
    """
    fresnel = FresnelOvoid(n0=n0, ni=ni, z0=z0, zi=zi)
    surface = fresnel.ovoid
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    with np.errstate(all="ignore"):
        z = surface.z(rho)
        r_cyl = surface.r(rho)
        cos_i, _ = fresnel._cosines(rho)
    alpha_obj = np.degrees(np.arctan2(r_cyl, z - z0))
    theta_inc = np.degrees(np.arccos(np.clip(cos_i, 0.0, 1.0)))
    theta_img = np.degrees(np.arctan2(r_cyl, zi - z))
    if alpha_obj.size == 1:
        return float(alpha_obj[0]), float(theta_inc[0]), float(theta_img[0])
    return alpha_obj, theta_inc, theta_img


def system_caption(a=None, *, n0=N0, ni=NI, z0=Z0, zi=ZI, lam=LAM):
    """One-line summary of the stigmatic single-diopter system for figures.

    Always reports ``n0, ni, z0, zi`` and the wavelength; when an aperture
    ``a`` is given, appends it and the resulting maximum object-side aperture
    angle ``alpha_obj``.
    """
    cap = (rf"sistema estigmático: $n_0$={n0}, $n_i$={ni}, "
           rf"$z_0$={z0} mm, $z_i$={zi} mm  ·  $\lambda$={lam * 1e6:.0f} nm")
    if a is not None:
        alpha_obj, _, _ = geometry_angles(a, n0=n0, ni=ni, z0=z0, zi=zi)
        cap += rf"  ·  apertura $a$={a:.2f} mm, $\alpha_{{obj}}$={alpha_obj:.1f}°"
    return cap


# ------------------------------------------------------------------ #
#  Vectorized Hankel transform                                         #
# ------------------------------------------------------------------ #

def hankel_matrix(order, r, q):
    """Matrix ``M`` such that ``M @ f`` == Hankel transform of ``f`` at ``q``.

    Uses Simpson weights on the (uniform) radial grid, matching
    ``vecdiff.hankel.HankelTransform`` while evaluating all output samples in
    a single matrix product.
    """
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    n = r.size
    if n < 3:
        raise ValueError("need at least three radial samples.")
    h = r[1] - r[0]
    w = np.zeros(n)
    m = n if n % 2 == 1 else n - 1  # largest odd prefix -> composite Simpson
    w[0:m:2] += 1.0
    w[1:m:2] += 4.0
    w[2:m:2] += 1.0
    w[: m] *= h / 3.0
    if m != n:  # trapezoid correction on the last interval
        w[-2] += 0.5 * h
        w[-1] += 0.5 * h
    return jv(order, np.outer(q, r)) * (w * r)[None, :]


def focal_kernels(r, pupil, t_plus, t_minus, q):
    """Return the co- and cross-polarized radial kernels ``(A, B)``."""
    pupil = np.asarray(pupil, dtype=complex)
    A = TAU * (hankel_matrix(0, r, q) @ (t_plus * pupil))
    B = TAU * (hankel_matrix(2, r, q) @ (t_minus * pupil))
    return A, B


# ------------------------------------------------------------------ #
#  Energies and spot metrics                                           #
# ------------------------------------------------------------------ #

def radial_energy(F, q):
    """``Int |F|^2 q dq`` (azimuthal factors handled by the caller)."""
    return float(np.trapezoid(np.abs(F) ** 2 * q, q))


def energy_report(A, B, q):
    """Energy bookkeeping of the focal field for kernels ``A``, ``B``."""
    eA = radial_energy(A, q)
    eB = radial_energy(B, q)
    total = TAU * (eA + eB)
    cross = np.pi * eB
    return {
        "energy_total": total,
        "energy_cross": cross,
        "f_cross": cross / (total + 1e-300),
        "peak_cross_over_peak_total": float(
            np.max(np.abs(B)) ** 2 / (np.max(np.abs(A) ** 2 + np.abs(B) ** 2) + 1e-300)
        ),
    }


def _half_max_radius(q, I):
    """First radius where the profile falls to half its central value."""
    I = np.asarray(I, dtype=float)
    ref = 0.5 * I[0]
    below = np.flatnonzero(I <= ref)
    if below.size == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    return float(np.interp(ref, [I[i], I[i - 1]], [q[i], q[i - 1]]))


def fwhm_radius(q, I):
    """Half-width at half maximum of a radial intensity profile (peak at q=0)."""
    return _half_max_radius(q, I)


def first_min_radius(q, I):
    """Radius of the first local minimum of the profile."""
    dI = np.diff(I)
    turns = np.flatnonzero((dI[:-1] < 0.0) & (dI[1:] >= 0.0))
    return float(q[turns[0] + 1]) if turns.size else np.nan


def encircled_radius(q, I_radial_avg, fraction=0.5):
    """Radius containing ``fraction`` of the azimuthally averaged energy."""
    cum = np.concatenate(([0.0], np.cumsum(np.diff(q) * 0.5 * ((I_radial_avg * q)[1:] + (I_radial_avg * q)[:-1]))))
    if cum[-1] <= 0.0:
        return np.nan
    return float(np.interp(fraction * cum[-1], cum, q))


def spot_metrics(A, B, q):
    """Spot-size metrics of the vectorial spot and its scalar (B=0) reference.

    Azimuthal cuts of the total intensity:
        phi = 0   : |A - B|^2
        phi = 90  : |A + B|^2
        phi = 45  : |A|^2 + |B|^2
    """
    I_x = np.abs(A - B) ** 2
    I_y = np.abs(A + B) ** 2
    I_avg = np.abs(A) ** 2 + np.abs(B) ** 2  # also the phi=45 cut
    I_scalar = np.abs(A) ** 2

    fx = fwhm_radius(q, I_x)
    fy = fwhm_radius(q, I_y)
    return {
        "hwhm_x": fx,
        "hwhm_y": fy,
        "hwhm_scalar": fwhm_radius(q, I_scalar),
        "ellipticity": fx / fy if np.isfinite(fx) and np.isfinite(fy) else np.nan,
        "first_min_x": first_min_radius(q, I_x),
        "first_min_y": first_min_radius(q, I_y),
        "first_min_scalar": first_min_radius(q, I_scalar),
        "r50": encircled_radius(q, I_avg, 0.5),
        "r50_scalar": encircled_radius(q, I_scalar, 0.5),
    }


# ------------------------------------------------------------------ #
#  2D reconstruction                                                   #
# ------------------------------------------------------------------ #

def maps_2d(A, B, q, half_size, n_img=460):
    """Reconstruct 2D intensity maps from the radial kernels.

    Returns ``(maps, extent)`` where ``maps`` holds ``I_vec`` (total),
    ``I_scalar`` (B=0), ``I_cross`` (|E_y|^2 clover) and ``I_co`` (|E_x|^2).
    """
    axis = np.linspace(-half_size, half_size, n_img)
    X, Y = np.meshgrid(axis, axis)
    rho = np.hypot(X, Y)
    phi = np.arctan2(Y, X)

    A2 = np.interp(rho, q, A.real) + 1j * np.interp(rho, q, A.imag)
    B2 = np.interp(rho, q, B.real) + 1j * np.interp(rho, q, B.imag)

    Ex = A2 - np.cos(2.0 * phi) * B2
    Ey = -np.sin(2.0 * phi) * B2

    maps = {
        "I_co": np.abs(Ex) ** 2,
        "I_cross": np.abs(Ey) ** 2,
        "I_vec": np.abs(Ex) ** 2 + np.abs(Ey) ** 2,
        "I_scalar": np.abs(A2) ** 2,
    }
    extent = [-half_size, half_size, -half_size, half_size]
    return maps, extent


def q_to_focal_lambda(q, *, ni=NI, zi=ZI):
    """Convert conjugate radial frequency q to focal radius in wavelengths."""
    return q * zi / (TAU * ni)
