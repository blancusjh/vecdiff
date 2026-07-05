"""Shared machinery for the central-spot investigation.

For a homogeneous, linearly polarized pupil at angle alpha the focal-plane
intensity of the vectorial model reduces exactly to

    I(rho, phi) = pi^2 [ |H0|^2 + |H2|^2 - 2 Re(H0 H2*) cos 2(phi - alpha) ]

with H0 = H_0[(tp+ts) P] and H2 = H_2[(tp-ts) P] (Hankel transforms of order
0 and 2 of the pupil P weighted by the Fresnel sum/difference).  The pattern
therefore rotates rigidly with alpha and every spot-shape question collapses
onto the two radial profiles H0(q), H2(q):

    parallel cut  (phi = alpha)      : I_par  = pi^2 |H0 - H2|^2
    perpendicular (phi = alpha+90deg): I_perp = pi^2 |H0 + H2|^2
    scalar reference (t- = 0)        : I_sca  = pi^2 |H0|^2
    ideal scalar (tp = ts = 1)       : I_idl  = pi^2 |H_0[2P]|^2
    circular incidence (or azimuthal
    mean of the linear case)         : I_circ = pi^2 (|H0|^2 + |H2|^2)

All focal-plane radii are expressed in wavelength units, s = rho/lambda, via
s = q * q_lambda with q_lambda = zi / (2 pi ni).
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.special import jv

from vecdiff.field_reconstruction import make_observation_grid
from vecdiff.fresnel import FresnelOvoid

π = np.pi

AIRY_FIRST_ZERO = 3.8317059702  # first zero of J1
AIRY_HWHM = 1.6163399689        # half max of [2 J1(x)/x]^2


# ---------------------------------------------------------------------
# System construction
# ---------------------------------------------------------------------

def grazing_radius(n0, ni, z0, zi, n_probe=4000):
    """Pupil radius where the incidence on the Cartesian oval becomes grazing."""
    rho = np.linspace(1e-4, 3.0 * abs(z0), n_probe)
    fresnel = FresnelOvoid(n0=n0, ni=ni, z0=z0, zi=zi)
    with np.errstate(all="ignore"):
        cos_i, _ = fresnel._cosines(rho)
        finite = np.isfinite(fresnel.ovoid.z(rho)) & np.isfinite(fresnel.ovoid.r(rho))
    return float(rho[np.nanargmin(np.where(finite, cos_i, np.nan))])


def fresnel_coefficients(n0, ni, z0, zi, r):
    """Exact (tp, ts) on the stigmatic diopter, with the r=0 indeterminacy repaired."""
    fresnel = FresnelOvoid(n0=n0, ni=ni, z0=z0, zi=zi)
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = np.asarray(fresnel.tp(r), dtype=float)
        ts = np.asarray(fresnel.ts(r), dtype=float)
    for t in (tp, ts):
        bad = ~np.isfinite(t)
        if bad.any():
            t[bad] = np.interp(r[bad], r[~bad], t[~bad])
    return tp, ts


def hankel_matrix(order, r, f_r, q):
    """Vectorized Hankel transform: same integral as vecdiff.HankelTransform."""
    kernel = jv(order, np.outer(q, r)) * r
    return simpson(kernel * np.asarray(f_r), x=r, axis=-1)


def spot_case(n0=1.0, ni=1.5, z0=-10.0, zi=6.0, lam=532e-6,
              a=None, a_frac=None, pupil=None,
              n_r=800, s_max=3.0, n_q=600):
    """Compute the radial ingredients H0, H2 (and the tp=ts=1 reference) of a
    focused, homogeneously polarized pupil.

    a is the aperture radius in mm; alternatively a_frac gives it as a fraction
    of the grazing radius.  pupil may be None (uniform), a callable P(r), or an
    array.  The q grid maps to a uniform focal grid s = q*q_lambda in [0, s_max]
    wavelengths.
    """
    r_graze = grazing_radius(n0, ni, z0, zi)
    if a is None:
        a = (0.97 if a_frac is None else a_frac) * r_graze
    if a >= r_graze:
        raise ValueError(f"aperture a={a:g} exceeds grazing radius {r_graze:g}")

    r = np.linspace(0.0, a, n_r)
    q_lambda = zi / (2.0 * π * ni)
    q = np.linspace(0.0, s_max / q_lambda, n_q)
    s = q * q_lambda

    tp, ts = fresnel_coefficients(n0, ni, z0, zi, r)
    if pupil is None:
        P = np.ones_like(r)
    elif callable(pupil):
        P = np.asarray(pupil(r))
    else:
        P = np.asarray(pupil)

    # Longitudinal channel from Maxwell transversality (EZ = -k_t E_p / k_z):
    # each pupil annulus maps to k_t = k r / zi, so w_z = k_t/k_z depends only
    # on r/zi; only the p-polarized part f0+f2 = 2 tp P contributes.
    if a >= zi:
        raise ValueError("aperture reaches the evanescent band (a >= zi)")
    w_z = (r / zi) / np.sqrt(1.0 - (r / zi) ** 2)

    return {
        "n0": n0, "ni": ni, "z0": z0, "zi": zi, "lam": lam,
        "a": a, "r_graze": r_graze,
        "alpha_obj_deg": float(np.degrees(np.arctan(a / abs(z0)))),
        "q_lambda": q_lambda,
        "r": r, "P": P, "tp": tp, "ts": ts, "w_z": w_z,
        "q": q, "s": s,
        "H0": hankel_matrix(0, r, (tp + ts) * P, q),
        "H2": hankel_matrix(2, r, (tp - ts) * P, q),
        "Hz": hankel_matrix(1, r, w_z * tp * P, q),
        "H0_ideal": hankel_matrix(0, r, 2.0 * P, q),
    }


# ---------------------------------------------------------------------
# Radial intensity profiles (unnormalized; common pi^2 factor dropped)
# ---------------------------------------------------------------------

def cuts(case):
    """Radial intensity profiles of all reference fields on the s grid.

    The *_tot entries include the longitudinal channel
    |Ez|^2/pi^2 = 2|Hz|^2 (1 + cos 2(phi-alpha)): Ez adds intensity along the
    incident polarization and vanishes across it (perp_tot == perp)."""
    H0, H2, H0i = case["H0"], case["H2"], case["H0_ideal"]
    Z = np.abs(case["Hz"]) ** 2
    return {
        "par": np.abs(H0 - H2) ** 2,
        "perp": np.abs(H0 + H2) ** 2,
        "sca": np.abs(H0) ** 2,
        "ideal": np.abs(H0i) ** 2,
        "circ": np.abs(H0) ** 2 + np.abs(H2) ** 2,
        "ez_par": 4.0 * Z,
        "par_tot": np.abs(H0 - H2) ** 2 + 4.0 * Z,
        "perp_tot": np.abs(H0 + H2) ** 2,
        "circ_tot": np.abs(H0) ** 2 + np.abs(H2) ** 2 + 2.0 * Z,
    }


def intensity_cut(case, psi, total=False):
    """Radial cut of the linear-incidence intensity at angle psi = phi - alpha.
    With total=True the longitudinal |Ez|^2 is included."""
    H0, H2 = case["H0"], case["H2"]
    A = np.abs(H0) ** 2 + np.abs(H2) ** 2
    B = np.real(H0 * np.conj(H2))
    if not total:
        return A - 2.0 * np.cos(2.0 * psi) * B
    Z = np.abs(case["Hz"]) ** 2
    return A + 2.0 * Z - 2.0 * np.cos(2.0 * psi) * (B - Z)


def intensity_maps(case, half_size, n_img=400, alpha=0.0):
    """2D focal-plane intensity maps built from the radial ingredients.

    Returns I_vec, I_sca, I_ideal, I_co, I_cross (co/cross relative to the
    incident direction alpha) and the imshow extent, all in lambda units.
    """
    s = case["s"]
    if half_size * np.sqrt(2.0) > s[-1]:
        raise ValueError("half_size*sqrt(2) exceeds the computed s range; raise s_max")
    A = np.abs(case["H0"]) ** 2 + np.abs(case["H2"]) ** 2
    B = np.real(case["H0"] * np.conj(case["H2"]))
    C = np.abs(case["H2"]) ** 2

    xx, yy, rho, varphi, extent = make_observation_grid(half_size, n_img)
    psi = varphi - alpha
    Ar, Br, Cr = (np.interp(rho, s, f) for f in (A, B, C))
    Zr = np.interp(rho, s, np.abs(case["Hz"]) ** 2)
    I_vec = Ar - 2.0 * np.cos(2.0 * psi) * Br
    I_cross = Cr * np.sin(2.0 * psi) ** 2
    I_z = 2.0 * Zr * (1.0 + np.cos(2.0 * psi))
    return {
        "I_vec": I_vec,
        "I_sca": np.interp(rho, s, np.abs(case["H0"]) ** 2),
        "I_ideal": np.interp(rho, s, np.abs(case["H0_ideal"]) ** 2),
        "I_co": I_vec - I_cross,
        "I_cross": I_cross,
        "I_z": I_z,
        "I_tot": I_vec + I_z,
        "extent": extent,
    }


def field_components(case, xx, yy, alpha=0.0, incidence="linear"):
    """Complex focal-plane field (Ex, Ey, Ez) of the PSF centered at the
    origin, evaluated at Cartesian points (xx, yy) in lambda units.

    The transverse components follow the conventions of
    vecdiff.field_reconstruction.reconstruct_2d_from_terms (cartesian basis);
    Ez is the closed radial form validated against vecdiff.longitudinal in
    notebook 05.  incidence='linear' is polarized at angle alpha;
    incidence='circular' uses e = (x_hat + i y_hat)/sqrt(2)."""
    s = case["s"]
    rho = np.hypot(xx, yy)
    varphi = np.arctan2(yy, xx)
    H0 = np.interp(rho, s, case["H0"])
    H2 = np.interp(rho, s, case["H2"])
    Hz = np.interp(rho, s, case["Hz"])
    if incidence == "circular":
        phase = np.exp(2.0j * varphi)
        Ex = π / np.sqrt(2.0) * (H0 - phase * H2)
        Ey = 1.0j * π / np.sqrt(2.0) * (H0 + phase * H2)
        Ez = -np.sqrt(2.0) * π * 1.0j * Hz * np.exp(1.0j * varphi)
        return Ex, Ey, Ez
    Ex = π * (np.cos(alpha) * H0 - np.cos(2.0 * varphi - alpha) * H2)
    Ey = π * (np.sin(alpha) * H0 - np.sin(2.0 * varphi - alpha) * H2)
    Ez = -2.0j * π * Hz * np.cos(varphi - alpha)
    return Ex, Ey, Ez


# ---------------------------------------------------------------------
# Spot metrics
# ---------------------------------------------------------------------

def hwhm(s, I):
    """Radius of the first crossing of half the central intensity (in s units)."""
    I = np.asarray(I, dtype=float)
    half = 0.5 * I[0]
    below = np.nonzero(I <= half)[0]
    if below.size == 0 or below[0] == 0:
        return np.nan
    j = below[0]
    return float(s[j - 1] + (half - I[j - 1]) * (s[j] - s[j - 1]) / (I[j] - I[j - 1]))


def first_min_radius(s, I):
    """Radius of the first local minimum (the first dark ring on this cut)."""
    dI = np.diff(I)
    idx = np.nonzero((dI[:-1] < 0.0) & (dI[1:] >= 0.0))[0]
    return float(s[idx[0] + 1]) if idx.size else np.nan


def sidelobe_level(s, I):
    """First-sidelobe peak intensity relative to the center, on this cut."""
    dI = np.diff(I)
    mins = np.nonzero((dI[:-1] < 0.0) & (dI[1:] >= 0.0))[0]
    if mins.size == 0:
        return np.nan
    j = mins[0] + 1
    return float(np.max(I[j:]) / I[0])


def encircled_radius(s, I, fraction=0.5):
    """Radius enclosing `fraction` of the energy of an azimuthally symmetric
    profile, relative to the energy inside the computed window s_max."""
    c = cumulative_trapezoid(I * s, s, initial=0.0)
    c /= c[-1]
    return float(np.interp(fraction, c, s))


def band_energies(case):
    """Exact (co, cross) channel energies of the focal field via Parseval:
    integral of |H_m[f]|^2 q dq over the full band equals integral of
    |f|^2 r dr, so no q grid (and no truncation/oscillation error) is needed."""
    r, P = case["r"], case["P"]
    e0 = simpson(np.abs((case["tp"] + case["ts"]) * P) ** 2 * r, x=r)
    e2 = simpson(np.abs((case["tp"] - case["ts"]) * P) ** 2 * r, x=r)
    return float(e0), float(e2)


def transmission(case):
    """Transmitted focal energy relative to the incident pupil energy (the
    tp=ts=1 field carries exactly the incident energy)."""
    e0, e2 = band_energies(case)
    e_in = simpson(np.abs(2.0 * case["P"]) ** 2 * case["r"], x=case["r"])
    return (e0 + e2) / e_in


def cross_fraction(case):
    """Energy fraction of the cross-polarized channel."""
    e0, e2 = band_energies(case)
    return e2 / (e0 + e2)


def z_energy(case):
    """Exact longitudinal-channel energy via Parseval (m=1), normalized so
    that the total focal energy is e0 + e2 + ez."""
    r, P = case["r"], case["P"]
    return float(2.0 * simpson(np.abs(case["w_z"] * case["tp"] * P) ** 2 * r, x=r))


def z_fraction(case):
    """Energy fraction of the longitudinal field |Ez|^2 in the focal plane."""
    e0, e2 = band_energies(case)
    ez = z_energy(case)
    return ez / (e0 + e2 + ez)


def hwhm_vs_psi(case, n_psi=181, total=False):
    """Half-max radius versus azimuth psi = phi - alpha (full circle)."""
    psi = np.linspace(0.0, 2.0 * π, n_psi)
    s = case["s"]
    radii = np.array([hwhm(s, intensity_cut(case, p, total=total)) for p in psi])
    return psi, radii


def spot_metrics(case):
    """Flat dict of the headline spot metrics for one configuration.
    *_tot metrics include the longitudinal |Ez|^2 in the intensity."""
    s = case["s"]
    c = cuts(case)
    R = {name: hwhm(s, I) for name, I in c.items()}
    z = {name: first_min_radius(s, I) for name, I in c.items()}
    return {
        "R_par": R["par"], "R_perp": R["perp"], "R_sca": R["sca"],
        "R_ideal": R["ideal"], "R_circ": R["circ"],
        "z_par": z["par"], "z_perp": z["perp"], "z_sca": z["sca"],
        "ellipticity": R["perp"] / R["par"],
        "anisotropy": (R["perp"] - R["par"]) / R["sca"],
        "sidelobe_par": sidelobe_level(s, c["par"]),
        "sidelobe_perp": sidelobe_level(s, c["perp"]),
        "f_cross": cross_fraction(case),
        "R_par_tot": R["par_tot"], "R_perp_tot": R["perp_tot"],
        "R_circ_tot": R["circ_tot"],
        "ellipticity_tot": R["perp_tot"] / R["par_tot"],
        "anisotropy_tot": (R["perp_tot"] - R["par_tot"]) / R["sca"],
        "f_z": z_fraction(case),
    }


def anisotropy_prediction(case, total=False):
    """First-order (in H2) prediction of the anisotropy A = (R_perp-R_par)/R_sca.

    Expanding I_par/perp = |H0 -+ H2|^2 around the scalar half-max radius gives
    R_par/perp ~ R_sca -+ 2 Re(H0 H2*)|_{R_sca} / |I_sca'(R_sca)|, hence
    A ~ 4 Re(H0 H2*) / (|I_sca'| R_sca): the deformation is governed by the
    *local* co/cross overlap at the half-max radius (first order in H2),
    not by the global energy fraction f_cross (second order in H2).

    With total=True the longitudinal term I_par gains +4|Hz|^2 (I_perp is
    untouched), so A_tot ~ 4 (B - Z) / (|I_sca'| R_sca): the elongation
    direction inverts exactly where Z = |Hz|^2 overtakes B = Re(H0 H2*) at
    the half-max radius."""
    s = case["s"]
    I_sca = np.abs(case["H0"]) ** 2
    B = np.real(case["H0"] * np.conj(case["H2"]))
    if total:
        B = B - np.abs(case["Hz"]) ** 2
    R = hwhm(s, I_sca)
    slope = np.interp(R, s, np.gradient(I_sca, s))
    return float(4.0 * np.interp(R, s, B) / (abs(slope) * R))


def airy_reference(case):
    """Analytic Airy radii (in lambda) of the uniform-pupil ideal scalar field."""
    scale = case["q_lambda"] / case["a"]
    return {"s_hwhm": AIRY_HWHM * scale, "s_zero": AIRY_FIRST_ZERO * scale}


# ---------------------------------------------------------------------
# Annotated focal maps: the measured radii drawn on the intensity pattern
# ---------------------------------------------------------------------

def display_half_size(case, n_lobes=3, factor=2.8):
    """Half-size of the display window for spot figures: tight around the
    central lobe (factor times R_perp), capped at the (n_lobes+1)-th minimum
    of the azimuthally averaged intensity so that at most n_lobes secondary
    maxima are ever visible."""
    s = case["s"]
    m = spot_metrics(case)
    A = np.abs(case["H0"]) ** 2 + np.abs(case["H2"]) ** 2
    interior = (A[1:-1] < A[:-2]) & (A[1:-1] <= A[2:])
    mins = s[1:-1][interior]
    half = min(factor * m["R_perp"], 0.999 * s[-1] / np.sqrt(2.0))
    if len(mins) > n_lobes:
        half = min(half, float(mins[n_lobes]))
    return float(half)


def plot_spot_with_radii(ax, case, alpha=0.0, half_size=None, n_img=400,
                         cmap="hot", incidence="linear", total=False):
    """Intensity map with the measured HWHM boundary and principal radii drawn
    and annotated on top of it.  White solid: measured half-max boundary of the
    vectorial field; cyan dashed: scalar (t-=0) half-max circle; arrows: the
    principal radii along/across the incident polarization.

    incidence='circular' draws the symmetric circular-incidence spot with its
    single radius instead.  total=True includes the longitudinal |Ez|^2 in the
    map and in every measured radius.  Returns (image, metrics dict)."""
    m = spot_metrics(case)
    s = case["s"]
    if half_size is None:
        half_size = display_half_size(case)

    theta = np.linspace(0.0, 2.0 * π, 361)
    if incidence == "circular":
        A = np.abs(case["H0"]) ** 2 + np.abs(case["H2"]) ** 2
        if total:
            A = A + 2.0 * np.abs(case["Hz"]) ** 2
        xx, yy, rho, _, extent = make_observation_grid(half_size, n_img)
        I = np.interp(rho, s, A)
        im = ax.imshow(I / I.max(), extent=extent, origin="lower", cmap=cmap)
        R = m["R_circ_tot"] if total else m["R_circ"]
        ax.plot(R * np.cos(theta), R * np.sin(theta), "w-", lw=1.4)
        ax.annotate("", xy=(R, 0), xytext=(-R, 0),
                    arrowprops=dict(arrowstyle="<->", color="w", lw=1.2))
        ax.text(0.03, 0.97, rf"$R_{{\rm circ}}$={R:.3f}$\lambda$",
                transform=ax.transAxes, va="top", color="w", fontsize=8)
    else:
        maps = intensity_maps(case, half_size, n_img, alpha=alpha)
        I = maps["I_tot"] if total else maps["I_vec"]
        im = ax.imshow(I / I.max(), extent=maps["extent"],
                       origin="lower", cmap=cmap)
        psi, radii = hwhm_vs_psi(case, 241, total=total)
        phi = psi + alpha
        ax.plot(radii * np.cos(phi), radii * np.sin(phi), "w-", lw=1.4)
        u = np.array([np.cos(alpha), np.sin(alpha)])
        v = np.array([-np.sin(alpha), np.cos(alpha)])
        R_par = m["R_par_tot"] if total else m["R_par"]
        R_perp = m["R_perp_tot"] if total else m["R_perp"]
        e = m["ellipticity_tot"] if total else m["ellipticity"]
        for R, w in [(R_par, u), (R_perp, v)]:
            ax.annotate("", xy=tuple(R * w), xytext=tuple(-R * w),
                        arrowprops=dict(arrowstyle="<->", color="w", lw=1.2))
        ax.text(*(1.28 * R_par * u), r"$R_\parallel$", color="w",
                fontsize=9, ha="center", va="center")
        ax.text(*(1.22 * R_perp * v), r"$R_\perp$", color="w",
                fontsize=9, ha="center", va="center")
        ax.text(0.03, 0.97,
                (rf"$R_\parallel$={R_par:.3f}$\lambda$" + "\n"
                 rf"$R_\perp$={R_perp:.3f}$\lambda$" + "\n"
                 rf"$e$={e:.3f}"),
                transform=ax.transAxes, va="top", color="w", fontsize=8)

    ax.plot(m["R_sca"] * np.cos(theta), m["R_sca"] * np.sin(theta),
            "--", color="cyan", lw=1.0)
    ax.text(0.97, 0.03, rf"$R_{{\rm sca}}$={m['R_sca']:.3f}$\lambda$",
            transform=ax.transAxes, ha="right", color="cyan", fontsize=8)
    ax.set(xlabel=r"$x/\lambda$", ylabel=r"$y/\lambda$")
    return im, m


# ---------------------------------------------------------------------
# Polarization aberration metrics (Chipman / McGuire-Chipman)
# ---------------------------------------------------------------------
#
# In the local s-p basis the Jones pupil of the diopter is diag(tp, ts) with
# the p axis radial.  Rotated to the Cartesian basis (Pauli expansion):
#
#     J(r, phi) = t_+ sigma_0 + t_- [cos(2 phi) sigma_1 + sin(2 phi) sigma_2]
#
# Real tp, ts (dielectric, below grazing) means zero retardance: the pupil is
# a pure diattenuator, with diattenuation oriented radially and magnitude
# D = (tp^2 - ts^2)/(tp^2 + ts^2).  Its lowest-order term is Chipman's
# *diattenuation defocus* (quadratic in r).

def diattenuation(case):
    """Diattenuation magnitude D(r) of the Jones pupil (radially oriented)."""
    Tp, Ts = case["tp"] ** 2, case["ts"] ** 2
    return (Tp - Ts) / (Tp + Ts)


def diattenuation_rms(case, weight="transmitted"):
    """RMS diattenuation over the pupil.  weight='incident' uses |P|^2 r dr;
    'transmitted' uses t_+^2 |P|^2 r dr (the co-channel intensity)."""
    D = diattenuation(case)
    r, P = case["r"], case["P"]
    w = np.abs(P) ** 2 * r
    if weight == "transmitted":
        w = w * (0.5 * (case["tp"] + case["ts"])) ** 2
    return float(np.sqrt(simpson(D ** 2 * w, x=r) / simpson(w, x=r)))


def cross_fraction_from_diattenuation(case):
    """Exact identity: with eps = t_-/t_+ one has, pointwise,
    eps^2/(1+eps^2) = (1 - sqrt(1-D^2))/2, so the Parseval cross fraction is

        f_cross = < (1 - sqrt(1-D^2))/2 >_w ,  w = (t_+^2 + t_-^2)|P|^2 r dr,

    whose leading order is <D^2>/4 (diattenuation-defocus squared)."""
    D = diattenuation(case)
    r, P = case["r"], case["P"]
    t_plus = 0.5 * (case["tp"] + case["ts"])
    t_minus = 0.5 * (case["tp"] - case["ts"])
    w = (t_plus ** 2 + t_minus ** 2) * np.abs(P) ** 2 * r
    frac = 0.5 * (1.0 - np.sqrt(np.maximum(0.0, 1.0 - D ** 2)))
    return float(simpson(frac * w, x=r) / simpson(w, x=r))


def fit_aberration_coefficients(case, fit_frac=0.3):
    """Fit the exact Fresnel channels over r <= fit_frac*a to the forms
    t_+ = g0 + c2 r^2 + c4 r^4 and t_- = d2 r^2 + d4 r^4 (Chipman-style
    piston/defocus/spherical terms).  D2 = 2 d2/g0 is the diattenuation-defocus
    coefficient of D(r) ~ D2 r^2."""
    r = case["r"]
    sel = (r > 0) & (r <= fit_frac * case["a"])
    rr = r[sel]
    t_plus = 0.5 * (case["tp"] + case["ts"])[sel]
    t_minus = 0.5 * (case["tp"] - case["ts"])[sel]
    g0, c2, c4 = np.linalg.lstsq(
        np.vstack([np.ones_like(rr), rr ** 2, rr ** 4]).T, t_plus, rcond=None)[0]
    d2, d4 = np.linalg.lstsq(
        np.vstack([rr ** 2, rr ** 4]).T, t_minus, rcond=None)[0]
    return {"g0": float(g0), "c2": float(c2), "c4": float(c4),
            "d2": float(d2), "d4": float(d4), "D2": float(2.0 * d2 / g0)}


def paraxial_theory(case):
    """Analytic second-order coefficients of the stigmatic diopter:

        xi  = n0 (z0-zi)^2 / (z0^2 zi^2 (n0^2-ni^2))
        t_+ = g0 + (n0+ni) (xi/2) r^2 ,   t_- = (n0-ni) (xi/2) r^2

    (matches quadratic fits of the exact FresnelOvoid; note that
    vecdiff.fresnel.FresnelOvoidParax carries a sign error in the r^2 term of
    t_s, which swaps/flips these coefficients)."""
    n0, ni, z0, zi = case["n0"], case["ni"], case["z0"], case["zi"]
    g0 = 2.0 * n0 / (n0 + ni)
    xi = n0 * (z0 - zi) ** 2 / (z0 ** 2 * zi ** 2 * (n0 ** 2 - ni ** 2))
    c2 = 0.5 * (n0 + ni) * xi
    d2 = 0.5 * (n0 - ni) * xi
    return {"g0": g0, "xi": xi, "c2": c2, "d2": d2, "D2": 2.0 * d2 / g0}
