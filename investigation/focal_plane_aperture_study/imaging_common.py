"""Shared two-diopter (4f-like) imaging machinery for studies 3 and 4.

System: D1 images the mask plane onto a Fourier plane where a circular
(optionally annular) pupil is applied; D2 forms the final image.  The
*scalar* model applies only ``t_plus`` at each diopter (``t_minus = 0``);
the *vectorial* model applies the full transverse diopter operator.  Both
share pupil and geometry, so image differences isolate the mixing term.
"""

import numpy as np

from vecdiff import CartesianSurface, FieldCartesian, Grid
from vecdiff.propagation import fresnel_coefficients_on_grid

LAM = 532e-6  # mm

D1 = dict(n0=1.0, ni=2.4, z0=-6.0, zi=3.6)
D2 = dict(n0=D1["ni"], ni=1.0, z0=-D1["zi"], zi=3.6)

def mag_of(d2=None, d1=None):
    """Transverse magnification of the two-diopter cascade (two Fourier stages)."""
    d1 = D1 if d1 is None else d1
    d2 = D2 if d2 is None else d2
    return (d2["zi"] * d1["ni"]) / (d1["zi"] * d2["ni"])


MAG = mag_of()

N = 641
L = 0.02  # mm
N_K = 1024

_x = np.linspace(-L / 2.0, L / 2.0, N)
X, Y = np.meshgrid(_x, _x)
DX = float(_x[1] - _x[0])


def Kc_of(r_a, d1=None):
    """Pupil cutoff (angular frequency) for aperture radius ``r_a``."""
    d1 = D1 if d1 is None else d1
    alpha = np.arctan(r_a / abs(d1["z0"]))
    return d1["n0"] * (2.0 * np.pi / LAM) * np.sin(alpha)


def airy_radius(r_a, d1=None):
    """Object-space Airy radius of the system at aperture ``r_a``."""
    return 3.8317059702075125 / Kc_of(r_a, d1)


def system_caption(r_a=None, *, d1=None, d2=None, lam=LAM):
    """Compact one-line summary of the two-diopter stigmatic imaging system.

    The diopters are conjugate: D2 runs the indices back (ni -> n0) and its
    object plane is fixed by D1's image plane, ``z0(D2) = -zi(D1)``.  So the
    whole cascade is set by the index pair and the three axial planes
    ``z0, zi1, zi2`` -- the rest is redundant and omitted.
    """
    d1 = D1 if d1 is None else d1
    d2 = D2 if d2 is None else d2
    cap = (rf"dioptrios conjugados: $n_0$={d1['n0']:g}, $n_i$={d1['ni']:g}  ·  "
           rf"$z_0$={d1['z0']:g}, $z_{{i1}}$={d1['zi']:g}, $z_{{i2}}$={d2['zi']:g} mm  ·  "
           rf"$\lambda$={lam * 1e6:.0f} nm")
    if r_a is not None:
        alpha = np.degrees(np.arctan(r_a / abs(d1["z0"])))
        cap += rf"  ·  $r_a$={r_a:g} mm, $\alpha_\mathrm{{max}}$={alpha:.1f}°"
    return cap


# ------------------------------------------------------------------ #
#  Masks                                                               #
# ------------------------------------------------------------------ #

def two_line_mask(theta, sep, line_w, line_len, xy=None):
    """Two lines whose separation axis makes ``theta`` with x.

    ``xy`` optionally overrides the sampling grid with a local ``(Xg, Yg)``
    pair (e.g. a fine grid cropped to a display window) instead of the module
    grid used for propagation.
    """
    Xg, Yg = (X, Y) if xy is None else xy
    ux, uy = np.cos(theta), np.sin(theta)
    vx, vy = -uy, ux
    T = np.zeros_like(Xg)
    for sign in (-1.0, 1.0):
        cx, cy = sign * 0.5 * sep * ux, sign * 0.5 * sep * uy
        u = (Xg - cx) * ux + (Yg - cy) * uy
        v = (Xg - cx) * vx + (Yg - cy) * vy
        T[(np.abs(u) <= 0.5 * line_w) & (np.abs(v) <= 0.5 * line_len)] = 1.0
    return T


def two_disc_mask(theta, sep, diameter, xy=None):
    """Two discs (canonical two-point objects) separated along ``theta``.

    ``xy`` optionally overrides the sampling grid, see ``two_line_mask``.
    """
    Xg, Yg = (X, Y) if xy is None else xy
    ux, uy = np.cos(theta), np.sin(theta)
    T = np.zeros_like(Xg)
    for sign in (-1.0, 1.0):
        cx, cy = sign * 0.5 * sep * ux, sign * 0.5 * sep * uy
        T[(Xg - cx) ** 2 + (Yg - cy) ** 2 <= (0.5 * diameter) ** 2] = 1.0
    return T


def fine_mask_grid(half_size_mm, n_img):
    """Local ``(Xg, Yg)`` Cartesian grid for a high-resolution display-only mask."""
    x = np.linspace(-half_size_mm, half_size_mm, n_img)
    return np.meshgrid(x, x)


# ------------------------------------------------------------------ #
#  Propagation                                                         #
# ------------------------------------------------------------------ #

def _kgrid(n, dx):
    k = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    KX, KY = np.meshgrid(k, k)
    return Grid.from_cartesian(KX, KY, domain="k")


def zoomed_focal_kgrid(half_size_mm, n_img, d2=None):
    """Dense k-grid covering only ``[-half_size_mm, half_size_mm]`` in D2's focal plane.

    The default k-grid built by ``_kgrid`` spans the *whole* Nyquist range at
    ``N_K`` points, so a pixel's physical size is the total field of view
    divided by ``N_K`` -- zooming into a sub-wavelength window then only shows
    a handful of those pixels (blocky renders).  ``FT2`` evaluates a custom,
    non-FFT-grid ``kgrid`` with a zoom FFT (``scipy.signal.zoom_fft``), so
    passing a k-grid restricted to the small window of interest gives ``n_img``
    points *inside that window specifically*, at the cost of one chirp-z
    transform instead of a full-field FFT -- i.e. a free "digital zoom" for
    figures, decoupled from the coarse default field of view.
    """
    d2 = D2 if d2 is None else d2
    scale2 = LAM * d2["zi"] / (2.0 * np.pi * d2["ni"])
    k_lim = half_size_mm / scale2
    k = np.linspace(-k_lim, k_lim, n_img)
    KX, KY = np.meshgrid(k, k)
    return Grid.from_cartesian(KX, KY, domain="k")


def apply_t_plus(field, diopter):
    """Scalar limit: multiply both components by t_plus = (tp + ts)/2."""
    support = (np.abs(field.x) > 0.0) | (np.abs(field.y) > 0.0)
    tp, ts = fresnel_coefficients_on_grid(field.grid.R, diopter, support=support)
    t_plus = 0.5 * (tp + ts)
    return FieldCartesian(x=t_plus * field.x, y=t_plus * field.y,
                          grid=field.grid, symmetric=False)


def stage1(mask, vectorial, d1=None):
    """Mask plane -> Fourier plane through D1 (no pupil yet)."""
    d1 = D1 if d1 is None else d1
    E0 = FieldCartesian(x=mask.astype(complex), y=np.zeros_like(mask, dtype=complex),
                        grid=Grid.from_cartesian(X, Y), symmetric=False)
    diopter = CartesianSurface(**d1)
    if vectorial:
        return E0.propagate_through_diopter(
            z=diopter.zi, ovoid=diopter, method="fft",
            kgrid=_kgrid(N_K, DX), transmission="vectorial")
    E0s = apply_t_plus(E0, diopter)
    return E0s.propagate_through_diopter(
        z=diopter.zi, ovoid=diopter, method="fft",
        kgrid=_kgrid(N_K, DX), transmission="identity")


def stage2(E1, r_a, vectorial, eps=0.0, edge_p=0.0, d2=None, d1=None, kgrid=None):
    """Pupil in the Fourier plane, then D2 to the image plane.

    ``eps`` is the central-obstruction fraction of the pupil cutoff (annulus).
    ``edge_p`` applies a smooth edge-weighting amplitude ``(|k|/kc)^edge_p``,
    which concentrates the pupil energy near the grazing rim -- where
    ``t_minus/t_plus`` peaks -- to amplify the polarization mixing without the
    hard side-lobes of a closed annulus.
    ``d1``/``d2`` override the first/second-diopter geometry.
    ``kgrid`` overrides the default full-field output k-grid, e.g. with
    ``zoomed_focal_kgrid`` for a densely sampled crop around a small region.
    """
    d1 = D1 if d1 is None else d1
    d2 = D2 if d2 is None else d2
    kx = E1.grid.X[0, :]
    ky = E1.grid.Y[:, 0]
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    kc2 = Kc_of(r_a, d1) ** 2
    pupil = ((K2 <= kc2) & (K2 >= (eps**2) * kc2)).astype(float)
    if edge_p > 0.0:
        pupil = pupil * (K2 / kc2) ** (0.5 * edge_p)

    scale = LAM * d1["zi"] / (2.0 * np.pi * d1["ni"])
    XF, YF = np.meshgrid(scale * kx, scale * ky)
    Ef = FieldCartesian(x=E1.x * pupil, y=E1.y * pupil,
                        grid=Grid.from_cartesian(XF, YF), symmetric=False)

    if kgrid is None:
        dxf = scale * float(kx[1] - kx[0])
        kgrid = _kgrid(N_K, dxf)
    diopter = CartesianSurface(**d2)
    if vectorial:
        return Ef.propagate_through_diopter(
            z=diopter.zi, ovoid=diopter, method="fft",
            kgrid=kgrid, output="focal",
            wavelength=LAM, transmission="vectorial")
    Efs = apply_t_plus(Ef, diopter)
    return Efs.propagate_through_diopter(
        z=diopter.zi, ovoid=diopter, method="fft",
        kgrid=kgrid, output="focal",
        wavelength=LAM, transmission="identity")


def image(mask, r_a, vectorial, eps=0.0, edge_p=0.0, d2=None, d1=None, kgrid=None, _cache={}):
    """Full pipeline mask -> image field (stage1 cached per mask/model/d1)."""
    d1_key = tuple(sorted((D1 if d1 is None else d1).items()))
    key = (mask.tobytes(), vectorial, d1_key)
    if key not in _cache:
        _cache.clear()  # keep at most one mask in memory per model pair
        _cache[key] = stage1(mask, vectorial, d1=d1)
        other = (mask.tobytes(), not vectorial, d1_key)
        _cache[other] = stage1(mask, not vectorial, d1=d1)
    return stage2(_cache[key], r_a, vectorial, eps=eps, edge_p=edge_p, d2=d2, d1=d1, kgrid=kgrid)


# ------------------------------------------------------------------ #
#  Measurement                                                         #
# ------------------------------------------------------------------ #

def _bilinear(I, xr, yr, px, py):
    ix = np.clip(np.searchsorted(xr, px) - 1, 0, xr.size - 2)
    iy = np.clip(np.searchsorted(yr, py) - 1, 0, yr.size - 2)
    wx = np.clip((px - xr[ix]) / (xr[ix + 1] - xr[ix]), 0.0, 1.0)
    wy = np.clip((py - yr[iy]) / (yr[iy + 1] - yr[iy]), 0.0, 1.0)
    return ((1 - wy) * ((1 - wx) * I[iy, ix] + wx * I[iy, ix + 1])
            + wy * ((1 - wx) * I[iy + 1, ix] + wx * I[iy + 1, ix + 1]))


def axis_profile(E, theta, sep, n=801, span=1.1, mag=MAG):
    """Total intensity along the separation axis (image coordinates)."""
    I = np.abs(E.x) ** 2 + np.abs(E.y) ** 2
    xr = E.grid.X[0, :]
    yr = E.grid.Y[:, 0]
    u = np.linspace(-1.0, 1.0, n) * (span * mag * sep)
    return u, _bilinear(I, xr, yr, u * np.cos(theta), u * np.sin(theta))


def valley_contrast(E, theta, sep, mag=MAG):
    """Two-feature valley contrast; feature images sit at u = +-mag*sep/2.

    Positive: a dip exists between the features (distinguishable).
    Negative or ~0: the valley is filled (indistinguishable).
    """
    u, profile = axis_profile(E, theta, sep, mag=mag)
    u_line = 0.5 * mag * sep
    peak_win = np.abs(np.abs(u) - u_line) <= 0.3 * u_line
    valley_win = np.abs(u) <= 0.4 * u_line
    peaks = 0.5 * (profile[peak_win & (u < 0)].max() + profile[peak_win & (u > 0)].max())
    valley = profile[valley_win].min()
    return float((peaks - valley) / (peaks + valley + 1e-300)), u, profile
