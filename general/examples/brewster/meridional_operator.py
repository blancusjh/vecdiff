"""The vertex-plane refraction operator on one meridian, with Hankel propagation.

This is the simple form of the eikonal-angular-spectrum method, written for
axisymmetric fields of azimuthal order m = 0 in the cylindrical basis
(E_rho, E_phi, E_z independent of phi). It complements vecdiff's
``MeridianTransport``, which needs the axial ray to meet no caustic on any
leg: the tip of a cone, sharp or rounded, focuses the rays near the axis,
and nested Brewster cones separated by millimetres of air put that focus
between two legs. Here the map is resampled with superposition where it
folds, and the near-axis rays are kept, so cones with sharp tips can be
stacked.

Between the vertex planes of consecutive surfaces the transverse pair
(E_rho, E_phi) is propagated with the exact angular spectrum: TM (E_rho) and
TE (E_phi) decouple and each is an order-1 quasi-discrete Hankel transform
(Guizar-Sicairos and Gutierrez-Vega, JOSA A 21, 53, 2004). At a surface
z = z_v + h(rho):

1. each sample of the vertex plane gets a ray direction from the
   intensity-weighted gradient of its phase, smoothed over ``SMOOTH``;
2. the ray goes to the surface (Newton on t u_z = h(rho + t u_rho)),
   the Fresnel transmission is applied in the local s/p frame, and the
   refracted ray goes back to the same vertex plane;
3. the field is resampled on the Hankel grid, with ray-tube energy
   conservation |E'|^2 = |t E|^2 (cos t uz rho d rho) / (cos i uz' rho' d rho').

Lengths in metres, time convention exp(-i omega t).
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import j0, j1, jn_zeros

# Intensity below which a sample carries no ray, relative to the maximum.
DARK = 1e-14
# Ray directions are smoothed over this length (m): a ray field varies on
# scales much larger than the wavelength.
SMOOTH = 5e-6


def _j2(x):
    return 2 * j1(x) / x - j0(x)


class HankelGrid:
    """Order-1 quasi-discrete Hankel transform on ``count`` nodes up to ``radius``."""

    def __init__(self, radius, count):
        zeros = jn_zeros(1, count + 1)
        self.S, self.j, self.R = zeros[-1], zeros[:-1], float(radius)
        self.rho = self.j * self.R / self.S
        self.k = self.j / self.R
        self.Jp1 = np.abs(_j2(self.j))
        self.T = 2 * j1(np.outer(self.j, self.j) / self.S) / (
            np.outer(self.Jp1, self.Jp1) * self.S
        )
        # Quadrature weights: integral f rho d rho ~ sum w f.
        self.w_rho = 2 * self.R**2 / (self.S**2 * self.Jp1**2)

    def forward(self, f):
        """Samples on rho -> orthogonal spectral vector (energy normalised)."""
        return self.T @ (f * self.R / self.Jp1)

    def inverse(self, F):
        return (self.T @ F) * self.Jp1 / self.R

    def evaluate(self, F, rho, order):
        """sum_m F_m 2 J_order(k_m rho) / (|J_2(j_m)| S R) at arbitrary rho.

        With order 1 it reproduces ``inverse``; order 0 gives E_z through
        E_z = i (k / k_z) E_rho in the spectral domain.
        """
        bessel = j1 if order == 1 else j0
        kernel = bessel(np.outer(np.atleast_1d(rho), self.k))
        return kernel @ (2 * F / (self.Jp1 * self.S * self.R))


def kz(grid, n, k0):
    return np.sqrt((n * k0) ** 2 - grid.k.astype(complex) ** 2)


@dataclass
class Field:
    """E_rho and E_phi of order m = 0 on a vertex plane, in medium of index n."""

    rho: np.ndarray
    phi: np.ndarray
    n: float


def propagate(grid, field, k0, dz):
    """Exact angular-spectrum step of (E_rho, E_phi) over dz; evanescent waves dropped."""
    phase = np.exp(1j * kz(grid, field.n, k0) * dz)
    phase[grid.k > field.n * k0] = 0.0
    return Field(
        grid.inverse(grid.forward(field.rho) * phase),
        grid.inverse(grid.forward(field.phi) * phase),
        field.n,
    )


def powers(grid, field, k0):
    """z-flux of the TM (E_rho) and TE (E_phi) parts, in one arbitrary unit."""
    n = field.n
    k_z = kz(grid, n, k0)
    keep = grid.k < n * k0
    a = np.abs(grid.forward(field.rho)) ** 2
    b = np.abs(grid.forward(field.phi)) ** 2
    tm = n * np.sum(a[keep] * (n * k0) / k_z[keep].real)
    te = n * np.sum(b[keep] * k_z[keep].real / (n * k0))
    return float(tm), float(te)


def spectra(grid, field):
    """The TM and TE spectral vectors of a field, to reuse across many planes."""
    return grid.forward(field.rho), grid.forward(field.phi)


def components(grid, field, k0, rho, dz, cached=None):
    """(E_rho, E_phi, E_z) at radii ``rho`` a distance dz past the plane.

    ``cached`` is the pair returned by ``spectra(grid, field)``.
    """
    n = field.n
    F_tm, F_te = cached if cached is not None else spectra(grid, field)
    k_z = kz(grid, n, k0)
    keep = grid.k < n * k0
    phase = np.where(keep, np.exp(1j * k_z * dz), 0)
    ratio = np.where(keep, grid.k / np.where(keep, k_z, 1), 0)
    e_rho = grid.evaluate(F_tm * phase, rho, 1)
    e_phi = grid.evaluate(F_te * phase, rho, 1)
    e_z = 1j * grid.evaluate(F_tm * phase * ratio, rho, 0)
    return e_rho, e_phi, e_z


@dataclass(frozen=True)
class Surface:
    """z = z_v + h(rho), normal toward +z, between indices n1 (before) and n2."""

    z_vertex: float
    sag: object
    slope: object
    n1: float
    n2: float
    name: str = ""


def cone(z_vertex, slope, n1, n2, tip=0.0, name=""):
    """Cone of slope g; with tip a > 0 its tip is the hyperbola g (sqrt(r² + a²) - a)."""
    g, a = float(slope), float(tip)
    if a == 0:
        return Surface(z_vertex, lambda r: g * r, lambda r: g + 0 * r, n1, n2, name)
    return Surface(
        z_vertex,
        lambda r: g * (np.sqrt(r * r + a * a) - a),
        lambda r: g * r / np.sqrt(r * r + a * a),
        n1,
        n2,
        name,
    )


def plane(z_vertex, n1, n2, name=""):
    return Surface(z_vertex, lambda r: 0 * r, lambda r: 0 * r, n1, n2, name)


def rays(grid, field, k0, smooth=None):
    """Reference phase and ray direction (u_rho, u_z) at each sample."""
    smooth = SMOOTH if smooth is None else smooth
    er, ef = field.rho, field.phi
    increment = er[1:] * np.conj(er[:-1]) + ef[1:] * np.conj(ef[:-1])
    dphi = np.angle(increment)
    phase = np.concatenate(([0.0], np.cumsum(dphi)))
    mid = 0.5 * (grid.rho[1:] + grid.rho[:-1])
    k_rho = np.interp(grid.rho, mid, dphi / np.diff(grid.rho))
    weight = np.abs(er) ** 2 + np.abs(ef) ** 2
    lit = weight > DARK * weight.max()
    k_rho = np.interp(grid.rho, grid.rho[lit], k_rho[lit])
    if smooth:
        sigma = smooth / np.mean(np.diff(grid.rho))
        wl = np.where(lit, weight, 0)
        k_rho = gaussian_filter1d(k_rho * wl, sigma) / np.maximum(
            gaussian_filter1d(wl, sigma), 1e-300
        )
        k_rho = np.interp(grid.rho, grid.rho[lit], k_rho[lit])
    u_rho = np.clip(k_rho / (field.n * k0), -0.999, 0.999)
    return phase, u_rho, np.sqrt(1 - u_rho**2)


def _hit(surface, rho, u_rho, u_z):
    """Signed leg t from (rho, z_v) along (u_rho, u_z) to z = z_v + h(rho + t u_rho)."""
    h, dh = surface.sag, surface.slope
    t = h(rho) / (u_z - dh(rho) * u_rho)  # tangent-plane start
    for _ in range(60):
        r = np.abs(rho + t * u_rho)
        f = t * u_z - h(r)
        df = u_z - dh(r) * u_rho * np.sign(rho + t * u_rho + 1e-300)
        step = f / df
        t = t - step
        if np.all(np.abs(step) < 1e-15 + 1e-12 * np.abs(t)):
            break
    return t


def refract(grid, field, surface, k0, smooth=None, diattenuation=True):
    """The refraction operator of ``surface`` on its vertex plane.

    With ``diattenuation=False`` the s amplitude transmission is replaced by
    the p one: the same geometry without its polarizing action, a reference.

    ``field`` lives on the vertex plane in the index before the surface; the
    result is the field of the refracted rays carried back to the same
    plane, in the index after it. Returns the field and a dict of
    diagnostics (incidence angle, |t_s/t_p|, ray map, folds).
    """
    n1, n2 = surface.n1, surface.n2
    rho = grid.rho
    phase_ref, ur, uz = rays(grid, field, k0, smooth)
    t1 = _hit(surface, rho, ur, uz)
    rho_s = rho + t1 * ur
    z_s = t1 * uz
    g = surface.slope(np.abs(rho_s))
    m = np.stack((-g, 0 * g, np.ones_like(g)), -1)
    m /= np.linalg.norm(m, axis=-1, keepdims=True)
    u = np.stack((ur, 0 * ur, uz), -1)
    ci = np.sum(u * m, -1)
    eta = n1 / n2
    weight = np.abs(field.rho) ** 2 + np.abs(field.phi) ** 2
    ok = (ci > 0.05) & (eta**2 * (1 - ci**2) < 1 - 1e-6) & (rho_s >= 0)
    ok &= weight > DARK * weight.max()
    ci = np.where(ok, ci, 1.0)
    ct = np.sqrt(1 - eta**2 * (1 - ci**2))
    u2 = eta * u + (ct - eta * ci)[:, None] * m
    ts = 2 * n1 * ci / (n1 * ci + n2 * ct)
    tp = 2 * n1 * ci / (n2 * ci + n1 * ct)
    if not diattenuation:
        ts = tp
    ts, tp = np.where(ok, ts, 0), np.where(ok, tp, 0)
    s = np.cross(u, m)
    norm = np.linalg.norm(s, axis=-1, keepdims=True)
    s = np.where(norm > 1e-12, s / np.maximum(norm, 1e-300), [0.0, 1.0, 0.0])
    p1, p2 = np.cross(s, u), np.cross(s, u2)
    e = np.stack((field.rho, field.phi, -ur * field.rho / uz), -1)
    e_slow = e * np.exp(-1j * phase_ref)[:, None]
    et = (ts * np.sum(e_slow * s, -1))[:, None] * s + (
        tp * np.sum(e_slow * p1, -1)
    )[:, None] * p2
    t2 = -z_s / u2[:, 2]
    rho_v = rho_s + t2 * u2[:, 0]
    total_phase = phase_ref + k0 * (n1 * t1 + n2 * t2)
    jac = (rho * np.gradient(rho)) / np.abs(rho_v * np.gradient(rho_v))
    et *= np.sqrt(np.abs(ct * uz * jac / (ci * u2[:, 2])))[:, None]
    total_phase = np.where(ok, total_phase, 0)
    et[~ok] = 0
    out = [np.zeros_like(rho, dtype=complex) for _ in range(2)]
    segments = np.flatnonzero(ok[1:] & ok[:-1])
    folds = int(np.sum(np.diff(rho_v)[segments] <= 0))
    # Each segment of the ray map is interpolated linearly; where the map
    # folds, the rays reaching a point are superposed.
    for i in segments:
        lo, hi = sorted((rho_v[i], rho_v[i + 1]))
        a, b = np.searchsorted(rho, lo), np.searchsorted(rho, hi)
        if b <= a:
            continue
        x = (rho[a:b] - rho_v[i]) / (rho_v[i + 1] - rho_v[i])
        ph = np.exp(1j * ((1 - x) * total_phase[i] + x * total_phase[i + 1]))
        for j in range(2):
            out[j][a:b] += ((1 - x) * et[i, j] + x * et[i + 1, j]) * ph
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ok, np.abs(ts / np.where(tp == 0, 1, tp)), np.nan)
    diagnostics = dict(
        incidence=np.degrees(np.arccos(ci)),
        ratio=ratio,
        rho_surface=rho_s,
        rho_virtual=rho_v,
        folds=folds,
        carried=ok,
    )
    return Field(out[0], out[1], n2), diagnostics


def run(grid, field, surfaces, k0, *, smooth=None, record=None, diattenuation=True):
    """Carry ``field``, given on the first surface's vertex plane, through ``surfaces``.

    ``record(index, surface)`` says after which surfaces the field on the
    vertex plane is kept (all by default). Returns the field on the last
    vertex plane, its z, the kept (index, z, Field) and the diagnostics.
    """
    z = surfaces[0].z_vertex
    recorded, diagnostics = [], []
    for index, surface in enumerate(surfaces):
        field = propagate(grid, field, k0, surface.z_vertex - z)
        field, diag = refract(grid, field, surface, k0, smooth, diattenuation)
        diagnostics.append(diag)
        z = surface.z_vertex
        if record is None or record(index, surface):
            recorded.append((index, z, field))
    return field, z, recorded, diagnostics
