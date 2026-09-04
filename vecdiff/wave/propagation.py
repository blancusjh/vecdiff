"""Propagators and the shared machinery of the spectral interface operator.

Three entry points matter:

``propagate``
    move a sampled field a distance ``dz`` by the angular spectrum.

``surface_spectrum``
    the source-driven interface operator — take a *named* incident field
    (a plane wave or a point source), refract or reflect it vectorially at a
    curved surface, and reduce the result to an angular spectrum by the exact
    surface integral.  For a surface of revolution the transform separates into
    azimuthal harmonics, which is what makes it cheap for interfaces that are
    hundreds of wavelengths across.

``surface_transform`` / ``incident_on_surface``
    the pieces the operator algebra of :mod:`vecdiff.wave.operators` reuses to
    map a *general* incident spectrum through the same local model.  The return
    integral is available both as the azimuthal Bessel kernel (surfaces of
    revolution) and as a fast, fully general NUFFT of the surface currents.
"""

from __future__ import annotations

import numpy as np
from scipy.special import jv

from .fields import Field
from .grids import Grid
from .interfaces import critical_angle, reflect_field, transmit_field
from .spectrum import AngularSpectrum
from .surfaces import Surface

__all__ = ["spectrum_of", "propagate", "surface_spectrum", "raised_cosine",
           "surface_transform", "incident_on_surface"]


def raised_cosine(u: np.ndarray, u1: float, u2: float) -> np.ndarray:
    """1 below ``u1``, cosine taper to 0 at ``u2``."""
    u = np.asarray(u, dtype=float)
    ramp = np.clip((u - u1) / max(u2 - u1, 1e-12), 0.0, 1.0)
    w = np.where(u > u1, 0.5 * (1 + np.cos(np.pi * ramp)), 1.0)
    return np.where(u >= u2, 0.0, w)


def spectrum_of(field: Field, sigma: int = +1) -> AngularSpectrum:
    """Angular spectrum of a sampled field (the inverse of ``AngularSpectrum.field``)."""
    E = np.stack(field.components)
    A = np.fft.fft2(np.fft.ifftshift(E, axes=(-2, -1)), axes=(-2, -1))
    A = A * (field.grid.dx * field.grid.dy)
    return AngularSpectrum(A, field.grid, field.wavelength, field.n, sigma)


def propagate(field: Field, dz: float, sigma: int = +1) -> Field:
    """Advance a field by ``dz`` (angular-spectrum, exact for the band-limited part)."""
    return spectrum_of(field, sigma).field(dz)


# ======================================================================
#  Incident fields on a surface
# ======================================================================
def _incident_on_surface(surface, rho2d, phi2d, z2d, k1, kind, polarization,
                         source_distance):
    """Incident direction and vector field for a *named* source (plane/point).

    The polarization callable specifies the designed transverse pair
    ``(Ex, Ey)``.  The longitudinal component is derived, never independently
    prescribed, from ``E.khat = 0``.  This is the pointwise analogue of the
    Fourier-space completion used by :meth:`Field.with_longitudinal`; it is
    exact here because the analytic source direction is known at every surface
    point, even though those points do not lie in one plane.
    """
    from .pupil import POLARIZATIONS
    pol = polarization if callable(polarization) else POLARIZATIONS[polarization]
    ex, ey = pol(rho2d / max(np.max(rho2d), 1e-12), phi2d)
    ex = np.asarray(ex, dtype=complex).ravel()
    ey = np.asarray(ey, dtype=complex).ravel()

    x = (rho2d * np.cos(phi2d)).ravel()
    y = (rho2d * np.sin(phi2d)).ravel()
    z = z2d.ravel()
    npts = x.size

    if kind == "plane":
        khat = np.zeros((3, npts))
        khat[2] = 1.0
        phase = np.exp(1j * k1 * z)
        scale = np.ones(npts)
    elif kind == "point":
        if source_distance is None:
            raise ValueError("kind='point' needs source_distance")
        d = float(source_distance)
        src = np.array([0.0, 0.0, -d])
        vec = np.stack([x - src[0], y - src[1], z - src[2]])
        R = np.linalg.norm(vec, axis=0)
        khat = vec / R
        phase = np.exp(1j * k1 * R) * (d / R)      # normalised to 1 at the vertex
        scale = np.ones(npts)
    else:  # pragma: no cover - user error path
        raise ValueError("kind must be 'plane' or 'point'")

    E = np.zeros((3, npts), dtype=complex)
    E[0] = ex * phase * scale
    E[1] = ey * phase * scale
    # Preserve the designed Cartesian transverse pair and derive the unique
    # longitudinal component for the known forward ray.  Named plane/point
    # sources used here have khat_z > 0; reject a grazing configuration rather
    # than silently returning a non-Maxwell field.
    kz_hat = khat[2]
    if np.any(np.abs(kz_hat) <= 64 * np.finfo(float).eps):
        raise ValueError("named-source Ez reconstruction is singular at grazing incidence")
    E[2] = -(khat[0] * E[0] + khat[1] * E[1]) / kz_hat
    return E, khat


def incident_on_surface(spec: AngularSpectrum, x, y, z, chunk: int = 2000):
    """Sample a spectrum and estimate one local energy-flow direction.

    Returns ``(E, khat)`` where ``E`` is the vector field ``(3, N)`` at the
    points and ``khat`` is the local propagation direction, taken as the
    local wavevector ``Im(E* . grad E)/|E|^2``.  The direction is exact for a
    single plane wave and is a geometrical-optics approximation where one ray
    reaches each point.  It is *not* a per-mode wavevector for an interfering
    superposition: applying Fresnel coefficients to this reconstructed
    direction is nonlinear in ``spec``.  ``InterfaceOperator`` therefore uses
    this function on one spectral mode at a time by default, and calls it on a
    total field only when ``incidence_model='local_ray'`` is explicit.
    """
    x = np.ascontiguousarray(np.atleast_1d(np.asarray(x, float)).ravel())
    y = np.ascontiguousarray(np.atleast_1d(np.asarray(y, float)).ravel())
    z = np.ascontiguousarray(np.atleast_1d(np.asarray(z, float)).ravel())
    k = spec.k
    KX, KY = spec.grid.KXY
    mask = spec.grid.propagating(k)
    # keep only modes the spectrum actually populates (a plane wave is one mode)
    nonzero = np.any(spec.A != 0, axis=0)
    mask = mask & nonzero
    kx = np.ascontiguousarray(KX[mask].astype(np.float64))
    ky = np.ascontiguousarray(KY[mask].astype(np.float64))
    kz = np.sqrt(np.maximum(k * k - kx * kx - ky * ky, 0.0)) * spec.sigma
    Amodes = spec.A[:, mask]                          # (3, M)
    area = (spec.grid.x.size * spec.grid.dx) * (spec.grid.y.size * spec.grid.dy)
    scale = 1.0 / area
    N = x.size
    E = np.zeros((3, N), dtype=complex)
    dE = np.zeros((3, 3, N), dtype=complex)           # dE[j, c] = dE_c/dx_j

    def _synth(cj):                                    # sum_m cj[m] exp(i k_m . Q)
        return _synth_modes(kx, ky, kz, np.ascontiguousarray(cj), x, y, z, chunk) * scale

    kvec = (kx, ky, kz)
    for c in range(3):
        Ac = Amodes[c]
        E[c] = _synth(Ac)
        for j in range(3):
            dE[j, c] = _synth(Ac * (1j * kvec[j]))
    den = np.sum(np.abs(E) ** 2, axis=0) + 1e-30
    kloc = np.stack([np.sum(np.imag(np.conj(E) * dE[j]), axis=0) / den
                     for j in range(3)])
    norm = np.linalg.norm(kloc, axis=0, keepdims=True)
    khat = np.divide(kloc, norm, out=np.tile([[0.0], [0.0], [1.0]], (1, N)),
                     where=norm > 1e-12)
    return E, khat


def _synth_modes(kx, ky, kz, cj, x, y, z, chunk):
    """``sum_m cj[m] exp(i (kx x + ky y + kz z))`` at scattered points.

    Uses a type-3 NUFFT (k-modes -> points) when finufft is available, so the
    cost is near-linear; falls back to a chunked direct sum otherwise."""
    try:
        import finufft
        return finufft.nufft3d3(kx, ky, kz, cj.astype(np.complex128),
                                x, y, z, isign=+1, eps=1e-8)
    except Exception:                                  # pragma: no cover
        N = x.size
        out = np.zeros(N, dtype=complex)
        for a in range(0, N, chunk):
            b = min(a + chunk, N)
            ph = np.outer(x[a:b], kx) + np.outer(y[a:b], ky) + np.outer(z[a:b], kz)
            out[a:b] = np.exp(1j * ph) @ cj
        return out


# ======================================================================
#  Surface sampling
# ======================================================================
def _axisym_samples(surface, aperture, n_rho, n_phi):
    """Sampling of a surface of revolution; the geometry the two return
    integrals share."""
    rho = np.linspace(aperture * 1e-4, aperture, int(n_rho))
    phi = np.arange(int(n_phi)) * 2 * np.pi / int(n_phi)
    RHO, PHI = np.meshgrid(rho, phi, indexing="ij")
    sag = surface.sag(rho)
    SAG = np.broadcast_to(sag[:, None], RHO.shape)
    nhat = surface.normal(RHO, PHI).reshape(3, -1)
    X = (RHO * np.cos(PHI)).ravel()
    Y = (RHO * np.sin(PHI)).ravel()
    Z = SAG.ravel()
    return dict(rho=rho, phi=phi, RHO=RHO, PHI=PHI, sag=sag, SAG=SAG,
                nhat=nhat, points=np.stack([X, Y, Z]), shape=RHO.shape)


def _freeform_samples(surface, aperture, n):
    """Sampling of a general (non-axisymmetric) surface on a Cartesian disk."""
    ax = np.linspace(-aperture, aperture, int(n))
    X, Y = np.meshgrid(ax, ax, indexing="xy")
    inside = (X * X + Y * Y) <= aperture * aperture
    x, y = X[inside], Y[inside]
    z = surface.sag_xy(x, y)
    nhat = surface.normal_xy(x, y)
    dxy = (ax[1] - ax[0]) ** 2
    dS = np.sqrt(1.0 + np.sum((nhat[:2] / nhat[2]) ** 2, axis=0)) * dxy
    return dict(points=np.stack([x, y, z]), nhat=nhat, dS=dS)


# ======================================================================
#  Return integrals  (surface currents -> angular spectrum)
# ======================================================================
def _return_integral_polar(datum, rho, sag, dsag, k_out, grid, m_max, n_kr,
                           sigma, wavelength, out_index,
                           datum_pair=None, normal_rz=None):
    """Azimuthal Bessel-kernel return integral for a surface of revolution.

    ``datum`` is the (already apodized) surface field, shape ``(3, n_rho, n_phi)``.

    With ``datum_pair`` and ``normal_rz`` given, the *Franz measure* is
    applied: every spectral component is weighted by the Kirchhoff obliquity
    pair ``(n.k_out + n.d)/2`` times the chart radiation factor ``k/kz`` —
    the stationary-phase content of the exact Franz/Stratton-Chu radiation of
    the surface currents, which reduces to unity in the planar limit.
    ``datum_pair`` carries ``(n.k_out)/2 * E`` on the surface grid (the
    Q-side half, azimuth-dependent for a general incident field);
    ``normal_rz = (n_rho(rho), n_z(rho))`` is the oriented meridional normal
    from which the direction-side half is assembled per outgoing angle.  Its
    axial part cancels ``k/kz`` exactly (``sigma * n_z/2`` survives), and the
    azimuthal cross term ``n_rho sin(theta') cos(psi - phi')`` folds back
    onto the *same* outgoing harmonic through ``J_{m-1} - J_{m+1} = 2 J_m'``.
    """
    phi_len = datum.shape[-1]
    cm = np.fft.fft(datum, axis=-1) / phi_len          # coeffs of exp(+i m phi)
    franz = datum_pair is not None
    if franz:
        cm_pair = np.fft.fft(datum_pair, axis=-1) / phi_len
        n_rho_arr, n_z_arr = normal_rz
    ms = np.arange(-int(m_max), int(m_max) + 1)

    kr_tab = np.linspace(0.0, k_out * 0.9995, int(n_kr))
    kz_tab = np.sqrt(np.maximum(k_out**2 - kr_tab**2, 0.0))
    inv_cos = k_out / np.maximum(kz_tab, 1e-9 * k_out)   # k/kz
    tan_t = kr_tab / np.maximum(kz_tab, 1e-9 * k_out)
    weight = rho * np.sqrt(1.0 + dsag ** 2) * (rho[1] - rho[0])
    phase_z = np.exp(-1j * sigma * np.outer(kz_tab, sag))  # (n_kr, n_rho)

    Am = np.zeros((3, ms.size, kr_tab.size), dtype=complex)
    for i, m in enumerate(ms):
        g = cm[:, :, m % phi_len] * weight[None, :]    # (3, n_rho)
        arg = np.outer(kr_tab, rho)
        Jm = jv(m, arg)                                # (n_kr, n_rho)
        kern = Jm * phase_z
        if not franz:
            Am[:, i, :] = 2 * np.pi * ((-1j) ** m) * (g @ kern.T)
            continue
        g1 = cm_pair[:, :, m % phi_len] * weight[None, :]
        Jpm = 0.5 * (jv(m - 1, arg) - jv(m + 1, arg))
        kern_p = Jpm * phase_z
        Am[:, i, :] = 2 * np.pi * ((-1j) ** m) * (
            (g1 @ kern.T) * inv_cos[None, :]
            + (g * (0.5 * sigma * n_z_arr)[None, :]) @ kern.T
            + 1j * ((g * (0.5 * n_rho_arr)[None, :]) @ kern_p.T)
            * tan_t[None, :]
        )

    KX, KY = grid.KXY
    KR = np.hypot(KX, KY)
    PHIK = np.arctan2(KY, KX)
    inside = KR <= kr_tab[-1]
    A = np.zeros((3, *grid.shape), dtype=complex)
    krq = np.where(inside, KR, 0.0)
    for i, m in enumerate(ms):
        ramp = np.exp(1j * m * PHIK)
        for c in range(3):
            re = np.interp(krq, kr_tab, Am[c, i].real)
            im = np.interp(krq, kr_tab, Am[c, i].imag)
            A[c] += (re + 1j * im) * ramp
    A *= inside[None]
    spec = AngularSpectrum(A, grid, wavelength, out_index, sigma)
    return spec.apodize_horizon().impose_transversality()


def _return_integral_nufft(points, Eout, dS, k_out, grid, sigma, wavelength,
                           out_index, eps: float = 1e-8,
                           pair_weight=None, nhat=None):
    """Fast, fully general return integral by a type-3 NUFFT of the currents.

    Computes ``A(k) = sum_Q Eout(Q) exp(-i k.Q) dS`` directly from the
    scattered surface points, so it needs no azimuthal symmetry — the same
    code serves a freeform surface and is near-linear in the number of
    samples.  Requires :mod:`finufft`.

    With ``pair_weight`` (``(n.k_out)/2`` per point) and ``nhat`` (the
    oriented unit normal, ``(3, N)``) given, the Franz measure is applied:
    ``A(k) = (k/kz) [ F(pair_weight E) + sum_j d_j F(n_j E / 2) ]`` — the
    Kirchhoff obliquity pair times the chart radiation factor, which is the
    stationary-phase content of the exact Franz radiation and reduces to the
    bare transform in the planar limit.  The ``n.d`` half separates over the
    three normal components, so the cost is four NUFFTs per field component.
    """
    import finufft
    x, y, z = (np.ascontiguousarray(points[i], dtype=np.float64) for i in range(3))
    KX, KY = grid.KXY
    mask = grid.propagating(k_out)
    kx = KX[mask].astype(np.float64)
    ky = KY[mask].astype(np.float64)
    kz = np.sqrt(np.maximum(k_out**2 - kx * kx - ky * ky, 0.0)) * sigma
    # want exp(-i (kx x + ky y + kz z)); finufft type-3 uses exp(+i (s x + ...))
    s, t, u = -kx, -ky, -kz
    w = np.asarray(dS, dtype=np.complex128)
    franz = pair_weight is not None
    if franz:
        dhat = np.stack([kx, ky, kz]) / k_out
        radiation = k_out / np.maximum(np.abs(kz), 1e-9 * k_out)
    A = np.zeros((3, *grid.shape), dtype=complex)
    for c in range(3):
        if not franz:
            cj = np.ascontiguousarray(Eout[c] * w, dtype=np.complex128)
            A[c][mask] = finufft.nufft3d3(x, y, z, cj, s, t, u, isign=+1,
                                          eps=eps)
            continue
        cj = np.ascontiguousarray(pair_weight * Eout[c] * w,
                                  dtype=np.complex128)
        fm = finufft.nufft3d3(x, y, z, cj, s, t, u, isign=+1, eps=eps)
        for j in range(3):
            cj = np.ascontiguousarray(0.5 * nhat[j] * Eout[c] * w,
                                      dtype=np.complex128)
            fm = fm + dhat[j] * finufft.nufft3d3(x, y, z, cj, s, t, u,
                                                 isign=+1, eps=eps)
        A[c][mask] = fm * radiation
    spec = AngularSpectrum(A, grid, wavelength, out_index, sigma)
    return spec.apodize_horizon().impose_transversality()


def surface_transform(points, Eout, dS, grid, *, k_out, sigma=+1,
                      wavelength=1.0, out_index=1.0, eps=1e-8,
                      pair_weight=None, nhat=None):
    """Public wrapper for the general (NUFFT) surface-current transform.

    Pass ``pair_weight`` and ``nhat`` to radiate with the Franz measure (see
    :func:`_return_integral_nufft`); omit them for the bare transform.
    """
    return _return_integral_nufft(points, Eout, dS, k_out, grid, sigma,
                                  wavelength, out_index, eps=eps,
                                  pair_weight=pair_weight, nhat=nhat)


# ======================================================================
#  The source-driven interface operator (unchanged public behaviour)
# ======================================================================
def _rim_tir_apodization(rho, phi_len, aperture, edge_softness, coeffs, mode,
                         n1, n2, tir_margin):
    """Soft rim and a smooth stand-off from the grazing/TIR limit."""
    vis = raised_cosine(rho, aperture * (1.0 - edge_softness), aperture)
    theta_c = critical_angle(n1, n2)
    if mode == "t" and np.isfinite(theta_c):
        cos_i = coeffs.cos_i.reshape(len(rho), phi_len).mean(axis=1)
        cos_c = np.cos(theta_c)
        vis = vis * 0.5 * (1 + np.tanh((cos_i - cos_c - tir_margin) /
                                       max(tir_margin, 1e-6)))
    return vis


def surface_spectrum(
    surface: Surface,
    grid: Grid,
    *,
    n1: float,
    n2: float,
    wavelength: float = 1.0,
    aperture: float | None = None,
    incident: str = "plane",
    polarization: str = "x",
    source_distance: float | None = None,
    mode: str = "t",
    m_max: int = 6,
    n_rho: int = 1200,
    n_phi: int = 64,
    n_kr: int = 512,
    edge_softness: float = 0.25,
    tir_margin: float = 0.04,
    sigma: int = +1,
    measure: str = "franz",
) -> AngularSpectrum:
    """Angular spectrum produced by a *named* incident field at an interface.

    The incident field (a plane wave or a point source) is sampled on the
    surface; its designed ``(Ex, Ey)`` pair is completed by ``E.k = 0`` before
    it is refracted (``mode='t'``) or reflected (``mode='r'``) with the full
    vector Fresnel operator, softly apodized near the rim and the
    total-internal-reflection limit, and transformed to an angular spectrum by
    the azimuthal Bessel kernel.  For a general incident spectrum, or a
    non-axisymmetric surface, use :class:`vecdiff.wave.operators.InterfaceOperator`.

    ``measure`` selects the amplitude measure of the return integral:
    ``"franz"`` (default) applies the Kirchhoff obliquity pair times the chart
    radiation factor — the stationary-phase content of the exact Franz
    radiation — while ``"flat"`` keeps the bare surface transform.  ``sigma``
    is the incident propagation sense; reflection reverses it in the returned
    spectrum.
    """
    if measure not in ("franz", "flat"):
        raise ValueError("measure must be 'franz' or 'flat'")
    k1 = 2 * np.pi * n1 / wavelength
    out_index = n2 if mode == "t" else n1
    k_out = 2 * np.pi * out_index / wavelength

    if aperture is None:
        aperture = surface.max_radius if np.isfinite(surface.max_radius) else 10.0 * wavelength
    aperture = float(aperture)

    smp = _axisym_samples(surface, aperture, n_rho, n_phi)
    rho, phi, RHO, PHI, SAG = smp["rho"], smp["phi"], smp["RHO"], smp["PHI"], smp["SAG"]
    nhat = smp["nhat"]

    E_in, khat = _incident_on_surface(surface, RHO, PHI, SAG, k1,
                                      incident, polarization, source_distance)
    if mode == "t":
        E_out, k_dir, coeffs = transmit_field(E_in, khat, nhat, n1, n2)
    else:
        E_out, k_dir, coeffs = reflect_field(E_in, khat, nhat, n1, n2)

    vis = _rim_tir_apodization(rho, len(phi), aperture, edge_softness,
                               coeffs, mode, n1, n2, tir_margin)
    datum = E_out * np.broadcast_to(vis[:, None], RHO.shape).ravel()[None, :]
    shape3 = (3, len(rho), len(phi))
    datum_pair = None
    normal_rz = None
    if measure == "franz":
        outward = nhat if mode == "t" else -nhat
        pair = 0.5 * np.abs(np.sum(outward * k_dir, axis=0))
        datum_pair = (datum * pair[None, :]).reshape(shape3)
        normal_rz = surface.normal(rho)
        if mode == "r":
            normal_rz = tuple(-component for component in normal_rz)
    datum = datum.reshape(shape3)

    out_sigma = sigma if mode == "t" else -sigma
    return _return_integral_polar(datum, rho, smp["sag"], surface.dsag(rho),
                                  k_out, grid, m_max, n_kr, out_sigma,
                                  wavelength, out_index,
                                  datum_pair=datum_pair, normal_rz=normal_rz)
