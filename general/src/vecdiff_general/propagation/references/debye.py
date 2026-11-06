"""Exact single-interface (first Debye term) fields for a cylinder and a sphere.

A plane wave travelling along +z meets one curved interface. The transmitted
wave carries no return and the reflected wave no re-entry: the transmitted and
reflected radial functions are one-way Hankel functions. Near the illuminated
vertex these are exact Maxwell solutions whose 1/(kR) expansion must equal the
local interface recurrence, order by order. They are references for local
orders, not for a closed dielectric body.

Geometry: ``interior=False`` places the centre at (0, 0, +R); the wave in
n_incident comes from outside and meets the lower vertex (convex toward the
incident side). ``interior=True`` places the centre at (0, 0, -R); the wave in
n_incident travels inside and meets the upper vertex (a lens exit). The
cylinder axis is y. Returned fields are complex SI phasors E (V/m) and H (A/m)
for unit incident |E|, x polarized for the sphere; the cylinder uses TE
(E along y) or TM (H along y, E initially along x). Time dependence exp(-i w t).

When n_incident > n_transmitted the exact field also contains a
critical-angle wave launched where incidence is critical. It is not local to
the vertex and is not represented by the local recurrence.
"""

import numpy as np
from scipy.special import (
    h1vp,
    h2vp,
    hankel1,
    hankel2,
    jv,
    jvp,
    spherical_jn,
    spherical_yn,
)

from vecdiff_general.media.medium import Z0


def _outgoing(interior):
    # (reflected kind, transmitted kind); h1 is outgoing from the centre.
    return ("h2", "h1") if interior else ("h1", "h2")


def _cylinder_radial(kind, m, x, derivative=False):
    if kind == "j":
        return jvp(m, x) if derivative else jv(m, x)
    if kind == "h1":
        return h1vp(m, x) if derivative else hankel1(m, x)
    return h2vp(m, x) if derivative else hankel2(m, x)


def cylinder_single_interface(
    points,
    wavelength0,
    radius,
    n_incident,
    n_transmitted,
    polarization,
    *,
    interior=False,
):
    """Incident, reflected and transmitted (E, H) at points (..., 3)."""
    if polarization not in ("TE", "TM"):
        raise ValueError("polarization must be TE or TM")
    points = np.asarray(points, float)
    shape = points.shape[:-1]
    p = points.reshape(-1, 3)
    centre = -radius if interior else radius
    X, Z = p[:, 0], p[:, 2] - centre
    r, phi = np.hypot(X, Z), np.arctan2(X, Z)
    k0 = 2 * np.pi / wavelength0
    k1, k2 = n_incident * k0, n_transmitted * k0
    x1, x2 = k1 * radius, k2 * radius
    M = int(max(x1, x2) + 30 * max(x1, x2) ** (1 / 3) + 40)
    m = np.arange(-M, M + 1)
    w1, w2 = (
        (1, 1) if polarization == "TE" else (1 / n_incident**2, 1 / n_transmitted**2)
    )
    amplitude = 1.0 if polarization == "TE" else n_incident  # E_y, or Z0 H_y
    ks, kt = _outgoing(interior)
    J, dJ = jv(m, x1), jvp(m, x1)
    S, dS = _cylinder_radial(ks, m, x1), _cylinder_radial(ks, m, x1, True)
    T, dT = _cylinder_radial(kt, m, x2), _cylinder_radial(kt, m, x2, True)
    # J + b S = t T ;  w1 k1 (J' + b S') = w2 k2 t T'
    det = S * w2 * k2 * dT - w1 * k1 * dS * T
    b = (w1 * k1 * dJ * T - J * w2 * k2 * dT) / det
    t = (S * w1 * k1 * dJ - w1 * k1 * dS * J) / det
    ok = np.isfinite(b) & np.isfinite(t)
    pre = amplitude * np.exp(1j * k1 * centre) * (1j) ** m
    angular = np.exp(1j * np.outer(phi, m))

    def field(kind, k, coefficient, n):
        c = np.where(ok, pre * coefficient, 0)[None]
        with np.errstate(all="ignore"):
            f = _cylinder_radial(kind, m[None], k * r[:, None])
            df = k * _cylinder_radial(kind, m[None], k * r[:, None], True)
        f, df = np.where(ok[None], f, 0), np.where(ok[None], df, 0)
        u = (angular * c * f).sum(1)
        dr = (angular * c * df).sum(1)
        dphi = (angular * 1j * m * c * f).sum(1)
        dX = np.sin(phi) * dr + np.cos(phi) / r * dphi
        dZ = np.cos(phi) * dr - np.sin(phi) / r * dphi
        zero = np.zeros_like(u)
        if polarization == "TE":
            E = np.stack((zero, u, zero), -1)
            Hbar = np.stack((-dZ, zero, dX), -1) / (1j * k0)
        else:
            Hbar = np.stack((zero, u, zero), -1)
            E = np.stack((dZ, zero, -dX), -1) / (1j * k0 * n**2)
        return E.reshape(shape + (3,)), (Hbar / Z0).reshape(shape + (3,))

    return {
        "incident": field("j", k1, np.ones(len(m)), n_incident),
        "reflected": field(ks, k1, b, n_incident),
        "transmitted": field(kt, k2, t, n_transmitted),
    }


def _spherical_radial(kind, n, x):
    j, dj = spherical_jn(n, x), spherical_jn(n, x, derivative=True)
    if kind == "j":
        return j, dj
    y, dy = spherical_yn(n, x), spherical_yn(n, x, derivative=True)
    sign = 1 if kind == "h1" else -1
    return j + sign * 1j * y, dj + sign * 1j * dy


def _sphere_single_interface_direct(
    points, wavelength0, radius, n_incident, n_transmitted, *, interior=False
):
    """Incident, reflected and transmitted (E, H) for an x-polarized wave."""
    points = np.asarray(points, float)
    shape = points.shape[:-1]
    centre = np.array([0.0, 0.0, -radius if interior else radius])
    p = points.reshape(-1, 3) - centre
    r = np.linalg.norm(p, axis=-1)
    th = np.arccos(np.clip(p[:, 2] / r, -1, 1))
    ph = np.arctan2(p[:, 1], p[:, 0])
    k0 = 2 * np.pi / wavelength0
    k1, k2 = n_incident * k0, n_transmitted * k0
    x1, x2 = k1 * radius, k2 * radius
    nmax = int(max(x1, x2) + 8 * max(x1, x2) ** (1 / 3) + 20)
    ns = np.arange(1, nmax + 1)
    ks, kt = _outgoing(interior)

    def z_and_psi(kind, x):
        z, dz = _spherical_radial(kind, ns, x)
        return z, (z + x * dz) / x

    zj, pj = z_and_psi("j", x1)
    zs, ps = z_and_psi(ks, x1)
    zt, pt = z_and_psi(kt, x2)
    # M_o1n (TE) coefficients: incident 1; N_e1n (TM): incident -i.
    te = np.linalg.solve(
        np.array([[zs, -zt], [n_incident * ps, -n_transmitted * pt]]).transpose(
            2, 0, 1
        ),
        -np.stack([zj, n_incident * pj], -1)[..., None],
    )[..., 0]
    tm = np.linalg.solve(
        np.array([[ps, -pt], [n_incident * zs, -n_transmitted * zt]]).transpose(
            2, 0, 1
        ),
        1j * np.stack([pj, n_incident * zj], -1)[..., None],
    )[..., 0]
    En = (1j) ** ns * (2 * ns + 1) / (ns * (ns + 1))
    pre = np.exp(1j * k1 * centre[2])
    mu = np.cos(th)
    pi = np.zeros((nmax + 1,) + mu.shape)
    tau = np.zeros_like(pi)
    pi[1] = 1
    for n in range(2, nmax + 1):
        pi[n] = (2 * n - 1) / (n - 1) * mu * pi[n - 1] - n / (n - 1) * pi[n - 2]
    for n in range(1, nmax + 1):
        tau[n] = n * mu * pi[n] - (n + 1) * pi[n - 1]
    pi, tau = pi[1:], tau[1:]
    st, ct, sp, cp = np.sin(th), mu, np.sin(ph), np.cos(ph)
    rh = np.stack((st * cp, st * sp, ct), -1)
    thh = np.stack((ct * cp, ct * sp, -st), -1)
    phh = np.stack((-sp, cp, 0 * sp), -1)
    nn = (ns * (ns + 1))[:, None]

    def wave(kind, k, alpha, beta, index):
        rho = k * r
        z, dz = _spherical_radial(kind, ns[:, None], rho[None])
        psi, zr = (z + rho * dz) / rho, z / rho

        def M(odd):
            if odd:
                return (cp * pi * z)[..., None] * thh - (sp * tau * z)[..., None] * phh
            return (-sp * pi * z)[..., None] * thh - (cp * tau * z)[..., None] * phh

        def N(odd):
            f, g = (sp, cp) if odd else (cp, -sp)
            return (
                (f * nn * st * pi * zr)[..., None] * rh
                + (f * tau * psi)[..., None] * thh
                + (g * pi * psi)[..., None] * phh
            )

        c, d = (En * alpha)[:, None, None], (En * beta)[:, None, None]
        E = pre * (c * M(True) + d * N(False)).sum(0)
        Hbar = -1j * index * pre * (c * N(True) + d * M(False)).sum(0)
        return E.reshape(shape + (3,)), (Hbar / Z0).reshape(shape + (3,))

    return {
        "incident": wave("j", k1, np.ones(nmax), -1j * np.ones(nmax), n_incident),
        "reflected": wave(ks, k1, te[:, 0], tm[:, 0], n_incident),
        "transmitted": wave(kt, k2, te[:, 1], tm[:, 1], n_transmitted),
    }


def _hankel_log_start(sign, x):
    """log h_1 and the ratio h_1/h_0 for h = j + sign*i*y (sign=+1: h1, -1: h2)."""
    log_h1 = np.log(-(x + 1j * sign) / x**2) + 1j * sign * x
    return log_h1, 1 / x - 1j * sign


def _hankel_log_table(sign, x, nmax):
    """log h_n(x) and the logarithmic derivative h_n'/h_n for n = 1..nmax.

    Ratios rho_n = h_n/h_(n-1) obey rho_(n+1) = (2n+1)/x - 1/rho_n, the dominant
    (stable) direction for Hankel functions; nothing overflows at n >> x.
    """
    log_h = np.empty(nmax, complex)
    derivative = np.empty(nmax, complex)
    log_h[0], rho = _hankel_log_start(sign, x)
    derivative[0] = 1 / rho - 2 / x
    for n in range(2, nmax + 1):
        rho = (2 * n - 1) / x - 1 / rho
        log_h[n - 1] = log_h[n - 2] + np.log(rho)
        derivative[n - 1] = 1 / rho - (n + 1) / x
    return log_h, derivative


def sphere_single_interface(
    points, wavelength0, radius, n_incident, n_transmitted, *, interior=False
):
    """Incident, reflected and transmitted (E, H) for an x-polarized wave.

    Numerically stable for macroscopic spheres (size parameter 10^4 and above).
    The one-way coefficients are solved for the surface values of each partial
    wave, a_n h_n(x1) and t_n h_n(x2), which stay bounded; radial functions are
    evaluated as ratios h_n(kr)/h_n(kR) from stable logarithmic recurrences.
    The series is summed order by order, so memory grows only with the number
    of points. The incident term is the analytic plane wave it represents.
    """
    points = np.asarray(points, float)
    shape = points.shape[:-1]
    centre = np.array([0.0, 0.0, -radius if interior else radius])
    p = points.reshape(-1, 3) - centre
    r = np.linalg.norm(p, axis=-1)
    th = np.arccos(np.clip(p[:, 2] / r, -1, 1))
    ph = np.arctan2(p[:, 1], p[:, 0])
    k0 = 2 * np.pi / wavelength0
    k1, k2 = n_incident * k0, n_transmitted * k0
    x1, x2 = k1 * radius, k2 * radius
    nmax = int(max(x1, x2) + 8 * max(x1, x2) ** (1 / 3) + 20)
    ns = np.arange(1, nmax + 1)
    ks, kt = _outgoing(interior)
    sign_s, sign_t = (1 if ks == "h1" else -1), (1 if kt == "h1" else -1)

    # Surface values: incident j_n, and log-derivatives of the one-way waves.
    zj = spherical_jn(ns, x1)
    pj = (zj + x1 * spherical_jn(ns, x1, derivative=True)) / x1
    log_s, dlog_s = _hankel_log_table(sign_s, x1, nmax)
    log_t, dlog_t = _hankel_log_table(sign_t, x2, nmax)
    gs, gt = 1 / x1 + dlog_s, 1 / x2 + dlog_t  # psi/z at the surface
    ni, nt = n_incident, n_transmitted
    # Unknowns alpha = a_n h_n(x1), tau = t_n h_n(x2) (TE, then TM).
    te_alpha = (ni * pj - nt * gt * zj) / (nt * gt - ni * gs)
    te_tau = te_alpha + zj
    tm_alpha = 1j * (nt * pj - ni * gt * zj) / (nt * gs - ni * gt)
    tm_tau = (ni * tm_alpha - 1j * ni * zj) / nt
    En = (1j) ** ns * (2 * ns + 1) / (ns * (ns + 1))

    pre = np.exp(1j * k1 * centre[2])
    mu = np.cos(th)
    st, sp, cp = np.sin(th), np.sin(ph), np.cos(ph)
    rh = np.stack((st * cp, st * sp, mu), -1)
    thh = np.stack((mu * cp, mu * sp, -st), -1)
    phh = np.stack((-sp, cp, 0 * sp), -1)

    def partial_waves(sign, k, log_surface, alpha, beta, index):
        rho_arg = k * r
        E = np.zeros(r.shape + (3,), complex)
        H = np.zeros(r.shape + (3,), complex)
        log_h, ratio = _hankel_log_start(sign, rho_arg)
        pi_prev, pi_n = np.zeros_like(mu), np.ones_like(mu)
        with np.errstate(over="ignore", invalid="ignore"):
            for n in range(1, nmax + 1):
                if n > 1:
                    ratio = (2 * n - 1) / rho_arg - 1 / ratio
                    log_h = log_h + np.log(ratio)
                    pi_prev, pi_n = (
                        pi_n,
                        ((2 * n - 1) * mu * pi_n - n * pi_prev) / (n - 1),
                    )
                tau_n = n * mu * pi_n - (n + 1) * pi_prev
                z = np.exp(log_h - log_surface[n - 1])
                psi = z * (1 / rho_arg + 1 / ratio - (n + 1) / rho_arg)
                zr = z / rho_arg
                c, d = En[n - 1] * alpha[n - 1], En[n - 1] * beta[n - 1]
                m_odd = (cp * pi_n * z)[:, None] * thh - (sp * tau_n * z)[:, None] * phh
                m_even = (-sp * pi_n * z)[:, None] * thh - (cp * tau_n * z)[
                    :, None
                ] * phh
                radial = n * (n + 1) * st * pi_n * zr
                n_odd = (
                    (sp * radial)[:, None] * rh
                    + (sp * tau_n * psi)[:, None] * thh
                    + (cp * pi_n * psi)[:, None] * phh
                )
                n_even = (
                    (cp * radial)[:, None] * rh
                    + (cp * tau_n * psi)[:, None] * thh
                    - (sp * pi_n * psi)[:, None] * phh
                )
                E += c * m_odd + d * n_even
                H += c * n_odd + d * m_even
        E = pre * E
        H = -1j * index * pre * H / Z0
        return E.reshape(shape + (3,)), H.reshape(shape + (3,))

    phase = np.exp(1j * k1 * points.reshape(-1, 3)[:, 2])
    E_inc = np.zeros(r.shape + (3,), complex)
    H_inc = np.zeros(r.shape + (3,), complex)
    E_inc[:, 0] = phase
    H_inc[:, 1] = n_incident / Z0 * phase

    return {
        "incident": (E_inc.reshape(shape + (3,)), H_inc.reshape(shape + (3,))),
        "reflected": partial_waves(sign_s, k1, log_s, te_alpha, tm_alpha, ni),
        "transmitted": partial_waves(sign_t, k2, log_t, te_tau, tm_tau, nt),
    }


def _riccati(kind, orders, x):
    """z_l(x) and D_l = (x z_l)'/x for spherical j, h1 (outgoing) or h2."""
    z, dz = _spherical_radial(kind, orders, x)
    return z, dz + z / x


def sphere_debye_coefficients(size, n_sphere, n_exterior, orders):
    """Single-interface coefficients R22, T21, R11, T12 per multipole order.

    size is k0 * n_exterior * radius. Keys "TE" (M waves, E ~ z) and "TM"
    (N waves, E ~ D). Outside: incoming h2 plus R22 h1 equals T21 h2 inside.
    Inside: outgoing h1 plus R11 h2 equals T12 h1 outside. The regular interior
    field of Debye term p is T21 R11^(p-1) j and its exterior field is
    T21 R11^(p-1) T12 h1 / 2; Lorenz-Mie is their sum (Hovenac and Lock 1992).
    """
    x, mx = size, size * n_sphere / n_exterior
    ne, ni = n_exterior, n_sphere
    out = {kind: _riccati(kind, orders, x) for kind in ("h1", "h2")}
    ins = {kind: _riccati(kind, orders, mx) for kind in ("h1", "h2")}
    result = {}
    for polarization in ("TE", "TM"):
        # (E function, H function) of a radial pair (z, D).
        e, h = (0, 1) if polarization == "TE" else (1, 0)

        def solve(source, source_n, back, back_n, through, through_n):
            # source + r back = t through, for tangential E and H.
            matrix = np.array(
                [
                    [back[e], -through[e]],
                    [back_n * back[h], -through_n * through[h]],
                ]
            ).transpose(2, 0, 1)
            rhs = -np.stack([source[e], source_n * source[h]], -1)[..., None]
            r, t = np.linalg.solve(matrix, rhs)[..., 0].T
            return r, t

        R22, T21 = solve(out["h2"], ne, out["h1"], ne, ins["h2"], ni)
        R11, T12 = solve(ins["h1"], ni, ins["h2"], ni, out["h1"], ne)
        # Lorenz-Mie interior coefficient T21 / (1 - R11), solved directly:
        # between kR and n kR both factors are tiny (tunnelling multipoles).
        regular = _riccati("j", orders, x)
        _, mie = solve(regular, ne, out["h1"], ne, _riccati("j", orders, mx), ni)
        result[polarization] = dict(R22=R22, T21=T21, R11=R11, T12=T12, mie=mie)
    return result
