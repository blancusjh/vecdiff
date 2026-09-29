"""Exact sphere fields under an x-polarized, axially symmetric beam.

A beam whose plane waves carry TM amplitude A(alpha) cos(beta) and TE amplitude
-B(alpha) sin(beta) (polar angle alpha, azimuth beta) contains only the m = 1
multipoles M_o1n and N_e1n. Its field is the plane-wave series of Bohren and
Huffman with beam-shape coefficients p_n (M) and q_n (N),

    p_n = 2 <P, m_n> / (n (n+1)),    q_n = 2 <P, n_n> / (n (n+1)),

where <P, .> projects the angular spectrum P on the far-field patterns
m_n = e_theta pi_n cos(beta) - e_phi tau_n sin(beta) and
n_n = e_theta tau_n cos(beta) - e_phi pi_n sin(beta) (Funk-Hecke). A single
plane wave along z gives p_n = q_n = 1. Every Debye term then costs one sum over
n per point instead of one Mie solution per plane wave, so dense maps of the
whole plane are cheap. The beam is the continuous one (no lattice images).
"""

from __future__ import annotations

import numpy as np

from ...media.medium import Z0
from .debye import SphereDebyeSeries, _riccati, sphere_debye_coefficients


def _pi_tau(orders, cosine):
    """Angular functions pi_n, tau_n (n = 1..orders) at the given cosines."""
    pi = np.zeros((orders + 1,) + np.shape(cosine))
    pi[1] = 1
    for n in range(2, orders + 1):
        pi[n] = (2 * n - 1) / (n - 1) * cosine * pi[n - 1] - n / (n - 1) * pi[n - 2]
    n = np.arange(1, orders + 1).reshape((-1,) + (1,) * np.ndim(cosine))
    return pi[1:], n * cosine * pi[1:] - (n + 1) * pi[:-1]


class AxialBeamSphere:
    """Lorenz-Mie and its Debye terms for an axially symmetric x-polarized beam.

    The beam travels along +z; its transverse E_x at the plane z = waist_z is a
    Gaussian of 1/e field radius ``waist`` (the same beam as an FFT spectrum of
    that plane restricted to propagating waves). The sphere of ``radius`` and
    index ``n_sphere`` is centred at ``center`` on the axis, in a medium of
    index ``n_exterior``. Terms follow SphereDebyeSeries, plus "incident"
    (regular, outside) and "mie" (the whole scattered or interior field).
    """

    def __init__(
        self,
        waist,
        waist_z,
        center_z,
        radius,
        n_sphere,
        n_exterior=1.0,
        wavelength0=1.0,
        *,
        order=None,
        nodes=None,
    ):
        self.radius, self.center = float(radius), np.array([0.0, 0.0, center_z])
        self.n_exterior, self.n_sphere = float(n_exterior), float(n_sphere)
        self.wavelength0 = float(wavelength0)
        self.k0 = 2 * np.pi / self.wavelength0
        k = self.k0 * self.n_exterior
        size = k * self.radius
        inner = size * self.n_sphere / self.n_exterior
        # Enough multipoles for the sphere and for the beam itself, which has
        # beam-shape coefficients up to about k (|waist_z - center_z| + waist).
        # For a beam much wider than the region of interest, pass ``order``:
        # about k times the largest distance from the centre is enough there.
        reach = k * (abs(waist_z - center_z) + 4 * waist)
        if order is None:
            order = int(
                np.ceil(
                    max(size, inner, reach)
                    + 8 * max(size, inner, reach) ** (1 / 3)
                    + 20
                )
            )
        self.order = order
        # Scattered coefficients have decayed past this order; beyond it the
        # outgoing h1 only adds roundoff times a huge number.
        self.exterior_order = min(order, int(np.ceil(size + 4 * size ** (1 / 3) + 10)))
        self.orders = np.arange(1, order + 1)
        # The sphere couples multipoles up to about its inner size; above that
        # only the incident beam remains (no scattering, no interior field), and
        # the Hankel functions there would overflow.
        coupled = min(order, int(np.ceil(inner + 8 * inner ** (1 / 3) + 20)))
        self.coefficients = sphere_debye_coefficients(
            size, self.n_sphere, self.n_exterior, self.orders[:coupled]
        )
        for c in self.coefficients.values():
            for key, value in c.items():
                pad = 1.0 if key == "R22" else 0.0
                c[key] = np.concatenate((value, np.full(order - coupled, pad, complex)))
        # Angular spectrum of E_x = exp(-rho^2 / w^2) at z = waist_z, per unit
        # solid angle, carried to the sphere centre: E_hat(k_t) k^2 cos(alpha).
        # It is below 1e-16 of its peak beyond sin(alpha) = 2 sqrt(37) / (k w).
        top = np.arcsin(min(1.0, 2 * np.sqrt(37) / (k * waist)))
        nodes = nodes or max(400, 2 * order)
        t, w = np.polynomial.legendre.leggauss(nodes)
        alpha = top / 2 * (t + 1)
        weight = top / 2 * w * np.sin(alpha)
        c = np.cos(alpha)
        spectrum = (
            waist**2
            / (4 * np.pi)
            * np.exp(-((k * np.sin(alpha) * waist) ** 2) / 4)
            * k**2
            * np.exp(1j * k * c * (center_z - waist_z))
        )
        pi, tau = _pi_tau(order, c)
        n = self.orders[:, None]
        # TM pattern A = spectrum (E_theta = E_hat cos(beta) / cos(alpha) times
        # the cos(alpha) of the measure); TE pattern B = spectrum cos(alpha).
        scale = 2 * np.pi / (n * (n + 1))
        self.p = scale[:, 0] * ((pi + c * tau) * spectrum * weight).sum(-1)
        self.q = scale[:, 0] * ((tau + c * pi) * spectrum * weight).sum(-1)

    def _alphas(self, side, terms):
        te, tm = self.coefficients["TE"], self.coefficients["TM"]
        rows = []
        for term in terms:
            pair = []
            for c in (te, tm):
                if term == "incident":
                    value = np.ones_like(c["mie"])
                elif term == "mie" and side == "interior":
                    value = c["mie"]
                elif term == "mie":
                    value = 0.5 * (c["R22"] - 1) + 0.5 * c["T12"] * c["mie"]
                elif isinstance(term, tuple):
                    value = c["mie"] * c["R11"] ** term[1]
                    if side == "exterior":
                        value = 0.5 * value * c["T12"]
                elif side == "interior":
                    value = c["T21"] * c["R11"] ** (term - 1)
                elif term == 0:
                    value = 0.5 * (c["R22"] - 1)
                else:
                    value = 0.5 * c["T21"] * c["R11"] ** (term - 1) * c["T12"]
                pair.append(value)
            rows.append((pair[0] * self.p, -1j * pair[1] * self.q))
        return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])

    def fields(self, points, side, terms, *, chunk=4096):
        """E and H (SI) of each term at points, shape (len(terms), P, 3) each.

        side 'interior' (points inside or on the surface) takes "mie" and p >= 1; 'exterior'
        takes "incident", "mie", 0 and p >= 1. ("after", P) is every term
        beyond P in closed form, as in SphereDebyeSeries.
        """
        points = np.asarray(points, float).reshape(-1, 3) - self.center
        r = np.linalg.norm(points, axis=-1)
        if side == "interior" and np.any(r > self.radius):
            raise ValueError("interior terms need points inside the sphere")
        if (
            side == "exterior"
            and set(terms) != {"incident"}
            and np.any(r <= self.radius)
        ):
            raise ValueError("exterior terms need points outside the sphere")
        index = self.n_sphere if side == "interior" else self.n_exterior
        A, B = self._alphas(side, terms)
        E = np.zeros((len(terms), len(points), 3), complex)
        H = np.zeros_like(E)
        ell = self.orders
        factor = 1j**ell * (2 * ell + 1) / (ell * (ell + 1))
        regular = np.array([t == "incident" for t in terms])
        for start in range(0, len(points), chunk):
            p = points[start : start + chunk]
            rho = (
                self.k0
                * index
                * np.maximum(np.linalg.norm(p, axis=-1), 1e-12 * self.radius)
            )
            for kind, rows, count in (
                ("j", regular | (side == "interior"), self.order),
                ("h1", ~regular & (side == "exterior"), self.exterior_order),
            ):
                if not rows.any():
                    continue
                z, D = _riccati(kind, ell[:count, None], rho[None])
                f = factor[:count]
                e, h = SphereDebyeSeries._local(
                    self, p, z, D, A[rows, :count] * f, B[rows, :count] * f, index
                )
                E[rows, start : start + chunk] = e
                H[rows, start : start + chunk] = h
        return E, H / Z0
