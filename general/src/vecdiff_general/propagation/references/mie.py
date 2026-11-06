"""Lorenz–Mie solution for a lossless, nonmagnetic homogeneous sphere.

SI E/H, exp(-i omega t), outgoing spherical Hankel functions of the first
kind. Coefficients follow the two independent tangential Maxwell conditions;
no local Fresnel approximation enters this reference.

Any propagating incident field E(r) = integral F(s) exp(i k s.(r - c)) dOmega
about the centre c expands in vector spherical waves,

    E = sum_nm A_nm z_n X_nm + B_nm W_nm,   W_nm = curl(z_n X_nm) / (i k),
    Z0 H = n sum_nm A_nm W_nm - B_nm z_n X_nm,
    A_nm = 4 pi i^n integral X_nm* . F dOmega,
    B_nm = 4 pi i^n integral (s x X_nm)* . F dOmega,

with X_nm = L Y_nm / sqrt(n(n+1)) and Y_nm = Pbar_n^|m|(cos theta) e^{i m phi}
(normalized, no Condon–Shortley phase: any orthonormal choice cancels between
projection and synthesis). The sphere multiplies A_nm (TE) and B_nm (TM) by
m-independent factors per n: Lorenz–Mie, or one Debye term.
"""

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from vecdiff_general.fields import ElectricField, ElectromagneticField
from vecdiff_general.fields.spectrum import PlaneWaveExpansion
from vecdiff_general.media.medium import Z0
from vecdiff_general.propagation.references.debye import (
    _riccati,
    sphere_debye_coefficients,
)
from vecdiff_general.sampling import PointGrid
from vecdiff_general.volumes import SphereVolume

# Relative size below which an azimuthal order's coefficients count as absent.
_ROUNDOFF = 64 * np.finfo(float).eps


def _radial(order, argument, outgoing=False):
    z = spherical_jn(order, argument)
    dz = spherical_jn(order, argument, derivative=True)
    if outgoing:
        z = z + 1j * spherical_yn(order, argument)
        dz = dz + 1j * spherical_yn(order, argument, derivative=True)
    return z, dz + z / argument


def _harmonics(order, m, cosine, sine):
    """Pbar, pibar = Pbar / sin and taubar = dPbar/dtheta for n = max(1, m)..order.

    Shape (orders, points) each. pibar follows the Pbar recursion from
    Pbar_m^m / sin, so no division by sin(theta) occurs at the poles; for
    m = 0, taubar = -sqrt(n(n+1)) Pbar_n^1.
    """
    mu = abs(m)
    first = max(mu, 1)
    count = order - first + 1
    P, pi, tau = (np.zeros((count,) + cosine.shape) for _ in range(3))

    def sequence(mu):
        # pibar_n^mu for n = mu..order (mu >= 1), or Pbar_n^0 for n = 0..order.
        start = np.full(cosine.shape, np.sqrt(1 / (4 * np.pi)))
        for k in range(1, mu + 1):
            start = start * np.sqrt((2 * k + 1) / (2 * k))
        if mu > 1:
            start = start * sine ** (mu - 1)
        previous, current = np.zeros(cosine.shape), start
        for n in range(mu, order + 1):
            yield n, current, previous
            nn = n + 1
            a = np.sqrt((4 * nn**2 - 1) / (nn**2 - mu**2))
            b = np.sqrt((n**2 - mu**2) / (4 * n**2 - 1)) if n > mu else 0.0
            previous, current = current, a * (cosine * current - b * previous)

    if mu == 0:
        for (n, p0, _), (_, p1, _) in zip(
            (item for item in sequence(0) if item[0] >= 1), sequence(1)
        ):
            P[n - 1] = p0
            tau[n - 1] = -np.sqrt(n * (n + 1)) * sine * p1
        return P, pi, tau

    for n, current, previous in sequence(mu):
        c = np.sqrt((2 * n + 1) * (n**2 - mu**2) / (2 * n - 1)) if n > mu else 0.0
        P[n - first] = sine * current
        pi[n - first] = current
        tau[n - first] = n * cosine * current - c * previous
    return P, pi, tau


def _spherical(local):
    radius = np.linalg.norm(local, axis=-1)
    safe = np.where(radius == 0, 1.0, radius)
    cosine = np.clip(np.where(radius == 0, 1.0, local[..., 2] / safe), -1, 1)
    sine = np.hypot(local[..., 0], local[..., 1]) / safe
    return radius, cosine, sine, np.arctan2(local[..., 1], local[..., 0])


class MieSphere:
    """Analytic sphere reference, truncated at ``order`` multipoles.

    ``incident`` is any propagating field of the exterior region: a
    PlaneWaveExpansion (one plane wave, or a sum such as an FFT spectrum) or a
    beam from fields (PlaneBeam, FocusedBeam), whose quadrature is chosen for
    the sphere. ``side`` permits evaluating both traces at the identical
    boundary points; ``term`` selects one Debye term. Increase order to verify
    convergence. The automatic order is a starting estimate, not an accuracy
    certificate.
    """

    def __init__(self, incident, volume, interior, *, order=None):
        if not isinstance(volume, SphereVolume):
            raise TypeError("MieSphere requires a SphereVolume")
        if isinstance(incident, PlaneWaveExpansion):
            if np.any(incident.wavevectors[np.any(incident.amplitudes != 0, -1)].imag):
                raise ValueError("Mie illumination must be propagating")
        elif not hasattr(incident, "quadrature"):
            raise TypeError(
                "MieSphere requires a PlaneWaveExpansion or a beam from fields"
            )
        self.incident, self.volume, self.interior = incident, volume, interior
        self.exterior = incident.region
        self.wavelength0 = incident.wavelength0
        self.k0 = 2 * np.pi / self.wavelength0
        ni, ne = interior.material.n, self.exterior.material.n
        x = self.k0 * ne * volume.radius
        if order is None:
            # Retain more of the incident-wave tail than the usual efficiency
            # truncation: near-boundary field continuity is a stricter test.
            order = int(np.ceil(x + 6 * x ** (1 / 3) + 12))
        if not isinstance(order, (int, np.integer)) or order < 1:
            raise ValueError("multipole order must be a positive integer")
        self.order = order
        ell = np.arange(1, order + 1)
        j, dj = _radial(ell, x)
        h, dh = _radial(ell, x, True)
        v, dv = _radial(ell, x * ni / ne)
        self.a = (ni * dj * v - ne * j * dv) / (ni * dh * v - ne * h * dv)
        self.b = (ni * j * dv - ne * dj * v) / (ni * h * dv - ne * dh * v)
        # Use both boundary equations to avoid division at Bessel zeros.
        self.c = (v * (j - self.b * h) + ni * dv * ne * (dj - self.b * dh)) / (
            v**2 + (ni * dv) ** 2
        )
        self.d = (dv * (dj - self.a * dh) + ni * v * ne * (j - self.a * h)) / (
            dv**2 + (ni * v) ** 2
        )
        if not all(np.isfinite(c).all() for c in (self.a, self.b, self.c, self.d)):
            raise ValueError("nonfinite Mie coefficients; reduce excessive order")
        self._debye = None
        self.basis, self.coefficients = self._project()

    def _project(self):
        """Local basis (columns u, v, axis) and {m: (A_n, B_n)} about the centre."""
        k = self.k0 * self.exterior.material.n
        n = np.arange(1, self.order + 1)
        scale = 4 * np.pi * 1j ** (n % 4) / np.sqrt(n * (n + 1))
        if isinstance(self.incident, PlaneWaveExpansion):
            basis, cosine, sine, harmonic, orders = self._modes(k, self.incident)
        elif getattr(self.incident, "is_plane_wave", False):
            waves = self.incident.plane_waves(0.0)
            basis, cosine, sine, harmonic, orders = self._modes(k, waves)
        else:
            basis, cosine, sine, harmonic, orders = self._beam_modes(k)
        coefficients = {}
        # Near the axis Pbar_n^|m| carries sin^|m|: skip orders that cannot reach roundoff.
        reach = np.max(sine, initial=0.0)
        for m in orders:
            if abs(m) > 1 and reach ** (abs(m) - 1) < _ROUNDOFF**2:
                continue
            g_theta, g_phi = harmonic(m)
            if not (np.any(g_theta) or np.any(g_phi)):
                continue
            P, pi, tau = _harmonics(self.order, m, cosine, sine)
            first = max(abs(m), 1)
            A = np.zeros(self.order, complex)
            B = np.zeros(self.order, complex)
            A[first - 1 :] = (-m * pi * g_theta + 1j * tau * g_phi).sum(-1)
            B[first - 1 :] = (-m * pi * g_phi - 1j * tau * g_theta).sum(-1)
            coefficients[m] = (scale * A, scale * B)
        peak = max(
            (np.max(np.abs(np.concatenate(c))) for c in coefficients.values()),
            default=0.0,
        )
        coefficients = {
            m: c
            for m, c in coefficients.items()
            if np.max(np.abs(np.concatenate(c))) > _ROUNDOFF * peak
        }
        return basis, coefficients

    def _modes(self, k, source):
        """Direct projection of discrete plane waves, axis along their mean direction."""
        keep = np.any(source.amplitudes != 0, axis=-1)
        wavevectors, amplitudes = source.wavevectors[keep].real, source.amplitudes[keep]
        amplitudes = (
            amplitudes
            * np.exp(1j * wavevectors @ (self.volume.center - source.origin))[:, None]
        )
        weight = np.linalg.norm(amplitudes, axis=-1) ** 2
        axis = weight @ wavevectors
        axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) else np.eye(3)[2]
        basis = _basis(axis)
        local = wavevectors @ basis / k
        _, cosine, sine, phi = _spherical(local)
        cp, sp = np.cos(phi), np.sin(phi)
        theta_hat = np.stack((cosine * cp, cosine * sp, -sine), -1) @ basis.T
        phi_hat = np.stack((-sp, cp, np.zeros_like(cp)), -1) @ basis.T
        f_theta = np.sum(amplitudes * theta_hat, -1)
        f_phi = np.sum(amplitudes * phi_hat, -1)

        def harmonic(m):
            # The azimuthal sum weights each wave by e^{-i m phi}.
            e = np.exp(-1j * m * phi)
            return e * f_theta, e * f_phi

        return basis, cosine, sine, harmonic, range(-self.order, self.order + 1)

    def _beam_modes(self, k):
        """Product quadrature of a beam: the azimuthal sums are one FFT per polar node."""
        beam = self.incident
        offset = self.volume.center - beam.frame.origin
        radius = np.linalg.norm(offset) + self.order / k
        theta, phi, weight, density = beam.quadrature(radius)
        s, theta_hat, phi_hat = beam._directions(theta, phi)
        density = (weight * np.exp(1j * k * s @ offset))[..., None] * density
        count = len(phi)
        f_theta = np.fft.fft(np.sum(density * theta_hat, -1), axis=1)
        f_phi = np.fft.fft(np.sum(density * phi_hat, -1), axis=1)
        m = np.fft.fftfreq(count, 1 / count).astype(int)
        present = np.max(np.abs(f_theta) + np.abs(f_phi), axis=0)
        orders = m[(present > _ROUNDOFF * present.max()) & (np.abs(m) <= self.order)]
        return (
            np.asarray(beam.frame.basis),
            np.cos(theta),
            np.sin(theta),
            lambda m: (f_theta[:, m], f_phi[:, m]),
            sorted(orders, key=abs),
        )

    def _factors(self, side, term):
        """Per-order multipliers of (A, B) and the radial function kind."""
        if term is None:
            if side == "interior":
                return self.c, self.d, "j"
            return -self.b, -self.a, "h1"
        if self._debye is None:
            x = self.k0 * self.exterior.material.n * self.volume.radius
            self._debye = sphere_debye_coefficients(
                x,
                self.interior.material.n,
                self.exterior.material.n,
                np.arange(1, self.order + 1),
            )
        pair = []
        for c in (self._debye["TE"], self._debye["TM"]):
            if isinstance(term, tuple):
                # ("after", P): every term beyond P, summed in closed form.
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
        return pair[0], pair[1], "j" if side == "interior" else "h1"

    def efficiencies(self):
        """Scattering/extinction cross sections divided by pi radius² (one plane wave)."""
        if not (
            isinstance(self.incident, PlaneWaveExpansion)
            and len(self.incident.amplitudes) == 1
        ):
            raise TypeError("efficiencies are defined for one incident plane wave")
        ell = np.arange(1, self.order + 1)
        x = self.k0 * self.exterior.material.n * self.volume.radius
        return {
            "scattering": float(
                2 / x**2 * np.sum((2 * ell + 1) * (abs(self.a) ** 2 + abs(self.b) ** 2))
            ),
            "extinction": float(
                2 / x**2 * np.sum((2 * ell + 1) * (self.a + self.b).real)
            ),
        }

    def _series(self, local, side, term):
        """Sphere-generated E and Z0 H at local points, global Cartesian components."""
        tA, tB, kind = self._factors(side, term)
        index = (
            self.interior.material.n if side == "interior" else self.exterior.material.n
        )
        radius, cosine, sine, phi = _spherical(local)
        # At the centre only n = 1 survives and its angular limit is regular.
        x = self.k0 * index * np.maximum(radius, 1e-12 * self.volume.radius)
        count = self.order
        if term is not None and side == "exterior":
            # Debye parts outside grow like h1 at high order and only cancel on
            # paper; the scattered coefficients have decayed past this order.
            size = self.k0 * self.exterior.material.n * self.volume.radius
            count = min(count, int(np.ceil(size + 4 * size ** (1 / 3) + 10)))
        n = np.arange(1, count + 1)
        z, D = _riccati(kind, n[:, None], x[None])
        root = np.sqrt(n * (n + 1))[:, None]
        Er, Et, Ep, Hr, Ht, Hp = (np.zeros(len(local), complex) for _ in range(6))
        for m, (A, B) in self.coefficients.items():
            first = max(abs(m), 1) - 1
            if first >= count:
                continue
            P, pi, tau = _harmonics(count, m, cosine, sine)
            a = (tA * A)[first:count, None]
            b = (tB * B)[first:count, None]
            zs, Ds, s = z[first:], D[first:], root[first:]
            e = np.exp(1j * m * phi)
            Er += e * np.sum(b * s * zs / x * P, 0)
            Et += e * np.sum((-m * a * zs * pi + b * Ds * tau) / s, 0)
            Ep += e * np.sum((-1j * a * zs * tau + 1j * m * b * Ds * pi) / s, 0)
            Hr += e * np.sum(a * s * zs / x * P, 0)
            Ht += e * np.sum((a * Ds * tau + m * b * zs * pi) / s, 0)
            Hp += e * np.sum((1j * m * a * Ds * pi + 1j * b * zs * tau) / s, 0)
        cp, sp = np.cos(phi), np.sin(phi)

        def cartesian(r, t, p):
            vector = np.stack(
                (
                    r * sine * cp + t * cosine * cp - p * sp,
                    r * sine * sp + t * cosine * sp + p * cp,
                    r * cosine - t * sine,
                ),
                -1,
            )
            return vector @ self.basis.T

        return cartesian(Er, Et, Ep), index * cartesian(Hr, Ht, Hp)

    def evaluate(self, grid, *, side=None, term=None, chunk_size=128):
        """Total fields, or one Debye term, with optional interior/exterior trace.

        term None is the total field: incident plus scattered outside, the
        Lorenz–Mie field inside. Debye term p >= 1 is the wave that has entered
        once and met the surface p - 1 more times from inside: inside it is the
        regular solution, outside what leaves on the p-th inside encounter.
        Exterior term 0 is (R22 - 1) / 2: the external reflection together with
        the diffraction that forms the shadow; it has no interior part.
        ("after", P) is every term beyond P in closed form, so terms 0..P plus
        it recover Lorenz–Mie minus the incident field outside.
        """
        if side not in (None, "interior", "exterior"):
            raise ValueError("side must be interior, exterior or None")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if term is not None and not (
            (isinstance(term, (int, np.integer)) and term >= 0)
            or (
                isinstance(term, tuple)
                and len(term) == 2
                and term[0] == "after"
                and term[1] >= 0
            )
        ):
            raise ValueError("term must be None, an integer p >= 0 or ('after', P)")
        points = grid.points.reshape(-1, 3)
        distance = np.linalg.norm(points - self.volume.center, axis=-1)
        eps = 1e-10 * self.volume.radius
        if side == "interior" and np.any(distance > self.volume.radius + eps):
            raise ValueError("interior evaluation outside the sphere")
        if side == "exterior" and np.any(distance < self.volume.radius - eps):
            raise ValueError("exterior evaluation inside the sphere")
        mask = (
            distance < self.volume.radius
            if side is None
            else np.full(len(points), side == "interior")
        )
        E, H = np.zeros_like(points, complex), np.zeros_like(points, complex)
        local = (points - self.volume.center) @ self.basis
        for inside in (False, True):
            if inside and term == 0:
                continue
            indices = np.flatnonzero(mask == inside)
            for start in range(0, len(indices), chunk_size):
                ids = indices[start : start + chunk_size]
                electric, magnetic = self._series(
                    local[ids], "interior" if inside else "exterior", term
                )
                E[ids] += electric
                H[ids] += magnetic / Z0
        outside = ~mask
        if term is None and np.any(outside):
            incident = self.incident.evaluate(PointGrid(points[outside]))
            E[outside] += incident.electric.values
            H[outside] += incident.magnetic
        if not np.isfinite(E).all() or not np.isfinite(H).all():
            raise ValueError("nonfinite Mie field; check order and observation radii")
        return ElectromagneticField(
            ElectricField(grid, self.wavelength0, E.reshape(grid.shape + (3,))),
            H.reshape(grid.shape + (3,)),
        )


def _basis(axis):
    """Right-handed orthonormal columns (u, v, axis)."""
    seed = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(seed, axis)
    u /= np.linalg.norm(u)
    return np.stack((u, np.cross(axis, u), axis), -1)
