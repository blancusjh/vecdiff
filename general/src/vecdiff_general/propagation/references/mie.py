"""Lorenz–Mie solution for a lossless, nonmagnetic homogeneous sphere.

SI E/H, exp(-i omega t), outgoing spherical Hankel functions of the first
kind. Coefficients follow the two independent tangential Maxwell conditions;
no local Fresnel approximation enters this reference.
"""

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from ...fields import ElectricField, ElectromagneticField
from ...fields.spectrum import PlaneWaveExpansion
from ...media.medium import Z0
from ...volumes import SphereVolume


def _radial(order, argument, outgoing=False):
    z = spherical_jn(order, argument)
    dz = spherical_jn(order, argument, derivative=True)
    if outgoing:
        z = z + 1j * spherical_yn(order, argument)
        dz = dz + 1j * spherical_yn(order, argument, derivative=True)
    return z, dz + z / argument


class MieSphere:
    """Analytic sphere reference, truncated at ``order`` multipoles.

    Illumination is one propagating plane wave with arbitrary direction and
    complex transverse polarization. ``side`` permits evaluating both traces
    at the identical boundary points. Increase order to verify convergence.
    The automatic order is a starting estimate, not an accuracy certificate.
    """

    def __init__(self, source, volume, interior, *, order=None):
        if not isinstance(volume, SphereVolume):
            raise TypeError("MieSphere requires a SphereVolume")
        if not isinstance(source, PlaneWaveExpansion) or len(source.amplitudes) != 1:
            raise TypeError("MieSphere requires one analytic plane wave")
        if np.any(source.wavevectors.imag):
            raise ValueError("Mie illumination must be propagating")
        self.source, self.volume, self.interior = source, volume, interior
        self.exterior = source.region
        self.wavelength0 = source.wavelength0
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
        self.direction = source.wavevectors[0].real / (self.k0 * ne)
        seed = np.eye(3)[np.argmin(abs(self.direction))]
        self.u = np.cross(seed, self.direction)
        self.u /= np.linalg.norm(self.u)
        self.v = np.cross(self.direction, self.u)

    def efficiencies(self):
        """Scattering/extinction cross sections divided by pi radius²."""
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

    def _polarized(self, local, inside):
        """Unit x polarized +z illumination; output Cartesian E and Z0 H."""
        radius = np.linalg.norm(local, axis=-1)
        center = radius == 0
        safe = np.where(center, 1.0, radius)
        ct = np.clip(local[:, 2] / safe, -1, 1)
        st = np.sqrt(np.maximum(0, 1 - ct**2))
        phi = np.arctan2(local[:, 1], local[:, 0])
        cp, sp = np.cos(phi), np.sin(phi)
        er = np.stack((st * cp, st * sp, ct), -1)
        et = np.stack((ct * cp, ct * sp, -st), -1)
        ep = np.stack((-sp, cp, np.zeros_like(cp)), -1)
        index = self.interior.material.n if inside else self.exterior.material.n
        rho = self.k0 * index * safe
        E, H = np.zeros_like(local, complex), np.zeros_like(local, complex)
        previous, pi = np.zeros(len(local)), np.ones(len(local))
        for ell in range(1, self.order + 1):
            tau = ell * ct * pi - (ell + 1) * previous
            z, derivative = _radial(ell, rho, outgoing=not inside)
            radial = ell * (ell + 1) * st * pi * z / rho
            mo = z[:, None] * (
                cp[:, None] * pi[:, None] * et - sp[:, None] * tau[:, None] * ep
            )
            me = z[:, None] * (
                -sp[:, None] * pi[:, None] * et - cp[:, None] * tau[:, None] * ep
            )
            no = (sp * radial)[:, None] * er + derivative[:, None] * (
                (sp * tau)[:, None] * et + (cp * pi)[:, None] * ep
            )
            ne = (cp * radial)[:, None] * er + derivative[:, None] * (
                (cp * tau)[:, None] * et - (sp * pi)[:, None] * ep
            )
            factor = 1j**ell * (2 * ell + 1) / (ell * (ell + 1))
            if inside:
                E += factor * (self.c[ell - 1] * mo - 1j * self.d[ell - 1] * ne)
                H -= index * factor * (self.d[ell - 1] * me + 1j * self.c[ell - 1] * no)
            else:
                E += factor * (-self.b[ell - 1] * mo + 1j * self.a[ell - 1] * ne)
                H += index * factor * (self.a[ell - 1] * me + 1j * self.b[ell - 1] * no)
            previous, pi = pi, ((2 * ell + 1) * ct * pi - (ell + 1) * previous) / ell
        if inside and np.any(center):
            E[center] = self.d[0] * np.array([1, 0, 0])
            H[center] = index * self.c[0] * np.array([0, 1, 0])
        return E, H

    def evaluate(self, grid, *, side=None, chunk_size=128):
        """Total fields, with optional interior/exterior boundary trace."""
        if side not in (None, "interior", "exterior"):
            raise ValueError("side must be interior, exterior or None")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
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
        phase = np.exp(
            1j * self.source.wavevectors[0] @ (self.volume.center - self.source.origin)
        )
        for inside in (False, True):
            indices = np.flatnonzero(mask == inside)
            for start in range(0, len(indices), chunk_size):
                ids = indices[start : start + chunk_size]
                for u, v in ((self.u, self.v), (self.v, -self.u)):
                    amplitude = phase * (self.source.amplitudes[0] @ u)
                    if abs(amplitude) == 0:
                        continue
                    basis = np.stack((u, v, self.direction), -1)
                    electric, magnetic = self._polarized(
                        (points[ids] - self.volume.center) @ basis, inside
                    )
                    E[ids] += amplitude * electric @ basis.T
                    H[ids] += amplitude * magnetic @ basis.T / Z0
        outside = ~mask
        if np.any(outside):
            from ...sampling import PointGrid

            incident = self.source.evaluate(PointGrid(points[outside]))
            E[outside] += incident.electric.values
            H[outside] += incident.magnetic
        if not np.isfinite(E).all() or not np.isfinite(H).all():
            raise ValueError("nonfinite Mie field; check order and observation radii")
        return ElectromagneticField(
            ElectricField(grid, self.wavelength0, E.reshape(grid.shape + (3,))),
            H.reshape(grid.shape + (3,)),
        )
