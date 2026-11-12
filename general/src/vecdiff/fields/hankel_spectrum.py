"""A homogeneous-region field held as Hankel spectra per azimuthal order."""

import numpy as np

from vecdiff._backend import bd, to_numpy
from vecdiff._printing import number
from vecdiff.fields.electromagnetic_field import (
    ElectricField,
    ElectromagneticField,
)
from vecdiff.fields.spectrum import Spectrum
from vecdiff.fourier.polar import hankel_sum
from vecdiff.media.medium import Z0
from vecdiff.sampling import PolarGrid

# Spectral panels: 64 nodes integrate exp(i w s) on [-1, 1] to 1e-13 for w <= 85.
_SPECTRAL_NODES, _SPECTRAL_PHASE = 64, 80.0


def composite_legendre(lo, hi, panel, count):
    """Gauss-Legendre nodes and weights on [lo, hi] in panels of width ``panel``."""
    x, w = np.polynomial.legendre.leggauss(count)
    edges = np.arange(lo, hi + panel, panel)
    edges[-1] = hi
    edges = np.unique(edges)
    half = np.diff(edges)[:, None] / 2
    middle = (edges[:-1, None] + edges[1:, None]) / 2

    return (middle + half * x).ravel(), (half * w).ravel()


def check_coaxial(frame, axis, wavelength, what):
    """Raise unless the frame's normal is the axis and its origin lies on it."""
    if not bd.allclose(frame.normal, axis.normal, rtol=0, atol=1e-12):
        raise ValueError(f"{what} is not perpendicular to the incident field's axis")

    offset = frame.origin - axis.origin
    transverse = offset - (offset @ axis.normal) * axis.normal
    scale = max(wavelength, float(bd.linalg.norm(offset)))

    if float(bd.linalg.norm(transverse)) > 1e-12 * scale:
        raise ValueError(f"{what} is not centred on the incident field's axis")


def azimuths(orders):
    """Uniform azimuths that resolve E_x +- i E_y of the given azimuthal orders."""
    top = max((abs(m) + 1 for m in orders), default=1)

    return 2 * bd.pi * bd.arange(4 * (top + 1)) / (4 * (top + 1))


def hankel_options(method, tolerance, chunk_size, threads):
    """The Spectrum summation options as options of hankel_sum.

    "nufft" is the fast Hankel sum where it pays ("auto" of hankel_sum).
    """
    if method not in ("auto", "nufft", "direct"):
        raise ValueError("method must be 'auto', 'nufft' or 'direct'")
    if threads is not None and (isinstance(threads, bool) or int(threads) < 0):
        raise ValueError("threads must be None or a nonnegative integer")
    options = dict(method="direct" if method == "direct" else "auto")
    if tolerance is not None:
        options["tolerance"] = float(tolerance)
    if chunk_size is not None:
        options["chunk_size"] = chunk_size

    return options


class HankelSpectrum(Spectrum):
    """Field of a homogeneous region from transverse components on a plane, by azimuthal order.

    ``harmonics`` maps m to (E_x + i E_y, E_x - i E_y) on the radial
    quadrature (radius, weights) of the plane through ``frame``: orders
    m + 1 and m - 1 of the azimuth about the frame's normal, measured from
    its u axis. The plane-wave spectrum is formed on first evaluation for the
    depth and radius asked, and reused for any request it resolves. Its
    chart is the polar grid of that quadrature, on azimuths that resolve
    every order it holds. ``direction`` is the sign of k_z along the normal:
    -1 travels back, as a reflection. The sums are Hankel sums: ``tolerance`` and
    ``chunk_size`` are theirs (None: 1e-13 and 1024), and their transforms
    run on one thread each, whatever ``threads`` says.
    """

    def __init__(
        self, frame, region, wavelength0, radius, weights, harmonics, direction=1
    ):
        from vecdiff.media.absorption import SPECTRUM, require_transparent

        require_transparent(region, "a Hankel spectrum", SPECTRUM)
        self.direction = direction
        self.frame, self.region, self.wavelength0 = frame, region, float(wavelength0)
        self.radius, self.weights, self.carried = radius, weights, harmonics
        self.wavenumber = 2 * np.pi * region.material.n / self.wavelength0
        self.extent = float(bd.max(radius))
        self.grid = PolarGrid(radius, azimuths(harmonics), frame, weights)
        self._cache = None

    def _size(self):
        orders = sorted(self.carried)
        listed = ", ".join(str(m) for m in orders[:8]) + (
            ", ..." if len(orders) > 8 else ""
        )
        return (
            f"harmonics m = ({listed}), {len(self.grid.radius)} radial nodes "
            f"to r={number(self.extent)}"
        )

    def _details(self):
        return [
            ("orders", "m carries azimuthal orders m + 1 and m - 1 of E_x +- i E_y")
        ]

    def relabeled(self, region):
        clone = HankelSpectrum(
            self.frame,
            region,
            self.wavelength0,
            self.radius,
            self.weights,
            self.carried,
            self.direction,
        )
        clone._cache = self._cache

        return clone

    def spectrum(self, depth, reach):
        """Polar nodes theta, weights and the order m + 1, m - 1 Hankel spectra.

        The panels resolve exp(i (kappa rho' + k_z z)) J(kappa r) for
        rho' <= extent, r <= reach and |z| <= depth.
        """
        k = self.wavenumber
        bound = k * np.hypot(self.extent + reach, depth)
        panels = max(1, int(np.ceil(bound * (np.pi / 2) / (2 * _SPECTRAL_PHASE))))

        if self._cache is not None and self._cache[0] >= panels:
            return self._cache[1]

        theta, w = composite_legendre(
            0.0, np.pi / 2, np.pi / 2 / panels, _SPECTRAL_NODES
        )
        theta, w = bd.asarray(theta), bd.asarray(w)
        kappa = k * bd.sin(theta)
        measure = self.radius * self.weights
        columns, orders = [], []

        for m, (plus, minus) in self.carried.items():
            columns += [measure * plus, measure * minus]
            orders += [m + 1, m - 1]

        H = hankel_sum(bd.stack(columns, -1), self.radius, kappa, order=orders)
        result = (theta, w, kappa, H[:, 0::2], H[:, 1::2])
        self._cache = (panels, result)

        return result

    def harmonics(self, plane, radii):
        """{m: (E_rho, E_phi, E_z)} at radii on a plane across the axis, and |E|² on it.

        The plane shares the frame's axes. Its power follows from Parseval,
        2 pi sum_m integral (|E+|² / 2 + |E-|² / 2 + |E_z|²) kappa d kappa.
        """
        check_coaxial(plane, self.frame, self.wavelength0, "the next plane")
        if not bd.allclose(plane.basis, self.frame.basis, rtol=0, atol=1e-12):
            raise ValueError("the next plane must share the field's axes")
        depth = float((plane.origin - self.frame.origin) @ self.frame.normal)
        radii = bd.asarray(radii, dtype=float)
        reach = float(bd.max(radii)) if len(radii) else 0.0
        theta, w, kappa, plus, minus = self.spectrum(abs(depth), reach)
        k = self.wavenumber
        sine, cosine = bd.sin(theta)[:, None], bd.cos(theta)[:, None]
        flat, steep = k**2 * sine * cosine * w[:, None], k**2 * sine**2 * w[:, None]
        phase = bd.exp(1j * self.direction * k * cosine * depth)
        orders = list(self.carried)
        columns = [flat * phase * plus, flat * phase * minus]
        columns.append(0.5j * self.direction * steep * phase * (plus - minus))
        bessel = [m + 1 for m in orders] + [m - 1 for m in orders] + orders
        S = hankel_sum(bd.concatenate(columns, -1), kappa, radii, order=bessel)
        count = len(orders)
        result = {}
        for index, m in enumerate(orders):
            upper, lower = S[:, index], S[:, count + index]
            axial = S[:, 2 * count + index]
            result[m] = ((upper + lower) / 2, (upper - lower) / 2j, axial)
        # |E_z|² kappa d kappa = (kappa / 2 k_z)² |E+ - E-|² kappa d kappa.
        axial = steep**2 / (4 * flat) * abs(plus - minus) ** 2
        power = (
            2 * bd.pi * bd.sum(flat * (abs(plus) ** 2 + abs(minus) ** 2) / 2 + axial)
        )

        return result, float(power)

    def _sums(self, points, magnetic, options):
        """Cartesian E (and Z0 H) at points, one plane of constant depth at a time."""
        local = to_numpy((bd.asarray(points) - self.frame.origin) @ self.frame.basis)
        radius = np.hypot(local[:, 0], local[:, 1])
        azimuth = np.arctan2(local[:, 1], local[:, 0])
        depth = local[:, 2]
        theta, w, kappa, plus, minus = self.spectrum(
            float(np.abs(depth).max(initial=0.0)), float(radius.max(initial=0.0))
        )
        k, k0 = self.wavenumber, 2 * np.pi / self.wavelength0
        sine, cosine = bd.sin(theta)[:, None], bd.cos(theta)[:, None]
        # kappa d kappa = k² sin cos d theta and (kappa / k_z) kappa d kappa = k² sin² d theta.
        flat, steep = k**2 * sine * cosine * w[:, None], k**2 * sine**2 * w[:, None]
        orders = list(self.carried)
        bessel = [m + 1 for m in orders] + [m - 1 for m in orders] + orders
        fields = 2 if magnetic else 1
        values = np.zeros((fields, len(radius), 3), complex)
        planes, index = np.unique(depth, return_inverse=True)

        for p, z in enumerate(planes):
            rows = np.flatnonzero(index == p)
            kz = self.direction * k * cosine
            phase = bd.exp(1j * kz * z)
            e_plus, e_minus = flat * phase * plus, flat * phase * minus
            e_z = 0.5j * self.direction * steep * phase * (plus - minus)
            columns = [e_plus, e_minus, e_z]
            if magnetic:
                # Z0 H = k x E / k0 for each plane wave, order by order.
                along = k * sine * e_z
                columns += [
                    (1j * kz * e_plus + along) / k0,
                    (-1j * kz * e_minus + along) / k0,
                    -k * sine * (e_plus + e_minus) / (2 * k0),
                ]
            S = to_numpy(
                hankel_sum(
                    bd.concatenate(columns, axis=-1),
                    kappa,
                    bd.asarray(radius[rows]),
                    order=bessel * fields,
                    **options,
                )
            ).reshape(len(rows), fields, 3, len(orders))
            turn = np.exp(1j * np.outer(azimuth[rows], orders))[:, None]
            spin = np.exp(1j * azimuth[rows])[:, None, None]
            upper = np.sum(S[:, :, 0] * turn * spin, -1)
            lower = np.sum(S[:, :, 1] * turn / spin, -1)
            axial = np.sum(S[:, :, 2] * turn, -1)
            values[:, rows] = np.stack(
                ((upper + lower) / 2, (upper - lower) / 2j, axial), -1
            ).transpose(1, 0, 2)

        return bd.asarray(values @ to_numpy(self.frame.basis).T)

    def electric(
        self, grid, *, method="auto", tolerance=None, chunk_size=None, threads=None
    ):
        """E on any grid."""
        options = hankel_options(method, tolerance, chunk_size, threads)
        values = self._sums(bd.asarray(grid.points).reshape(-1, 3), False, options)[0]

        return ElectricField(grid, self.wavelength0, values.reshape(grid.shape + (3,)))

    def evaluate(
        self, grid, *, method="auto", tolerance=None, chunk_size=None, threads=None
    ):
        """SI E and H on any grid; method "direct" sums the Hankel integrals densely."""
        options = hankel_options(method, tolerance, chunk_size, threads)
        E, H = self._sums(bd.asarray(grid.points).reshape(-1, 3), True, options)

        return ElectromagneticField(
            ElectricField(grid, self.wavelength0, E.reshape(grid.shape + (3,))),
            H.reshape(grid.shape + (3,)) / Z0,
        )
