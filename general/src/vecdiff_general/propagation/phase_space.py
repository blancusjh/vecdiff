"""Experimental phase-space fields: local plane-wave atoms and exact R on them.

A plane field is a sum of atoms E(x) = sum_a u_a(x - x_a) exp(i k_a . x): each
atom has a centre x_a, a carrier k_a (the local transverse wavevector) and a
smooth vector envelope u_a sampled on its own small grid. Carriers are removed,
so envelopes are sampled at their own bandwidth, not at the Nyquist rate of the
field; the Nyquist lattice of the whole plane is never formed.

R (homogeneous propagation over z) is exact per atom: with
kz(k_a + xi) = kz(k_a) + v_a . xi + delta(xi), v_a = -k_a / kz(k_a),
the constant is a phase, the linear term moves the centre by -v_a z along the
atom's ray, and delta is applied by one small FFT of the envelope. Envelope
grids must hold the atom after its own diffraction; ``grid_margin`` reserves
that room, and the propagated envelope energy near the grid edge is reported.

This module is a research prototype for docs/phase_space_operators.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import erf

from ..fourier.nufft import NUFFT


@dataclass(frozen=True, eq=False)
class AtomField:
    """Atoms on a plane (local coordinates x, y; the plane's own z = 0).

    centres (A, 2), carriers (A, 2), envelopes (A, M, M, 3) sampled at
    ``step`` with the atom centre at index M // 2. ``wavenumber`` is n k0.
    """

    centres: np.ndarray
    carriers: np.ndarray
    envelopes: np.ndarray
    step: float
    wavenumber: float

    @property
    def size(self):
        return self.envelopes.shape[1]

    @property
    def samples(self):
        return self.envelopes.shape[0] * self.size**2

    @classmethod
    def from_function(
        cls,
        field,
        gradient,
        wavenumber,
        *,
        extent,
        tile,
        step,
        size,
        sharpness=6.0,
        tolerance=1e-8,
    ):
        """Atoms of an analytic field on the square |x|, |y| <= extent.

        field(xy) -> (..., 3) complex E; gradient(xy) -> (..., 2) local
        transverse wavevector (phase gradient). Tiles of plateau ``tile`` form
        an erf partition of unity; atoms whose envelope norm is below
        tolerance x the largest are dropped.
        """
        count = int(np.ceil(2 * extent / tile)) + 1
        axis = (np.arange(count) - (count - 1) / 2) * tile
        cx, cy = np.meshgrid(axis, axis)
        centres = np.column_stack((cx.ravel(), cy.ravel()))
        carriers = gradient(centres)
        local = (np.arange(size) - size // 2) * step
        envelopes = np.empty((len(centres), size, size, 3), complex)
        LX, LY = np.meshgrid(local, local)
        for a, (centre, carrier) in enumerate(zip(centres, carriers)):
            x = centre[0] + LX
            y = centre[1] + LY
            window = _window_value(x, centre[0], tile) * _window_value(
                y, centre[1], tile
            )
            xy = np.stack((x, y), -1)
            envelopes[a] = (
                field(xy)
                * (window * np.exp(-1j * (carrier[0] * x + carrier[1] * y)))[..., None]
            )
        norms = np.linalg.norm(envelopes.reshape(len(centres), -1), axis=1)
        keep = norms > tolerance * norms.max()
        return cls(centres[keep], carriers[keep], envelopes[keep], step, wavenumber)

    def propagate(self, distance):
        """Exact homogeneous R over ``distance`` (atoms keep their carriers)."""
        k = self.wavenumber
        M, h = self.size, self.step
        xi = 2 * np.pi * np.fft.fftfreq(M, h)
        XU, XV = np.meshgrid(xi, xi)
        ku = self.carriers[:, 0][:, None, None] + XU[None]
        kv = self.carriers[:, 1][:, None, None] + XV[None]
        kz = np.sqrt((k**2 - ku**2 - kv**2).astype(complex))
        kz0 = np.sqrt(k**2 - np.sum(self.carriers**2, 1))
        velocity = -self.carriers / kz0[:, None]  # d kz / d k at the carrier
        residual = (
            kz
            - kz0[:, None, None]
            - velocity[:, 0][:, None, None] * XU
            - velocity[:, 1][:, None, None] * XV
        )
        spectrum = np.fft.fft2(
            np.fft.ifftshift(self.envelopes, axes=(1, 2)), axes=(1, 2)
        )
        spectrum *= np.exp(1j * residual * distance)[..., None]
        envelopes = np.fft.fftshift(np.fft.ifft2(spectrum, axes=(1, 2)), axes=(1, 2))
        shift = -velocity * distance
        # A shift of the centre changes the carrier phase reference: E = u(x - x_a) e^{i k.x}.
        envelopes *= np.exp(1j * kz0 * distance)[:, None, None, None]
        return AtomField(self.centres + shift, self.carriers, envelopes, self.step, k)

    def edge_fraction(self, width=2):
        """Envelope energy within ``width`` samples of the grid edge (aliasing check)."""
        e = np.sum(abs(self.envelopes) ** 2, axis=-1)
        edge = e.copy()
        edge[:, width:-width, width:-width] = 0
        return float(edge.sum() / e.sum())

    def evaluate(self, points, *, tolerance=1e-12):
        """E at transverse points (P, 2): band-limited envelope interpolation (NUFFT)."""
        points = np.asarray(points, float)
        M, h = self.size, self.step
        half = (M // 2) * h
        xi = 2 * np.pi * np.fft.fftfreq(M, h)
        XU, XV = np.meshgrid(xi, xi)
        freq = np.column_stack((XU.ravel(), XV.ravel()))
        out = np.zeros((len(points), 3), complex)
        for centre, carrier, envelope in zip(
            self.centres, self.carriers, self.envelopes
        ):
            rel = points - centre
            inside = np.all(np.abs(rel) <= half, axis=1)
            if not inside.any():
                continue
            coeff = (
                np.fft.fft2(
                    np.fft.ifftshift(envelope, axes=(0, 1)), axes=(0, 1)
                ).reshape(-1, 3)
                / M**2
            )
            values = NUFFT(
                coeff, freq, rel[inside], sign=1, backend="nufft", tolerance=tolerance
            )
            out[inside] += values * np.exp(1j * points[inside] @ carrier)[:, None]
        return out


def _window_value(x, centre, plateau, sharpness=6.0):
    sigma = plateau / sharpness
    return 0.5 * (
        erf((x - centre + plateau / 2) / sigma)
        - erf((x - centre - plateau / 2) / sigma)
    )
