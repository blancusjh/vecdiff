"""Azimuthal harmonics, with explicit Fourier-series normalization."""
import numpy as np
from scipy.special import jv


def harmonics(values):
    """Last axis samples phi=2*pi*j/N; returns integer m and series coefficients."""
    n = values.shape[-1]
    return np.rint(np.fft.fftfreq(n)*n).astype(int), np.fft.fft(values, axis=-1)/n


def cylindrical_transform(radius, height, weighted_values, kr, kz, *, max_order=8, tail_tolerance=1e-10):
    """Azimuthal Fourier-Bessel transform at paired (kr,kz) coordinates.

weighted_values has shape (channels, radial_nodes, phi_nodes), includes ALL
surface quadrature weights, and uses phi=2*pi*j/N. Returns orders and their
transformed coefficients, to be multiplied by exp(i*m*phi_k) by the caller.
The discarded harmonic tail is checked, never silently accepted.
    """
    n = weighted_values.shape[-1]
    orders = np.rint(np.fft.fftfreq(n)*n).astype(int)
    cm = np.fft.fft(weighted_values, axis=-1)
    keep = abs(orders) <= max_order
    if np.linalg.norm(cm[..., ~keep]) > tail_tolerance*max(np.linalg.norm(cm), 1e-300):
        raise ValueError("azimuthal truncation is unresolved; increase max_order and phi sampling")
    out = []
    for index in np.flatnonzero(keep):
        m = int(orders[index])
        transformed = np.empty(cm.shape[:-2]+(len(kr),), complex)
        # Bound temporary memory independently of macroscopic radius and
        # spectral-table length; the quadrature and harmonic content are unchanged.
        for start in range(0, len(kr), 256):
            stop = start+256
            kernel = np.exp(-1j*np.outer(kz[start:stop], height))*jv(m, np.outer(kr[start:stop], radius))
            transformed[..., start:stop] = (-1j)**m*(cm[..., index] @ kernel.T)
        out.append(transformed)
    return orders[keep], np.asarray(out)


def cylindrical_synthesize(kr, kz, coefficients, orders, points):
    """Integrate azimuth analytically using the Jacobi–Anger identity.

Coefficients have shape (polar_nodes, orders, channels), including polar
quadrature and the full 2*pi azimuthal measure. This evaluates the SAME
propagating spectrum without discretizing its rapidly oscillating observation
phase in azimuth. Radial/polar quadrature must still converge.
    """
    p = np.asarray(points, float)
    shape = p.shape[:-1]; p = p.reshape(-1, 3)
    rho, phi = np.linalg.norm(p[:, :2], axis=-1), np.arctan2(p[:, 1], p[:, 0])
    out = np.zeros((len(p), coefficients.shape[-1]), complex)
    phase = np.exp(1j*np.outer(kz, p[:, 2]))
    for j, m in enumerate(orders):
        kernel = phase*jv(int(m), np.outer(kr, rho))*(1j**int(m))*np.exp(1j*m*phi)
        out += kernel.T @ coefficients[:, j]
    return out.reshape(shape+(coefficients.shape[-1],))
