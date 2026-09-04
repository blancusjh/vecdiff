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
    phase = np.exp(-1j*np.outer(kz, height))
    out = []
    for index in np.flatnonzero(keep):
        m = int(orders[index])
        kernel = phase*jv(m, np.outer(kr, radius))
        out.append((-1j)**m*(cm[..., index] @ kernel.T))
    return orders[keep], np.asarray(out)
