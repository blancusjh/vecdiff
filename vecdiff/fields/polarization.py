"""Polarization observables in an explicitly chosen transverse component pair."""
import numpy as np


def stokes(e1, e2):
    """S3=2 Im(conj(e1)*e2), for exp(-i omega t)."""
    cross = np.conj(e1)*e2
    return np.stack((abs(e1)**2+abs(e2)**2, abs(e1)**2-abs(e2)**2,
                     2*cross.real, 2*cross.imag), axis=-1)
