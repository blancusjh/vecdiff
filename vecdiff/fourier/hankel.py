"""Radial Fourier-Bessel quadrature; no optical apodization is implicit."""
import numpy as np
from scipy.special import jv
from scipy.integrate import trapezoid


def transform(values, radius, radial_frequency, order=0):
    r = np.asarray(radius)
    return trapezoid(np.asarray(values)[..., None, :]*jv(order, np.outer(radial_frequency, r))*r,
                     x=r, axis=-1)
