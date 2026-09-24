"""Mathematical Fourier transforms with explicit Cartesian or polar coordinates.

transform/inverse use the continuous convention F(q)=integral f(x) exp(-iq.x) dx,
with inverse measure dq/(2*pi)^2. Lower-level fft_modes retains its separately
named Fourier-series normalization for existing modal propagation algorithms.
"""

from . import cartesian, polar
from .cartesian import fft_modes, frequency_grid, inverse_fft_modes
from .nufft import synthesize
from .polar import (
    azimuthal_coefficients,
    cylindrical_synthesize,
    hankel_transform,
    radial_weights,
)


def transform(values, *, mode, coordinates, frequencies=None, weights=None):
    """Transform numerical samples in a declared coordinate system.

    Cartesian: coordinates=(u,v), values[v,u,...], unshifted FFT output.
    Polar: coordinates=(r,phi), frequencies=(q,theta), values[r,phi,...].
    This function accepts numerical arrays, never an optical field or Grid.
    """
    if mode == "cartesian":
        if frequencies is not None or weights is not None:
            raise ValueError(
                "Cartesian FFT mode determines its frequency lattice and weights"
            )

        return cartesian.transform(values, coordinates)

    if mode == "polar":
        return polar.transform(values, coordinates, frequencies, weights=weights)

    raise ValueError("mode must be 'cartesian' or 'polar'")


def inverse(values, *, mode, coordinates, frequencies=None, weights=None):
    """Inverse transform on explicitly requested spatial coordinates.

    Polar weights, when supplied, integrate the input radial frequency dq.
    Cartesian coordinates identify the original FFT cell. The polar inverse
    is a finite-quadrature approximation to the continuous inverse integral.
    """
    if mode == "cartesian":
        if frequencies is not None or weights is not None:
            raise ValueError(
                "Cartesian FFT mode determines its frequency lattice and weights"
            )

        return cartesian.inverse(values, coordinates)

    if mode == "polar":
        return polar.inverse(values, coordinates, frequencies, weights=weights)

    raise ValueError("mode must be 'cartesian' or 'polar'")


__all__ = [
    "transform",
    "inverse",
    "fft_modes",
    "inverse_fft_modes",
    "frequency_grid",
    "hankel_transform",
    "radial_weights",
    "azimuthal_coefficients",
    "cylindrical_synthesize",
    "synthesize",
]
