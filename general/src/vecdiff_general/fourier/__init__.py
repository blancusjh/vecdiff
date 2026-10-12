"""Mathematical Fourier transforms with explicit Cartesian, polar or nonuniform coordinates.

transform/inverse use the continuous convention F(q)=integral f(x) exp(-iq.x) dx,
with inverse measure d^d q/(2*pi)^d. FFT2/IFFT2 provide the centered discrete
transform pair with NumPy normalization, without coordinate measures.
"""

from vecdiff_general.fourier import cartesian, nufft, polar
from vecdiff_general.fourier.cartesian import FFT2, IFFT2, frequency_grid
from vecdiff_general.fourier.nufft import NUFFT
from vecdiff_general.fourier.polar import (
    FT2,
    HT,
    IFT2,
    IHT,
    azimuthal_coefficients,
    cylindrical_synthesize,
    radial_weights,
)


def transform(g, *, mode, coordinates, frequencies=None, weights=None, **options):
    """Transform numerical samples in a declared coordinate system.

    Cartesian: coordinates=(u,v), values[v,u,...], centered FFT output.
    Polar: coordinates=(r,phi), frequencies=(q,theta), values[r,phi,...].
    Nonuniform: coordinates=(nodes,dimension), frequencies=(...,dimension),
    values[nodes,...]; explicit quadrature weights are required. Backend options
    are passed to NUFFT only in this mode. No optical field or Grid is accepted.
    """
    if options and mode != "nonuniform":
        raise TypeError("backend options apply only to nonuniform transforms")

    if mode == "cartesian":
        if frequencies is not None or weights is not None:
            raise ValueError(
                "Cartesian FFT mode determines its frequency lattice and weights"
            )

        return cartesian.transform(g, coordinates)

    if mode == "polar":
        return polar.FT2(g, coordinates, frequencies, weights=weights)

    if mode == "nonuniform":
        if frequencies is None or weights is None:
            raise ValueError(
                "nonuniform transforms require frequencies and quadrature weights"
            )

        return nufft.transform(g, coordinates, frequencies, weights=weights, **options)

    raise ValueError("mode must be 'cartesian', 'polar' or 'nonuniform'")


def inverse(G, *, mode, coordinates, frequencies=None, weights=None, **options):
    """Inverse transform on explicitly requested spatial coordinates.

    Polar weights, when supplied, integrate the input radial frequency dq.
    Cartesian coordinates identify the original FFT cell. The polar inverse
    is a finite-quadrature approximation to the continuous inverse integral.
    Nonuniform weights integrate frequency-space measure d^d q; that inverse
    likewise approximates the integral, not an arbitrary matrix inversion.
    """
    if options and mode != "nonuniform":
        raise TypeError("backend options apply only to nonuniform transforms")

    if mode == "cartesian":
        if frequencies is not None or weights is not None:
            raise ValueError(
                "Cartesian FFT mode determines its frequency lattice and weights"
            )

        return cartesian.inverse(G, coordinates)

    if mode == "polar":
        return polar.IFT2(G, coordinates, frequencies, weights=weights)

    if mode == "nonuniform":
        if frequencies is None or weights is None:
            raise ValueError(
                "nonuniform transforms require frequencies and quadrature weights"
            )

        return nufft.inverse(G, coordinates, frequencies, weights=weights, **options)

    raise ValueError("mode must be 'cartesian', 'polar' or 'nonuniform'")


__all__ = [
    "transform",
    "inverse",
    "FFT2",
    "IFFT2",
    "frequency_grid",
    "FT2",
    "IFT2",
    "HT",
    "IHT",
    "radial_weights",
    "azimuthal_coefficients",
    "cylindrical_synthesize",
    "NUFFT",
]
