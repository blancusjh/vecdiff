"""Centered Cartesian Fourier transforms of numerical arrays."""

import numpy as np
import scipy.fft


def FFT2(g: np.ndarray) -> np.ndarray:
    """Centered 2D FFT on axes (0, 1), with NumPy's default normalization.

    Computed with scipy.fft on all available cores.

    Arrays have order (y, x, ...). Trailing components or batches are untouched.
    The spatial reference is index (ny // 2, nx // 2); zero frequency is centered.
    """
    g = np.asarray(g)

    if g.ndim < 2 or 0 in g.shape[:2] or not np.isfinite(g).all():
        raise ValueError("FFT input must be finite with two nonempty spatial axes")

    g_ = np.fft.ifftshift(g, axes=(0, 1))
    G_ = scipy.fft.fft2(g_, axes=(0, 1), workers=-1)
    G = np.fft.fftshift(G_, axes=(0, 1))

    return G


def IFFT2(G: np.ndarray) -> np.ndarray:
    """Centered inverse 2D FFT on axes (0, 1), including division by ny * nx.

    This is the inverse of FFT2 for both odd and even sizes. Trailing dimensions
    remain independent components or batches, never transform axes.
    """
    G = np.asarray(G)

    if G.ndim < 2 or 0 in G.shape[:2] or not np.isfinite(G).all():
        raise ValueError("FFT input must be finite with two nonempty spatial axes")

    G_ = np.fft.ifftshift(G, axes=(0, 1))
    g_ = scipy.fft.ifft2(G_, axes=(0, 1), workers=-1)
    g = np.fft.fftshift(g_, axes=(0, 1))

    return g


def _coordinate_axes(coordinates):
    """Read uniform axes from (x, y) vectors or an xy-indexed mesh pair."""
    if coordinates is None or len(coordinates) != 2:
        raise ValueError("Cartesian coordinates must be a pair (x, y)")

    x, y = (np.asarray(axis, dtype=float) for axis in coordinates)

    if x.ndim == 2 and y.ndim == 2:
        if x.shape != y.shape or min(x.shape) < 2:
            raise ValueError("x and y meshes must have the same shape, at least (2, 2)")

        if not np.array_equal(
            x, np.broadcast_to(x[0, :], x.shape)
        ) or not np.array_equal(y, np.broadcast_to(y[:, 0, None], y.shape)):
            raise ValueError("coordinate meshes must be rectilinear with xy indexing")

        x, y = x[0, :], y[:, 0]

    for axis in (x, y):
        if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
            raise ValueError("coordinate axes must be finite one-dimensional arrays")

        steps = np.diff(axis)

        if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-10, atol=0):
            raise ValueError("Cartesian FFT coordinates must increase uniformly")

    return x, y


def frequency_grid(grid) -> tuple[np.ndarray, np.ndarray]:
    """Return centered (fx, fy) meshes in cycles per coordinate unit.

    grid is a numerical (x, y) pair: either 1D axes or 2D meshes constructed
    with indexing="xy". Both axes must be uniformly spaced and increasing.
    Use 2*pi*fx and 2*pi*fy when angular frequencies are required.
    """
    x, y = _coordinate_axes(grid)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    nx, ny = len(x), len(y)

    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    fx, fy = np.meshgrid(fx, fy, indexing="xy")

    return fx, fy


def _measure_and_phase(g, coordinates):
    """Integration weight and translation from the centered sample to zero."""
    x, y = _coordinate_axes(coordinates)
    nx, ny = len(x), len(y)

    if g.shape[:2] != (ny, nx):
        raise ValueError("Cartesian values must have leading shape (len(y), len(x))")

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    fx, fy = frequency_grid((x, y))

    phase = np.exp(-2j * np.pi * (fx * x[nx // 2] + fy * y[ny // 2]))
    phase = phase.reshape((ny, nx) + (1,) * (g.ndim - 2))

    return dx * dy, phase


def transform(g: np.ndarray, coordinates) -> np.ndarray:
    """Approximate integral g(x,y) exp(-i(qx*x + qy*y)) dx dy by centered FFT.

    coordinates is a numerical (x,y) pair. Output corresponds to angular
    frequencies (qx,qy) = 2*pi*frequency_grid(coordinates). Coordinate offsets
    enter the phase, and dx*dy supplies the continuous integration measure.
    """
    g = np.asarray(g)
    area, phase = _measure_and_phase(g, coordinates)

    G = area * phase * FFT2(g)

    return G


def inverse(G: np.ndarray, coordinates) -> np.ndarray:
    """Recover values on the original (x,y) coordinates from centered FT data.

    Exact discrete inverse of transform on the declared periodic cell, with
    the continuous inverse convention integral G(q) exp(i*q.x) d²q / (2*pi)².
    """
    G = np.asarray(G)
    area, phase = _measure_and_phase(G, coordinates)

    g = IFFT2(G / phase) / area

    return g
