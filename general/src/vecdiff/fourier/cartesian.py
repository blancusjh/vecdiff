"""Centered Cartesian Fourier transforms of numerical arrays."""

import numpy as np

from vecdiff._backend import bd, fft2, ifft2, is_device_array


def FFT2(g: np.ndarray) -> np.ndarray:
    """Centered 2D FFT on axes (0, 1), with NumPy's default normalization.

    Computed with scipy.fft on all CPU cores, or cuFFT on the CUDA backend.

    Arrays have order (y, x, ...). Trailing components or batches are untouched.
    The spatial reference is index (ny // 2, nx // 2); zero frequency is centered.
    """
    g = bd.asarray(g)

    if g.ndim < 2 or 0 in g.shape[:2] or not bd.isfinite(g).all():
        raise ValueError("FFT input must be finite with two nonempty spatial axes")

    return _centered(g)


def IFFT2(G: np.ndarray) -> np.ndarray:
    """Centered inverse 2D FFT on axes (0, 1), including division by ny * nx.

    This is the inverse of FFT2 for both odd and even sizes. Trailing dimensions
    remain independent components or batches, never transform axes.
    """
    G = bd.asarray(G)

    if G.ndim < 2 or 0 in G.shape[:2] or not bd.isfinite(G).all():
        raise ValueError("FFT input must be finite with two nonempty spatial axes")

    return _centered(G, inverse=True)


def _alternate(source, scale, out):
    """out = scale (-1)^(i + j) source over the leading axes (i, j).

    out may be source. On the host the rows of a contiguous complex array are
    multiplied as real rows, one contiguous pass per row parity.
    """
    ny, nx = source.shape[:2]
    sign = bd.ones(nx)
    sign[1::2] = -1
    if (
        not is_device_array(source)
        and source.flags.c_contiguous
        and out.flags.c_contiguous
    ):
        real = source.real.dtype
        rows, into = source.view(real).reshape(ny, -1), out.view(real).reshape(ny, -1)
        row = bd.repeat(scale * sign, rows.shape[1] // nx)
        bd.multiply(rows[0::2], row, out=into[0::2])
        bd.multiply(rows[1::2], -row, out=into[1::2])
        return out
    column = bd.ones(ny)
    column[1::2] = -1
    pattern = (scale * column[:, None] * sign[None, :]).reshape(
        (ny, nx) + (1,) * (source.ndim - 2)
    )
    return bd.multiply(source, pattern, out=out)


def _centered(g, *, inverse=False, scale=1.0, overwrite=False):
    """scale times the centered (inverse) 2D FFT of g on axes (0, 1), unchecked.

    For even sizes the shifts are exact sign modulations: with
    S = (-1)^(i + j), fftshift(fft2(ifftshift(g))) = (-1)^((ny + nx) / 2)
    S fft2(S g), and the same for ifft2. So the transform runs in place on
    one modulated copy, and scale costs nothing; overwrite lets it consume g
    when g is complex. Odd sizes take the shifts.
    """
    transform = ifft2 if inverse else fft2
    ny, nx = g.shape[:2]

    if ny % 2 or nx % 2:
        shifted = bd.fft.ifftshift(g, axes=(0, 1))
        G = bd.fft.fftshift(
            transform(shifted, axes=(0, 1), overwrite_x=True), axes=(0, 1)
        )
        return G if scale == 1 else G * scale

    if not bd.iscomplexobj(g):
        g, overwrite = g.astype(complex), True

    work = g if overwrite else bd.empty(g.shape, g.dtype)
    work = transform(_alternate(g, 1.0, work), axes=(0, 1), overwrite_x=True)

    return _alternate(work, (-1) ** ((ny + nx) // 2) * scale, work)


def _coordinate_axes(coordinates):
    """Read uniform axes from (x, y) vectors or an xy-indexed mesh pair."""
    if coordinates is None or len(coordinates) != 2:
        raise ValueError("Cartesian coordinates must be a pair (x, y)")

    x, y = (bd.asarray(axis, dtype=float) for axis in coordinates)

    if x.ndim == 2 and y.ndim == 2:
        if x.shape != y.shape or min(x.shape) < 2:
            raise ValueError("x and y meshes must have the same shape, at least (2, 2)")

        if not bd.array_equal(
            x, bd.broadcast_to(x[0, :], x.shape)
        ) or not bd.array_equal(y, bd.broadcast_to(y[:, 0, None], y.shape)):
            raise ValueError("coordinate meshes must be rectilinear with xy indexing")

        x, y = x[0, :], y[:, 0]

    for axis in (x, y):
        if axis.ndim != 1 or len(axis) < 2 or not bd.isfinite(axis).all():
            raise ValueError("coordinate axes must be finite one-dimensional arrays")

        steps = bd.diff(axis)

        if steps[0] <= 0 or not bd.allclose(steps, steps[0], rtol=1e-10, atol=0):
            raise ValueError("Cartesian FFT coordinates must increase uniformly")

    return x, y


def frequency_grid(grid) -> tuple[np.ndarray, np.ndarray]:
    """Return centered (fx, fy) meshes in cycles per coordinate unit.

    grid is a numerical (x, y) pair: either 1D axes or 2D meshes constructed
    with indexing="xy". Both axes must be uniformly spaced and increasing.
    Use 2*pi*fx and 2*pi*fy when angular frequencies are required.
    """
    fx, fy = _frequency_axes(grid)
    fx, fy = bd.meshgrid(fx, fy, indexing="xy")

    return fx, fy


def _frequency_axes(grid):
    """The centered (fx, fy) axes of frequency_grid, one-dimensional."""
    x, y = _coordinate_axes(grid)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    nx, ny = len(x), len(y)

    fx = bd.fft.fftshift(bd.fft.fftfreq(nx, d=dx))
    fy = bd.fft.fftshift(bd.fft.fftfreq(ny, d=dy))

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

    phase = bd.exp(-2j * bd.pi * (fx * x[nx // 2] + fy * y[ny // 2]))
    phase = phase.reshape((ny, nx) + (1,) * (g.ndim - 2))

    return dx * dy, phase


def transform(g: np.ndarray, coordinates) -> np.ndarray:
    """Approximate integral g(x,y) exp(-i(qx*x + qy*y)) dx dy by centered FFT.

    coordinates is a numerical (x,y) pair. Output corresponds to angular
    frequencies (qx,qy) = 2*pi*frequency_grid(coordinates). Coordinate offsets
    enter the phase, and dx*dy supplies the continuous integration measure.
    """
    g = bd.asarray(g)
    area, phase = _measure_and_phase(g, coordinates)

    G = area * phase * FFT2(g)

    return G


def inverse(G: np.ndarray, coordinates) -> np.ndarray:
    """Recover values on the original (x,y) coordinates from centered FT data.

    Exact discrete inverse of transform on the declared periodic cell, with
    the continuous inverse convention integral G(q) exp(i*q.x) d²q / (2*pi)².
    """
    G = bd.asarray(G)
    area, phase = _measure_and_phase(G, coordinates)

    g = IFFT2(G / phase) / area

    return g
