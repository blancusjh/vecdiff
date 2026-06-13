"""Fourier transform utilities."""

import numpy as np
from scipy.signal import zoom_fft

from .grid import Grid

π = np.pi


# ================================================================ #
#  1D Fourier Transform                                             #
# ================================================================ #

def FT(g, dx=1.0, physical=True, shift=True):
    """Return the one-dimensional Fourier transform of ``g``.

    When ``physical`` is true, the result is scaled by ``dx`` to approximate
    ``integral g(x) exp(-i k x) dx`` at the sampled spectral points.
    """
    g = np.asarray(g, dtype=complex)
    if shift:
        g = np.fft.ifftshift(g)
    G = np.fft.fft(g)
    if physical:
        G = dx * G
    if shift:
        G = np.fft.fftshift(G)
    return G


def IFT(G, dx=1.0, physical=True, shift=True):
    """Return the one-dimensional inverse Fourier transform of ``G``."""
    G = np.asarray(G, dtype=complex)
    if shift:
        G = np.fft.ifftshift(G)
    if physical:
        G = G / dx
    g = np.fft.ifft(G)
    if shift:
        g = np.fft.fftshift(g)
    return g


def KAXIS(N, dx=1.0, shift=True):
    """Return the one-dimensional angular spectral axis ``k = 2*pi*f``."""
    k = 2.0 * π * np.fft.fftfreq(N, d=dx)
    if shift:
        k = np.fft.fftshift(k)
    return k


# ================================================================ #
#  2D Fourier Transform                                             #
# ================================================================ #

def _ft2_array(g, dx=1.0, dy=1.0, physical=True, shift=True):
    """Return the two-dimensional Fourier transform of ``g[y, x]``."""
    g = np.asarray(g, dtype=complex)
    if shift:
        g = np.fft.ifftshift(g)
    G = np.fft.fft2(g)
    if physical:
        G = dx * dy * G
    if shift:
        G = np.fft.fftshift(G)
    return G


def _ift2_array(G, dx=1.0, dy=1.0, physical=True, shift=True):
    """Return the two-dimensional inverse Fourier transform of ``G``."""
    G = np.asarray(G, dtype=complex)
    if shift:
        G = np.fft.ifftshift(G)
    if physical:
        G = G / (dx * dy)
    g = np.fft.ifft2(G)
    if shift:
        g = np.fft.fftshift(g)
    return g


def _grid_from_shape_or_spacing(shape, grid=None, dx=1.0, dy=1.0) -> Grid:
    if grid is None:
        return Grid.from_spacing(shape, dx=dx, dy=dy)
    if not isinstance(grid, Grid):
        raise TypeError("grid must be a vecdiff.grid.Grid instance.")
    return grid


def _separable_axis_from_grid_values(values: np.ndarray, axis: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} grid must be two-dimensional.")
    if axis == 1:
        axis_values = values[0, :]
        if not np.allclose(values, axis_values[None, :]):
            raise ValueError(f"{name} grid must be separable with rows constant along y.")
    elif axis == 0:
        axis_values = values[:, 0]
        if not np.allclose(values, axis_values[:, None]):
            raise ValueError(f"{name} grid must be separable with columns constant along x.")
    else:
        raise ValueError("axis must be 0 or 1.")
    return axis_values


def _uniform_axis(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} axis must be one-dimensional.")
    if values.size < 2:
        return values
    diffs = np.diff(values)
    spacing = float(np.mean(diffs))
    if not np.allclose(diffs, spacing):
        raise ValueError(f"{name} axis must be uniformly sampled.")
    return values


def _custom_k_axes(kgrid: Grid) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(kgrid, Grid):
        raise TypeError("kgrid must be a vecdiff.grid.Grid instance.")
    if kgrid.type != "cartesian":
        raise ValueError("kgrid must be Cartesian.")
    kx = _separable_axis_from_grid_values(kgrid.X, axis=1, name="kx")
    ky = _separable_axis_from_grid_values(kgrid.Y, axis=0, name="ky")
    return _uniform_axis(kx, "kx"), _uniform_axis(ky, "ky")


def _direct_ft2_on_kgrid(
    g: np.ndarray,
    grid: Grid,
    kx: np.ndarray,
    ky: np.ndarray,
    *,
    physical: bool = True,
) -> np.ndarray:
    x = np.asarray(grid.X[0, :], dtype=float)
    y = np.asarray(grid.Y[:, 0], dtype=float)
    dx, dy = grid.spacing()

    x_kernel = np.exp(-1.0j * np.outer(kx, x))
    y_kernel = np.exp(-1.0j * np.outer(ky, y))
    G = y_kernel @ g @ x_kernel.T
    if physical:
        G = dx * dy * G
    return G


def _zoom_ft_axis(values: np.ndarray, sample_axis: np.ndarray, k_axis: np.ndarray, axis: int) -> np.ndarray:
    if k_axis.size == 1:
        kernel = np.exp(-1.0j * k_axis[0] * sample_axis)
        transformed = np.tensordot(values, kernel, axes=([axis], [0]))
        return np.expand_dims(transformed, axis=axis)

    spacing = float(np.mean(np.diff(sample_axis)))
    fs = 1.0 / spacing
    frequencies = k_axis / (2.0 * π)
    transformed = zoom_fft(
        values,
        [float(frequencies[0]), float(frequencies[-1])],
        m=int(k_axis.size),
        fs=fs,
        endpoint=True,
        axis=axis,
    )
    phase = np.exp(-1.0j * k_axis * float(sample_axis[0]))
    if axis == 0:
        return phase[:, None] * transformed
    if axis == 1:
        return transformed * phase[None, :]
    raise ValueError("axis must be 0 or 1.")


def _zoom_ft2_on_kgrid(
    g: np.ndarray,
    grid: Grid,
    kx: np.ndarray,
    ky: np.ndarray,
    *,
    physical: bool = True,
) -> np.ndarray:
    x = np.asarray(grid.X[0, :], dtype=float)
    y = np.asarray(grid.Y[:, 0], dtype=float)
    dx, dy = grid.spacing()

    G = _zoom_ft_axis(g, x, kx, axis=1)
    G = _zoom_ft_axis(G, y, ky, axis=0)
    if physical:
        G = dx * dy * G
    return G


def FT2(
    g,
    grid: Grid | None = None,
    *,
    dx=1.0,
    dy=1.0,
    kgrid: Grid | None = None,
    physical=True,
    shift=True,
    method: str = "auto",
):
    """Return ``(G, kgrid)`` for the field samples ``g`` on ``grid``.

    When ``physical`` is true, the result is scaled by ``dx * dy`` to
    approximate ``integral integral g(x, y) exp[-i(kx*x + ky*y)] dx dy``.
    If ``grid`` is omitted, a centered Cartesian grid is built from ``dx`` and ``dy``.
    If ``kgrid`` is provided, the transform is evaluated on that separable
    Cartesian k-grid using a zoom FFT by default.
    """
    g = np.asarray(g, dtype=complex)
    grid = _grid_from_shape_or_spacing(g.shape, grid=grid, dx=dx, dy=dy)
    if g.shape != grid.shape:
        raise ValueError("g must have the same shape as grid.")
    if method not in {"auto", "fft", "zoom", "direct"}:
        raise ValueError("method must be one of: 'auto', 'fft', 'zoom', 'direct'.")

    dx, dy = grid.spacing()
    if kgrid is not None:
        if method == "fft":
            raise ValueError("method='fft' cannot be used with a custom kgrid.")
        kx, ky = _custom_k_axes(kgrid)
        if kgrid.shape != (ky.size, kx.size):
            raise ValueError("kgrid shape must match its separable axes.")
        if kgrid.dual is None:
            kgrid = Grid.from_cartesian(kgrid.X, kgrid.Y, domain=kgrid.domain, dual=grid)

        if method == "direct":
            return _direct_ft2_on_kgrid(g, grid, kx, ky, physical=physical), kgrid
        return _zoom_ft2_on_kgrid(g, grid, kx, ky, physical=physical), kgrid

    if method in {"zoom", "direct"}:
        raise ValueError(f"method={method!r} requires a custom kgrid.")

    G = _ft2_array(g, dx=dx, dy=dy, physical=physical, shift=shift)
    return G, grid.kgrid(shift=shift)


def IFT2(G, kgrid: Grid | None = None, *, dx=1.0, dy=1.0, physical=True, shift=True):
    """Return ``(g, grid)`` from samples ``G`` on a Fourier ``kgrid``."""
    G = np.asarray(G, dtype=complex)
    if kgrid is None:
        grid = Grid.from_spacing(G.shape, dx=dx, dy=dy)
        kgrid = grid.kgrid(shift=shift)
    elif not isinstance(kgrid, Grid):
        raise TypeError("kgrid must be a vecdiff.grid.Grid instance.")
    elif kgrid.dual is None:
        raise ValueError("kgrid must carry its dual spatial grid.")
    else:
        grid = kgrid.dual

    if G.shape != kgrid.shape:
        raise ValueError("G must have the same shape as kgrid.")

    dx, dy = grid.spacing()
    return _ift2_array(G, dx=dx, dy=dy, physical=physical, shift=shift), grid


def KGRID2(grid: Grid | None = None, *, shape=None, dx=1.0, dy=1.0, shift=True):
    """Return angular spectral grids ``(KX, KY)`` for a Cartesian ``Grid``."""
    if grid is None:
        if shape is None:
            raise TypeError("KGRID2 requires either grid or shape.")
        grid = Grid.from_spacing(shape, dx=dx, dy=dy)
    elif not isinstance(grid, Grid):
        raise TypeError("grid must be a vecdiff.grid.Grid instance.")

    kgrid = grid.kgrid(shift=shift)
    return kgrid.X, kgrid.Y


# ================================================================ #
#  3D Fourier Transform                                             #
# ================================================================ #

def FT3(g, dx=1.0, dy=1.0, dz=1.0, physical=True, shift=True):
    """Return the three-dimensional Fourier transform of ``g[z, y, x]``.

    When ``physical`` is true, the result is scaled by ``dx * dy * dz`` to
    approximate the corresponding Fourier integral.
    """
    g = np.asarray(g, dtype=complex)
    if shift:
        g = np.fft.ifftshift(g)
    G = np.fft.fftn(g, axes=(0, 1, 2))
    if physical:
        G = dx * dy * dz * G
    if shift:
        G = np.fft.fftshift(G)
    return G


def IFT3(G, dx=1.0, dy=1.0, dz=1.0, physical=True, shift=True):
    """Return the three-dimensional inverse Fourier transform of ``G``."""
    G = np.asarray(G, dtype=complex)
    if shift:
        G = np.fft.ifftshift(G)
    if physical:
        G = G / (dx * dy * dz)
    g = np.fft.ifftn(G, axes=(0, 1, 2))
    if shift:
        g = np.fft.fftshift(g)
    return g


def KGRID3(shape, dx=1.0, dy=1.0, dz=1.0, shift=True):
    """Return angular spectral grids ``(KX, KY, KZ)`` for ``g[z, y, x]``."""
    Nz, Ny, Nx = shape
    kx = 2.0 * π * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * π * np.fft.fftfreq(Ny, d=dy)
    kz = 2.0 * π * np.fft.fftfreq(Nz, d=dz)
    if shift:
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        kz = np.fft.fftshift(kz)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    return KX, KY, KZ


# ================================================================ #
#  Centered 2D Compatibility Helpers                                #
# ================================================================ #

def FFT2(g: np.ndarray) -> np.ndarray:
    """Return the centered two-dimensional FFT of a centered input field."""
    g_ = np.fft.ifftshift(g)
    G_ = np.fft.fft2(g_)
    return np.fft.fftshift(G_)


def IFFT2(G: np.ndarray) -> np.ndarray:
    """Return the centered inverse two-dimensional FFT of a centered spectrum."""
    G_ = np.fft.ifftshift(G)
    g_ = np.fft.ifft2(G_)
    return np.fft.fftshift(g_)


# ================================================================ #
#  Cartesian Frequency Grids                                        #
# ================================================================ #

def _as_cartesian_grid(grid) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(grid, "X") and hasattr(grid, "Y"):
        return np.asarray(grid.X, dtype=float), np.asarray(grid.Y, dtype=float)

    try:
        x, y = grid
    except (TypeError, ValueError) as exc:
        raise TypeError("grid must be a Grid instance or an (x, y) pair.") from exc

    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _uniform_spacing(values: np.ndarray, axis: int, name: str) -> float:
    diffs = np.diff(values, axis=axis)
    if diffs.size == 0:
        raise ValueError(f"{name} grid axis must contain at least two samples.")

    spacing = float(np.mean(diffs))
    if not np.allclose(diffs, spacing):
        raise ValueError(f"{name} grid axis must be uniformly sampled.")
    if np.isclose(spacing, 0.0):
        raise ValueError(f"{name} grid spacing must be nonzero.")
    return spacing


def frequency_grid(grid) -> tuple[np.ndarray, np.ndarray]:
    """Return centered ordinary-frequency grids ``(fx, fy)``.

    ``grid`` may be a :class:`vecdiff.grid.Grid` instance or an ``(x, y)``
    pair of Cartesian sampling grids.
    """
    x, y = _as_cartesian_grid(grid)

    if x.shape != y.shape:
        raise ValueError("x and y grids must have the same shape.")
    if x.ndim != 2:
        raise ValueError("x and y must be 2D arrays.")

    ny, nx = x.shape
    dx = _uniform_spacing(x, axis=1, name="x")
    dy = _uniform_spacing(y, axis=0, name="y")

    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    return np.meshgrid(fx, fy, indexing="xy")
