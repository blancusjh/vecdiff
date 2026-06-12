"""Fourier transform utilities."""

import numpy as np

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


def FT2(g, grid: Grid | None = None, *, dx=1.0, dy=1.0, physical=True, shift=True):
    """Return ``(G, kgrid)`` for the field samples ``g`` on ``grid``.

    When ``physical`` is true, the result is scaled by ``dx * dy`` to
    approximate ``integral integral g(x, y) exp[-i(kx*x + ky*y)] dx dy``.
    If ``grid`` is omitted, a centered Cartesian grid is built from ``dx`` and ``dy``.
    """
    g = np.asarray(g, dtype=complex)
    grid = _grid_from_shape_or_spacing(g.shape, grid=grid, dx=dx, dy=dy)
    if g.shape != grid.shape:
        raise ValueError("g must have the same shape as grid.")

    dx, dy = grid.spacing()
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
