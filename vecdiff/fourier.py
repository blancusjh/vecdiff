"""Fourier transform utilities."""

import numpy as np

π = np.pi


# ================================================================ #
#  1D Fourier Transform                                             #
# ================================================================ #

def FT(g, dx=1.0, physical=False, shift=False):
    """Return the one-dimensional Fourier transform of ``g``.

    When ``physical`` is true, the result is scaled by ``dx`` to approximate
    ``integral g(x) exp(-i k x) dx`` at the sampled spectral points.
    """
    G = np.fft.fft(np.asarray(g, dtype=complex))
    if physical:
        G = dx * G
    if shift:
        G = np.fft.fftshift(G)
    return G


def IFT(G, dx=1.0, physical=False, shift=False):
    """Return the one-dimensional inverse Fourier transform of ``G``."""
    G = np.asarray(G, dtype=complex)
    if shift:
        G = np.fft.ifftshift(G)
    if physical:
        G = G / dx
    return np.fft.ifft(G)


def KAXIS(N, dx=1.0, shift=False):
    """Return the one-dimensional angular spectral axis ``k = 2*pi*f``."""
    k = 2.0 * π * np.fft.fftfreq(N, d=dx)
    if shift:
        k = np.fft.fftshift(k)
    return k


# ================================================================ #
#  2D Fourier Transform                                             #
# ================================================================ #

def FT2(g, dx=1.0, dy=1.0, physical=False, shift=False):
    """Return the two-dimensional Fourier transform of ``g[y, x]``.

    When ``physical`` is true, the result is scaled by ``dx * dy`` to
    approximate ``integral integral g(x, y) exp[-i(kx*x + ky*y)] dx dy``.
    """
    G = np.fft.fft2(np.asarray(g, dtype=complex))
    if physical:
        G = dx * dy * G
    if shift:
        G = np.fft.fftshift(G)
    return G


def IFT2(G, dx=1.0, dy=1.0, physical=False, shift=False):
    """Return the two-dimensional inverse Fourier transform of ``G``."""
    G = np.asarray(G, dtype=complex)
    if shift:
        G = np.fft.ifftshift(G)
    if physical:
        G = G / (dx * dy)
    return np.fft.ifft2(G)


def KGRID2(shape, dx=1.0, dy=1.0, shift=False):
    """Return angular spectral grids ``(KX, KY)`` for an array ``g[y, x]``."""
    Ny, Nx = shape
    kx = 2.0 * π * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * π * np.fft.fftfreq(Ny, d=dy)
    if shift:
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
    return np.meshgrid(kx, ky, indexing="xy")


# ================================================================ #
#  3D Fourier Transform                                             #
# ================================================================ #

def FT3(g, dx=1.0, dy=1.0, dz=1.0, physical=False, shift=False):
    """Return the three-dimensional Fourier transform of ``g[z, y, x]``.

    When ``physical`` is true, the result is scaled by ``dx * dy * dz`` to
    approximate the corresponding Fourier integral.
    """
    G = np.fft.fftn(np.asarray(g, dtype=complex), axes=(0, 1, 2))
    if physical:
        G = dx * dy * dz * G
    if shift:
        G = np.fft.fftshift(G)
    return G


def IFT3(G, dx=1.0, dy=1.0, dz=1.0, physical=False, shift=False):
    """Return the three-dimensional inverse Fourier transform of ``G``."""
    G = np.asarray(G, dtype=complex)
    if shift:
        G = np.fft.ifftshift(G)
    if physical:
        G = G / (dx * dy * dz)
    return np.fft.ifftn(G, axes=(0, 1, 2))


def KGRID3(shape, dx=1.0, dy=1.0, dz=1.0, shift=False):
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
