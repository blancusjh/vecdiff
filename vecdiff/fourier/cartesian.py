"""Physical Fourier convention: A(k)=integral f(x) exp(-ik.x) dx.

Inverse discrete synthesis divides by the sampled period area. Axes may have
arbitrary origins; no hidden fftshift convention enters the physics.
"""
import numpy as np


def transform(values, grid):
    kx, ky = grid.kxy
    return np.fft.fft2(values, axes=(-2, -1))*grid.dx*grid.dy*np.exp(-1j*(kx*grid.x[0]+ky*grid.y[0]))


def inverse(amplitudes, grid):
    kx, ky = grid.kxy
    return np.fft.ifft2(amplitudes*np.exp(1j*(kx*grid.x[0]+ky*grid.y[0])), axes=(-2, -1))/(grid.dx*grid.dy)
