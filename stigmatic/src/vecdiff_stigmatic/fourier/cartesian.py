"""Cartesian Fourier transforms of arrays, independent of optical objects."""

import numpy as np


def fft_modes(values):
    """Fourier-series coefficients: FFT(values)/(N_v*N_u) on axes (0, 1).

    Trailing dimensions are independent components or batches. Coefficients
    have the same units as the input; the phase origin is sample (0, 0).
    """
    values = np.asarray(values, dtype=complex)

    if values.ndim < 2 or not np.isfinite(values).all():
        raise ValueError("FFT values must be finite with at least two axes")

    sample_count = np.prod(values.shape[:2])

    return np.fft.fft2(values, axes=(0, 1)) / sample_count


def inverse_fft_modes(coefficients):
    """Inverse of fft_modes with the same first-two-axis convention."""
    coefficients = np.asarray(coefficients, dtype=complex)

    if coefficients.ndim < 2 or not np.isfinite(coefficients).all():
        raise ValueError("FFT coefficients must be finite with at least two axes")

    sample_count = np.prod(coefficients.shape[:2])

    return np.fft.ifft2(coefficients, axes=(0, 1)) * sample_count


def frequency_grid(shape, spacing):
    """Return angular-frequency arrays for a uniform two-dimensional lattice.

    shape is (N_v, N_u); spacing is (delta_u, delta_v). Frequencies are
    radians per coordinate unit. No physical Grid or material is required.
    """

    if len(shape) != 2 or len(spacing) != 2:
        raise ValueError("provide two sample counts and two coordinate spacings")

    if any(
        isinstance(count, bool) or int(count) != count or count <= 0 for count in shape
    ):
        raise ValueError("sample counts must be positive integers")

    if not np.isfinite(spacing).all() or np.any(np.asarray(spacing) <= 0):
        raise ValueError("coordinate spacings must be finite and positive")

    count_v, count_u = shape
    spacing_u, spacing_v = spacing

    frequency_u = 2 * np.pi * np.fft.fftfreq(int(count_u), spacing_u)
    frequency_v = 2 * np.pi * np.fft.fftfreq(int(count_v), spacing_v)

    return np.meshgrid(frequency_u, frequency_v, indexing="xy")


def _coordinate_axes(coordinates):
    if coordinates is None or len(coordinates) != 2:
        raise ValueError("Cartesian coordinates must be (u, v) axes")

    axes = tuple(np.asarray(axis, dtype=float) for axis in coordinates)

    for axis in axes:
        if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
            raise ValueError("coordinate axes must be finite one-dimensional arrays")

        steps = np.diff(axis)

        if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-10, atol=0):
            raise ValueError("Cartesian FFT coordinates must increase uniformly")

    return axes


def _measure_and_phase(values, coordinates):
    axis_u, axis_v = _coordinate_axes(coordinates)
    shape = (len(axis_v), len(axis_u))

    if values.shape[:2] != shape:
        raise ValueError("Cartesian values must have leading shape (len(v), len(u))")

    spacing = (axis_u[1] - axis_u[0], axis_v[1] - axis_v[0])
    frequency_u, frequency_v = frequency_grid(shape, spacing)

    cell_area = np.prod(shape) * np.prod(spacing)
    phase = np.exp(-1j * (frequency_u * axis_u[0] + frequency_v * axis_v[0]))
    phase = phase.reshape(shape + (1,) * (values.ndim - 2))

    return cell_area, phase


def transform(values, coordinates):
    """Approximate integral f(u,v) exp(-i(q_u u + q_v v)) du dv by FFT.

    coordinates=(u,v); arrays have order (v,u,...). Output frequencies are the
    unshifted FFT frequencies returned by frequency_grid. The original axis
    origins contribute to the phase. Trailing components are independent.
    """
    values = np.asarray(values, dtype=complex)
    cell_area, phase = _measure_and_phase(values, coordinates)

    return cell_area * phase * fft_modes(values)


def inverse(values, coordinates):
    """Recover Cartesian samples from their unshifted continuous-FT samples.

    coordinates are the original (u,v) axes, not frequency coordinates. This
    is the exact discrete inverse of transform on the declared periodic cell.
    Its frequency-cell measure includes the continuous inverse 1/(2*pi)^2.
    """
    values = np.asarray(values, dtype=complex)
    cell_area, phase = _measure_and_phase(values, coordinates)

    return inverse_fft_modes(values / phase) / cell_area
