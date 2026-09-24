"""Direct nonuniform Fourier synthesis and stable complex phase factors."""

from __future__ import annotations

import numpy as np


def phase_factor(wavevectors, displacements):
    """Stable forward phase, with an explicit failure for overflowing growth."""
    exponent = 1j * (displacements @ wavevectors.T)

    if np.any(exponent.real > 700):
        raise ValueError(
            "complex exponential overflows; shift the expansion origin or reduce the evaluation domain"
        )

    return np.exp(exponent)


def synthesize(
    wavevectors,
    amplitudes,
    displacements,
    *,
    backend="direct",
    tolerance=1e-12,
    chunk_size=256,
):
    """Evaluate weighted nonuniform Fourier modes at coordinate displacements.

    Arrays have shapes (modes, 3), (modes, components), and (..., 3). Modal
    amplitudes already include quadrature weights. FINUFFT is optional and
    supports real wavevectors; complex exponential phases use direct evaluation.
    """
    wavevectors = np.asarray(wavevectors, dtype=complex)
    amplitudes = np.asarray(amplitudes, dtype=complex)
    displacements = np.asarray(displacements, dtype=float)

    if wavevectors.ndim != 2 or wavevectors.shape[1] != 3 or amplitudes.ndim != 2:
        raise ValueError(
            "expected (modes, 3) wavevectors and (modes, components) amplitudes"
        )

    if amplitudes.shape[0] != len(wavevectors) or displacements.shape[-1:] != (3,):
        raise ValueError(
            "modal coefficients and observation coordinates are incompatible"
        )

    if not all(
        np.isfinite(value).all() for value in (wavevectors, amplitudes, displacements)
    ):
        raise ValueError("Fourier inputs must be finite")

    occupied = np.any(amplitudes != 0, axis=-1)
    wavevectors, amplitudes = wavevectors[occupied], amplitudes[occupied]
    points = displacements.reshape(-1, 3)
    result = np.zeros((len(points), amplitudes.shape[1]), dtype=complex)

    if backend == "nufft":
        if np.any(wavevectors.imag):
            raise ValueError("FINUFFT does not accept complex frequencies; use direct")

        if not 0 < tolerance < 1:
            raise ValueError("NUFFT tolerance must lie in (0, 1)")

        try:
            import finufft
        except ImportError as error:
            raise ImportError("install the nufft extra to use this backend") from error

        if len(wavevectors):
            coordinates = [
                np.ascontiguousarray(wavevectors.real[:, axis]) for axis in range(3)
            ]
            targets = [np.ascontiguousarray(points[:, axis]) for axis in range(3)]
            result = finufft.nufft3d3(
                *coordinates,
                np.ascontiguousarray(amplitudes.T),
                *targets,
                isign=1,
                eps=tolerance,
                nthreads=1,
            ).T
    elif backend == "direct":
        if int(chunk_size) != chunk_size or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        for start in range(0, len(points), int(chunk_size)):
            stop = start + int(chunk_size)
            result[start:stop] = (
                phase_factor(wavevectors, points[start:stop]) @ amplitudes
            )
    else:
        raise ValueError("backend must be 'direct' or 'nufft'")

    return result.reshape(displacements.shape[:-1] + (amplitudes.shape[-1],))
