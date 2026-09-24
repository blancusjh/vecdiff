"""Nonuniform Fourier sums and quadrature in one, two or three dimensions."""

import numpy as np


def phase_factor(q, x, *, sign=1):
    """Return exp(sign * i * x.q), rejecting overflowing complex growth."""
    if sign not in (-1, 1):
        raise ValueError("Fourier sign must be -1 or +1")

    exponent = sign * 1j * (np.asarray(x) @ np.asarray(q).T)

    if np.any(exponent.real > 700):
        raise ValueError(
            "complex exponential overflows; reduce the exponent's real part"
        )

    phase = np.exp(exponent)

    return phase


def _sum_inputs(g, x, q):
    g = np.asarray(g, dtype=complex)
    x = np.asarray(x, dtype=complex)
    q = np.asarray(q, dtype=complex)

    if x.ndim != 2 or x.shape[1] not in (1, 2, 3):
        raise ValueError(
            "source coordinates must have shape (nodes, dimension), dimension 1–3"
        )

    if g.ndim < 1 or g.shape[0] != len(x):
        raise ValueError("the first values axis must match the source coordinates")

    if q.ndim < 1 or q.shape[-1] != x.shape[1]:
        raise ValueError("source and target coordinate dimensions must agree")

    if not all(np.isfinite(array).all() for array in (g, x, q)):
        raise ValueError("Fourier inputs must be finite")

    return g, x, q


def _direct_sum(g, x, q, sign, chunk_size):
    if isinstance(chunk_size, bool) or int(chunk_size) != chunk_size or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    G = np.zeros((len(q), g.shape[1]), dtype=complex)

    for start in range(0, len(q), int(chunk_size)):
        stop = start + int(chunk_size)
        phase = phase_factor(x, q[start:stop], sign=sign)

        G[start:stop] = phase @ g

    return G


def _fast_sum(g, x, q, sign, tolerance):
    if np.any(x.imag) or np.any(q.imag):
        raise ValueError("FINUFFT requires real coordinates; use backend='direct'")

    if not 0 < tolerance < 1:
        raise ValueError("NUFFT tolerance must lie in (0, 1)")

    try:
        import finufft
    except ImportError as error:
        raise ImportError("install the nufft extra to use this backend") from error

    if not len(x) or not len(q) or not g.shape[1]:
        return np.zeros((len(q), g.shape[1]), dtype=complex)

    dimension = x.shape[1]
    source_axes = [np.ascontiguousarray(x.real[:, axis]) for axis in range(dimension)]
    target_axes = [np.ascontiguousarray(q.real[:, axis]) for axis in range(dimension)]
    transform = getattr(finufft, f"nufft{dimension}d3")

    G = transform(
        *source_axes,
        np.ascontiguousarray(g.T),
        *target_axes,
        isign=sign,
        eps=tolerance,
        nthreads=1,
    ).T

    return G


def NUFFT(
    g: np.ndarray, x, q, *, sign=-1, backend="nufft", tolerance=1e-12, chunk_size=256
):
    """Evaluate G[k] = sum_j g[j] exp(sign * i * q[k].x[j]).

    x has shape (nodes, dimension); q has shape (..., dimension), dimension 1–3.
    g has shape (nodes, ...) with independent trailing components or batches.
    Output shape is q.shape[:-1] + g.shape[1:]. Frequencies are angular: no 2*pi
    is inserted. There is no normalization or implied quadrature weight.

    The default negative sign gives the forward sum. Positive sign gives the
    reverse-sign sum, not a matrix inverse. The direct backend supports complex
    coordinates; FINUFFT accelerates real coordinates with a type-3 transform.
    """
    if sign not in (-1, 1):
        raise ValueError("Fourier sign must be -1 or +1")

    g, x, q = _sum_inputs(g, x, q)
    output_shape = q.shape[:-1] + g.shape[1:]
    component_count = int(np.prod(g.shape[1:]))

    g = g.reshape(len(x), component_count)
    q = q.reshape(-1, x.shape[1])

    # Zero terms contribute nothing, even when their exponential would overflow.
    occupied = np.any(g != 0, axis=1)
    g, x = g[occupied], x[occupied]

    if backend == "direct":
        G = _direct_sum(g, x, q, sign, chunk_size)
    elif backend == "nufft":
        G = _fast_sum(g, x, q, sign, tolerance)
    else:
        raise ValueError("backend must be 'direct' or 'nufft'")

    return G.reshape(output_shape)


def _weighted_values(g, weights):
    g = np.asarray(g, dtype=complex)
    weights = np.asarray(weights, dtype=float)

    if g.ndim < 1 or weights.shape != g.shape[:1]:
        raise ValueError("quadrature weights must match the first values axis")

    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("quadrature weights must be finite and nonnegative")

    weights = weights.reshape(weights.shape + (1,) * (g.ndim - 1))

    return weights * g


def transform(
    g: np.ndarray,
    coordinates,
    frequencies,
    *,
    weights,
    backend="nufft",
    tolerance=1e-12,
    chunk_size=256,
):
    """Approximate G(q) = integral g(x) exp(-i*q.x) d^d x.

    weights explicitly supply the spatial integration measure at each node.
    Coordinates and frequency arrays use the NUFFT shape convention.
    """
    g_weighted = _weighted_values(g, weights)

    G = NUFFT(
        g_weighted,
        coordinates,
        frequencies,
        sign=-1,
        backend=backend,
        tolerance=tolerance,
        chunk_size=chunk_size,
    )

    return G


def inverse(
    G: np.ndarray,
    coordinates,
    frequencies,
    *,
    weights,
    backend="nufft",
    tolerance=1e-12,
    chunk_size=256,
):
    """Approximate g(x) = integral G(q) exp(+i*q.x) d^d q / (2*pi)^d.

    weights supply frequency-space measure; frequencies has shape (nodes, d).
    Finite support and quadrature limit this inverse integral. It is not an
    exact algebraic inverse of an arbitrary nonuniform forward matrix.
    """
    G_weighted = _weighted_values(G, weights)

    g = NUFFT(
        G_weighted,
        frequencies,
        coordinates,
        sign=1,
        backend=backend,
        tolerance=tolerance,
        chunk_size=chunk_size,
    )
    dimension = np.asarray(frequencies).shape[1]
    g = g / (2 * np.pi) ** dimension

    return g
