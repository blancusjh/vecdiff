"""Nonuniform Fourier sums and quadrature in one, two or three dimensions."""

import math
import os

import numpy as np

from vecdiff._backend import bd, to_numpy


def phase_factor(q, x, *, sign=1):
    """Return exp(sign * i * x.q), rejecting overflowing complex growth."""
    if sign not in (-1, 1):
        raise ValueError("Fourier sign must be -1 or +1")

    exponent = sign * 1j * (bd.asarray(x) @ bd.asarray(q).T)

    if bd.any(exponent.real > 700):
        raise ValueError(
            "complex exponential overflows; reduce the exponent's real part"
        )

    phase = bd.exp(exponent)

    return phase


def _sum_inputs(g, x, q):
    g = bd.asarray(g, dtype=complex)
    x = bd.asarray(x, dtype=complex)
    q = bd.asarray(q, dtype=complex)

    if x.ndim != 2 or x.shape[1] not in (1, 2, 3):
        raise ValueError(
            "source coordinates must have shape (nodes, dimension), dimension 1–3"
        )

    if g.ndim < 1 or g.shape[0] != len(x):
        raise ValueError("the first values axis must match the source coordinates")

    if q.ndim < 1 or q.shape[-1] != x.shape[1]:
        raise ValueError("source and target coordinate dimensions must agree")

    if not all(bd.isfinite(array).all() for array in (g, x, q)):
        raise ValueError("Fourier inputs must be finite")

    return g, x, q


def _direct_sum(g, x, q, sign, chunk_size):
    if isinstance(chunk_size, bool) or int(chunk_size) != chunk_size or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    G = bd.zeros((len(q), g.shape[1]), dtype=complex)
    real = x.dtype.kind == "f" and q.dtype.kind == "f"

    for start in range(0, len(q), int(chunk_size)):
        stop = start + int(chunk_size)
        if real:
            # A real phase: a real product, and cos and sin in place of the
            # complex exponential, which costs 1.4 times as much.
            angle = q[start:stop] @ x.T
            phase = bd.empty(angle.shape, complex)
            phase.real[...] = bd.cos(angle)
            phase.imag[...] = bd.sin(angle) if sign == 1 else -bd.sin(angle)
        else:
            phase = phase_factor(x, q[start:stop], sign=sign)

        G[start:stop] = phase @ g

    return G


def _threads(threads):
    """FINUFFT thread count: the argument, else VECDIFF_NUFFT_THREADS, else 0 (all cores)."""
    if threads is None:
        threads = os.environ.get("VECDIFF_NUFFT_THREADS", 0)
    threads = int(threads)
    if threads < 0:
        raise ValueError("threads must be a nonnegative integer (0: all cores)")
    return threads


def _plane(q, x, tolerance):
    """(centre, axes) of a plane through every target, or None.

    The targets q (n, 3) lie on the plane centre + a axes[:, 0] + b axes[:, 1]
    when dropping their offsets d from it changes no phase x.q by more than
    ``tolerance``: |x| |d| <= tolerance, an error within the transform's own.
    The plane is the least-squares one through the targets, so a tilted plane
    is found as well as an axis-aligned one.
    """
    if len(q) < 3:
        return None
    # Pairwise sums along each coordinate: a column sum drifts by n eps.
    centre = np.ascontiguousarray(q.T).mean(axis=1)
    offset = q - centre
    _, vectors = np.linalg.eigh(offset.T @ offset)
    # The sum of squares tilts the normal by ~ sqrt(n) eps; one least-squares
    # step on the (small, accurate) depths removes the tilt.
    planar = offset @ vectors[:, 1:]
    tilt = np.linalg.lstsq(planar, offset @ vectors[:, 0], rcond=None)[0]
    normal = vectors[:, 0] - vectors[:, 1:] @ tilt
    normal /= np.linalg.norm(normal)
    depth = np.max(np.abs(offset @ normal))
    reach = np.sqrt(np.max(np.sum(x * x, axis=1)))
    if reach * depth > tolerance:
        return None
    first = vectors[:, 1] - normal * (vectors[:, 1] @ normal)
    first /= np.linalg.norm(first)
    return centre, np.stack((first, np.cross(normal, first)), -1)


def _host_inputs(g, x, q, tolerance):
    """Host copies for FINUFFT, coplanar 3D targets folded into two dimensions."""
    x, q, g = to_numpy(x).real, to_numpy(q).real, to_numpy(g)
    plane = _plane(q, x, tolerance) if x.shape[1] == 3 else None
    if plane is not None:
        # Coplanar targets: the phase x.(centre + offset) splits into a
        # per-source factor and an in-plane phase, a 2D type-3 transform.
        centre, axes = plane
        return g, x @ axes, (q - centre) @ axes, np.exp(1j * (x @ centre))
    return g, x, q, None


def _fast_sum(g, x, q, sign, tolerance, threads=None, host=None):
    if bd.any(x.imag) or bd.any(q.imag):
        raise ValueError("FINUFFT requires real coordinates; use method='direct'")

    if not 0 < tolerance < 1:
        raise ValueError("NUFFT tolerance must lie in (0, 1)")

    try:
        import finufft
    except ImportError as error:
        raise ImportError("install the nufft extra to use this method") from error

    if not len(x) or not len(q) or not g.shape[1]:
        return bd.zeros((len(q), g.shape[1]), dtype=complex)

    # FINUFFT runs on the host for either backend; it outpaces the GPU's
    # type-3 transform in double precision on consumer cards.
    g, x, q, fold = host or _host_inputs(g, x, q, tolerance)
    if fold is not None:
        g = g * (fold if sign == 1 else np.conj(fold))[:, None]
    dimension = x.shape[1]
    source_axes = [np.ascontiguousarray(x[:, axis]) for axis in range(dimension)]
    target_axes = [np.ascontiguousarray(q[:, axis]) for axis in range(dimension)]
    transform = getattr(finufft, f"nufft{dimension}d3")

    G = transform(
        *source_axes,
        np.ascontiguousarray(g.T),
        *target_axes,
        isign=sign,
        eps=tolerance,
        nthreads=_threads(threads),
    ).T

    return bd.asarray(G)


# method="auto": the direct sum costs about 20 ns per (source, target) pair,
# the type-3 transform about 0.7 us per unit of its space-bandwidth product
# prod_d (half-width of sources x half-width of targets) and ~3 kB of memory
# (measured, 10 cores). The direct sum is taken below 2^18 pairs, where the
# transform's start-up dominates, wherever it is the cheaper estimate, and
# beyond a space-bandwidth product of 2^21 (~7 GB of transform grid).
_DIRECT_PAIRS = 2**18
_PAIRS_PER_UNIT = 35
_LARGEST_TRANSFORM = 2**21


def _automatic(g, x, q, tolerance):
    """None for the direct sum, else the host inputs of the transform."""
    import importlib.util

    if len(x) * len(q) <= _DIRECT_PAIRS or importlib.util.find_spec("finufft") is None:
        return None
    if x.dtype.kind != "f" or q.dtype.kind != "f":
        return None
    host = _host_inputs(g, x, q, tolerance)
    size = math.prod(
        max(1.0, float(np.ptp(host[1][:, d])) * float(np.ptp(host[2][:, d])) / 4)
        for d in range(host[1].shape[1])
    )
    if size > _LARGEST_TRANSFORM or size * _PAIRS_PER_UNIT >= len(x) * len(q):
        return None
    return host


def NUFFT(
    g: np.ndarray,
    x,
    q,
    *,
    sign=-1,
    method="nufft",
    tolerance=1e-12,
    chunk_size=256,
    threads=None,
):
    """Evaluate G[k] = sum_j g[j] exp(sign * i * q[k].x[j]).

    x has shape (nodes, dimension); q has shape (..., dimension), dimension 1–3.
    g has shape (nodes, ...) with independent trailing components or batches.
    Output shape is q.shape[:-1] + g.shape[1:]. Frequencies are angular: no 2*pi
    is inserted. There is no normalization or implied quadrature weight.

    The default negative sign gives the forward sum. Positive sign gives the
    reverse-sign sum, not a matrix inverse. The direct method supports complex
    coordinates; FINUFFT accelerates real coordinates with a type-3 transform,
    on ``threads`` threads: None reads VECDIFF_NUFFT_THREADS, else all cores (0).
    Three-dimensional targets on one plane (a field map) take a two-dimensional
    transform, whose fine grid has no third axis to pad. method "auto" takes
    the transform when FINUFFT is installed, the coordinates are real and its
    estimated cost is below the direct sum's, else the direct sum.
    """
    if sign not in (-1, 1):
        raise ValueError("Fourier sign must be -1 or +1")

    g, x, q = _sum_inputs(g, x, q)
    output_shape = q.shape[:-1] + g.shape[1:]
    component_count = int(math.prod(g.shape[1:]))

    g = g.reshape(len(x), component_count)
    q = q.reshape(-1, x.shape[1])

    # Zero terms contribute nothing, even when their exponential would overflow.
    occupied = bd.any(g != 0, axis=1)
    g, x = g[occupied], x[occupied]

    # Real coordinates stay real: the direct phase is then a real product.
    if not bd.any(x.imag) and not bd.any(q.imag):
        x, q = bd.ascontiguousarray(x.real), bd.ascontiguousarray(q.real)

    if method not in ("auto", "direct", "nufft"):
        raise ValueError("method must be 'auto', 'direct' or 'nufft'")
    host = _automatic(g, x, q, tolerance) if method == "auto" else None
    if method == "direct" or (method == "auto" and host is None):
        G = _direct_sum(g, x, q, sign, chunk_size)
    else:
        G = _fast_sum(g, x, q, sign, tolerance, threads, host)

    return G.reshape(output_shape)


def _weighted_values(g, weights):
    g = bd.asarray(g, dtype=complex)
    weights = bd.asarray(weights, dtype=float)

    if g.ndim < 1 or weights.shape != g.shape[:1]:
        raise ValueError("quadrature weights must match the first values axis")

    if not bd.isfinite(weights).all() or bd.any(weights < 0):
        raise ValueError("quadrature weights must be finite and nonnegative")

    weights = weights.reshape(weights.shape + (1,) * (g.ndim - 1))

    return weights * g


def transform(
    g: np.ndarray,
    coordinates,
    frequencies,
    *,
    weights,
    method="nufft",
    tolerance=1e-12,
    chunk_size=256,
    threads=None,
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
        method=method,
        tolerance=tolerance,
        chunk_size=chunk_size,
        threads=threads,
    )

    return G


def inverse(
    G: np.ndarray,
    coordinates,
    frequencies,
    *,
    weights,
    method="nufft",
    tolerance=1e-12,
    chunk_size=256,
    threads=None,
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
        method=method,
        tolerance=tolerance,
        chunk_size=chunk_size,
        threads=threads,
    )
    dimension = bd.asarray(frequencies).shape[1]
    g = g / (2 * bd.pi) ** dimension

    return g
