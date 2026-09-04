"""Nonuniform Fourier sums, without field or interface dependencies."""
import numpy as np


def synthesize(frequencies, coefficients, points, *, backend="direct", eps=1e-10, chunk=512):
    """sum_j c[j] exp(i k[j].x); final coefficient axis indexes modes.

FINUFFT requires real coordinates. Complex (evanescent) wavevectors use the
direct backend. An explicitly requested unavailable backend raises an error.
    """
    k, c, x = np.asarray(frequencies), np.asarray(coefficients), np.asarray(points)
    if k.ndim != 2 or k.shape[1] != 3 or x.ndim != 2 or x.shape[1] != 3 or c.shape[-1] != len(k):
        raise ValueError("expected k=(m,3), c=(...,m), points=(p,3)")
    if backend == "nufft":
        if np.any(np.imag(k)):
            raise ValueError("FINUFFT does not accept complex wavevectors; select direct")
        import finufft
        args = [np.ascontiguousarray(k[:, j].real, dtype=float) for j in range(3)]
        targets = [np.ascontiguousarray(x[:, j], dtype=float) for j in range(3)]
        flat = np.ascontiguousarray(c.reshape(-1, len(k)), dtype=complex)
        if len(k) == 0:
            return np.zeros(c.shape[:-1] + (len(x),), complex)
        result = finufft.nufft3d3(*args, flat, *targets, isign=1, eps=eps)
        return result.reshape(c.shape[:-1] + (len(x),))
    if backend != "direct":
        raise ValueError("backend must be 'direct' or 'nufft'")
    out = np.empty(c.shape[:-1] + (len(x),), complex)
    for start in range(0, len(x), chunk):
        out[..., start:start+chunk] = c @ np.exp(1j*k @ x[start:start+chunk].T)
    return out
