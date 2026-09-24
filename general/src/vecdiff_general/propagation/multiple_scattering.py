"""Coherent linear feedback with a reported algebraic convergence residual."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


@dataclass(frozen=True)
class FeedbackResult:
    amplitudes: np.ndarray
    relative_residual: float
    iterations: int


def coherent_feedback(injection, round_trip, *, tolerance=1e-10, max_iterations=200):
    """Solve amplitudes = injection + round_trip(amplitudes) with GMRES.

    The supplied round-trip operator must be linear and use a fixed modal
    representation. This residual certifies the discrete feedback equation;
    physical boundary residuals must be checked separately.
    """
    injection = np.asarray(injection, dtype=complex)

    if not np.isfinite(injection).all() or not 0 < tolerance < 1:
        raise ValueError("finite injection and a tolerance in (0, 1) are required")

    shape = injection.shape
    iterations = 0

    def apply(vector):
        values = vector.reshape(shape)
        returning = np.asarray(round_trip(values), dtype=complex)

        if returning.shape != shape or not np.isfinite(returning).all():
            raise ValueError("round-trip operator returned incompatible amplitudes")

        return (values - returning).ravel()

    def count_iteration(residual):
        nonlocal iterations
        iterations += 1

    operator = LinearOperator(
        (injection.size, injection.size), matvec=apply, dtype=complex
    )
    amplitudes, status = gmres(
        operator,
        injection.ravel(),
        rtol=tolerance,
        atol=0,
        maxiter=max_iterations,
        callback=count_iteration,
        callback_type="pr_norm",
    )
    norm = np.linalg.norm(injection)
    relative_residual = np.linalg.norm(apply(amplitudes) - injection.ravel()) / max(
        norm, 1e-300
    )

    if status != 0 or relative_residual > tolerance:
        raise RuntimeError(
            f"coherent feedback did not converge: residual={relative_residual:.3g}"
        )

    return FeedbackResult(
        amplitudes.reshape(shape), float(relative_residual), iterations
    )
