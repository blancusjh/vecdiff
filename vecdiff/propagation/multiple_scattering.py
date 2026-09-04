"""Coherent feedback: sum complex amplitudes over repeated interactions.

The caller supplies a linear round-trip map in one fixed state representation.
The fixed point satisfies state = injection + round_trip(state). Convergence
of this algebra does not establish accuracy of the supplied boundary map.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


class ConvergenceError(RuntimeError):
    """The requested physical/numerical residual was not achieved."""


@dataclass(frozen=True)
class FeedbackResult:
    state: np.ndarray
    relative_residual: float
    iterations: int
    residual_history: tuple
    method: str

    def __post_init__(self):
        state = np.array(self.state, complex, copy=True)
        state.setflags(write=False)
        object.__setattr__(self, "state", state)


def coherent_feedback(injection, round_trip, *, rtol=1e-10, max_iterations=1000,
                      method="successive", restart=50):
    """Solve the coherent feedback equation with a true fixed-point residual.

successive explicitly accumulates all encounters. gmres solves the same
all-orders equation when simple round-trip accumulation converges too slowly.
Neither method silently returns a nonconverged answer.
    """
    b = np.asarray(injection, complex)
    if not np.isfinite(b).all() or b.size == 0:
        raise ValueError("injection must be a nonempty finite complex state")
    if method not in ("successive", "gmres"):
        raise ValueError("method must be 'successive' or 'gmres'")
    if not 0 < rtol < 1 or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("require 0 < rtol < 1 and positive max_iterations")
    if not isinstance(restart, int) or restart < 1:
        raise ValueError("restart must be a positive integer")
    scale = np.linalg.norm(b)
    if scale == 0:
        return FeedbackResult(np.zeros_like(b), 0., 0, (), method)
    def apply(x):
        y = np.asarray(round_trip(np.asarray(x).reshape(b.shape)), complex)
        if y.shape != b.shape or not np.isfinite(y).all():
            raise ConvergenceError("round-trip map returned invalid or divergent state")
        return y
    history = []
    if method == "successive":
        state = b.copy()
        for iteration in range(1, max_iterations+1):
            correction = b+apply(state)-state
            residual = float(np.linalg.norm(correction)/scale)
            history.append(residual)
            if residual <= rtol:
                return FeedbackResult(state, residual, iteration, tuple(history), method)
            state += correction
    elif method == "gmres":
        operator = LinearOperator((b.size, b.size), matvec=lambda x: x-apply(x).ravel(), dtype=complex)
        def record(residual):
            history.append(float(residual))
            if len(history) >= max_iterations and residual > rtol:
                raise ConvergenceError(f"gmres feedback exceeded {max_iterations} inner iterations; residual={residual:.3g}")
        state, info = gmres(operator, b.ravel(), rtol=rtol, atol=0., maxiter=max_iterations,
                            restart=min(restart, max_iterations), callback=record, callback_type="pr_norm")
        state = state.reshape(b.shape)
        residual = float(np.linalg.norm(state-b-apply(state))/scale)
        if info == 0 and residual <= rtol:
            return FeedbackResult(state, residual, len(history), tuple(map(float, history)), method)
    else:
        raise ValueError("method must be 'successive' or 'gmres'")
    raise ConvergenceError(f"{method} feedback failed: residual={residual:.3g}, tolerance={rtol:.3g}, iterations={len(history)}")
