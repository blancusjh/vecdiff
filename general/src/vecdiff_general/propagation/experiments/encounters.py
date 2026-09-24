"""Compose encounter count independently of inverse-wavenumber order."""

import numpy as np


def apply_orders(operator, amplitudes):
    """b_m = sum_{j=0}^m T_j a_{m-j}; never retain orders above the budget."""
    operator, amplitudes = np.asarray(operator), np.asarray(amplitudes)
    if (
        operator.ndim != 3
        or amplitudes.ndim != 2
        or operator.shape[0] != amplitudes.shape[0]
        or operator.shape[2] != amplitudes.shape[1]
    ):
        raise ValueError("expected (orders,out,in) operator and (orders,in) state")
    result = np.zeros((len(amplitudes), operator.shape[1]), complex)
    for m in range(len(amplitudes)):
        for j in range(m + 1):
            result[m] += operator[j] @ amplitudes[m - j]
    return result


def layer_returns(
    entry, upper_reflection, lower_reflection, propagation, incident, *, pairs
):
    """Internal up/down spectra including zero through 2*pairs-1 returns.

    Upward coefficients are referenced to the lower mean plane, downward ones
    to the upper mean plane. Each propagation is through the same homogeneous
    layer. No spectral-radius assumption or infinite-sum convergence is made.
    """
    if not isinstance(pairs, int) or pairs < 1:
        raise ValueError("pairs must be a positive integer")
    state = apply_orders(entry, incident)
    up, down = np.zeros_like(state), np.zeros_like(state)
    propagation = np.asarray(propagation)
    if propagation.shape != (state.shape[1],):
        raise ValueError("propagation must supply one phase per internal mode")
    for _ in range(pairs):
        up += state
        returning = apply_orders(upper_reflection, state * propagation)
        down += returning
        state = apply_orders(lower_reflection, returning * propagation)
    return up, down
