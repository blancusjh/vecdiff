"""Small validation helpers; inputs are copied before being made read-only."""

import numpy as np


def frozen_array(value, *, dtype=float):
    array = np.array(value, dtype=dtype, copy=True)
    if not np.isfinite(array).all():
        raise ValueError("arrays must contain finite values")
    array.setflags(write=False)
    return array


def positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def positive_integer(value, name):
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
