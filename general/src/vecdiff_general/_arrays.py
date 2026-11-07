"""Small validation helpers; inputs are copied before being made read-only.

CuPy arrays cannot be flagged read-only, so on the GPU this is a convention."""

from vecdiff_general._backend import bd, is_device_array


def frozen_array(value, *, dtype=float):
    array = bd.array(value, dtype=dtype, copy=True)
    if not bd.isfinite(array).all():
        raise ValueError("arrays must contain finite values")
    if not is_device_array(array):
        array.setflags(write=False)
    return array


def positive(value, name):
    value = float(value)
    if not bd.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def positive_integer(value, name):
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
