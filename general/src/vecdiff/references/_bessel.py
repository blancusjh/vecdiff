"""Bessel functions of large order and argument for the separable references.

Riccati-Bessel functions psi_n(x) = x j_n(x), xi_n(x) = x h1_n(x) (spherical)
and Bessel functions J_n(x), H1_n(x) (cylindrical) for n = 0..count at real
x > 0. A function is returned with its logarithmic scale, so neither the
regular functions, which decay beyond n = x, nor the Hankel functions, which
grow there, under- or overflow at orders far beyond the argument.

The regular functions come from Miller's downward recurrence, normalized by
the exact zeroth or first order; the Hankel functions from the upward
recurrence of their values, the dominant direction, rescaled by exact powers
of two. The imaginary part of a Hankel logarithmic derivative is the
Wronskian divided by |f|², 1 / |xi|² or 2 / (pi x |H|²): taken from there it
keeps its relative accuracy where it is far smaller than the real part.
"""

import numpy as np
from scipy.special import hankel1, j0, j1

# Rescaling threshold of the downward recurrence; one step grows by < 1e100.
_HUGE = 1e200
_LOG_HUGE = np.log(_HUGE)
# Rescaling threshold of the upward recurrence, a power of two.
_BINARY_HUGE = 2.0**500
_LOG2 = np.log(2.0)


def _step(spherical):
    # Coefficient c_n of f_{n-1} + f_{n+1} = c_n f_n.
    return (lambda n, x: (2 * n + 1) / x) if spherical else (lambda n, x: 2 * n / x)


def regular(count, x, spherical=True):
    """psi_n(x) (spherical) or J_n(x) (cylindrical) for n = 0..count.

    Returns value, derivative, scale with shape (count + 1,) + x.shape:
    f_n = value exp(scale), f'_n = derivative exp(scale), and
    max(|value|, |derivative|) = 1.
    """
    x = np.asarray(x, float)
    if np.any(x <= 0) or not np.all(np.isfinite(x)):
        raise ValueError("Bessel arguments must be finite and positive")
    step = _step(spherical)
    big = max(float(x.max(initial=0.0)), 1.0)
    top = int(max(count, big) + 10 * big ** (1 / 3) + 40)
    value = np.empty((count + 2,) + x.shape)
    scale = np.empty((count + 2,) + x.shape)
    above, current = np.zeros(x.shape), np.full(x.shape, 1e-300)
    level = np.zeros(x.shape)
    for n in range(top, 0, -1):
        if n <= count + 1:
            value[n], scale[n] = current, level
        below = step(n, x) * current - above
        above, current = current, below
        large = np.abs(current) > _HUGE
        if np.any(large):
            above = np.where(large, above / _HUGE, above)
            current = np.where(large, current / _HUGE, current)
            level = level + np.where(large, _LOG_HUGE, 0.0)
    value[0], scale[0] = current, level
    if spherical:
        exact0, exact1 = np.sin(x), np.sin(x) / x - np.cos(x)
    else:
        exact0, exact1 = j0(x), j1(x)
    first = np.abs(exact0) >= np.abs(exact1)
    exact = np.where(first, exact0, exact1)
    reference = np.where(first, value[0], value[1])
    level = np.where(first, scale[0], scale[1])
    sign = np.sign(exact) * np.sign(reference)
    scale = scale + (np.log(np.abs(exact)) - np.log(np.abs(reference)) - level)
    value = sign * value
    # f'_n = f_{n-1} - (n / x) f_n, both expressed at the scale of order n.
    n = np.arange(1, count + 1).reshape((-1,) + (1,) * x.ndim)
    derivative = np.empty_like(value[: count + 1])
    derivative[1:] = (
        value[:count] * np.exp(scale[:count] - scale[1 : count + 1])
        - (n / x) * value[1 : count + 1]
    )
    if spherical:
        derivative[0] = np.cos(x) * np.exp(-scale[0])
    else:
        derivative[0] = -value[1] * np.exp(scale[1] - scale[0])
    value = value[: count + 1]
    scale = scale[: count + 1]
    size = np.maximum(np.abs(value), np.abs(derivative))
    return value / size, derivative / size, scale + np.log(size)


def _hankel(count, x, spherical):
    """f_n = value_n 2^exponent_n, n = 0..count, by the upward recurrence.

    Upward is the dominant direction for the Hankel functions: each step adds
    a roundoff of order eps relative to the function, so the values keep a
    relative accuracy of a few times sqrt(n) eps, with their phase in the
    mantissa. A pair is rescaled by a power of two, exactly, once a value
    exceeds 2^500 (beyond n = x, where f_n grows), so the mantissas are
    bitwise those of the unscaled recurrence and nothing overflows while one
    step grows by less than 2^500 (2 count / x < 1e150).
    """
    step = _step(spherical)
    value = np.empty((count + 1,) + x.shape, complex)
    exponent = np.zeros((count + 1,) + x.shape)
    if spherical:
        value[0] = -1j * np.exp(1j * x)
        current = value[0] * (1 / x - 1j)
    else:
        value[0] = hankel1(0, x)
        current = hankel1(1, x)
    previous, level = value[0], np.zeros(x.shape)
    for n in range(1, count + 1):
        if n > 1:
            previous, current = current, step(n - 1, x) * current - previous
        size = np.maximum(np.abs(current.real), np.abs(current.imag))
        large = size > _BINARY_HUGE
        if np.any(large):
            shift = np.where(large, np.frexp(size)[1], 0)
            factor = np.exp2(-shift)
            previous, current = previous * factor, current * factor
            level = level + shift
        value[n], exponent[n] = current, level
    return value, exponent


def outgoing(count, x, spherical=True):
    """log xi_n(x) (spherical) or log H1_n(x) (cylindrical), n = 0..count.

    Returns the complex logarithm, imaginary part in (-pi, pi], and the
    logarithmic derivative f'_n / f_n, shape (count + 1,) + x.shape.

    The logarithm is taken of the mantissa of the upward recurrence of the
    values (_hankel), plus its binary exponent times log 2. A logarithm
    continuous in n, a sum of log ratios, carries a phase of order x and so
    an absolute roundoff of order eps x in every term (4e-11 at x = 9400);
    here the phase is reduced to (-pi, pi] and the error of the logarithm is
    the relative error of the value, a few times 1e-14 to x = 3e4, plus eps
    |Re log| from the exponent where the function is far beyond the float
    range.
    """
    x = np.asarray(x, float)
    if np.any(x <= 0) or not np.all(np.isfinite(x)):
        raise ValueError("Bessel arguments must be finite and positive")
    value, exponent = _hankel(count, x, spherical)
    log = np.log(value) + _LOG2 * exponent
    derivative = np.empty_like(log)
    # f'_n / f_n = f_{n-1} / f_n - n / x (f'_0 / f_0 = i, -H1_1 / H1_0).
    if spherical:
        derivative[0] = 1j
    elif count:
        derivative[0] = -value[1] / value[0] * np.exp2(exponent[1])
    else:
        derivative[0] = -hankel1(1, x) / value[0]
    n = np.arange(1, count + 1).reshape((-1,) + (1,) * x.ndim)
    derivative[1:] = (
        value[:-1] / value[1:] * np.exp2(exponent[:-1] - exponent[1:]) - n / x
    )
    wronskian = 1.0 if spherical else 2 / (np.pi * x)
    derivative = derivative.real + 1j * wronskian * np.exp(-2 * log.real)
    return log, derivative


def outgoing_values(count, x):
    """xi_n(x) and xi_n'(x), n = 0..count, by the upward recurrence of values.

    Shape (count + 1,) + x.shape. Each value keeps a relative accuracy of
    about 1e-14 up to x ~ 1e4 (the mantissas of ``outgoing``); beyond n = x
    the values overflow to infinity where xi_n exceeds the float range.
    """
    x = np.asarray(x, float)
    if np.any(x <= 0) or not np.all(np.isfinite(x)):
        raise ValueError("Bessel arguments must be finite and positive")
    mantissa, exponent = _hankel(count, x, True)
    exponent = exponent.astype(int)
    value = np.empty_like(mantissa)
    with np.errstate(over="ignore", invalid="ignore"):
        # Per component, so a part within the float range stays finite.
        value.real = np.ldexp(mantissa.real, exponent)
        value.imag = np.ldexp(mantissa.imag, exponent)
        derivative = np.empty_like(value)
        derivative[0] = np.exp(1j * x)
        n = np.arange(1, count + 1).reshape((-1,) + (1,) * x.ndim)
        derivative[1:] = value[:-1] - (n / x) * value[1:]
    return value, derivative
