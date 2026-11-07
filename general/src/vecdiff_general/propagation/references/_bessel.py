"""Bessel functions of large order and argument for the separable references.

Riccati-Bessel functions psi_n(x) = x j_n(x), xi_n(x) = x h1_n(x) (spherical)
and Bessel functions J_n(x), H1_n(x) (cylindrical) for n = 0..count at real
x > 0. A function is returned with its logarithmic scale, so neither the
regular functions, which decay beyond n = x, nor the Hankel functions, which
grow there, under- or overflow at orders far beyond the argument.

The regular functions come from Miller's downward recurrence, normalized by
the exact zeroth or first order; the Hankel functions from the upward
recurrence of their ratios, which is the dominant direction. The imaginary
part of a Hankel logarithmic derivative is the Wronskian divided by |f|²,
1 / |xi|² or 2 / (pi x |H|²): taken from there it keeps its relative accuracy
where it is far smaller than the real part.
"""

import numpy as np
from scipy.special import hankel1, j0, j1

# Rescaling threshold of the downward recurrence; one step grows by < 1e100.
_HUGE = 1e200
_LOG_HUGE = np.log(_HUGE)


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


def outgoing(count, x, spherical=True):
    """log xi_n(x) (spherical) or log H1_n(x) (cylindrical), n = 0..count.

    Returns the complex logarithm (continuous in n) and the logarithmic
    derivative f'_n / f_n, shape (count + 1,) + x.shape.
    """
    x = np.asarray(x, float)
    if np.any(x <= 0) or not np.all(np.isfinite(x)):
        raise ValueError("Bessel arguments must be finite and positive")
    step = _step(spherical)
    log = np.empty((count + 1,) + x.shape, complex)
    derivative = np.empty_like(log)
    if spherical:
        log[0] = -0.5j * np.pi + 1j * x
        ratio = 1 / x - 1j
        derivative[0] = 1j
    else:
        h0, h1 = hankel1(0, x), hankel1(1, x)
        log[0] = np.log(h0)
        ratio = h1 / h0
        derivative[0] = -ratio
    for n in range(1, count + 1):
        if n > 1:
            ratio = step(n - 1, x) - 1 / ratio
        log[n] = log[n - 1] + np.log(ratio)
        derivative[n] = 1 / ratio - n / x
    wronskian = 1.0 if spherical else 2 / (np.pi * x)
    derivative = derivative.real + 1j * wronskian * np.exp(-2 * log.real)
    return log, derivative
