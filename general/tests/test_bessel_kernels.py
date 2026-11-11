"""Hankel kernels of the separable references against 50-digit mpmath."""

import numpy as np
import pytest
from vecdiff.references import _bessel


def _reference(spherical, x, orders):
    """log f_n and f'_n / f_n by the upward recurrence at 50 digits.

    Upward is the dominant direction, so the recurrence loses a few digits at
    most; it is seeded by xi_0, xi_1 in closed form or mpmath's H1_0, H1_1.
    """
    mpmath = pytest.importorskip("mpmath")
    with mpmath.workdps(50):
        X = mpmath.mpf(x)
        if spherical:
            previous = -1j * mpmath.expj(X)
            current = previous * (1 / X - 1j)
        else:
            previous = mpmath.hankel1(0, X, maxprec=100000)
            current = mpmath.hankel1(1, X, maxprec=100000)
        log, dlog, wanted = [], [], set(int(n) for n in orders)
        for n in range(1, int(max(orders)) + 1):
            if n > 1:
                c = (2 * n - 1) / X if spherical else 2 * (n - 1) / X
                previous, current = current, c * current - previous
            if n in wanted:
                log.append(complex(mpmath.log(current)))
                dlog.append(complex(previous / current - n / X))
    return np.array(log), np.array(dlog)


@pytest.mark.parametrize("spherical", [True, False], ids=["xi", "H1"])
@pytest.mark.parametrize("x", [1e-6, 0.5, 30.0, 700.0, 3000.0, 9400.0, 3e4])
def test_outgoing_keeps_its_relative_accuracy_at_large_argument_and_order(spherical, x):
    """Orders 1..1.3 x + 50, through the turning point to where f_n overflows.

    The error of the logarithm is the relative error of the value; where
    |f_n| is far beyond the float range it is bounded by roundoff of the
    logarithm itself, eps |Re log|. The summed-ratio logarithm reached 1e-10
    at x = 3e4 (an absolute roundoff of order eps x in its phase).
    """
    top = int(1.3 * x + 50)
    orders = np.unique(
        np.r_[
            np.linspace(1, top, 30).astype(int),
            np.clip(int(x) + np.arange(-5, 6), 1, top),
        ]
    )
    log, dlog = _bessel.outgoing(top, np.array([x]), spherical=spherical)
    log, dlog = log[orders, 0], dlog[orders, 0]
    exact_log, exact_dlog = _reference(spherical, x, orders)
    difference = log - exact_log
    difference = difference.real + 1j * np.angle(np.exp(1j * difference.imag))
    assert np.all(np.abs(log.imag) <= np.pi)
    assert np.all(np.abs(difference) <= 1e-13 * np.maximum(1, np.abs(exact_log.real)))
    assert np.all(np.abs(dlog - exact_dlog) <= 2e-13 * np.abs(exact_dlog))
    # Im D is the Wronskian over |f|², to relative accuracy where it is tiny.
    wronskian = 1.0 if spherical else 2 / (np.pi * x)
    tiny = wronskian * np.exp(-2 * exact_log.real)
    normal = tiny > 1e-300
    assert np.all(np.abs(dlog.imag[normal] - tiny[normal]) <= 3e-13 * tiny[normal])


def test_outgoing_values_are_the_rescaled_mantissas_of_outgoing():
    x = np.array([0.3, 40.0, 900.0])
    value, derivative = _bessel.outgoing_values(1000, x)
    log, dlog = _bessel.outgoing(1000, x)
    finite = np.isfinite(value) & np.isfinite(derivative)
    assert finite[:, 2].all() and not finite[:, 0].all()
    value, derivative, log, dlog = (t[finite] for t in (value, derivative, log, dlog))
    assert np.all(
        np.abs(np.log(value) - log) <= 1e-13 * np.maximum(1, np.abs(log.real))
    )
    assert np.all(np.abs(derivative / value - dlog) <= 1e-13 * np.abs(dlog))
