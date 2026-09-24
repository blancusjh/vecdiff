"""Local orders 0..3 against exact single-interface cylinder and sphere fields.

These are independent Maxwell solutions (Bessel/Hankel partial waves), not
residual checks of the recurrence. Near the vertex the error through order N
must fall as (kR)^-(N+1): halving it for N=0, dividing by 16 for N=3.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from vecdiff_general.media.medium import Z0
from vecdiff_general.propagation.experiments.taylor_interface import interface_expansion
from vecdiff_general.propagation.references.debye import (
    cylinder_single_interface,
    sphere_single_interface,
)

UV = np.array([[0, 0], [0.2, 0.1], [-0.15, 0.25], [0.3, -0.2], [-0.25, -0.05]])


def _errors(kR, reference, height, n1, n2, uv, amplitude):
    wavelength = 1.0
    radius = kR / (2 * np.pi)
    expansion = interface_expansion(
        height,
        uv * radius,
        n_incident=n1,
        n_transmitted=n2,
        direction=(0, 0, 1),
        amplitude=amplitude,
        order=3,
        length_scale=radius,
    )
    exact = reference(expansion.points, wavelength, radius)
    errors = np.zeros((4, 2))
    for b, branch in enumerate(("reflected", "transmitted")):
        e, h = exact[branch]
        scale = np.sum(abs(e) ** 2 + abs(Z0 * h) ** 2)
        for n in range(4):
            t = expansion.trace(branch, wavelength, order=n)
            d = np.sum(
                abs(t.electric.values - e) ** 2 + abs(Z0 * (t.magnetic - h)) ** 2
            )
            errors[n, b] = np.sqrt(d / scale)
    return errors, expansion, exact


def _assert_rates(coarse, fine):
    ratios = coarse / fine
    for n in range(4):
        assert_allclose(ratios[n], 2 ** (n + 1), rtol=0.3)


@pytest.mark.parametrize("interior", [False, True])
def test_sphere_orders_converge_at_asymptotic_rate(interior):
    sign = -1 if interior else 1

    def height(u, v):
        return sign * (1 - (1 - u * u - v * v) ** 0.5)

    def reference(points, wavelength, radius):
        return sphere_single_interface(
            points, wavelength, radius, 1.0, 1.5, interior=interior
        )

    coarse, *_ = _errors(50, reference, height, 1.0, 1.5, UV, (1, 0, 0))
    fine, *_ = _errors(100, reference, height, 1.0, 1.5, UV, (1, 0, 0))
    _assert_rates(coarse, fine)
    assert fine[3].max() < 2e-6
    assert fine[0].max() < 6e-3


@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_cylinder_orders_converge_at_asymptotic_rate(polarization):
    uv = np.column_stack((np.linspace(-0.3, 0.3, 7), np.zeros(7)))

    def height(u, v):
        return 1 - (1 - u * u) ** 0.5

    def reference(points, wavelength, radius):
        return cylinder_single_interface(
            points, wavelength, radius, 1.0, 1.5, polarization
        )

    amplitude = (0, 1, 0) if polarization == "TE" else (1, 0, 0)
    coarse, *_ = _errors(50, reference, height, 1.0, 1.5, uv, amplitude)
    fine, *_ = _errors(100, reference, height, 1.0, 1.5, uv, amplitude)
    _assert_rates(coarse, fine)
    assert fine[3].max() < 1e-6


def test_exact_reference_satisfies_interface_conditions():
    def height(u, v):
        return -(1 - (1 - u * u - v * v) ** 0.5)

    def reference(points, wavelength, radius):
        return sphere_single_interface(
            points, wavelength, radius, 1.5, 1.0, interior=True
        )

    _, expansion, exact = _errors(60, reference, height, 1.5, 1.0, UV, (1, 0, 0))
    n = expansion.normals
    for field in (0, 1):
        minus = exact["incident"][field] + exact["reflected"][field]
        jump = np.cross(n, minus - exact["transmitted"][field])
        scale = Z0 if field else 1
        assert abs(jump).max() * scale < 1e-12
