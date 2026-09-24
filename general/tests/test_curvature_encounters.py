"""Independent total-order bookkeeping and planar reference limits."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from vecdiff_general.propagation.experiments.encounters import layer_returns
from vecdiff_general.propagation.references.periodic_layer import PeriodicLayerReference


def test_encounters_preserve_total_order_against_polynomial_products():
    entry = np.array([0.7, 0.03, -0.002, 0.0003], complex)
    upper = np.array([0.3, 0.01, 0.003, -0.0005], complex)
    lower = np.array([-0.4, 0.02, -0.005, 0.0007], complex)
    phase = np.exp(0.7j)
    source = np.zeros((4, 1), complex)
    source[0] = 1
    up, down = layer_returns(
        entry[:, None, None],
        upper[:, None, None],
        lower[:, None, None],
        np.array([phase]),
        source,
        pairs=5,
    )
    expected_up = np.zeros(4, complex)
    expected_down = np.zeros(4, complex)
    state = entry
    for _ in range(5):
        expected_up += state[:4]
        returning = np.polynomial.polynomial.polymul(upper, state) * phase
        expected_down += returning[:4]
        state = np.polynomial.polynomial.polymul(lower, returning) * phase
    assert_allclose(up[:, 0], expected_up, atol=1e-15)
    assert_allclose(down[:, 0], expected_down, atol=1e-15)


@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_independent_periodic_reference_exact_resonant_flat_slab(polarization):
    def zero(x):
        return np.zeros_like(x)

    reference = PeriodicLayerReference(
        7.4, 5 / 6, 1, 1, 3, zero, zero, zero, zero, polarization, 8
    )
    expected_t = np.zeros(17, complex)
    expected_t[8] = -1
    assert_allclose(reference.transmitted, expected_t, atol=2e-13)
    assert_allclose(reference.reflected, 0, atol=2e-13)
    assert reference.solve_residual < 1e-13
    assert_allclose(reference.up[8], 2 / 3 if polarization == "TE" else 2, atol=1e-13)
    assert_allclose(
        reference.down[8], -1 / 3 if polarization == "TE" else 1, atol=1e-13
    )
