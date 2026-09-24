"""Tangential electric-basis admittances for coherent planar layers."""

import numpy as np


def tangential_coefficients(
    normal_wavenumber_1, normal_wavenumber_2, material_1, material_2, vacuum_wavenumber
):
    """s/p coefficients in a fixed tangential E basis, used by layer recursion."""
    first, second = np.broadcast_arrays(normal_wavenumber_1, normal_wavenumber_2)

    if np.any(np.abs(first) == 0) or np.any(np.abs(second) == 0):
        raise ValueError("exact grazing/critical layers need a limiting solution")

    admittance_1 = np.stack(
        (first / vacuum_wavenumber, material_1.n**2 * vacuum_wavenumber / first),
        axis=-1,
    )
    admittance_2 = np.stack(
        (second / vacuum_wavenumber, material_2.n**2 * vacuum_wavenumber / second),
        axis=-1,
    )
    denominator = admittance_1 + admittance_2

    return (admittance_1 - admittance_2) / denominator, 2 * admittance_1 / denominator
