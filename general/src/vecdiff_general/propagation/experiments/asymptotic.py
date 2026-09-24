"""Derivative adapter for auditing supplied inverse-wavenumber coefficients.

This constructs no interface coefficients and does not alter Fresnel T. A
coefficient generator must supply physical Cartesian derivatives, including
normal derivatives from its eikonal/transport continuation.
"""

import numpy as np

from ...fields import ElectricField
from ...observables.maxwell import ElectricFieldJet


def wkb_electric_jet(
    grid,
    wavelength0,
    phase,
    phase_gradient,
    phase_hessian,
    amplitudes,
    amplitude_gradients,
    amplitude_hessians,
    *,
    order,
):
    """Differentiate exp(i k0 S) sum_{m=0}^N A_m/(i k0)^m completely.

    Arrays have leading order axis followed by grid axes; derivative axes
    precede Cartesian vector components. Phase is optical path in metres.
    Request derivative depth independently of retained order N. H obtained
    from the returned jet includes curl(A_N)/(i k0)^(N+1).
    """
    if not np.isfinite(wavelength0) or wavelength0 <= 0:
        raise ValueError("wavelength0 must be finite and positive")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 0:
        raise ValueError("order must be a nonnegative integer")
    arrays = [
        np.asarray(v)
        for v in (
            phase,
            phase_gradient,
            phase_hessian,
            amplitudes,
            amplitude_gradients,
            amplitude_hessians,
        )
    ]
    s, w, ds, a, da, dda = arrays
    if any(not np.isfinite(v).all() for v in arrays):
        raise ValueError("phase and amplitude derivatives must be finite")
    if (
        s.shape != grid.shape
        or w.shape != grid.shape + (3,)
        or ds.shape != grid.shape + (3, 3)
    ):
        raise ValueError("phase derivatives must match the physical grid")
    for v, tail in ((a, (3,)), (da, (3, 3)), (dda, (3, 3, 3))):
        if (
            v.ndim != 1 + len(grid.shape) + len(tail)
            or v.shape[0] <= order
            or v.shape[1:] != grid.shape + tail
        ):
            raise ValueError(
                "amplitudes need an order axis and complete Cartesian derivatives"
            )
    ik = 2j * np.pi / wavelength0
    powers = ik ** -np.arange(order + 1)
    b, db, ddb = [
        np.einsum("m,m...->...", powers, v[: order + 1]) for v in (a, da, dda)
    ]
    exponential = np.exp(ik * s)
    gradient = db + ik * w[..., :, None] * b[..., None, :]
    hessian = (
        ddb
        + ik
        * (
            w[..., :, None, None] * db[..., None, :, :]
            + w[..., None, :, None] * db[..., :, None, :]
            + ds[..., :, :, None] * b[..., None, None, :]
        )
        + ik**2 * w[..., :, None, None] * w[..., None, :, None] * b[..., None, None, :]
    )
    return ElectricFieldJet(
        ElectricField(grid, wavelength0, exponential[..., None] * b),
        exponential[..., None, None] * gradient,
        exponential[..., None, None, None] * hessian,
    )
