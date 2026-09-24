"""Polar Fourier transforms: radial Bessel integrals and azimuthal harmonics."""

from __future__ import annotations

import numpy as np
from scipy.special import jv


def radial_weights(radius):
    """Trapezoidal dr weights on an increasing nonnegative radial grid."""
    radius = np.asarray(radius, dtype=float)

    if (
        radius.ndim != 1
        or len(radius) < 2
        or np.any(radius < 0)
        or np.any(np.diff(radius) <= 0)
    ):
        raise ValueError(
            "radius must contain at least two increasing nonnegative nodes"
        )

    if not np.isfinite(radius).all():
        raise ValueError("radial coordinates must be finite")

    steps = np.diff(radius)

    return np.concatenate(
        ([steps[0] / 2], (steps[:-1] + steps[1:]) / 2, [steps[-1] / 2])
    )


def hankel_transform(order: int, radius, values, radial_frequency, *, weights=None):
    """Evaluate integral f(r) J_order(q*r) r dr; q is in radians/metre.

    The last values axis corresponds to radius. Optional weights are dr
    quadrature weights, not r*dr. No 2*pi factor is hidden in this definition.
    """

    if int(order) != order:
        raise ValueError("Hankel order must be an integer")

    radius = np.asarray(radius, dtype=float)
    values = np.asarray(values, dtype=complex)
    frequency = np.atleast_1d(np.asarray(radial_frequency, dtype=float))
    integration_weights = (
        radial_weights(radius) if weights is None else np.asarray(weights, dtype=float)
    )

    if values.shape[-1:] != radius.shape or integration_weights.shape != radius.shape:
        raise ValueError("values and quadrature weights must match the radial grid")

    if not all(
        np.isfinite(a).all() for a in (radius, values, frequency, integration_weights)
    ):
        raise ValueError("transform inputs must be finite")

    kernel = jv(order, frequency[:, None] * radius[None, :])

    return (values * radius * integration_weights) @ kernel.T


def azimuthal_coefficients(values, *, axis=-1, max_order=None, tail_tolerance=1e-10):
    """Fourier series coefficients on uniform phi = 2*pi*j/N rings.

    If max_order is supplied, unresolved energy in omitted coefficients raises
    an error. Trailing components are transformed independently; symmetry of
    the coordinate domain does not remove harmonics of the sampled function.
    """
    values = np.asarray(values, dtype=complex)
    count = values.shape[axis]
    orders = np.rint(np.fft.fftfreq(count) * count).astype(int)
    coefficients = np.fft.fft(values, axis=axis) / count

    if max_order is None:
        return orders, coefficients

    if int(max_order) != max_order or max_order < 0:
        raise ValueError("max_order must be a nonnegative integer")

    keep = np.abs(orders) <= max_order
    discarded = np.take(coefficients, np.flatnonzero(~keep), axis=axis)

    if np.linalg.norm(discarded) > tail_tolerance * max(
        np.linalg.norm(coefficients), 1e-300
    ):
        raise ValueError("azimuthal harmonic tail exceeds the requested tolerance")

    return orders[keep], np.take(coefficients, np.flatnonzero(keep), axis=axis)


def cylindrical_synthesize(
    radial_wavevectors, axial_wavevectors, coefficients, orders, points
):
    """Integrate azimuth analytically for cylindrical vector modal coefficients.

    coefficients has shape (radial_nodes, harmonics, components), including the
    full azimuthal measure and radial quadrature weights. The phase is
    exp(i*(k_r*r*cos(phi-phi_obs) + k_z*z)).
    """
    points = np.asarray(points, dtype=float)
    radius = np.linalg.norm(points[..., :2], axis=-1).ravel()
    azimuth = np.arctan2(points[..., 1], points[..., 0]).ravel()
    axial = points[..., 2].ravel()
    coefficients = np.asarray(coefficients, dtype=complex)
    result = np.zeros((len(radius), coefficients.shape[-1]), dtype=complex)
    phase_axial = np.exp(1j * axial[:, None] * np.asarray(axial_wavevectors)[None, :])

    for index, order in enumerate(orders):
        bessel = jv(order, radius[:, None] * np.asarray(radial_wavevectors)[None, :])
        phase_azimuth = (1j ** int(order)) * np.exp(1j * order * azimuth)
        result += phase_azimuth[:, None] * (
            (phase_axial * bessel) @ coefficients[:, index]
        )

    return result.reshape(points.shape[:-1] + (coefficients.shape[-1],))


def _polar_coordinates(coordinates):
    if coordinates is None or len(coordinates) != 2:
        raise ValueError("polar coordinates must be (radius, azimuth) axes")

    radius, azimuth = (np.asarray(axis, dtype=float) for axis in coordinates)
    radial_weights(radius)

    if azimuth.ndim != 1 or len(azimuth) < 2 or not np.isfinite(azimuth).all():
        raise ValueError("azimuth must be a finite one-dimensional periodic axis")

    expected_step = 2 * np.pi / len(azimuth)

    if not np.allclose(np.diff(azimuth), expected_step, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "azimuth must cover one uniform period without a duplicate endpoint"
        )

    return radius, azimuth


def _polar_transform(values, source, target, weights, *, inverse):
    source_radius, source_azimuth = _polar_coordinates(source)
    target_radius, target_azimuth = _polar_coordinates(target)
    values = np.asarray(values, dtype=complex)

    if values.shape[:2] != (len(source_radius), len(source_azimuth)):
        raise ValueError("polar values must have leading shape (radius, azimuth)")

    if not np.isfinite(values).all():
        raise ValueError("polar transform samples must be finite")

    integration_weights = (
        radial_weights(source_radius)
        if weights is None
        else np.asarray(weights, dtype=float)
    )

    if integration_weights.shape != source_radius.shape:
        raise ValueError("radial integration weights must match the source radial axis")

    if not np.isfinite(integration_weights).all() or np.any(integration_weights <= 0):
        raise ValueError("radial integration weights must be finite and positive")

    components = values.reshape(len(source_radius), len(source_azimuth), -1)
    orders, harmonics = azimuthal_coefficients(components, axis=1)

    # FFT harmonics use phi-phi_0. Restore the explicitly supplied angular origin.
    harmonics *= np.exp(-1j * orders * source_azimuth[0])[None, :, None]

    radial_measure = source_radius * integration_weights
    result = np.zeros(
        (len(target_radius), len(target_azimuth), components.shape[-1]), dtype=complex
    )
    phase_sign = 1j if inverse else -1j
    normalization = 1 / (2 * np.pi) if inverse else 2 * np.pi

    for index, order in enumerate(orders):
        kernel = jv(order, target_radius[:, None] * source_radius[None, :])
        radial_integral = kernel @ (radial_measure[:, None] * harmonics[:, index])
        angular_factor = phase_sign ** int(order) * np.exp(1j * order * target_azimuth)

        result += (
            normalization * radial_integral[:, None, :] * angular_factor[None, :, None]
        )

    return result.reshape((len(target_radius), len(target_azimuth)) + values.shape[2:])


def transform(values, coordinates, frequencies, *, weights=None):
    """Polar continuous Fourier transform with exp(-i q dot r).

    coordinates=(r,phi), frequencies=(q,theta), array order (r,phi,...).
    Each angular Fourier harmonic has its own Bessel order. Optional weights
    integrate dr, not r*dr. Both angular axes cover one full uniform period.
    """
    return _polar_transform(values, coordinates, frequencies, weights, inverse=False)


def inverse(values, coordinates, frequencies, *, weights=None):
    """Polar inverse integral with exp(+i q dot r)/(2*pi)^2.

    coordinates are the requested (r,phi); frequencies describe input (q,theta).
    Optional weights integrate dq. Finite radial support and quadrature make
    this an approximate continuous inverse, not a matrix-inverse correction.
    """
    return _polar_transform(values, frequencies, coordinates, weights, inverse=True)
