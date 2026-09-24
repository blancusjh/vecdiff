"""Polar Fourier and Hankel integrals, independent of optical objects."""

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


def _radial_measure(radius, weights):
    """Validate radial quadrature and return its r*dr measure."""
    default_weights = radial_weights(radius)
    weights = default_weights if weights is None else np.asarray(weights, dtype=float)

    if weights.shape != radius.shape:
        raise ValueError("radial integration weights must match the radial axis")

    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("radial integration weights must be finite and positive")

    return radius * weights


def HT(g: np.ndarray, r, q, *, order=0, weights=None) -> np.ndarray:
    """Hankel transform G(q) = integral g(r) J_order(q*r) r dr.

    The first axis of g corresponds to r; trailing components are independent.
    q is angular radial frequency. Optional weights integrate dr, not r*dr.
    There is no hidden 2*pi factor. The order is an integer, including negative
    angular harmonics. Radial nodes must increase and be nonnegative.
    """
    if isinstance(order, bool) or not np.isfinite(order) or int(order) != order:
        raise ValueError("Hankel order must be an integer")

    g = np.asarray(g, dtype=complex)
    r = np.asarray(r, dtype=float)
    q = np.atleast_1d(np.asarray(q, dtype=float))
    measure = _radial_measure(r, weights)

    if g.ndim < 1 or g.shape[0] != len(r):
        raise ValueError("the first values axis must match the radial coordinates")

    if q.ndim != 1 or not len(q) or np.any(q < 0):
        raise ValueError("radial frequencies must be a nonempty nonnegative axis")

    if not np.isfinite(g).all() or not np.isfinite(q).all():
        raise ValueError("Hankel inputs must be finite")

    components = g.reshape(len(r), -1)
    bessel = jv(int(order), q[:, None] * r[None, :])
    integrand = measure[:, None] * components

    G = bessel @ integrand

    return G.reshape((len(q),) + g.shape[1:])


def IHT(G: np.ndarray, q, r, *, order=0, weights=None) -> np.ndarray:
    """Inverse Hankel integral g(r) = integral G(q) J_order(q*r) q dq.

    Optional weights integrate dq. The continuous Hankel pair is self-reciprocal;
    its finite quadrature is approximate, not an exact discrete matrix inverse.
    """
    g = HT(G, q, r, order=order, weights=weights)

    return g


def azimuthal_coefficients(g, *, axis=-1, max_order=None, tail_tolerance=1e-10):
    """Fourier series coefficients on uniform phi = 2*pi*j/N rings.

    If max_order is supplied, unresolved energy in omitted coefficients raises
    an error. Trailing components are transformed independently; symmetry of
    the coordinate domain does not remove harmonics of the sampled function.
    """
    g = np.asarray(g, dtype=complex)
    count = g.shape[axis]

    G = np.fft.fft(g, axis=axis)
    coefficients = G / count
    orders = np.rint(np.fft.fftfreq(count) * count).astype(int)

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
        radial_integral = (phase_axial * bessel) @ coefficients[:, index]

        result += phase_azimuth[:, None] * radial_integral

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


def _integrate_harmonics(g, source, target, weights, *, sign):
    r, phi = _polar_coordinates(source)
    q, theta = _polar_coordinates(target)
    g = np.asarray(g, dtype=complex)

    if g.shape[:2] != (len(r), len(phi)):
        raise ValueError("polar values must have leading shape (radius, azimuth)")

    if not np.isfinite(g).all():
        raise ValueError("polar transform values must be finite")

    # Angular Fourier series: g(r,phi) = sum_m c_m(r) exp(i*m*phi).
    components = g.reshape(len(r), len(phi), -1)
    orders, coefficients = azimuthal_coefficients(components, axis=1)
    origin_phase = np.exp(-1j * orders * phi[0])
    coefficients = coefficients * origin_phase[None, :, None]

    G = np.zeros((len(q), len(theta), components.shape[-1]), dtype=complex)

    for index, order in enumerate(orders):
        # Each harmonic has its own radial Hankel integral and angular phase.
        radial_integral = HT(coefficients[:, index], r, q, order=order, weights=weights)
        angular_phase = (sign * 1j) ** int(order) * np.exp(1j * order * theta)

        G += radial_integral[:, None, :] * angular_phase[None, :, None]

    return G.reshape((len(q), len(theta)) + g.shape[2:])


def FT2(g: np.ndarray, coordinates, frequencies, *, weights=None) -> np.ndarray:
    """Polar Fourier integral G(q,theta) = integral g(r,phi) exp(-i*q.r) d²r.

    coordinates=(r,phi), frequencies=(q,theta), array order (r,phi,...).
    q is angular radial frequency. Optional weights integrate dr. Angular axes
    cover a full uniform period, with arbitrary origins and no duplicate end.
    This is angular FFT plus radial quadrature, not a 2D FFT of the polar array.
    """
    G = _integrate_harmonics(g, coordinates, frequencies, weights, sign=-1)
    G = 2 * np.pi * G

    return G


def IFT2(G: np.ndarray, coordinates, frequencies, *, weights=None) -> np.ndarray:
    """Polar inverse g(r,phi) = integral G(q,theta) exp(+i*q.r) d²q / (2*pi)².

    coordinates=(r,phi) describes output; frequencies=(q,theta) describes input.
    Optional weights integrate dq. The angular integral supplies 2*pi; the
    remaining normalization is 1/(2*pi). Finite radial quadrature is approximate.
    """
    g = _integrate_harmonics(G, frequencies, coordinates, weights, sign=1)
    g = g / (2 * np.pi)

    return g
