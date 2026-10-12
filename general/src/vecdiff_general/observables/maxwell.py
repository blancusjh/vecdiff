"""Maxwell audits using physical Cartesian derivatives and area quadrature.

These diagnostics accept the field actually evaluated, including all derivatives
of a retained approximation. Surface-coordinate derivatives are not Cartesian
derivatives. No boundary identity is inferred from the field's construction.
"""

from dataclasses import dataclass

import numpy as np

from vecdiff_general._arrays import frozen_array, positive
from vecdiff_general.fields import ElectricField, ElectromagneticField
from vecdiff_general.media.medium import Z0


def _norm(values, weights):
    squared = np.abs(values) ** 2
    if squared.ndim == weights.ndim + 1:
        squared = squared.sum(axis=-1)
    if squared.shape != weights.shape:
        raise ValueError("quadrature and field shapes must agree")
    return float(np.sqrt(np.sum(weights * squared)))


def _weights(weights, shape):
    weights = np.asarray(weights, float)
    if weights.shape != shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("quadrature weights must be finite, positive and match points")
    return weights


@dataclass(frozen=True)
class ElectricFieldJet:
    """E, d_i E_j and d_i d_j E_l at physical points, in SI units.

    Derivative axes precede the vector-component axis. Derivatives must be from
    one side of an interface and in a single Cartesian frame. Hessians commute
    within a smooth homogeneous region. H is derived from the COMPLETE curl.
    """

    electric: ElectricField
    gradient: np.ndarray
    hessian: np.ndarray

    def __post_init__(self):
        shape = self.electric.grid.shape
        for name, tail in (("gradient", (3, 3)), ("hessian", (3, 3, 3))):
            value = frozen_array(getattr(self, name), dtype=complex)
            if value.shape != shape + tail:
                raise ValueError(f"{name} has incorrect derivative shape")
            object.__setattr__(self, name, value)

    @property
    def curl(self):
        g = self.gradient
        return np.stack(
            (
                g[..., 1, 2] - g[..., 2, 1],
                g[..., 2, 0] - g[..., 0, 2],
                g[..., 0, 1] - g[..., 1, 0],
            ),
            axis=-1,
        )

    @property
    def field(self):
        k0 = 2 * np.pi / self.electric.wavelength0
        return ElectromagneticField(self.electric, self.curl / (1j * k0 * Z0))

    def bulk_residuals(self, index, weights):
        """Relative L2 divergence, Helmholtz and full curl-curl residuals.

        Norms are divided by k||E|| or k²||E||. This tests the smooth local
        continuation, not radiation conditions or global boundary-value stability.
        Faraday's law is an identity when H is derived by ``field``.
        """
        k = 2 * np.pi * positive(index, "index") / self.electric.wavelength0
        weights = _weights(weights, self.electric.grid.shape)
        scale = _norm(self.electric.values, weights)
        if scale == 0:
            raise ValueError("relative residual needs a nonzero electric field")
        divergence = np.trace(self.gradient, axis1=-2, axis2=-1)
        laplacian = np.einsum("...iij->...j", self.hessian)
        grad_div = np.einsum("...ijj->...i", self.hessian)
        helmholtz = laplacian + k**2 * self.electric.values
        return {
            "divergence_E": _norm(divergence, weights) / (k * scale),
            "helmholtz_E": _norm(helmholtz, weights) / (k**2 * scale),
            "curl_curl_E": _norm(grad_div - helmholtz, weights) / (k**2 * scale),
        }


def plane_wave_jet(expansion, grid, *, chunk_size=256):
    """Direct modal sum and exact Cartesian derivatives, independent of Q/FFT.

    Intended for verification on modest point sets. Empty modes are skipped;
    the two derivatives use iK and -K tensor K, including complex wavevectors.
    """
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    occupied = np.any(expansion.amplitudes != 0, axis=-1)
    k = expansion.wavevectors[occupied]
    a = expansion.amplitudes[occupied]
    points = (grid.points - expansion.origin).reshape(-1, 3)
    electric = np.empty((len(points), 3), complex)
    gradient = np.empty((len(points), 3, 3), complex)
    hessian = np.empty((len(points), 3, 3, 3), complex)
    da = 1j * k[:, :, None] * a[:, None, :]
    dda = -np.einsum("mi,mj,ml->mijl", k, k, a)
    for start in range(0, len(points), chunk_size):
        block = slice(start, start + chunk_size)
        phase = np.exp(1j * points[block] @ k.T)
        electric[block] = phase @ a
        gradient[block] = (phase @ da.reshape(len(k), 9)).reshape(-1, 3, 3)
        hessian[block] = (phase @ dda.reshape(len(k), 27)).reshape(-1, 3, 3, 3)
    return ElectricFieldJet(
        ElectricField(grid, expansion.wavelength0, electric.reshape(grid.shape + (3,))),
        gradient.reshape(grid.shape + (3, 3)),
        hessian.reshape(grid.shape + (3, 3, 3)),
    )


def interface_l2_residuals(minus, plus, incident, grid, *, n_minus, n_plus):
    """Same-point one-sided Maxwell jumps, integrated with physical dS.

    Normalize E, Z0 H and Z0 B/mu0 by ||E_inc||_dS; D/epsilon0 also by
    n_minus². This follows the supplied audit's convention (no extra n_minus
    factor in H). Compare only equal conventions. ``minus`` is total incident
    plus reflected field. Report signed local Poynting mismatch separately;
    interference is retained before computing total flux.
    """
    n_minus, n_plus = positive(n_minus, "n_minus"), positive(n_plus, "n_plus")
    weights = _weights(grid.weights, grid.shape)
    for value in (minus, plus, incident):
        if value.electric.wavelength0 != incident.electric.wavelength0:
            raise ValueError("one-sided fields must have the same wavelength")
        if not np.array_equal(value.electric.grid.points, grid.points):
            raise ValueError("one-sided fields must use identical physical points")
    norm = _norm(incident.electric.values, weights)
    if norm == 0:
        raise ValueError("relative boundary residual needs nonzero illumination")
    normals = grid.normals
    de = minus.electric.values - plus.electric.values
    dh = Z0 * (minus.magnetic - plus.magnetic)
    dd = n_minus**2 * minus.electric.values - n_plus**2 * plus.electric.values
    incoming = np.sum(incident.poynting() * normals, axis=-1)
    jump = np.sum((minus.poynting() - plus.poynting()) * normals, axis=-1)
    flux_scale = _norm(incoming, weights)
    return {
        "tangential_E": _norm(np.cross(normals, de), weights) / norm,
        "tangential_H": _norm(np.cross(normals, dh), weights) / norm,
        "normal_D": _norm(np.sum(normals * dd, axis=-1), weights) / (n_minus**2 * norm),
        "normal_B": _norm(np.sum(normals * dh, axis=-1), weights) / norm,
        "normal_flux_L2": _norm(jump, weights) / flux_scale if flux_scale else None,
    }
