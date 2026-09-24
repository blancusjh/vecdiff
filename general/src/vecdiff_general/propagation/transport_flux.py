"""Geometrical flux transport between a graph and a parallel plane.

This is the leading eikonal amplitude law, not an extra gain for ASM. It
relates fields at ray-mapped coordinates. Exact homogeneous Q already includes
geometrical spreading in its high-frequency limit. No reference fields enter.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RayPlaneFlux:
    plane_xy: np.ndarray
    path_length: np.ndarray
    area_jacobian: np.ndarray
    amplitude: np.ndarray
    regular: np.ndarray
    caustic_crossed: np.ndarray


def ray_plane_flux(xy, height, gradient, direction, direction_gradient, plane_z=0.0):
    """Return the flux-derived graph-to-plane map in one homogeneous medium.

    All coordinates/directions use the graph's orthonormal local frame.
    direction_gradient[..., component, coordinate] is the derivative of the
    real unit ray direction along each of the two graph coordinates (1/metre).

    Let q=u_xy/u_z, L=plane_z-h, y=x+L*q, J=det(dy/dx). At corresponding
    points E_plane(y) has amplitude |E_surface(x)| times

        sqrt((u_z - u_xy.grad(h)) / (u_z * J)).

    Polarization and exp(i*k*path_length) must also be transported to y.
    Multiplying a field at the original x by this factor is not this map.
    Fresnel transmission/reflection are separate interface operations.

    This single-branch geometrical law is invalid at a caustic or after a
    caustic crossing; no gain cap substitutes for wave propagation there.
    Such samples have regular=False and amplitude=NaN. The complete segment
    is checked, including a focus crossed with positive endpoint Jacobian.
    Local regularity does not establish global injectivity of the ray map.
    """
    arrays = [
        np.asarray(v) for v in (xy, height, gradient, direction, direction_gradient)
    ]
    if any(np.iscomplexobj(v) or not np.isfinite(v).all() for v in arrays):
        raise ValueError("ray geometry must be finite and real")
    xy, height, gradient, u, du = [np.asarray(v, float) for v in arrays]
    shape = height.shape
    if (
        xy.shape != shape + (2,)
        or gradient.shape != shape + (2,)
        or u.shape != shape + (3,)
        or du.shape != shape + (3, 2)
    ):
        raise ValueError("ray geometry arrays have incompatible shapes")
    plane_z = np.asarray(plane_z)
    if plane_z.shape != () or np.iscomplexobj(plane_z) or not np.isfinite(plane_z):
        raise ValueError("plane_z must be a finite real scalar")
    if not np.allclose(np.sum(u * u, axis=-1), 1, rtol=0, atol=1e-9):
        raise ValueError("ray directions must be unit vectors")
    orthogonal = np.einsum("...i,...ij->...j", u, du)
    if np.any(
        np.linalg.norm(orthogonal, axis=-1)
        > 1e-8 * np.maximum(np.linalg.norm(du, axis=(-2, -1)), 1e-300)
    ):
        raise ValueError("direction derivatives must be tangent to the unit sphere")
    forward = u[..., 2] > 1e-10
    uz = np.where(forward, u[..., 2], 1.0)
    q = u[..., :2] / uz[..., None]
    dq = (du[..., :2, :] - q[..., :, None] * du[..., 2, None, :]) / uz[..., None, None]
    length = (plane_z - height) / uz
    jacobian = (
        np.eye(2)
        - q[..., :, None] * gradient[..., None, :]
        + (plane_z - height)[..., None, None] * dq
    )
    determinant = np.linalg.det(jacobian)

    # Projected area of a ray tube at any fraction t of this segment is a
    # quadratic polynomial. Terms from derivatives of path length lie along
    # u and vanish in the scalar triple product.
    ru = np.stack((np.ones(shape), np.zeros(shape), gradient[..., 0]), -1)
    rv = np.stack((np.zeros(shape), np.ones(shape), gradient[..., 1]), -1)

    def dot(a):
        return np.sum(u * a, axis=-1)

    c0 = dot(np.cross(ru, rv))
    c1 = length * dot(np.cross(du[..., :, 0], rv) + np.cross(ru, du[..., :, 1]))
    c2 = length**2 * dot(np.cross(du[..., :, 0], du[..., :, 1]))
    vertex = np.clip(-c1 / (2 * np.where(c2 > 0, c2, 1.0)), 0, 1)
    minimum = np.minimum(c0, c0 + c1 + c2)
    minimum = np.where(
        c2 > 0, np.minimum(minimum, c0 + c1 * vertex + c2 * vertex**2), minimum
    )
    caustic = minimum <= 1e-10 * np.maximum(abs(c0), 1e-300)
    regular = forward & (c0 > 1e-10) & (determinant > 1e-10) & ~caustic
    amplitude = np.full(shape, np.nan)
    np.sqrt(
        np.divide(c0, uz * determinant, out=np.zeros(shape), where=regular),
        out=amplitude,
        where=regular,
    )
    return RayPlaneFlux(
        xy + (plane_z - height)[..., None] * q,
        length,
        determinant,
        amplitude,
        regular,
        caustic,
    )
