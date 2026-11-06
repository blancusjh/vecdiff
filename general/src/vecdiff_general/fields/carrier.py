"""Reference ray families: the declared carrier of a high-frequency field.

A carrier is a geometric path function psi(r) in metres with |grad psi| = 1.
Its gradient is the unit ray direction s(r) and its Hessian is the direction
Jacobian ds_i/dr_j. In a homogeneous region of index n the reference eikonal
(optical path) is n psi, so one carrier serves every region it crosses. It is
declared with a field, never inferred from an interfering total field.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from vecdiff_general._arrays import frozen_array


@dataclass(frozen=True)
class Carrier:
    """path(points), gradient(points) and hessian(points) for points of shape (..., 3)."""

    path: Callable
    gradient: Callable
    hessian: Callable

    def __post_init__(self):
        if not all(callable(f) for f in (self.path, self.gradient, self.hessian)):
            raise TypeError("carrier path, gradient and hessian must be callable")

    @classmethod
    def converging(cls, point):
        """Rays toward ``point``; psi = -|point - r| vanishes there."""
        point = _position(point)

        def gradient(r):
            return _unit(point - np.asarray(r, float))

        def hessian(r):
            d = point - np.asarray(r, float)
            return -_projector(_unit(d)) / np.linalg.norm(d, axis=-1)[..., None, None]

        return cls(
            lambda r: -np.linalg.norm(point - np.asarray(r, float), axis=-1),
            gradient,
            hessian,
        )

    @classmethod
    def diverging(cls, point):
        """Rays from ``point``; psi = |r - point| vanishes there."""
        point = _position(point)

        def gradient(r):
            return _unit(np.asarray(r, float) - point)

        def hessian(r):
            d = np.asarray(r, float) - point
            return _projector(_unit(d)) / np.linalg.norm(d, axis=-1)[..., None, None]

        return cls(
            lambda r: np.linalg.norm(np.asarray(r, float) - point, axis=-1),
            gradient,
            hessian,
        )

    @classmethod
    def collimated(cls, direction, origin=(0.0, 0.0, 0.0)):
        """Parallel rays along ``direction``; psi = direction . (r - origin)."""
        direction = _position(direction)
        norm = np.linalg.norm(direction)

        if norm == 0:
            raise ValueError("collimated carrier needs a nonzero direction")

        direction, origin = direction / norm, _position(origin)

        return cls(
            lambda r: (np.asarray(r, float) - origin) @ direction,
            lambda r: np.broadcast_to(direction, np.shape(r)).copy(),
            lambda r: np.zeros(np.shape(r)[:-1] + (3, 3)),
        )

    def directions(self, points):
        """Unit ray directions; raises where the carrier is singular or not unit."""
        points = np.asarray(points, float)
        direction = np.broadcast_to(
            np.asarray(self.gradient(points), dtype=float), points.shape
        )

        if not np.isfinite(direction).all() or not np.allclose(
            np.linalg.norm(direction, axis=-1), 1, rtol=0, atol=1e-12
        ):
            raise ValueError(
                "carrier gradient must be a finite unit vector at every point"
            )

        return direction

    def jacobians(self, points):
        """Direction Jacobians ds_i/dr_j, shape (..., 3, 3), checked symmetric."""
        points = np.asarray(points, float)
        jacobian = np.broadcast_to(
            np.asarray(self.hessian(points), dtype=float), points.shape + (3,)
        )
        scale = np.max(np.abs(jacobian), axis=(-2, -1), keepdims=True)

        if not np.isfinite(jacobian).all() or np.any(
            np.abs(jacobian - np.swapaxes(jacobian, -1, -2)) > 1e-12 * scale
        ):
            raise ValueError("carrier hessian must be finite and symmetric")

        return jacobian


def _position(value):
    value = frozen_array(value)

    if value.shape != (3,):
        raise ValueError("expected a Cartesian 3-vector")

    return value


def _unit(d):
    with np.errstate(invalid="ignore", divide="ignore"):
        return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _projector(s):
    return np.eye(3) - s[..., :, None] * s[..., None, :]
