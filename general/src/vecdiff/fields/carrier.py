"""Reference ray families: the declared carrier of a high-frequency field.

A carrier is a geometric path function psi(r), a length, with |grad psi| = 1.
Its gradient is the unit ray direction s(r) and its Hessian is the direction
Jacobian ds_i/dr_j. In a homogeneous region of index n the reference eikonal
(optical path) is n psi, so one carrier serves every region it crosses. It is
declared with a field, never inferred from an interfering total field.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from vecdiff._arrays import frozen_array
from vecdiff._backend import bd
from vecdiff._printing import call, guarded, name_of, vector


@dataclass(frozen=True, repr=False)
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
            return _unit(point - bd.asarray(r, float))

        def hessian(r):
            d = point - bd.asarray(r, float)
            return -_projector(_unit(d)) / bd.linalg.norm(d, axis=-1)[..., None, None]

        return cls(
            lambda r: -bd.linalg.norm(point - bd.asarray(r, float), axis=-1),
            gradient,
            hessian,
        )

    @classmethod
    def diverging(cls, point):
        """Rays from ``point``; psi = |r - point| vanishes there."""
        point = _position(point)

        def gradient(r):
            return _unit(bd.asarray(r, float) - point)

        def hessian(r):
            d = bd.asarray(r, float) - point
            return _projector(_unit(d)) / bd.linalg.norm(d, axis=-1)[..., None, None]

        return cls(
            lambda r: bd.linalg.norm(bd.asarray(r, float) - point, axis=-1),
            gradient,
            hessian,
        )

    @classmethod
    def collimated(cls, direction, origin=(0.0, 0.0, 0.0)):
        """Parallel rays along ``direction``; psi = direction . (r - origin)."""
        direction = _position(direction)
        norm = bd.linalg.norm(direction)

        if norm == 0:
            raise ValueError("collimated carrier needs a nonzero direction")

        direction, origin = direction / norm, _position(origin)

        return cls(
            lambda r: (bd.asarray(r, float) - origin) @ direction,
            lambda r: bd.broadcast_to(direction, bd.shape(r)).copy(),
            lambda r: bd.zeros(bd.shape(r)[:-1] + (3, 3)),
        )

    def family(self):
        """Which family built the carrier and its defining vectors.

        ("converging", {"point": ...}) for Carrier.converging, likewise
        "diverging" and "collimated" (direction, origin); ("custom", {})
        for one built from its own functions.
        """
        qualname = getattr(self.path, "__qualname__", "")
        for kind in ("converging", "diverging", "collimated"):
            if qualname.startswith(f"Carrier.{kind}.<locals>"):
                cells = self.path.__closure__ or ()
                names = self.path.__code__.co_freevars
                return kind, {n: c.cell_contents for n, c in zip(names, cells)}
        return "custom", {}

    @guarded
    def __repr__(self):
        """The family of rays and the point or direction that defines it."""
        kind, vectors = self.family()
        if kind == "custom":
            return call("Carrier", path=name_of(self.path))
        order = ("point", "direction", "origin")
        pairs = {n: vector(vectors[n]) for n in order if n in vectors}
        return call(f"Carrier.{kind}", **pairs)

    def directions(self, points):
        """Unit ray directions; raises where the carrier is singular or not unit."""
        points = bd.asarray(points, float)
        direction = bd.broadcast_to(
            bd.asarray(self.gradient(points), dtype=float), points.shape
        )

        if not bd.isfinite(direction).all() or not bd.allclose(
            bd.linalg.norm(direction, axis=-1), 1, rtol=0, atol=1e-12
        ):
            raise ValueError(
                "carrier gradient must be a finite unit vector at every point"
            )

        return direction

    def jacobians(self, points):
        """Direction Jacobians ds_i/dr_j, shape (..., 3, 3), checked symmetric."""
        points = bd.asarray(points, float)
        jacobian = bd.broadcast_to(
            bd.asarray(self.hessian(points), dtype=float), points.shape + (3,)
        )
        scale = bd.max(bd.abs(jacobian), axis=(-2, -1), keepdims=True)

        if not bd.isfinite(jacobian).all() or bd.any(
            bd.abs(jacobian - bd.swapaxes(jacobian, -1, -2)) > 1e-12 * scale
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
        return d / bd.linalg.norm(d, axis=-1, keepdims=True)


def _projector(s):
    return bd.eye(3) - s[..., :, None] * s[..., None, :]
