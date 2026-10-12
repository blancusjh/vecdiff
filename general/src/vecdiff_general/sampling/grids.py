"""Physical evaluation coordinates and integration measures."""

from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np

from vecdiff_general._arrays import frozen_array
from vecdiff_general.geometry.frames import Frame


@runtime_checkable
class Grid(Protocol):
    @property
    def points(self) -> np.ndarray: ...

    @property
    def shape(self) -> tuple: ...


@dataclass(frozen=True, eq=False)
class PointGrid:
    """Arbitrary Cartesian positions in metres; no implied quadrature."""

    points: np.ndarray

    def __post_init__(self):
        points = frozen_array(self.points)

        if points.ndim < 2 or points.shape[-1] != 3 or points.size == 0:
            raise ValueError("points must have nonempty shape (..., 3)")

        object.__setattr__(self, "points", points)

    @property
    def shape(self):
        return self.points.shape[:-1]


@dataclass(frozen=True, eq=False)
class SurfaceGrid:
    """Physical surface quadrature with oriented unit normals and area weights."""

    points: np.ndarray
    normals: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        points = PointGrid(self.points).points
        normals = frozen_array(self.normals)
        weights = frozen_array(self.weights)

        if normals.shape != points.shape or weights.shape != points.shape[:-1]:
            raise ValueError("surface points, normals and weights must agree")

        if not np.allclose(np.linalg.norm(normals, axis=-1), 1, atol=1e-12, rtol=0):
            raise ValueError("surface normals must be unit vectors")

        if np.any(weights <= 0):
            raise ValueError("surface area weights must be positive")

        for name, value in (
            ("points", points),
            ("normals", normals),
            ("weights", weights),
        ):
            object.__setattr__(self, name, value)

    @property
    def shape(self):
        return self.points.shape[:-1]


@dataclass(frozen=True, eq=False)
class CartesianGrid:
    """Uniform embedded plane lattice, array order (v, u).

    Integration uses rectangular cells of area du*dv: the periodic FFT cell
    spans Nu*du by Nv*dv. Do not include a duplicate periodic endpoint.
    """

    coordinate_system: ClassVar[str] = "cartesian"
    array_order: ClassVar[tuple[str, str]] = ("v", "u")

    u: np.ndarray
    v: np.ndarray
    frame: Frame = field(default_factory=Frame)

    def __post_init__(self):
        if not isinstance(self.frame, Frame):
            raise TypeError("frame must be a Frame")

        for name in ("u", "v"):
            axis = frozen_array(getattr(self, name))

            if axis.ndim != 1 or len(axis) < 2:
                raise ValueError("plane axes must be 1D with at least two points")

            steps = np.diff(axis)

            if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-10, atol=0):
                raise ValueError("plane axes must be increasing and uniformly spaced")

            object.__setattr__(self, name, axis)

    @property
    def shape(self):
        return (len(self.v), len(self.u))

    @property
    def spacing(self):
        return (self.u[1] - self.u[0], self.v[1] - self.v[0])

    @property
    def points(self):
        u, v = np.meshgrid(self.u, self.v, indexing="xy")

        return (
            self.frame.origin
            + u[..., None] * self.frame.u
            + v[..., None] * self.frame.v
        )

    @property
    def normals(self):
        return np.broadcast_to(self.frame.normal, self.shape + (3,))

    @property
    def weights(self):
        du, dv = self.spacing

        return np.full(self.shape, du * dv)

    @property
    def coordinates(self):
        """Numerical chart axes (u,v); array order is explicitly (v,u)."""
        return self.u, self.v

    def translated(self, displacement):
        return CartesianGrid(self.u, self.v, self.frame.translated(displacement))


@dataclass(frozen=True, eq=False)
class PolarGrid:
    """Embedded polar plane, array order (radius, azimuth).

    points[i,j] = origin + radius[i]*(cos(azimuth[j])*u + sin(azimuth[j])*v).
    Azimuth covers one uniform period, without a duplicate endpoint. The
    optional dr_weights integrate dr; area weights include r*dphi. Sampling
    coordinates do not change Cartesian electric-vector component conventions.
    """

    coordinate_system: ClassVar[str] = "polar"
    array_order: ClassVar[tuple[str, str]] = ("radius", "azimuth")

    radius: np.ndarray
    azimuth: np.ndarray
    frame: Frame = field(default_factory=Frame)
    dr_weights: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.frame, Frame):
            raise TypeError("frame must be a Frame")

        radius = frozen_array(self.radius)
        azimuth = frozen_array(self.azimuth)

        if (
            radius.ndim != 1
            or len(radius) < 2
            or np.any(radius < 0)
            or np.any(np.diff(radius) <= 0)
        ):
            raise ValueError("polar radii must be increasing and nonnegative")

        if azimuth.ndim != 1 or len(azimuth) < 2:
            raise ValueError("polar azimuths must form a one-dimensional periodic axis")

        if not np.allclose(
            np.diff(azimuth), 2 * np.pi / len(azimuth), rtol=1e-10, atol=1e-12
        ):
            raise ValueError(
                "azimuth must cover one uniform period without a duplicate endpoint"
            )

        if self.dr_weights is None:
            steps = np.diff(radius)
            weights = np.concatenate(
                ([steps[0] / 2], (steps[:-1] + steps[1:]) / 2, [steps[-1] / 2])
            )
        else:
            weights = np.asarray(self.dr_weights)

        weights = frozen_array(weights)

        if weights.shape != radius.shape or np.any(weights <= 0):
            raise ValueError("dr_weights must be positive and match the radial axis")

        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "azimuth", azimuth)
        object.__setattr__(self, "dr_weights", weights)

    @property
    def coordinates(self):
        return self.radius, self.azimuth

    @property
    def shape(self):
        return len(self.radius), len(self.azimuth)

    @property
    def points(self):
        radius, azimuth = np.meshgrid(self.radius, self.azimuth, indexing="ij")
        local_u = radius * np.cos(azimuth)
        local_v = radius * np.sin(azimuth)

        return (
            self.frame.origin
            + local_u[..., None] * self.frame.u
            + local_v[..., None] * self.frame.v
        )

    @property
    def normals(self):
        return np.broadcast_to(self.frame.normal, self.shape + (3,))

    @property
    def weights(self):
        radial_area = self.radius * self.dr_weights * (2 * np.pi / len(self.azimuth))

        return np.broadcast_to(radial_area[:, None], self.shape)

    def translated(self, displacement):
        return PolarGrid(
            self.radius,
            self.azimuth,
            self.frame.translated(displacement),
            self.dr_weights,
        )


# Existing callers retain the old name; there is only one Cartesian grid type.
PlaneGrid = CartesianGrid
