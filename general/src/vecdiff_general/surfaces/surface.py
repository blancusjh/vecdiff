"""Placed optical graph surfaces, independent of fields and material assignments."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from vecdiff_general._backend import bd
from vecdiff_general.geometry.frames import Frame


class GraphSurface(ABC):
    """Local height z=h(x,y), embedded by a right-handed physical frame."""

    frame: Frame

    @abstractmethod
    def height(self, xy):
        """Local height in metres at (..., 2) local Cartesian coordinates."""

    @abstractmethod
    def gradient(self, xy):
        """Dimensionless local derivatives (dh/dx, dh/dy)."""

    def points(self, xy):
        xy = bd.asarray(xy, dtype=float)

        if xy.shape[-1:] != (2,) or not bd.isfinite(xy).all():
            raise ValueError("surface coordinates must be finite (..., 2) arrays")

        local = bd.concatenate((xy, self.height(xy)[..., None]), axis=-1)

        return self.frame.origin + local @ self.frame.basis.T

    def normals(self, xy):
        gradient = self.gradient(bd.asarray(xy, dtype=float))
        normal = bd.concatenate(
            (-gradient, bd.ones(gradient.shape[:-1] + (1,))), axis=-1
        )
        normal /= bd.linalg.norm(normal, axis=-1)[..., None]

        return normal @ self.frame.basis.T

    def area_jacobian(self, xy):
        return bd.sqrt(1 + bd.sum(self.gradient(xy) ** 2, axis=-1))

    def sample(self, *, u, v):
        """Sample a uniform chart using the existing physical grid types."""
        from vecdiff_general.sampling.grids import CartesianGrid, SurfaceGrid

        plane = CartesianGrid(u, v, self.frame)
        if isinstance(self, Plane):
            return plane
        x, y = bd.meshgrid(plane.u, plane.v)
        xy = bd.stack((x, y), axis=-1)
        return SurfaceGrid(
            self.points(xy), self.normals(xy), plane.weights * self.area_jacobian(xy)
        )


@dataclass(frozen=True)
class Plane(GraphSurface):
    frame: Frame = field(default_factory=Frame)

    def height(self, xy):
        return bd.zeros(bd.asarray(xy).shape[:-1])

    def gradient(self, xy):
        return bd.zeros_like(bd.asarray(xy, dtype=float))

    @classmethod
    def z(cls, value):
        """Global coordinate plane z=value, chart (u,v)=(x,y)."""
        return cls(Frame([0, 0, value]))

    @classmethod
    def y(cls, value):
        """Global coordinate plane y=value, chart (u,v)=(x,z)."""
        return cls(Frame([0, value, 0], [[1, 0, 0], [0, 0, -1], [0, 1, 0]]))

    @classmethod
    def x(cls, value):
        """Global coordinate plane x=value, chart (u,v)=(y,z)."""
        return cls(Frame([value, 0, 0], [[0, 0, 1], [1, 0, 0], [0, 1, 0]]))


@dataclass(frozen=True)
class FreeformSurface(GraphSurface):
    height_function: Callable
    gradient_function: Callable
    frame: Frame = field(default_factory=Frame)

    def height(self, xy):
        return bd.broadcast_to(
            bd.asarray(self.height_function(xy), dtype=float), bd.asarray(xy).shape[:-1]
        )

    def gradient(self, xy):
        return bd.broadcast_to(
            bd.asarray(self.gradient_function(xy), dtype=float), bd.asarray(xy).shape
        )
