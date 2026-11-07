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

    def domain(self, xy):
        """True where the graph z = h(x, y) is defined; all points by default."""
        return bd.ones(bd.asarray(xy).shape[:-1], dtype=bool)

    def hessian(self, xy):
        """Second derivatives [[h_xx, h_xy], [h_xy, h_yy]] in 1/m, shape (..., 2, 2).

        Eikonal transport differentiates the surface normal (manuscript
        Appendix A); a surface that does not declare its Hessian cannot be used.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares no Hessian; eikonal transport needs one"
        )

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


class ImplicitSurface(ABC):
    """A surface F(Q) = 0 that rays meet beyond its graph chart.

    A graph surface may also declare its implicit form; eikonal transport
    then intersects rays with the whole surface F = 0, while regions, charts
    and targets keep the graph. Points and directions are frame-local.
    """

    @abstractmethod
    def implicit(self, points):
        """F, its gradient (..., 3) and Hessian (..., 3, 3) at local points.

        The gradient points from the minus side into the plus side, as the
        graph normal (-grad h, 1) does where both exist.
        """

    @abstractmethod
    def crossing(self, points, directions):
        """Signed leg from each point to where its ray first crosses F = 0
        from the minus side into the plus side; NaN where it never does."""


@dataclass(frozen=True)
class Plane(GraphSurface):
    frame: Frame = field(default_factory=Frame)

    def height(self, xy):
        return bd.zeros(bd.asarray(xy).shape[:-1])

    def gradient(self, xy):
        return bd.zeros_like(bd.asarray(xy, dtype=float))

    def hessian(self, xy):
        return bd.zeros(bd.asarray(xy).shape + (2,))

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
    """z = height_function(xy); gradient and optional Hessian functions alongside."""

    height_function: Callable
    gradient_function: Callable
    frame: Frame = field(default_factory=Frame)
    hessian_function: Callable | None = None

    def height(self, xy):
        return bd.broadcast_to(
            bd.asarray(self.height_function(xy), dtype=float), bd.asarray(xy).shape[:-1]
        )

    def gradient(self, xy):
        return bd.broadcast_to(
            bd.asarray(self.gradient_function(xy), dtype=float), bd.asarray(xy).shape
        )

    def hessian(self, xy):
        if self.hessian_function is None:
            return super().hessian(xy)

        return bd.broadcast_to(
            bd.asarray(self.hessian_function(xy), dtype=float),
            bd.asarray(xy).shape + (2,),
        )
