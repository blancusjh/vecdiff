"""Placed optical graph surfaces, independent of fields and material assignments."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from vecdiff._backend import bd
from vecdiff.geometry.frames import Frame


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

    def reversed(self):
        """The same surface seen from its other side.

        Its frame is this frame reversed (axes u, -v, -normal), its graph
        -h(x, -y) and its normal the opposite one: what was its plus side
        is the reversed surface's minus side. Light that meets a surface
        travelling against its normal meets its reverse along the normal.
        """
        return ReversedSurface.of(self)

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
        from vecdiff.sampling.grids import CartesianGrid, SurfaceGrid

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
    def crossing(self, points, directions, *, leaving=False):
        """Signed leg from each point to where its line crosses F = 0 from the
        minus side into the plus side (``leaving``: from the plus side into the
        minus side); NaN where it never does."""

    def far_side(self):
        """The same closed surface with its orientation reversed, charted from
        the vertex opposite its own.

        Light inside it, travelling along the normal, leaves through its far
        side: that crossing is the minus-to-plus crossing of the far side.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares no far side; light cannot be "
            "carried from inside it"
        )


_TURN = (1.0, -1.0, -1.0)  # local (x, y, z) of a frame -> of its reverse


@dataclass(frozen=True)
class ReversedSurface(GraphSurface):
    """A graph surface seen from its other side (GraphSurface.reversed).

    ``base`` is the surface; ``frame`` is its frame reversed, in which the
    graph is -h(x, -y) and the domain that of (x, -y). A frame rotated
    about the normal is accepted for a surface of revolution, whose
    graph does not depend on the azimuth.
    """

    base: GraphSurface
    frame: Frame = None

    def __post_init__(self):
        if self.frame is None:
            object.__setattr__(self, "frame", self.base.frame.reversed())

    @staticmethod
    def of(surface):
        """The reverse of ``surface``, implicit as well where it is."""
        if isinstance(surface, ImplicitSurface):
            return ReversedClosedSurface(surface)
        return ReversedSurface(surface)

    def reversed(self):
        return self.base

    @staticmethod
    def _mirror(xy):
        xy = bd.asarray(xy, dtype=float)
        return xy * bd.asarray([1.0, -1.0])

    def domain(self, xy):
        return self.base.domain(self._mirror(xy))

    def height(self, xy):
        return -self.base.height(self._mirror(xy))

    def gradient(self, xy):
        return self.base.gradient(self._mirror(xy)) * bd.asarray([-1.0, 1.0])

    def hessian(self, xy):
        signs = bd.asarray([[-1.0, 1.0], [1.0, -1.0]])
        return self.base.hessian(self._mirror(xy)) * signs


@dataclass(frozen=True)
class ReversedClosedSurface(ReversedSurface, ImplicitSurface):
    """A closed surface seen from its other side: F'(x, y, z) = -F(x, -y, -z).

    Its minus-to-plus crossing is the base's plus-to-minus one.
    """

    def implicit(self, points):
        turn = bd.asarray(_TURN)
        value, gradient, hessian = self.base.implicit(bd.asarray(points) * turn)
        return -value, -gradient * turn, -hessian * turn[:, None] * turn[None, :]

    def crossing(self, points, directions, *, leaving=False):
        turn = bd.asarray(_TURN)
        return self.base.crossing(
            bd.asarray(points) * turn,
            bd.asarray(directions) * turn,
            leaving=not leaving,
        )


@dataclass(frozen=True)
class Plane(GraphSurface):
    frame: Frame = field(default_factory=Frame)

    def height(self, xy):
        return bd.zeros(bd.asarray(xy).shape[:-1])

    def gradient(self, xy):
        return bd.zeros_like(bd.asarray(xy, dtype=float))

    def hessian(self, xy):
        return bd.zeros(bd.asarray(xy).shape + (2,))

    def reversed(self):
        return Plane(self.frame.reversed())

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
