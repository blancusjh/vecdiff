"""Rotationally symmetric graph surfaces and spherical caps."""

from collections.abc import Callable
from dataclasses import dataclass, field

from vecdiff_general._backend import bd, divide
from vecdiff_general.geometry.frames import Frame
from vecdiff_general.surfaces.surface import GraphSurface


def _radial_hessian(xy, slope, curvature):
    """Hessian of h(r) at points xy from h'(r) and h''(r); h''(0) on the axis."""
    xy = bd.asarray(xy, dtype=float)
    radius = bd.linalg.norm(xy, axis=-1)
    on_axis = radius == 0
    unit = divide(xy, radius[..., None], ~on_axis[..., None])
    radial = unit[..., :, None] * unit[..., None, :]
    tangential = bd.eye(2) - radial
    # h'(r) / r tends to h''(0) on the axis of a smooth surface of revolution.
    bending = bd.where(on_axis, curvature, divide(slope, radius, ~on_axis))

    return curvature[..., None, None] * radial + bending[..., None, None] * tangential


@dataclass(frozen=True)
class AxisymmetricSurface(GraphSurface):
    """z = sag(r); slope h'(r) and optional curvature h''(r) functions alongside."""

    sag_function: Callable
    slope_function: Callable
    frame: Frame = field(default_factory=Frame)
    curvature_function: Callable | None = None

    def sag(self, radius):
        return bd.asarray(self.sag_function(bd.asarray(radius)), dtype=float)

    def slope(self, radius):
        return bd.asarray(self.slope_function(bd.asarray(radius)), dtype=float)

    def height(self, xy):
        return self.sag(bd.linalg.norm(xy, axis=-1))

    def gradient(self, xy):
        xy = bd.asarray(xy, dtype=float)
        radius = bd.linalg.norm(xy, axis=-1)

        return divide(
            self.slope(radius)[..., None] * xy,
            radius[..., None],
            radius[..., None] != 0,
        )

    def hessian(self, xy):
        if self.curvature_function is None:
            return super().hessian(xy)

        radius = bd.linalg.norm(bd.asarray(xy, dtype=float), axis=-1)
        curvature = bd.asarray(self.curvature_function(radius), dtype=float)

        return _radial_hessian(xy, self.slope(radius), curvature)


@dataclass(frozen=True)
class SphericalCap(GraphSurface):
    radius: float
    frame: Frame = field(default_factory=Frame)

    def __post_init__(self):
        if not bd.isfinite(self.radius) or self.radius == 0:
            raise ValueError("signed curvature radius must be finite and nonzero")

    def domain(self, xy):
        return bd.sum(bd.asarray(xy, dtype=float) ** 2, axis=-1) < self.radius**2

    def _root(self, xy):
        radial_squared = bd.sum(bd.asarray(xy) ** 2, axis=-1)

        if bd.any(radial_squared >= self.radius**2):
            raise ValueError("spherical-cap coordinates reach or exceed the equator")

        return bd.sqrt(self.radius**2 - radial_squared)

    def height(self, xy):
        # Rationalized form avoids cancellation close to the optical axis.
        return (
            bd.sign(self.radius)
            * bd.sum(bd.asarray(xy) ** 2, axis=-1)
            / (abs(self.radius) + self._root(xy))
        )

    def gradient(self, xy):
        return bd.sign(self.radius) * bd.asarray(xy) / self._root(xy)[..., None]

    def hessian(self, xy):
        xy = bd.asarray(xy, dtype=float)
        root = self._root(xy)[..., None, None]
        outer = xy[..., :, None] * xy[..., None, :]

        return bd.sign(self.radius) * (bd.eye(2) / root + outer / root**3)
