"""Rotationally symmetric graph surfaces and spherical caps."""

from collections.abc import Callable
from dataclasses import dataclass, field

from vecdiff_general._backend import bd, divide
from vecdiff_general.geometry.frames import Frame
from vecdiff_general.surfaces.surface import GraphSurface, ImplicitSurface


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


@dataclass(frozen=True)
class Sphere(SphericalCap, ImplicitSurface):
    """The closed sphere whose front cap is the SphericalCap of the same radius.

    Regions, charts and targets use the cap's graph, as for SphericalCap;
    eikonal transport meets the whole sphere, so a ray that passes the cap's
    equator still enters (or leaves) the ball on its far side. With
    F(Q) = (R² - |Q - c|²) / (2 R) and c = (0, 0, R), grad F = (c - Q) / R is
    the unit normal from the minus into the plus side and the Hessian is
    -I / R; R is signed as for the cap.
    """

    def _centre(self):
        return bd.asarray([0.0, 0.0, float(self.radius)])

    def implicit(self, points):
        offset = bd.asarray(points, dtype=float) - self._centre()
        R = float(self.radius)
        value = (R**2 - bd.sum(offset**2, axis=-1)) / (2 * R)
        hessian = bd.broadcast_to(-bd.eye(3) / R, offset.shape + (3,)).copy()

        return value, -offset / R, hessian

    def crossing(self, points, directions):
        offset = bd.asarray(points, dtype=float) - self._centre()
        s = bd.asarray(directions, dtype=float)
        b = bd.sum(s * offset, axis=-1)
        c = bd.sum(offset**2, axis=-1) - float(self.radius) ** 2
        disc = b * b - c
        root = bd.sqrt(bd.where(disc >= 0, disc, bd.nan))
        # The crossing into the plus side has s . grad F = -s . offset / R > 0:
        # the near root for a convex sphere, the far one for a concave one.
        if self.radius > 0:
            # Stable form of -b - root.
            return divide(c, -b + root, (-b + root) != 0)
        return -b + root
