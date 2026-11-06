"""Rotationally symmetric graph surfaces and spherical caps."""

from collections.abc import Callable
from dataclasses import dataclass, field

from vecdiff_general._backend import bd, divide
from vecdiff_general.geometry.frames import Frame
from vecdiff_general.surfaces.surface import GraphSurface


@dataclass(frozen=True)
class AxisymmetricSurface(GraphSurface):
    sag_function: Callable
    slope_function: Callable
    frame: Frame = field(default_factory=Frame)

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


@dataclass(frozen=True)
class SphericalCap(GraphSurface):
    radius: float
    frame: Frame = field(default_factory=Frame)

    def __post_init__(self):
        if not bd.isfinite(self.radius) or self.radius == 0:
            raise ValueError("signed curvature radius must be finite and nonzero")

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
