"""Rotationally symmetric graph surfaces and spherical caps."""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..geometry.frames import Frame
from .surface import GraphSurface


@dataclass(frozen=True)
class AxisymmetricSurface(GraphSurface):
    sag_function: Callable
    slope_function: Callable
    frame: Frame = field(default_factory=Frame)

    def sag(self, radius):
        return np.asarray(self.sag_function(np.asarray(radius)), dtype=float)

    def slope(self, radius):
        return np.asarray(self.slope_function(np.asarray(radius)), dtype=float)

    def height(self, xy):
        return self.sag(np.linalg.norm(xy, axis=-1))

    def gradient(self, xy):
        xy = np.asarray(xy, dtype=float)
        radius = np.linalg.norm(xy, axis=-1)

        return np.divide(
            self.slope(radius)[..., None] * xy,
            radius[..., None],
            out=np.zeros_like(xy),
            where=radius[..., None] != 0,
        )


@dataclass(frozen=True)
class SphericalCap(GraphSurface):
    radius: float
    frame: Frame = field(default_factory=Frame)

    def __post_init__(self):
        if not np.isfinite(self.radius) or self.radius == 0:
            raise ValueError("signed curvature radius must be finite and nonzero")

    def _root(self, xy):
        radial_squared = np.sum(np.asarray(xy) ** 2, axis=-1)

        if np.any(radial_squared >= self.radius**2):
            raise ValueError("spherical-cap coordinates reach or exceed the equator")

        return np.sqrt(self.radius**2 - radial_squared)

    def height(self, xy):
        # Rationalized form avoids cancellation close to the optical axis.
        return (
            np.sign(self.radius)
            * np.sum(np.asarray(xy) ** 2, axis=-1)
            / (abs(self.radius) + self._root(xy))
        )

    def gradient(self, xy):
        return np.sign(self.radius) * np.asarray(xy) / self._root(xy)[..., None]
