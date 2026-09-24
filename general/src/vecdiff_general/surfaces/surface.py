"""Placed optical graph surfaces, independent of fields and material assignments."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..geometry.frames import Frame


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
        xy = np.asarray(xy, dtype=float)

        if xy.shape[-1:] != (2,) or not np.isfinite(xy).all():
            raise ValueError("surface coordinates must be finite (..., 2) arrays")

        local = np.concatenate((xy, self.height(xy)[..., None]), axis=-1)

        return self.frame.origin + local @ self.frame.basis.T

    def normals(self, xy):
        gradient = self.gradient(np.asarray(xy, dtype=float))
        normal = np.concatenate(
            (-gradient, np.ones(gradient.shape[:-1] + (1,))), axis=-1
        )
        normal /= np.linalg.norm(normal, axis=-1)[..., None]

        return normal @ self.frame.basis.T

    def area_jacobian(self, xy):
        return np.sqrt(1 + np.sum(self.gradient(xy) ** 2, axis=-1))


@dataclass(frozen=True)
class Plane(GraphSurface):
    frame: Frame = field(default_factory=Frame)

    def height(self, xy):
        return np.zeros(np.asarray(xy).shape[:-1])

    def gradient(self, xy):
        return np.zeros_like(np.asarray(xy, dtype=float))


@dataclass(frozen=True)
class FreeformSurface(GraphSurface):
    height_function: Callable
    gradient_function: Callable
    frame: Frame = field(default_factory=Frame)

    def height(self, xy):
        return np.broadcast_to(
            np.asarray(self.height_function(xy), dtype=float), np.asarray(xy).shape[:-1]
        )

    def gradient(self, xy):
        return np.broadcast_to(
            np.asarray(self.gradient_function(xy), dtype=float), np.asarray(xy).shape
        )
