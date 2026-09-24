"""Even asphere with a conic base and coefficients A4, A6, ... ."""

from dataclasses import dataclass, field

import numpy as np

from ..geometry.frames import Frame
from .surface import GraphSurface


@dataclass(frozen=True)
class EvenAsphere(GraphSurface):
    curvature: float
    conic: float = 0.0
    coefficients: tuple[float, ...] = ()
    frame: Frame = field(default_factory=Frame)

    def __post_init__(self):
        coefficients = tuple(float(value) for value in self.coefficients)

        if not np.isfinite((self.curvature, self.conic, *coefficients)).all():
            raise ValueError("asphere design coefficients must be finite")

        object.__setattr__(self, "coefficients", coefficients)

    def _root(self, radius):
        radicand = 1 - (1 + self.conic) * self.curvature**2 * radius**2

        if np.any(radicand <= 0):
            raise ValueError("asphere coordinates exceed the real graph branch")

        return np.sqrt(radicand)

    def height(self, xy):
        radius = np.linalg.norm(xy, axis=-1)
        height = self.curvature * radius**2 / (1 + self._root(radius))

        for order, coefficient in enumerate(self.coefficients, start=2):
            height += coefficient * radius ** (2 * order)

        return height

    def gradient(self, xy):
        xy = np.asarray(xy, dtype=float)
        radius = np.linalg.norm(xy, axis=-1)
        slope = self.curvature * radius / self._root(radius)

        for order, coefficient in enumerate(self.coefficients, start=2):
            slope += 2 * order * coefficient * radius ** (2 * order - 1)

        return np.divide(
            slope[..., None] * xy,
            radius[..., None],
            out=np.zeros_like(xy),
            where=radius[..., None] != 0,
        )
