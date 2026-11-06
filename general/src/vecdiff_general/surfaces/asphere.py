"""Even asphere with a conic base and coefficients A4, A6, ... ."""

from dataclasses import dataclass, field

import numpy as np

from vecdiff_general._backend import bd, divide
from vecdiff_general.geometry.frames import Frame
from vecdiff_general.surfaces.surface import GraphSurface


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

    def domain(self, xy):
        radius = bd.linalg.norm(bd.asarray(xy, dtype=float), axis=-1)

        return 1 - (1 + self.conic) * self.curvature**2 * radius**2 > 0

    def _root(self, radius):
        radicand = 1 - (1 + self.conic) * self.curvature**2 * radius**2

        if bd.any(radicand <= 0):
            raise ValueError("asphere coordinates exceed the real graph branch")

        return bd.sqrt(radicand)

    def height(self, xy):
        radius = bd.linalg.norm(bd.asarray(xy, dtype=float), axis=-1)
        height = self.curvature * radius**2 / (1 + self._root(radius))

        for order, coefficient in enumerate(self.coefficients, start=2):
            height += coefficient * radius ** (2 * order)

        return height

    def gradient(self, xy):
        xy = bd.asarray(xy, dtype=float)
        radius = bd.linalg.norm(xy, axis=-1)
        slope = self.curvature * radius / self._root(radius)

        for order, coefficient in enumerate(self.coefficients, start=2):
            slope += 2 * order * coefficient * radius ** (2 * order - 1)

        return divide(slope[..., None] * xy, radius[..., None], radius[..., None] != 0)

    def hessian(self, xy):
        from vecdiff_general.surfaces.axisymmetric import _radial_hessian

        radius = bd.linalg.norm(bd.asarray(xy, dtype=float), axis=-1)
        root = self._root(radius)
        slope = self.curvature * radius / root
        curvature = self.curvature / root**3

        for order, coefficient in enumerate(self.coefficients, start=2):
            slope += 2 * order * coefficient * radius ** (2 * order - 1)
            curvature += (
                2 * order * (2 * order - 1) * coefficient * radius ** (2 * order - 2)
            )

        return _radial_hessian(xy, slope, curvature)
