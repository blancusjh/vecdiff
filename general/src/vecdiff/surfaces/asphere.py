"""Even asphere with a conic base and coefficients A4, A6, ... ."""

from dataclasses import dataclass, field

import numpy as np

from vecdiff._backend import bd, divide
from vecdiff._printing import number, vector
from vecdiff.geometry.frames import Frame
from vecdiff.surfaces.surface import GraphSurface


@dataclass(frozen=True, repr=False)
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

    def _terms(self):
        """The aspheric coefficients by name: A4, A6, ..."""
        return [(f"A{2 * j}", number(a)) for j, a in enumerate(self.coefficients, 2)]

    def _parameters(self):
        count = len(self.coefficients)
        if count <= 6:
            coefficients = vector(self.coefficients) if count else None
        else:
            coefficients = f"<A4..A{2 * count + 2}, {count} terms>"
        conic = number(self.conic) if self.conic else None
        return [
            ("curvature", number(self.curvature)),
            ("conic", conic),
            ("coefficients", coefficients),
        ]

    def _details(self):
        radius = number(1 / self.curvature) if self.curvature else "plane (inf)"
        return [
            ("curvature", number(self.curvature)),
            ("radius", radius),
            ("conic", number(self.conic)),
            *self._terms(),
        ]

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
        from vecdiff.surfaces.axisymmetric import _radial_hessian

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

    def reversed(self):
        """The asphere seen from its other side: every coefficient negated."""
        return EvenAsphere(
            -self.curvature,
            self.conic,
            tuple(-a for a in self.coefficients),
            self.frame.reversed(),
        )
