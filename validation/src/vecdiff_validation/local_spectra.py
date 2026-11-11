"""Plane-wave expansions valid in a ball, or beyond a plane, of a radiating source.

They describe the radiation of prescribed surface currents
(comparisons.surface_radiation), not a region's field: they are kept with
the comparisons that use them, outside the library.
"""

from dataclasses import dataclass

import numpy as np
from vecdiff._backend import bd
from vecdiff.fields.spectrum import PlaneWaveExpansion


@dataclass(frozen=True)
class LocalSpectrum:
    """Bounds compare with the same discrete surface-current Green integral.

    They exclude source quadrature, boundary-model error and roundoff. Electric
    and magnetic absolute vector-error bounds are V/m and A/m respectively.
    """

    expansion: PlaneWaveExpansion
    radius: float
    electric_error_bound: float
    magnetic_error_bound: float

    def __post_init__(self):
        for value in (
            self.radius,
            self.electric_error_bound,
            self.magnetic_error_bound,
        ):
            if not bd.isfinite(value) or value < 0:
                raise ValueError(
                    "radius and absolute error bounds must be finite and nonnegative"
                )

    @property
    def center(self):
        return self.expansion.origin

    def evaluate(self, grid, **kwargs):
        distance = bd.linalg.norm(grid.points - self.center, axis=-1)
        tolerance = 32 * np.finfo(float).eps * max(1.0, bd.max(bd.abs(self.center)))

        if bd.any(distance > self.radius + tolerance):
            raise ValueError("observation leaves the local spectrum's validity ball")

        return self.expansion.evaluate(grid, **kwargs)

    def electric(self, grid, **kwargs):
        return self.evaluate(grid, **kwargs).electric


@dataclass(frozen=True)
class HalfSpaceSpectrum:
    """Propagating Weyl quadrature valid beyond the source z envelope."""

    expansion: PlaneWaveExpansion
    boundary_z: float
    direction: int

    def evaluate(self, grid, **kwargs):
        if bd.any(self.direction * (grid.points[..., 2] - self.boundary_z) < 0):
            raise ValueError(
                "Weyl representation is valid only beyond the source z envelope"
            )

        return self.expansion.evaluate(grid, **kwargs)
