"""Assignment of constitutive matter to geometrical volumes."""

from dataclasses import dataclass

import numpy as np

from .medium import Dielectric


@dataclass(frozen=True)
class MaterialRegion:
    name: str
    geometry: object
    material: Dielectric

    def __post_init__(self):
        if not callable(getattr(self.geometry, "contains", None)):
            raise TypeError("material region geometry must provide spatial membership")

        if not isinstance(self.material, Dielectric):
            raise TypeError("a material region needs dielectric constitutive data")


@dataclass(frozen=True)
class MaterialDistribution:
    background: Dielectric
    regions: tuple[MaterialRegion, ...] = ()

    def __post_init__(self):
        if not isinstance(self.background, Dielectric):
            raise TypeError("background must be a dielectric material")

        object.__setattr__(self, "regions", tuple(self.regions))

    def refractive_index(self, points):
        """Piecewise n(r); overlapping bodies require an explicit topology."""
        points = np.asarray(points, dtype=float)

        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("positions must be finite Cartesian coordinates")

        indices = np.full(points.shape[:-1], self.background.n)
        occupied = np.zeros(points.shape[:-1], dtype=bool)

        for region in self.regions:
            inside = region.geometry.contains(points)

            if np.any(occupied & inside):
                raise ValueError(
                    "overlapping material regions have no unique assignment"
                )

            indices[inside] = region.material.n
            occupied |= inside

        return indices
