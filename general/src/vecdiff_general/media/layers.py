"""Spatial prescription for parallel homogeneous dielectric layers."""

from dataclasses import dataclass, field

import numpy as np

from ..geometry.frames import Frame
from .medium import HomogeneousRegion


@dataclass(frozen=True)
class LayerStack:
    """Entrance half-space, finite layers, and exit half-space.

    thicknesses has one entry per interior region, in metres. The first
    interface is at frame.origin. Boundaries advance along frame.normal.
    """

    regions: tuple[HomogeneousRegion, ...]
    thicknesses: tuple[float, ...]
    frame: Frame = field(default_factory=Frame)

    def __post_init__(self):
        regions = tuple(self.regions)
        thicknesses = tuple(float(value) for value in self.thicknesses)

        if len(regions) < 2 or len(thicknesses) != len(regions) - 2:
            raise ValueError(
                "provide two half-spaces and one thickness per interior region"
            )

        if not all(isinstance(region, HomogeneousRegion) for region in regions):
            raise TypeError("all stack entries must be homogeneous material regions")

        if any(not np.isfinite(value) or value <= 0 for value in thicknesses):
            raise ValueError("interior layer thicknesses must be finite and positive")

        if not isinstance(self.frame, Frame):
            raise TypeError("a layer stack requires a physical Frame")

        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "thicknesses", thicknesses)

    @property
    def boundaries(self):
        return np.concatenate(([0.0], np.cumsum(self.thicknesses)))
