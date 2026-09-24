"""An oriented planar boundary joining two homogeneous material regions."""

from __future__ import annotations

from dataclasses import dataclass

from ..geometry.frames import Frame
from ..media.medium import HomogeneousRegion


@dataclass(frozen=True)
class PlanarInterface:
    """Frame normal points from the minus region into the plus region."""

    frame: Frame
    minus: HomogeneousRegion
    plus: HomogeneousRegion

    def __post_init__(self):
        if not isinstance(self.frame, Frame):
            raise TypeError("interface requires a Frame")

        if not all(isinstance(r, HomogeneousRegion) for r in (self.minus, self.plus)):
            raise TypeError("interface sides must be homogeneous regions")

        if self.minus == self.plus:
            raise ValueError(
                "name the two spatial regions distinctly, even for equal materials"
            )


@dataclass(frozen=True)
class DielectricInterface:
    """A placed surface separating minus and plus material regions.

    The surface's graph normal points from minus to plus. Incidence belongs to
    an encounter, not permanently to the surface itself.
    """

    surface: object
    minus: HomogeneousRegion
    plus: HomogeneousRegion

    def __post_init__(self):
        from ..surfaces.surface import GraphSurface

        if not isinstance(self.surface, GraphSurface):
            raise TypeError("interface requires a placed GraphSurface")

        if not all(
            isinstance(region, HomogeneousRegion) for region in (self.minus, self.plus)
        ):
            raise TypeError("interface sides must be homogeneous material regions")

        if self.minus == self.plus:
            raise ValueError("the two spatial regions need distinct identities")
