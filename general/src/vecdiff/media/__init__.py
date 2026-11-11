"""Media: the materials that fill space (``Dielectric``, ``air``, ``vacuum``) and
the regions of space they fill (``HomogeneousRegion``)."""

from vecdiff.media.distribution import MaterialDistribution, MaterialRegion
from vecdiff.media.medium import (
    C0,
    EPSILON0,
    MU0,
    Z0,
    Dielectric,
    HomogeneousRegion,
    air,
    vacuum,
)

__all__ = [
    "Dielectric",
    "HomogeneousRegion",
    "air",
    "vacuum",
    "EPSILON0",
    "MU0",
    "Z0",
    "C0",
    "MaterialRegion",
    "MaterialDistribution",
]
