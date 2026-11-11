"""Media: the materials that fill space (``Medium``, lossless or absorbing;
``Dielectric``, the lossless one; ``air``, ``vacuum``) and the regions of space
they fill (``HomogeneousRegion``). ``AbsorbingMediumError`` is the refusal of
an absorbing medium where a computation is exact only in a transparent one."""

from vecdiff.media.absorption import AbsorbingMediumError
from vecdiff.media.distribution import MaterialDistribution, MaterialRegion
from vecdiff.media.medium import (
    C0,
    EPSILON0,
    MU0,
    Z0,
    Dielectric,
    HomogeneousRegion,
    Medium,
    air,
    vacuum,
)

__all__ = [
    "Medium",
    "Dielectric",
    "AbsorbingMediumError",
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
