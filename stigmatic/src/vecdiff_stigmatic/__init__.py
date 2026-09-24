"""Independent stigmatic reference-sphere optics; all lengths in metres."""

from .fields import ElectricField, ElectromagneticField, TransverseElectricField
from .fields.sources import cartesian_source
from .fields.transfer_result import TransferResult
from .geometry import Frame
from .geometry.rays import RayBundle
from .interfaces import StigmaticSystem
from .media import EPSILON0, MU0, Z0, Dielectric
from .propagation.debye import DebyeFocus
from .propagation.fresnel import fresnel_focus
from .propagation.interface_transform import transfer
from .sampling import CartesianGrid, PlaneGrid, PointGrid, PolarGrid, SurfaceGrid
from .surfaces.cartesian_oval import CartesianOval

__all__ = [
    "CartesianGrid",
    "PolarGrid",
    "fresnel_focus",
    "Frame",
    "PlaneGrid",
    "PointGrid",
    "SurfaceGrid",
    "Dielectric",
    "Z0",
    "EPSILON0",
    "MU0",
    "ElectricField",
    "TransverseElectricField",
    "ElectromagneticField",
    "CartesianOval",
    "StigmaticSystem",
    "RayBundle",
    "cartesian_source",
    "transfer",
    "TransferResult",
    "DebyeFocus",
]
