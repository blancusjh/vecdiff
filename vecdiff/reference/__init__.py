"""Independent reference solvers used to arbitrate modelling conventions."""

from .currents import (
    PhysicalOpticsCurrents,
    PrescribedSphericalCapCurrents,
    SurfaceCurrents,
)
from .stratton_chu import (
    default_focal_cut,
    focal_field_reference,
    franz_integral,
    franz_integral_chunked,
)

__all__ = [
    "PhysicalOpticsCurrents",
    "PrescribedSphericalCapCurrents",
    "SurfaceCurrents",
    "default_focal_cut",
    "focal_field_reference",
    "franz_integral",
    "franz_integral_chunked",
]
