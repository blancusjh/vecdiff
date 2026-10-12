"""Public operations with explicit physical model and grid contracts."""

from vecdiff_general.sampling.grids import (
    CartesianGrid,
    Grid,
    PlaneGrid,
    PointGrid,
    PolarGrid,
    SurfaceGrid,
)
from vecdiff_general.sampling.near_surface import near_surface_grid
from vecdiff_general.sampling.surface_sampling import sample_surface

__all__ = [
    "CartesianGrid",
    "PolarGrid",
    "Grid",
    "PlaneGrid",
    "PointGrid",
    "SurfaceGrid",
    "sample_surface",
    "near_surface_grid",
]
