"""Explicit prescribed-current comparisons, independent of operational propagation."""

from vecdiff_general.comparisons.curved_interface import (
    SurfaceInteraction,
    scatter_surface,
)
from vecdiff_general.comparisons.surface_radiation import SurfaceRadiation

__all__ = ["SurfaceInteraction", "SurfaceRadiation", "scatter_surface"]
