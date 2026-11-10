"""Explicit prescribed-current comparisons, independent of operational propagation."""

from vecdiff_validation.comparisons.curved_interface import (
    SurfaceInteraction,
    scatter_surface,
)
from vecdiff_validation.comparisons.surface_radiation import SurfaceRadiation

__all__ = ["SurfaceInteraction", "SurfaceRadiation", "scatter_surface"]
