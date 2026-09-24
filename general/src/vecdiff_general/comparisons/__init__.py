"""Explicit prescribed-current comparisons, independent of operational propagation."""

from .curved_interface import SurfaceInteraction, scatter_surface
from .surface_radiation import SurfaceRadiation

__all__ = ["SurfaceInteraction", "SurfaceRadiation", "scatter_surface"]
