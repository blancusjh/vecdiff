"""Other methods, kept for checks and for cases outside the ray transport.

Wave transport of a field to and across a curved surface: the exact
homogeneous transfer onto the surface (``SurfaceTransport``) with a Fresnel
treatment resolved by direction (``SpectralFresnel``) or taken locally
(``LocalFresnel``), and the exact planar boundary (``scatter_plane``).
"""

from vecdiff.alternatives.surface_interaction import (
    CarrierFresnel,
    DirectionResolvedFresnel,
    InterfaceSolution,
    LocalFresnel,
    SpectralFresnel,
    scatter_plane,
)
from vecdiff.alternatives.surface_transfer import InverseReport, SurfaceTransport

__all__ = [
    "SurfaceTransport",
    "InverseReport",
    "SpectralFresnel",
    "LocalFresnel",
    "CarrierFresnel",
    "DirectionResolvedFresnel",
    "InterfaceSolution",
    "scatter_plane",
]
