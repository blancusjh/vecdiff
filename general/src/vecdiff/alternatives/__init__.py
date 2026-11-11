"""Other methods, kept for checks and for cases outside the ray transport.

Wave transport through an optical system (``WavePropagator``), returning the
same Solution as ``propagation.Propagator``; its pieces, the exact
homogeneous transfer onto a surface (``SurfaceTransport``) and a Fresnel
treatment resolved by direction (``SpectralFresnel``) or taken locally
(``LocalFresnel``); and the exact planar boundary (``scatter_plane``).
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
from vecdiff.alternatives.wave_propagator import WavePropagator

__all__ = [
    "WavePropagator",
    "SurfaceTransport",
    "InverseReport",
    "SpectralFresnel",
    "LocalFresnel",
    "CarrierFresnel",
    "DirectionResolvedFresnel",
    "InterfaceSolution",
    "scatter_plane",
]
