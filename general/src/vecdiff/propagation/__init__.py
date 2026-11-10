"""Homogeneous propagation, surface transfer, and boundary interactions."""

from vecdiff.propagation.coverage import CoverageReport
from vecdiff.propagation.eikonal import (
    EikonalReport,
    EikonalTransport,
    InterfaceMap,
    RayMap,
)
from vecdiff.propagation.homogeneous import (
    PlaneTransport,
    propagate,
    propagate_spectrum,
)
from vecdiff.propagation.pipeline import (
    PropagationPipeline,
    PropagationSolution,
)
from vecdiff.propagation.surface_interaction import (
    CarrierFresnel,
    DirectionResolvedFresnel,
    InterfaceSolution,
    LocalFresnel,
    SpectralFresnel,
    scatter_plane,
)
from vecdiff.propagation.surface_transfer import InverseReport, SurfaceTransport

__all__ = [
    "CoverageReport",
    "PlaneTransport",
    "propagate",
    "propagate_spectrum",
    "SurfaceTransport",
    "InverseReport",
    "EikonalTransport",
    "EikonalReport",
    "InterfaceMap",
    "RayMap",
    "CarrierFresnel",
    "DirectionResolvedFresnel",
    "LocalFresnel",
    "SpectralFresnel",
    "InterfaceSolution",
    "scatter_plane",
    "PropagationPipeline",
    "PropagationSolution",
]
