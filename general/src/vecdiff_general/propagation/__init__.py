"""Homogeneous propagation, surface transfer, and boundary interactions."""

from vecdiff_general.propagation.coverage import CoverageReport
from vecdiff_general.propagation.eikonal import (
    EikonalReport,
    EikonalTransport,
    InterfaceMap,
    RayMap,
)
from vecdiff_general.propagation.homogeneous import (
    PlaneTransport,
    propagate,
    propagate_spectrum,
)
from vecdiff_general.propagation.pipeline import (
    PropagationPipeline,
    PropagationSolution,
)
from vecdiff_general.propagation.surface_interaction import (
    CarrierFresnel,
    DirectionResolvedFresnel,
    InterfaceSolution,
    LocalFresnel,
    SpectralFresnel,
    scatter_plane,
)
from vecdiff_general.propagation.surface_transfer import InverseReport, SurfaceTransport

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
