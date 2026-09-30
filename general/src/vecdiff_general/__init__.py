"""General vector optics: SI phasors with explicit geometry and material context."""

from .fields import ElectricField, ElectromagneticField, TransverseElectricField
from .fields.eikonal import EikonalField
from .fields.spectrum import PlaneSpectrum, PlaneWaveExpansion
from .geometry import Frame
from .interfaces import PlanarInterface
from .interfaces.dielectric_interface import DielectricInterface
from .interfaces.optical_system import OpticalSystem
from .media import EPSILON0, MU0, Z0, Dielectric, HomogeneousRegion
from .media.distribution import MaterialDistribution, MaterialRegion
from .media.layers import LayerStack
from .observables.diagnostics import boundary_residuals
from .propagation import InterfaceSolution, scatter_plane
from .propagation.angular_spectrum import angular_spectrum
from .propagation.high_frequency import HighFrequencySolution, propagate_high_frequency
from .propagation.interface_assembly import (
    propagate_coherent_planes,
    propagate_interfaces,
)
from .propagation.layered_propagation import LayeredSolution, propagate_layers
from .propagation.operators import (
    FresnelEncounter,
    InverseReport,
    PlanarASM,
    SurfaceTransport,
    fresnel_transmission,
)
from .propagation.propagation import propagate
from .propagation.tiled_transport import TiledSurfaceTransport
from .sampling import CartesianGrid, PlaneGrid, PointGrid, PolarGrid, SurfaceGrid
from .sampling.surface_sampling import sample_surface
from .surfaces import (
    AxisymmetricSurface,
    CartesianOval,
    EvenAsphere,
    FreeformSurface,
    GraphSurface,
    Plane,
    SphericalCap,
    gots_parameters,
)
from .volumes import SphereVolume, VolumeBetweenSurfaces

__all__ = [
    "CartesianGrid",
    "PolarGrid",
    "EikonalField",
    "VolumeBetweenSurfaces",
    "SphereVolume",
    "DielectricInterface",
    "OpticalSystem",
    "LayerStack",
    "MaterialRegion",
    "MaterialDistribution",
    "propagate",
    "propagate_high_frequency",
    "HighFrequencySolution",
    "propagate_layers",
    "LayeredSolution",
    "propagate_interfaces",
    "propagate_coherent_planes",
    "sample_surface",
    "GraphSurface",
    "Plane",
    "FreeformSurface",
    "AxisymmetricSurface",
    "SphericalCap",
    "EvenAsphere",
    "CartesianOval",
    "gots_parameters",
    "Frame",
    "PlaneGrid",
    "PointGrid",
    "SurfaceGrid",
    "Dielectric",
    "HomogeneousRegion",
    "Z0",
    "EPSILON0",
    "MU0",
    "ElectricField",
    "TransverseElectricField",
    "ElectromagneticField",
    "PlaneWaveExpansion",
    "PlaneSpectrum",
    "angular_spectrum",
    "PlanarInterface",
    "InterfaceSolution",
    "scatter_plane",
    "boundary_residuals",
    "PlanarASM",
    "SurfaceTransport",
    "TiledSurfaceTransport",
    "FresnelEncounter",
    "InverseReport",
    "fresnel_transmission",
]
