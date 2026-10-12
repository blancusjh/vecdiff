"""Vector optics: fields, physical systems, transport, interaction and pipelines."""

from vecdiff_general.catalog import thorlabs_al1225
from vecdiff_general.fields import (
    ElectricField,
    ElectromagneticField,
    TransverseElectricField,
)
from vecdiff_general.fields.angular_spectrum import angular_spectrum
from vecdiff_general.fields.eikonal import EikonalField
from vecdiff_general.fields.spectrum import PlaneSpectrum, PlaneWaveExpansion
from vecdiff_general.geometry import Frame
from vecdiff_general.media import EPSILON0, MU0, Z0, Dielectric, HomogeneousRegion
from vecdiff_general.media.distribution import MaterialDistribution, MaterialRegion
from vecdiff_general.observables.diagnostics import boundary_residuals
from vecdiff_general.propagation import (
    CoverageReport,
    InverseReport,
    PlaneTransport,
    SurfaceTransport,
    propagate,
)
from vecdiff_general.propagation.pipeline import (
    PropagationPipeline,
    PropagationSolution,
)
from vecdiff_general.propagation.surface_interaction import (
    DirectionResolvedFresnel,
    InterfaceSolution,
    LocalFresnel,
    SpectralFresnel,
    scatter_plane,
)
from vecdiff_general.sampling import (
    CartesianGrid,
    PlaneGrid,
    PointGrid,
    PolarGrid,
    SurfaceGrid,
)
from vecdiff_general.sampling.surface_sampling import sample_surface
from vecdiff_general.surfaces import (
    AxisymmetricSurface,
    CartesianOval,
    EvenAsphere,
    FreeformSurface,
    GraphSurface,
    Plane,
    SphericalCap,
    gots_parameters,
)
from vecdiff_general.system import DielectricInterface, OpticalSystem, PlanarInterface
from vecdiff_general.volumes import SphereVolume, VolumeBetweenSurfaces

__all__ = [
    "CoverageReport",
    "thorlabs_al1225",
    "ElectricField",
    "ElectromagneticField",
    "TransverseElectricField",
    "EikonalField",
    "PlaneSpectrum",
    "PlaneWaveExpansion",
    "angular_spectrum",
    "Frame",
    "EPSILON0",
    "MU0",
    "Z0",
    "Dielectric",
    "HomogeneousRegion",
    "MaterialDistribution",
    "MaterialRegion",
    "boundary_residuals",
    "CartesianGrid",
    "PlaneGrid",
    "PointGrid",
    "PolarGrid",
    "SurfaceGrid",
    "sample_surface",
    "AxisymmetricSurface",
    "CartesianOval",
    "EvenAsphere",
    "FreeformSurface",
    "GraphSurface",
    "Plane",
    "SphericalCap",
    "gots_parameters",
    "DielectricInterface",
    "OpticalSystem",
    "PlanarInterface",
    "PlaneTransport",
    "SurfaceTransport",
    "InverseReport",
    "propagate",
    "DirectionResolvedFresnel",
    "LocalFresnel",
    "SpectralFresnel",
    "scatter_plane",
    "InterfaceSolution",
    "PropagationPipeline",
    "PropagationSolution",
    "SphereVolume",
    "VolumeBetweenSurfaces",
]
