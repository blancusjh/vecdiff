"""Public operations with explicit physical model and grid contracts."""

from .angular_spectrum import angular_spectrum
from .high_frequency import HighFrequencySolution, propagate_high_frequency
from .interface_assembly import (
    propagate_coherent_planes,
    propagate_interfaces,
)
from .interface_transform import InterfaceSolution, scatter_plane
from .layered_propagation import LayeredSolution, propagate_layers
from .multiple_scattering import FeedbackResult, coherent_feedback
from .operators import (
    FresnelEncounter,
    InverseReport,
    PlanarASM,
    SurfaceTransport,
    fresnel_transmission,
)
from .propagation import propagate, propagate_spectrum
from .tiled_transport import TiledSurfaceTransport

__all__ = [
    "angular_spectrum",
    "propagate",
    "propagate_spectrum",
    "InterfaceSolution",
    "scatter_plane",
    "LayeredSolution",
    "propagate_layers",
    "HighFrequencySolution",
    "propagate_high_frequency",
    "propagate_interfaces",
    "propagate_coherent_planes",
    "FeedbackResult",
    "coherent_feedback",
    "PlanarASM",
    "SurfaceTransport",
    "TiledSurfaceTransport",
    "FresnelEncounter",
    "InverseReport",
    "fresnel_transmission",
]
