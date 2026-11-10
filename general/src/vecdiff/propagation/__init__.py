"""General propagation: homogeneous media, and a field through a whole optical system.

The transport of a field across each surface is in ``vecdiff.transport``;
the pipeline here composes it with homogeneous propagation between surfaces.
"""

from vecdiff.propagation.coverage import CoverageReport
from vecdiff.propagation.homogeneous import (
    PlaneTransport,
    propagate,
    propagate_spectrum,
)
from vecdiff.propagation.pipeline import (
    PropagationPipeline,
    PropagationSolution,
)

__all__ = [
    "CoverageReport",
    "PlaneTransport",
    "propagate",
    "propagate_spectrum",
    "PropagationPipeline",
    "PropagationSolution",
]
