"""Propagation of light: in homogeneous media, and through a whole optical system.

``Propagator(system)(light)`` is the method: it composes the transport of
the field across each surface (``vecdiff.transport``) with its propagation
as plane waves between surfaces, and returns a ``Solution``.
"""

from vecdiff.propagation.coverage import CoverageReport
from vecdiff.propagation.homogeneous import (
    PlaneTransport,
    propagate,
    propagate_spectrum,
)
from vecdiff.propagation.propagator import Propagator, Solution

__all__ = [
    "Propagator",
    "Solution",
    "CoverageReport",
    "PlaneTransport",
    "propagate",
    "propagate_spectrum",
]
