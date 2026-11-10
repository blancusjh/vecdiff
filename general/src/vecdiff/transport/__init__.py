"""Transport of a field across a surface, along the rays of its carrier.

A point of a reference plane follows its carrier ray to the surface, is
refracted there, and reaches the same geometric plane in the next medium:
``InterfaceMap`` is that map and ``RayMap`` its rays. A leg carries a field
along them, on a lattice for any geometry (``LatticeTransport``) or on one
meridian for a system of revolution (``MeridianTransport``), and reports what
it could not carry (``TransportReport``).
"""

from vecdiff.transport.dyadic import carried, check_window, ray_weights
from vecdiff.transport.lattice import LatticeTransport
from vecdiff.transport.meridian import MeridianTransport
from vecdiff.transport.rays import InterfaceMap, RayMap, RefractedRays
from vecdiff.transport.report import TransportReport

__all__ = [
    "InterfaceMap",
    "RayMap",
    "RefractedRays",
    "LatticeTransport",
    "MeridianTransport",
    "TransportReport",
    "carried",
    "ray_weights",
    "check_window",
]
