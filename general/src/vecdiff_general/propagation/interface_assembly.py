"""Explicit sequential propagation and exact coherent parallel-plane assemblies."""

import numpy as np

from ..media.layers import LayerStack
from ..surfaces.surface import Plane
from .layered_propagation import propagate_layers


def propagate_interfaces(source, system, grid, **kwargs):
    """Forward operational path R Q^-1 T Q R; see propagate_high_frequency."""
    from .high_frequency import propagate_high_frequency

    return propagate_high_frequency(source, system, grid, **kwargs)


def propagate_coherent_planes(source, system):
    """Convert a parallel-plane physical prescription to the all-orders layer solver."""

    if any(aperture is not None for aperture in system.apertures):
        raise ValueError(
            "coherent plane layers describe infinite faces; finite-aperture "
            "diffraction requires a different model"
        )

    first = system.interfaces[0]

    if not all(isinstance(interface.surface, Plane) for interface in system.interfaces):
        raise ValueError(
            "exact coherent assembly currently requires parallel planar interfaces"
        )

    frame = first.surface.frame
    positions = []

    for interface in system.interfaces:
        if not np.allclose(
            interface.surface.frame.basis, frame.basis, rtol=0, atol=1e-12
        ):
            raise ValueError("coherent planes must share a frame orientation")

        positions.append(
            np.dot(interface.surface.frame.origin - frame.origin, frame.normal)
        )

    thicknesses = tuple(np.diff(positions))
    regions = (first.minus,) + tuple(interface.plus for interface in system.interfaces)

    return propagate_layers(source, LayerStack(regions, thicknesses, frame))
