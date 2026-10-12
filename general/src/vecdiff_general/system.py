"""An oriented planar boundary joining two homogeneous material regions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vecdiff_general.geometry.frames import Frame
from vecdiff_general.media.medium import HomogeneousRegion


@dataclass(frozen=True)
class PlanarInterface:
    """Frame normal points from the minus region into the plus region."""

    frame: Frame
    minus: HomogeneousRegion
    plus: HomogeneousRegion

    def __post_init__(self):
        if not isinstance(self.frame, Frame):
            raise TypeError("interface requires a Frame")

        if not all(isinstance(r, HomogeneousRegion) for r in (self.minus, self.plus)):
            raise TypeError("interface sides must be homogeneous regions")

        if self.minus == self.plus:
            raise ValueError(
                "name the two spatial regions distinctly, even for equal materials"
            )


@dataclass(frozen=True)
class DielectricInterface:
    """A placed surface separating minus and plus material regions.

    The surface's graph normal points from minus to plus. Incidence belongs to
    an encounter, not permanently to the surface itself.
    """

    surface: object
    minus: HomogeneousRegion
    plus: HomogeneousRegion

    def __post_init__(self):
        from vecdiff_general.surfaces.surface import GraphSurface

        if not isinstance(self.surface, GraphSurface):
            raise TypeError("interface requires a placed GraphSurface")

        if not all(
            isinstance(region, HomogeneousRegion) for region in (self.minus, self.plus)
        ):
            raise TypeError("interface sides must be homogeneous material regions")

        if self.minus == self.plus:
            raise ValueError("the two spatial regions need distinct identities")


@dataclass(frozen=True)
class OpticalSystem:
    """Forward encounters from each interface's minus side to its plus side.

    This prescription does not itself claim a closed material body, or include
    coherent returning branches. Those require a chosen solver/topology.

    Each aperture is the circular edge of that surface in its local (x, y)
    chart, in metres; None means the surface is unbounded. An edge is where
    the material face ends, not a pupil: it never cuts the field.

    Each stop is an opaque pupil at that surface: a circular hole of the given
    radius in an opaque screen lying on the surface's chart (the surface inside
    its edge, flat at rim height beyond). None means no pupil there.
    These describe the prescription; PropagationPipeline currently rejects
    finite edges and stops because it does not implement their scattering.
    """

    interfaces: tuple[DielectricInterface, ...] = ()
    apertures: tuple[float | None, ...] | None = None
    stops: tuple[float | None, ...] | None = None
    incident_field: object | None = None
    incident_region: HomogeneousRegion | None = None

    def __post_init__(self):
        interfaces = tuple(self.interfaces)

        if not all(isinstance(value, DielectricInterface) for value in interfaces):
            raise ValueError("optical paths contain dielectric interfaces")

        if self.incident_region is not None:
            if not isinstance(self.incident_region, HomogeneousRegion):
                raise TypeError("incident_region must be a HomogeneousRegion")
            if interfaces and self.incident_region != interfaces[0].minus:
                raise ValueError("incident_region disagrees with the first interface")

        for previous, following in zip(interfaces[:-1], interfaces[1:]):
            if previous.plus != following.minus:
                raise ValueError(
                    "adjacent encounters disagree about the intervening material region"
                )

        apertures = (
            tuple(self.apertures)
            if self.apertures is not None
            else (None,) * len(interfaces)
        )

        if len(apertures) != len(interfaces) or any(
            value is not None and (not np.isfinite(value) or value <= 0)
            for value in apertures
        ):
            raise ValueError("provide one positive aperture or None per encounter")

        stops = (
            tuple(self.stops) if self.stops is not None else (None,) * len(interfaces)
        )

        if len(stops) != len(interfaces) or any(
            value is not None and (not np.isfinite(value) or value <= 0)
            for value in stops
        ):
            raise ValueError("provide one positive stop radius or None per encounter")

        object.__setattr__(self, "interfaces", interfaces)
        object.__setattr__(self, "apertures", apertures)
        object.__setattr__(self, "stops", stops)
