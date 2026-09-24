"""A physically explicit ordered forward path through dielectric interfaces."""

from dataclasses import dataclass

import numpy as np

from .dielectric_interface import DielectricInterface


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
    its edge, flat at rim height beyond). A stop cuts every field reaching it,
    inside and outside the material. None means no pupil there.
    """

    interfaces: tuple[DielectricInterface, ...]
    apertures: tuple[float | None, ...] | None = None
    stops: tuple[float | None, ...] | None = None

    def __post_init__(self):
        interfaces = tuple(self.interfaces)

        if not interfaces or not all(
            isinstance(value, DielectricInterface) for value in interfaces
        ):
            raise ValueError("an optical path needs dielectric interfaces")

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
