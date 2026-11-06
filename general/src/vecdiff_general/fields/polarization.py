"""Transverse polarization states of beams, apart from their scalar profiles.

A state has two complex components on one of two bases of a beam's plane or
entrance pupil: the fixed Cartesian axes (u, v) of its frame, or the local
polar axes (radial, azimuthal) at each point. Components are amplitudes and
are not normalized: cartesian(1, 1j) carries twice the intensity of x.
"""

from dataclasses import dataclass

from vecdiff_general._backend import bd


@dataclass(frozen=True)
class Jones:
    """Components (a, b) on the "cartesian" (u, v) or "polar" (radial, azimuthal) basis."""

    components: tuple
    basis: str = "cartesian"

    def __post_init__(self):
        components = bd.asarray(self.components, dtype=complex)

        if components.shape != (2,) or not bd.isfinite(components).all():
            raise ValueError("a polarization state has two finite components")

        if self.basis not in ("cartesian", "polar"):
            raise ValueError("polarization basis must be 'cartesian' or 'polar'")

        object.__setattr__(self, "components", tuple(complex(c) for c in components))

    def vectors(self, azimuth):
        """(u, v) components at points of the given plane azimuths, shape (..., 2)."""
        a, b = self.components
        azimuth = bd.asarray(azimuth, float)

        if self.basis == "cartesian":
            return bd.broadcast_to(bd.array([a, b]), azimuth.shape + (2,))

        c, s = bd.cos(azimuth), bd.sin(azimuth)

        return bd.stack((a * c - b * s, a * s + b * c), axis=-1)


def cartesian(a, b):
    """a along the frame's u axis plus b along its v axis."""
    return Jones((a, b), "cartesian")


def polar(a, b):
    """a along the local radial direction plus b along the local azimuthal one."""
    return Jones((a, b), "polar")


x = cartesian(1, 0)
y = cartesian(0, 1)
radial = polar(1, 0)
azimuthal = polar(0, 1)
