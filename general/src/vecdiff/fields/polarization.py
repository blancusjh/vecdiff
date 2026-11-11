"""Transverse polarization states of beams, apart from their scalar profiles.

A state has two complex components on one of two bases of a beam's plane or
entrance pupil: the fixed Cartesian axes (u, v) of its frame, or the local
polar axes (radial, azimuthal) at each point. Components are amplitudes and
are not normalized: cartesian(1, 1j) carries twice the intensity of x.
"""

import cmath
import math
from dataclasses import dataclass

from vecdiff._backend import bd
from vecdiff._printing import block, guarded, number, vector


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

    def describe(self):
        """The state in words: x-polarized, radial, linear at 30 deg from u, ..."""
        a, b = self.components
        if self.basis == "polar":
            names = {(1, 0): "radial", (0, 1): "azimuthal"}
            return names.get((a, b), "polar")
        if b == 0:
            return "x-polarized" if a == 1 else "linear along u"
        if a == 0:
            return "y-polarized" if b == 1 else "linear along v"
        phase = cmath.phase(b / a)
        if abs(math.sin(phase)) < 1e-12:
            angle = math.degrees(math.atan2((b / a).real, 1))
            return f"linear at {number(angle)} deg from u"
        if abs(abs(a) - abs(b)) < 1e-12 * abs(a) and abs(math.cos(phase)) < 1e-12:
            return "circular"
        return "elliptical"

    @guarded
    def __repr__(self):
        """Its name in words, its components and their basis."""
        return (
            f"Jones({self.describe()}, components={vector(self.components)}, "
            f"basis={self.basis!r})"
        )

    @guarded
    def __str__(self):
        axes = "(u, v)" if self.basis == "cartesian" else "(radial, azimuthal)"
        return block(
            "Jones polarization state",
            [
                ("state", self.describe()),
                ("components", f"{vector(self.components)} on {axes}"),
            ],
        )

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
