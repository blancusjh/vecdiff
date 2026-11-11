"""Lossless nondispersive matter and homogeneous solution context."""

from dataclasses import dataclass

from vecdiff._arrays import positive

# SI constants. mu0 uses the conventional value to the precision needed here.
C0 = 299792458.0
MU0 = 1.25663706212e-6
EPSILON0 = 1.0 / (MU0 * C0 * C0)
Z0 = MU0 * C0


@dataclass(frozen=True)
class Dielectric:
    n: float
    name: str = "dielectric"

    def __post_init__(self):
        object.__setattr__(self, "n", positive(self.n, "refractive index"))

    @property
    def epsilon(self):
        return EPSILON0 * self.n**2


@dataclass(frozen=True)
class HomogeneousRegion:
    """A part of space filled with one material.

    Its boundary is the system's: the surfaces of the path around it. Two
    regions of the same material are distinct parts of space when their
    names differ.
    """

    name: str
    material: Dielectric

    def __post_init__(self):
        if not isinstance(self.material, Dielectric):
            raise TypeError("region material must be a Dielectric")


# The common media; air is taken as index 1, as vacuum.
vacuum = Dielectric(1.0, "vacuum")
air = Dielectric(1.0, "air")
