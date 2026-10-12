"""Lossless nondispersive matter and homogeneous solution context."""

from dataclasses import dataclass

from vecdiff_general._arrays import positive

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
    """Material context. A system/interface supplies its spatial boundary."""

    name: str
    material: Dielectric

    def __post_init__(self):
        if not isinstance(self.material, Dielectric):
            raise TypeError("region material must be a Dielectric")
