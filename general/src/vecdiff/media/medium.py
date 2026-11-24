"""Linear isotropic nonmagnetic matter, lossless or absorbing, and the
homogeneous regions it fills.

The fields vary in time as exp(-i omega t). A medium is its refractive index
n at the wavelength of the light: n = n' + i kappa with n' > 0 and kappa >= 0,
so that a plane wave exp(i n k0 z) decays as exp(-kappa k0 z). ``Medium`` is
any such matter, a metal (silver in the visible has n' of a few hundredths
and kappa of three to four) or a lossy glass among them; ``Dielectric`` is
the lossless one, of real n.
"""

import math
from dataclasses import dataclass

from vecdiff._printing import guarded, number

# SI constants. mu0 uses the conventional value to the precision needed here.
C0 = 299792458.0
MU0 = 1.25663706212e-6
EPSILON0 = 1.0 / (MU0 * C0 * C0)
Z0 = MU0 * C0


def refractive_index(value):
    """``value`` as a refractive index: a float when it is real, else a complex.

    Refuses what is not a passive medium of the exp(-i omega t) convention:
    a real part that is not positive, or a negative imaginary part (gain).
    """
    try:
        n = complex(value)
    except (TypeError, ValueError):
        raise TypeError(
            "a refractive index is a number: n, or n' + i kappa for an absorbing "
            f"medium, not {value!r}"
        ) from None
    if not (math.isfinite(n.real) and math.isfinite(n.imag)):
        raise ValueError("refractive index must be finite")
    shown = f"{n.real:g}" if n.imag == 0 else f"{n:g}"
    if n.imag < 0:
        raise ValueError(
            f"refractive index {shown} has a negative imaginary part. The fields "
            "here vary as exp(-i omega t), so an absorbing medium has "
            "n = n' + i kappa with kappa >= 0, and kappa < 0 would amplify the "
            "light (gain). An index tabulated for exp(+i omega t) as n' - i kappa "
            "is the complex conjugate of the one meant here"
        )
    if n.real <= 0:
        raise ValueError(
            f"refractive index {shown} must have a positive real part (a metal "
            "has a small one, such as 0.05 + 3.3i, but not zero or negative)"
        )
    return n.real if n.imag == 0 else n


@dataclass(frozen=True, eq=False)
class Medium:
    """Linear, isotropic, nonmagnetic matter of refractive index ``n``.

    ``n`` is n' + i kappa at the wavelength of the light (exp(-i omega t)),
    n' > 0 and the extinction coefficient kappa >= 0: a float when the
    medium is lossless, else a complex. Two media are equal when their index
    and name are, whichever class names them.
    """

    n: complex
    name: str = "medium"

    def __post_init__(self):
        object.__setattr__(self, "n", refractive_index(self.n))

    def __eq__(self, other):
        if not isinstance(other, Medium):
            return NotImplemented
        return (self.n, self.name) == (other.n, other.name)

    def __hash__(self):
        return hash((self.n, self.name))

    @property
    def n_real(self):
        """n', the real part of the index: the phase index."""
        return float(complex(self.n).real)

    @property
    def extinction(self):
        """The extinction coefficient kappa = Im n >= 0."""
        return float(complex(self.n).imag)

    @property
    def absorbing(self):
        """Whether the medium absorbs: kappa > 0."""
        return self.extinction > 0

    @property
    def lossless(self):
        """Whether the medium is transparent: a real index."""
        return not self.absorbing

    @property
    def relative_permittivity(self):
        """epsilon / epsilon0 = n^2, complex when the medium absorbs."""
        return self.n**2

    @property
    def epsilon(self):
        """The permittivity epsilon0 n^2 (complex when the medium absorbs)."""
        return EPSILON0 * self.n**2

    @guarded
    def __repr__(self):
        """Its name and refractive index: Dielectric('BK7-like', n=1.5)."""
        if self.absorbing:
            index = f"{number(self.n.real)}+{number(self.n.imag)}i"
        else:
            index = number(self.n)
        return f"{type(self).__name__}({self.name!r}, n={index})"


@dataclass(frozen=True, eq=False, repr=False)
class Dielectric(Medium):
    """Lossless nondispersive matter: a medium of real refractive index ``n``.

    An absorbing medium (complex n) is a ``Medium``.
    """

    n: float
    name: str = "dielectric"

    def __post_init__(self):
        super().__post_init__()
        if self.absorbing:
            raise ValueError(
                f"a Dielectric is lossless, of real index; n = {self.n:g} absorbs: "
                "an absorbing medium (a metal, a lossy glass) is a media.Medium"
            )


@dataclass(frozen=True)
class HomogeneousRegion:
    """A part of space filled with one material.

    Its boundary is the system's: the surfaces of the path around it. Two
    regions of the same material are distinct parts of space when their
    names differ.
    """

    name: str
    material: Medium

    def __post_init__(self):
        if not isinstance(self.material, Medium):
            raise TypeError("region material must be a media.Medium or Dielectric")

    @guarded
    def __repr__(self):
        """Its name and the material filling it."""
        return f"HomogeneousRegion({self.name!r}, {self.material!r})"


# The common media; air is taken as index 1, as vacuum.
vacuum = Dielectric(1.0, "vacuum")
air = Dielectric(1.0, "air")
