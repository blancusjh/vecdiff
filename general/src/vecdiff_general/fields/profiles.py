"""Scalar transverse amplitudes of beams, functions of plane coordinates in metres.

A profile p(x, y) multiplies a polarization state; it carries no vector
direction. x and y are the u and v coordinates of a beam's plane (PlaneBeam)
or of its entrance pupil (FocusedBeam). Any callable p(x, y) is a profile; the
classes below also declare their support, spatial-frequency bandwidth and
edges, so quadratures need not be told. Profiles multiply: ``Gaussian(w) *
Annulus(a, b)`` is a Gaussian behind an annular aperture.
"""

from dataclasses import dataclass

import numpy as np

from vecdiff_general._arrays import positive
from vecdiff_general._backend import bd

# exp(-ROUNDOFF_EXPONENT) is below double-precision roundoff relative to 1.
ROUNDOFF_EXPONENT = -np.log(np.finfo(float).eps / 2)


class _Profile:
    """Products with other profiles and with plain callables p(x, y)."""

    def __mul__(self, other):
        if not callable(other):
            return NotImplemented

        return Product((self, other))

    def __rmul__(self, other):
        if not callable(other):
            return NotImplemented

        return Product((other, self))


def _radii(x, y):
    x, y = bd.broadcast_arrays(bd.asarray(x, float), bd.asarray(y, float))

    return bd.hypot(x, y)


@dataclass(frozen=True)
class Uniform(_Profile):
    """Unit amplitude on the closed disk of ``radius``; unbounded if radius is None."""

    radius: float | None = None

    def __post_init__(self):
        if self.radius is not None:
            object.__setattr__(self, "radius", positive(self.radius, "profile radius"))

    def __call__(self, x, y):
        rho = _radii(x, y)

        if self.radius is None:
            return bd.ones(rho.shape)

        return (rho <= self.radius).astype(float)

    @property
    def support(self):
        """Radius beyond which the profile vanishes (None: unbounded)."""
        return self.radius

    @property
    def bandwidth(self):
        """A hard edge leaves every transverse wavenumber populated."""
        return None

    @property
    def edges(self):
        """Radii where the amplitude jumps."""
        return () if self.radius is None else (self.radius,)


@dataclass(frozen=True)
class Annulus(_Profile):
    """Unit amplitude on the closed ring ``inner`` <= rho <= ``outer``."""

    inner: float
    outer: float

    def __post_init__(self):
        inner = positive(self.inner, "annulus inner radius")
        outer = positive(self.outer, "annulus outer radius")

        if outer <= inner:
            raise ValueError("annulus outer radius must exceed the inner radius")

        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "outer", outer)

    def __call__(self, x, y):
        rho = _radii(x, y)

        return ((rho >= self.inner) & (rho <= self.outer)).astype(float)

    @property
    def support(self):
        return self.outer

    @property
    def bandwidth(self):
        return None

    @property
    def edges(self):
        return (self.inner, self.outer)


@dataclass(frozen=True)
class Zones(_Profile):
    """Constant values on concentric zones: stops, apertures and phase plates.

    ``values[0]`` holds for rho <= ``edges[0]``, ``values[i]`` for
    ``edges[i - 1]`` < rho <= ``edges[i]`` and ``values[-1]`` beyond the last
    edge. Values may be complex: a binary phase plate alternates 1 and -1, a
    zero blocks its zone.
    """

    edges: tuple
    values: tuple

    def __post_init__(self):
        edges = tuple(positive(edge, "zone edge") for edge in self.edges)
        values = np.asarray(self.values)

        if values.ndim != 1 or values.dtype.kind not in "biufc":
            raise TypeError("zone values must be a sequence of numbers")

        if len(values) != len(edges) + 1:
            raise ValueError("Zones takes one value more than it has edges")

        if any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError("zone edges must increase")

        if not values.any():
            raise ValueError("at least one zone must be nonzero")

        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "values", tuple(values.tolist()))

    def __call__(self, x, y):
        zone = bd.searchsorted(bd.asarray(self.edges, float), _radii(x, y))

        return bd.asarray(self.values)[zone]

    @property
    def support(self):
        """The edge beyond which every zone is zero (None: unbounded)."""
        last = max(i for i, value in enumerate(self.values) if value != 0)

        return None if last == len(self.edges) else self.edges[last]

    @property
    def bandwidth(self):
        return None


@dataclass(frozen=True)
class Product(_Profile):
    """Pointwise product of profiles; also written ``a * b``.

    Its support is the smallest declared support, its edges the union of the
    factors' edges inside that support, and its bandwidth the sum of the
    factors' bandwidths when every factor declares one. A plain callable
    counts as unbounded, edgeless and of unknown bandwidth.
    """

    factors: tuple

    def __post_init__(self):
        flat = []

        for factor in self.factors:
            if not callable(factor):
                raise TypeError("profile factors must be callables p(x, y)")

            flat.extend(factor.factors if isinstance(factor, Product) else [factor])

        object.__setattr__(self, "factors", tuple(flat))

    def __call__(self, x, y):
        result = self.factors[0](x, y)

        for factor in self.factors[1:]:
            result = result * factor(x, y)

        return result

    @property
    def support(self):
        supports = [getattr(f, "support", None) for f in self.factors]
        supports = [s for s in supports if s is not None]

        return min(supports) if supports else None

    @property
    def bandwidth(self):
        bandwidths = [getattr(f, "bandwidth", None) for f in self.factors]

        return None if None in bandwidths else sum(bandwidths)

    @property
    def edges(self):
        support = self.support
        edges = {e for f in self.factors for e in getattr(f, "edges", ())}

        return tuple(sorted(e for e in edges if support is None or e <= support))


@dataclass(frozen=True)
class Gaussian(_Profile):
    """exp(-(x² + y²) / waist²): ``waist`` is the 1/e field radius."""

    waist: float

    def __post_init__(self):
        object.__setattr__(self, "waist", positive(self.waist, "Gaussian waist"))

    def __call__(self, x, y):
        x, y = bd.broadcast_arrays(bd.asarray(x, float), bd.asarray(y, float))

        return bd.exp(-(x**2 + y**2) / self.waist**2)

    @property
    def support(self):
        """Radius beyond which the amplitude is below double-precision roundoff."""
        return self.waist * float(np.sqrt(ROUNDOFF_EXPONENT))

    @property
    def bandwidth(self):
        """Transverse wavenumber beyond which the transform exp(-k² w²/4) is roundoff."""
        return 2 * float(np.sqrt(ROUNDOFF_EXPONENT)) / self.waist

    @property
    def edges(self):
        return ()
