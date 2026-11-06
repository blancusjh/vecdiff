"""Scalar transverse amplitudes of beams, functions of plane coordinates in metres.

A profile p(x, y) multiplies a polarization state; it carries no vector
direction. x and y are the u and v coordinates of a beam's plane (PlaneBeam)
or of its entrance pupil (FocusedBeam). Any callable p(x, y) is a profile; the
classes below also declare their support and spatial-frequency bandwidth, so
quadratures need not be told.
"""

from dataclasses import dataclass

import numpy as np

from vecdiff_general._arrays import positive
from vecdiff_general._backend import bd

# exp(-ROUNDOFF_EXPONENT) is below double-precision roundoff relative to 1.
ROUNDOFF_EXPONENT = -bd.log(np.finfo(float).eps / 2)


@dataclass(frozen=True)
class Uniform:
    """Unit amplitude on the closed disk of ``radius``; unbounded if radius is None."""

    radius: float | None = None

    def __post_init__(self):
        if self.radius is not None:
            object.__setattr__(self, "radius", positive(self.radius, "profile radius"))

    def __call__(self, x, y):
        x, y = bd.broadcast_arrays(bd.asarray(x, float), bd.asarray(y, float))

        if self.radius is None:
            return bd.ones(x.shape)

        return (bd.hypot(x, y) <= self.radius).astype(float)

    @property
    def support(self):
        """Radius beyond which the profile vanishes (None: unbounded)."""
        return self.radius

    @property
    def bandwidth(self):
        """A hard edge leaves every transverse wavenumber populated."""
        return None


@dataclass(frozen=True)
class Gaussian:
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
        return self.waist * bd.sqrt(ROUNDOFF_EXPONENT)

    @property
    def bandwidth(self):
        """Transverse wavenumber beyond which the transform exp(-k² w²/4) is roundoff."""
        return 2 * bd.sqrt(ROUNDOFF_EXPONENT) / self.waist
