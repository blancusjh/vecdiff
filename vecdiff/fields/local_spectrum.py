"""A homogeneous electric spectrum with a finite domain of approximation."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LocalElectricSpectrum:
    """Plane waves referenced to a centre, valid within an explicit ball.

    The error bounds concern the radiation of the supplied discrete currents.
    They exclude source quadrature error and error in the interface traces.
    """
    spectrum: object
    center: np.ndarray
    radius: float
    electric_error_bound: float
    magnetic_error_bound: float

    def __post_init__(self):
        center = np.array(self.center, float, copy=True)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("center must be a finite three-vector")
        if not np.isfinite([self.radius, self.electric_error_bound, self.magnetic_error_bound]).all() or min(self.radius, self.electric_error_bound, self.magnetic_error_bound) < 0:
            raise ValueError("radius and error bounds must be finite and nonnegative")
        center.setflags(write=False); object.__setattr__(self, "center", center)

    def evaluate(self, points, *, backend="direct"):
        p = np.asarray(points, float)
        if p.shape[-1:] != (3,) or not np.isfinite(p).all():
            raise ValueError("points must be finite (...,3)")
        offsets = p-self.center
        if np.any(np.linalg.norm(offsets, axis=-1) > self.radius*(1+1e-12)):
            raise ValueError("observation outside the local spectrum's certified ball")
        return self.spectrum.evaluate(offsets, backend=backend)
