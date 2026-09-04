"""Sampling coordinates; grids do not own fields or physical propagation."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, eq=False)
class CartesianGrid:
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        for name in ("x", "y"):
            a = np.array(getattr(self, name), dtype=float, copy=True)
            if a.ndim != 1 or a.size < 2 or not np.isfinite(a).all():
                raise ValueError("axes must contain at least two finite coordinates")
            d = np.diff(a)
            if d[0] <= 0 or not np.allclose(d, d[0], rtol=1e-10, atol=1e-14*abs(d[0])):
                raise ValueError("CartesianGrid requires uniform increasing axes")
            a.setflags(write=False)
            object.__setattr__(self, name, a)

    @classmethod
    def from_spacing(cls, spacing, count):
        x = (np.arange(count) - count // 2) * spacing
        return cls(x, x)

    @property
    def shape(self): return (self.y.size, self.x.size)
    @property
    def dx(self): return self.x[1] - self.x[0]
    @property
    def dy(self): return self.y[1] - self.y[0]
    @property
    def xy(self): return np.meshgrid(self.x, self.y)
    @property
    def kxy(self):
        return np.meshgrid(2*np.pi*np.fft.fftfreq(self.x.size, self.dx),
                           2*np.pi*np.fft.fftfreq(self.y.size, self.dy))
    @property
    def period_area(self): return self.x.size*self.y.size*self.dx*self.dy


@dataclass(frozen=True, eq=False)
class PointSampling:
    points: np.ndarray

    def __post_init__(self):
        a = np.array(self.points, float, copy=True)
        if a.ndim < 2 or a.shape[-1] != 3 or not np.isfinite(a).all():
            raise ValueError("points must have shape (..., 3)")
        a.setflags(write=False)
        object.__setattr__(self, "points", a)

    @property
    def shape(self): return self.points.shape[:-1]
