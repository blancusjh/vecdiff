"""Surfaces of revolution and complete spheres have distinct charts."""
from abc import abstractmethod
import numpy as np
from .surface import Surface
from ..geometry.frames import Frame


class AxisymmetricSurface(Surface):
    def __init__(self, frame=None): self.frame = frame or Frame()
    @abstractmethod
    def sag(self, radius): pass
    @abstractmethod
    def slope(self, radius): pass
    def position(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        return self.frame.points(np.stack((u*np.cos(v), u*np.sin(v), self.sag(u)), axis=-1))
    def tangents(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        return (self.frame.vectors(np.stack((np.cos(v), np.sin(v), self.slope(u)), axis=-1)),
                self.frame.vectors(np.stack((-u*np.sin(v), u*np.cos(v), np.zeros_like(u)), axis=-1)))


class SphericalCap(AxisymmetricSurface):
    """Graph cap with vertex at frame.origin; signed radius places its center."""
    def __init__(self, radius, frame=None):
        super().__init__(frame)
        if not np.isfinite(radius) or radius == 0: raise ValueError("radius must be finite and nonzero")
        self.radius = radius
    def _root(self, r):
        if np.any(np.abs(r) >= abs(self.radius)): raise ValueError("cap chart requires rho < abs(radius)")
        return np.sqrt(self.radius**2-np.asarray(r)**2)
    def sag(self, r): return self.radius-np.sign(self.radius)*self._root(r)
    def slope(self, r): return np.sign(self.radius)*np.asarray(r)/self._root(r)


class Sphere(Surface):
    """Closed sphere: u=cos(theta), v=azimuth; normal points outward.

Chart order (phi, mu) would give outward orientation; the explicit normal
below records the chosen outward orientation with (mu, phi) coordinates.
    """
    is_closed = True

    def __init__(self, radius, frame=None):
        if not np.isfinite(radius) or radius <= 0: raise ValueError("radius must be positive")
        self.radius, self.frame = radius, frame or Frame()
    def position(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        if np.any(abs(u) > 1): raise ValueError("mu must lie in [-1,1]")
        s = np.sqrt(1-u*u)
        return self.frame.points(self.radius*np.stack((s*np.cos(v), s*np.sin(v), u), axis=-1))
    def tangents(self, u, v):
        u, v = np.broadcast_arrays(u, v); s = np.sqrt(1-u*u)
        if np.any(s == 0): raise ValueError("sphere chart excludes the poles")
        return (self.frame.vectors(self.radius*np.stack((-u/s*np.cos(v), -u/s*np.sin(v), np.ones_like(u)), axis=-1)),
                self.frame.vectors(self.radius*np.stack((-s*np.sin(v), s*np.cos(v), np.zeros_like(u)), axis=-1)))
    def normal_and_jacobian(self, u, v):
        p = self.position(u, v)
        return (p-self.frame.origin)/self.radius, np.full(p.shape[:-1], self.radius**2)
