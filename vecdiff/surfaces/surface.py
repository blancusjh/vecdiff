"""Parametric surfaces expose positions and both tangents on the same chart."""
from abc import ABC, abstractmethod
import numpy as np
from ..geometry.frames import Frame


class Surface(ABC):
    @abstractmethod
    def position(self, u, v): pass

    @abstractmethod
    def tangents(self, u, v): pass

    def normal_and_jacobian(self, u, v):
        a, b = self.tangents(u, v)
        c = np.cross(a, b)
        jac = np.linalg.norm(c, axis=-1)
        if np.any(jac <= 0): raise ValueError("surface chart is singular at these coordinates")
        return c/jac[..., None], jac


class Plane(Surface):
    def __init__(self, frame=None): self.frame = frame or Frame()
    def position(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        return self.frame.points(np.stack((u, v, np.zeros_like(u)), axis=-1))
    def tangents(self, u, v):
        shape = np.broadcast_shapes(np.shape(u), np.shape(v))+(3,)
        return np.broadcast_to(self.frame.rotation[:, 0], shape), np.broadcast_to(self.frame.rotation[:, 1], shape)


class FreeformSurface(Surface):
    """Graph z=sag(x,y), with a supplied analytic or measured gradient."""
    def __init__(self, sag, gradient, frame=None):
        self.sag, self.gradient, self.frame = sag, gradient, frame or Frame()
    def position(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        return self.frame.points(np.stack((u, v, np.broadcast_to(self.sag(u, v), u.shape)), axis=-1))
    def tangents(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        gx, gy = self.gradient(u, v)
        one, zero = np.ones_like(u), np.zeros_like(u)
        return (self.frame.vectors(np.stack((one, zero, np.broadcast_to(gx, u.shape)), axis=-1)),
                self.frame.vectors(np.stack((zero, one, np.broadcast_to(gy, u.shape)), axis=-1)))
