"""Cartesian-oval geometry defined by optical path, without a field solver."""
import numpy as np
from scipy.optimize import newton
from .axisymmetric import AxisymmetricSurface


class CartesianOval(AxisymmetricSurface):
    def __init__(self, object_z, image_z, n1, n2, frame=None):
        super().__init__(frame)
        if not (object_z < 0 < image_z and n1 > 0 and n2 > 0 and n1 != n2):
            raise ValueError("requires object_z < 0 < image_z and distinct positive design indices")
        self.object_z, self.image_z, self.n1, self.n2 = object_z, image_z, n1, n2
    def _derivatives(self, r, z):
        a, b = np.hypot(r, z-self.object_z), np.hypot(r, z-self.image_z)
        return r*(self.n1/a+self.n2/b), self.n1*(z-self.object_z)/a+self.n2*(z-self.image_z)/b
    def sag(self, radius):
        r = np.asarray(radius, float)
        path = self.n1*abs(self.object_z)+self.n2*abs(self.image_z)
        f = lambda z: self.n1*np.hypot(r, z-self.object_z)+self.n2*np.hypot(r, z-self.image_z)-path
        z = newton(f, np.zeros_like(r), fprime=lambda z: self._derivatives(r, z)[1], tol=1e-11, maxiter=100)
        if not np.isfinite(z).all() or np.any(abs(f(z)) > 1e-9*path):
            raise ValueError("oval graph did not converge at this radius")
        return z
    def slope(self, radius):
        fr, fz = self._derivatives(np.asarray(radius), self.sag(radius))
        if np.any(abs(fz) < 1e-12): raise ValueError("oval graph has a vertical tangent")
        return -fr/fz
