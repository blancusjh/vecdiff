"""Rotational conics and even aspheres; all lengths share the caller's unit."""
import numpy as np
from .axisymmetric import AxisymmetricSurface


class EvenAsphere(AxisymmetricSurface):
    """z = c*r²/(1+sqrt(1-(1+K)*c²*r²)) + sum A_j*r**(2*j+4).

    A zero curvature is planar unless polynomial coefficients are supplied.
    Coefficients start at r**4, matching the native prescription CSV dialect.
    """
    def __init__(self, curvature=0., conic=0., coefficients=(), frame=None):
        super().__init__(frame)
        if not np.isfinite([curvature, conic, *coefficients]).all():
            raise ValueError("asphere parameters must be finite")
        self.curvature, self.conic = float(curvature), float(conic)
        self.coefficients = tuple(float(a) for a in coefficients)

    def _root(self, r):
        r = np.asarray(r, float)
        d = 1-(1+self.conic)*(self.curvature*r)**2
        if not np.isfinite(r).all() or np.any(d <= 0):
            raise ValueError("asphere chart requires a finite radius and positive radicand")
        return r, np.sqrt(d)

    def sag(self, radius):
        r, root = self._root(radius)
        return self.curvature*r*r/(1+root) + sum(a*r**(2*j+4) for j, a in enumerate(self.coefficients))

    def slope(self, radius):
        r, root = self._root(radius)
        return self.curvature*r/root + sum((2*j+4)*a*r**(2*j+3) for j, a in enumerate(self.coefficients))
