"""A single smooth phase and its vector electric envelope (leading WKB order)."""
from dataclasses import dataclass
import numpy as np
from ..media.medium import Medium


@dataclass(frozen=True)
class EikonalElectricField:
    """E(r)=A(r) exp(i k0 L(r)), with |grad L|=n and A.grad L=0.

optical_path, gradient and envelope are callables of global (...,3) positions.
L is optical path LENGTH; gradient is its analytic spatial gradient. Direction
is grad L/n: the leading phase-gradient term of the momentum operator -i grad.
The envelope derivative is omitted, so this representation is explicitly a
leading high-frequency field, not an exact divergence-free Maxwell spectrum.
Do not assign one phase to an unresolved coherent superposition of branches.
    """
    optical_path: object
    gradient: object
    envelope: object
    wavelength: float = 1.
    medium: Medium = Medium()

    def __post_init__(self):
        if not all(callable(f) for f in (self.optical_path,self.gradient,self.envelope)):
            raise TypeError('optical_path, gradient and envelope must be callable')
        self.medium.wavenumber(self.wavelength)

    def directions(self,points):
        p=np.asarray(points,float);g=np.broadcast_to(np.asarray(self.gradient(p),float),p.shape)
        if not np.isfinite(g).all() or not np.allclose(np.linalg.norm(g,axis=-1),self.medium.n,rtol=1e-9,atol=1e-12):
            raise ValueError('phase gradient must satisfy |grad L|=n')
        return g/self.medium.n

    def amplitudes(self,points):
        p=np.asarray(points,float);a=np.broadcast_to(np.asarray(self.envelope(p),complex),p.shape).copy()
        if not np.isfinite(a).all() or np.any(abs(np.sum(a*self.directions(p),axis=-1))>1e-9*np.linalg.norm(a,axis=-1)+1e-14):
            raise ValueError('electric envelope must be finite and transverse to the phase gradient')
        return a
