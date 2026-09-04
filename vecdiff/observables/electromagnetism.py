"""Maxwell and energy diagnostics, using H_normalized=Z0*H_SI."""
import numpy as np
from scipy.constants import physical_constants

Z0 = physical_constants["characteristic impedance of vacuum"][0]


def poynting(E, H):
    """Time averaged W/m^2 for E in V/m and normalized H."""
    return .5*np.real(np.cross(E, np.conj(H)))/Z0


def boundary_residuals(E1, H1, E2, H2, normals, medium1, medium2, *, weights=None, electric_scale=None, magnetic_scale=None):
    """RMS jumps for a charge-free, current-free dielectric interface.

Tangential E,H and normal D,B are tested independently. Normal D is scaled
by max(epsilon_r)*Escale; B/mu0 by Hscale. Fixed incident scales are preferred
for convergence studies to avoid hiding errors in a changing denominator.
    """
    e1, h1, e2, h2, n = map(np.asarray, (E1, H1, E2, H2, normals))
    if e1.shape[-1:] != (3,) or any(a.shape != e1.shape for a in (h1, e2, h2)):
        raise ValueError("E and H must have identical (..., 3) shapes")
    if not all(np.isfinite(a).all() for a in (e1, h1, e2, h2, n)):
        raise ValueError("Boundary fields and normals must be finite")
    n = np.broadcast_to(n, e1.shape)
    if not np.allclose(np.linalg.norm(n, axis=-1), 1, rtol=1e-12, atol=1e-12):
        raise ValueError("Normals must have unit length")
    w = np.ones(e1.shape[:-1]) if weights is None else np.asarray(weights)
    if w.shape != e1.shape[:-1] or not np.isfinite(w).all() or np.any(w <= 0) or w.size == 0:
        raise ValueError("weights must be positive, finite, and match the observation shape")
    def rms(a):
        square = np.sum(abs(a)**2, axis=-1) if a.ndim == e1.ndim else abs(a)**2
        return float(np.sqrt(np.sum(w*square)/np.sum(w)))
    es = max(rms(e1), rms(e2), 1e-30) if electric_scale is None else electric_scale
    hs = max(rms(h1), rms(h2), 1e-30) if magnetic_scale is None else magnetic_scale
    if not np.isfinite(es) or not np.isfinite(hs) or es <= 0 or hs <= 0:
        raise ValueError("normalization scales must be finite and positive")
    return dict(tangential_E=rms(np.cross(n, e2-e1))/es,
                tangential_H=rms(np.cross(n, h2-h1))/hs,
                normal_D=rms(np.sum(n*(medium2.epsilon_r*e2-medium1.epsilon_r*e1), axis=-1))/(max(medium1.epsilon_r, medium2.epsilon_r)*es),
                normal_B=rms(np.sum(n*(h2-h1), axis=-1))/hs)
