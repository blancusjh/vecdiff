"""Surface quadrature owns nodes and weights, never Fresnel physics."""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss


@dataclass(frozen=True, eq=False)
class SurfaceSampling:
    surface: object
    points: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    parameter_shape: tuple | None = None

    def __post_init__(self):
        for key in ("points", "normals", "weights"):
            a = np.array(getattr(self, key), float, copy=True)
            if not np.isfinite(a).all(): raise ValueError("surface samples must be finite")
            a.setflags(write=False); object.__setattr__(self, key, a)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or self.normals.shape != self.points.shape or self.weights.shape != (len(self.points),):
            raise ValueError("expected points,normals=(n,3), weights=(n,)")
        if np.any(self.weights <= 0) or not np.allclose(np.linalg.norm(self.normals, axis=-1), 1, rtol=1e-12, atol=1e-12):
            raise ValueError("weights must be positive and normals unit length")


def sample_surface(surface, u_bounds, v_bounds, nu, nv, *, periodic_v=True):
    """Gauss-Legendre in u; periodic trapezoidal or Gauss-Legendre in v."""
    if nu < 2 or nv < 2: raise ValueError("at least two nodes per coordinate are required")
    if u_bounds[1] <= u_bounds[0] or v_bounds[1] <= v_bounds[0]: raise ValueError("bounds must increase")
    u, wu = leggauss(nu)
    u = (u+1)*(u_bounds[1]-u_bounds[0])/2+u_bounds[0]; wu *= (u_bounds[1]-u_bounds[0])/2
    if periodic_v:
        v = np.linspace(*v_bounds, nv, endpoint=False); wv = np.full(nv, (v_bounds[1]-v_bounds[0])/nv)
    else:
        v, wv = leggauss(nv); v = (v+1)*(v_bounds[1]-v_bounds[0])/2+v_bounds[0]; wv *= (v_bounds[1]-v_bounds[0])/2
    u, v = np.meshgrid(u, v, indexing="ij")
    normals, jac = surface.normal_and_jacobian(u, v)
    return SurfaceSampling(surface, surface.position(u, v).reshape(-1, 3), normals.reshape(-1, 3), (wu[:, None]*wv*jac).ravel(), u.shape)
