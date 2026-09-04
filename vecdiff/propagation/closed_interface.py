"""Self-consistent Maxwell boundary matching with auxiliary dipole sources.

This dense method of fundamental solutions (MFS) is not a Mie formula or a
local Fresnel approximation. Internal interactions are implicit in a complex
linear solve. Convergence requires source-count, source-placement, and held-out
boundary checks. Only one closed interface between homogeneous media is supported.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import lstsq
from ..media.medium import Medium


def _tangents(normals):
    axis = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
    a = np.cross(normals, axis)
    a /= np.linalg.norm(a, axis=1)[:, None]
    return np.stack((a, np.cross(normals, a)), axis=1)


def _dipoles(points, sources, directions, wavelength, medium):
    """E and Z0 H kernels, shape (observation, Cartesian, source)."""
    delta = points[:, None, :] - sources[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    if np.any(distance == 0):
        raise ValueError("Cannot evaluate an auxiliary source at its singularity")
    u = delta / distance[..., None]
    k0 = 2*np.pi/wavelength
    k = medium.n*k0
    g = np.exp(1j*k*distance)/(4*np.pi*distance)
    gp = (1j*k-1/distance)*g
    gpp = ((1j*k-1/distance)**2+1/distance**2)*g
    ud = np.sum(u*directions[None, :, :], axis=-1)[..., None]*u
    hessian = gpp[..., None]*ud + (gp/distance)[..., None]*(directions-ud)
    e = 1j*k0*(g[..., None]*directions+hessian/k**2)
    h = np.cross(gp[..., None]*u, directions)
    return e.transpose(0, 2, 1), h.transpose(0, 2, 1)


@dataclass(frozen=True)
class ClosedInterfaceField:
    """Total fields with explicit region selection.

    Auxiliary sources are not physical emitters. Evaluate each expansion only
    in its physical region. The generic surface API cannot infer containment;
    callers must check this, including all supplied auxiliary-source offsets.
    """
    incident: object
    interior_medium: Medium
    interior_sources: np.ndarray
    exterior_sources: np.ndarray
    directions: np.ndarray
    coefficients: np.ndarray
    fit_residual: float
    rank: int
    condition_number: float

    def __post_init__(self):
        for name in ("interior_sources", "exterior_sources", "directions", "coefficients"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def evaluate(self, points, *, region, chunk=128):
        if region not in ("interior", "exterior"):
            raise ValueError("region must be 'interior' or 'exterior'")
        p = np.asarray(points, dtype=float)
        if p.shape[-1:] != (3,) or not np.all(np.isfinite(p)):
            raise ValueError("points must be finite (..., 3) coordinates")
        if not isinstance(chunk, int) or chunk < 1:
            raise ValueError("chunk must be a positive integer")
        flat = p.reshape(-1, 3)
        count = len(self.interior_sources)
        outside = region == "exterior"
        sources = self.interior_sources if outside else self.exterior_sources
        medium = self.incident.medium if outside else self.interior_medium
        c = self.coefficients[:count] if outside else self.coefficients[count:]
        e = np.empty_like(flat, dtype=complex)
        h = np.empty_like(e)
        for start in range(0, len(flat), chunk):
            q = flat[start:start+chunk]
            ek, hk = _dipoles(q, sources, self.directions, self.incident.wavelength, medium)
            e[start:start+chunk], h[start:start+chunk] = ek@c, hk@c
        if outside:
            ei, hi = self.incident.evaluate(flat)
            e += ei
            h += hi
        return e.reshape(p.shape), h.reshape(p.shape)


def solve_closed_interface(incident, interior_medium, boundary, source_sampling,
                           *, inward_offset, outward_offset, rcond=1e-12):
    """Fit tangential E/H continuity on an outward-oriented closed boundary.

    Both SurfaceSampling arguments must sample the same physical surface. Two
    tangential electric dipoles per source location span each medium's expansion.
    Sources displaced inward radiate the exterior scattered field; outward
    sources represent the interior total field. Positive offsets must keep the
    sources in the intended opposite regions (not guaranteed for nonconvex bodies).

    The weighted, column-scaled least-squares residual is NOT an accuracy
    certificate. Check all four Maxwell boundary conditions on different points
    and compare successive discretizations. Dense memory scales as boundary count
    times source count. This method is intended for small objects, not macroscopic
    optical systems. Singular-value truncation is controlled by rcond.
    """
    if boundary.surface is not source_sampling.surface:
        raise ValueError("boundary and sources must sample the same surface object")
    if not boundary.surface.is_closed:
        raise ValueError("The physical surface must declare is_closed=True")
    if not isinstance(interior_medium, Medium):
        raise TypeError("interior_medium must be a Medium")
    if not (np.isfinite(inward_offset) and np.isfinite(outward_offset)
            and inward_offset > 0 and outward_offset > 0):
        raise ValueError("Auxiliary-source offsets must be finite and positive")
    if not 0 < rcond < 1:
        raise ValueError("rcond must lie in (0, 1)")
    if len(boundary.points) < 2*len(source_sampling.points):
        raise ValueError("Use at least twice as many boundary as source points")
    directions = _tangents(source_sampling.normals).reshape(-1, 3)
    inside = np.repeat(source_sampling.points-inward_offset*source_sampling.normals, 2, axis=0)
    outside = np.repeat(source_sampling.points+outward_offset*source_sampling.normals, 2, axis=0)
    tangent = _tangents(boundary.normals)
    eo, ho = _dipoles(boundary.points, inside, directions, incident.wavelength, incident.medium)
    et, ht = _dipoles(boundary.points, outside, directions, incident.wavelength, interior_medium)
    project = lambda a: np.einsum("pti,pis->pts", tangent, a).reshape(-1, a.shape[-1])
    matrix = np.concatenate((np.concatenate((project(eo), -project(et)), axis=1),
                             np.concatenate((project(ho), -project(ht)), axis=1)/incident.medium.n), axis=0)
    ei, hi = incident.evaluate(boundary.points)
    rhs = -np.concatenate((np.einsum("pti,pi->pt", tangent, ei).ravel(),
                           np.einsum("pti,pi->pt", tangent, hi).ravel()/incident.medium.n))
    weights = np.tile(np.repeat(np.sqrt(boundary.weights/np.mean(boundary.weights)), 2), 2)
    matrix *= weights[:, None]
    rhs *= weights
    scale = np.linalg.norm(matrix, axis=0)
    if np.any(scale == 0) or not np.all(np.isfinite(matrix)):
        raise ValueError("Degenerate auxiliary-source expansion")
    matrix /= scale
    solution, _, rank, singular = lstsq(matrix, rhs, cond=rcond, lapack_driver="gelsd")
    residual = np.linalg.norm(matrix@solution-rhs)/max(np.linalg.norm(rhs), np.finfo(float).tiny)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else np.inf
    return ClosedInterfaceField(incident, interior_medium, inside, outside, directions,
                                 solution/scale, float(residual), rank, condition)
