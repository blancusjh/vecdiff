"""Evaluate fixed surface currents in separate, bounded observation patches."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RadiationEvaluation:
    electric: np.ndarray
    magnetic: np.ndarray
    electric_error_bound: float
    magnetic_error_bound: float
    patch_count: int


def evaluate_local(radiation, points, *, radius, backend='nufft'):
    """Recompute a local spectrum per patch; never translate a fixed field.

Each observation is evaluated at its actual global position. The returned
absolute vector-norm bounds are uniform over all observations and compare
against the full Green kernel of the SAME discrete surface currents. They
exclude surface quadrature, Fourier evaluation/roundoff, and errors in the
supplied dielectric traces.

radius limits observation patch size; it does not define an isoplanatic field
of view. The illuminating field and Fresnel traces must be recomputed when the
object changes. Source quadrature must resolve the residual phase for every
patch, even if the on-axis response was already converged.
    """
    p=np.asarray(points,float)
    if p.ndim<2 or p.shape[-1]!=3 or not np.isfinite(p).all() or p.size==0:
        raise ValueError('points must be nonempty finite (...,3) coordinates')
    if not np.isfinite(radius) or radius<=0:
        raise ValueError('radius must be finite and positive')
    if backend not in ('direct','nufft','auto'):
        raise ValueError('backend must be direct, nufft or auto')
    flat=p.reshape(-1,3)
    # Cube half-diagonal <= radius, independent of the cloud's orientation.
    keys=np.floor((flat-flat.min(axis=0))/(2*radius/np.sqrt(3))).astype(np.int64)
    _,labels=np.unique(keys,axis=0,return_inverse=True)
    ordering=np.argsort(labels,kind='stable')
    groups=np.split(ordering,np.flatnonzero(np.diff(labels[ordering]))+1)
    e=np.empty_like(flat,complex); h=np.empty_like(flat,complex)
    eb=hb=0.
    for indices in groups:
        q=flat[indices]; center=(q.min(axis=0)+q.max(axis=0))/2
        actual_radius=float(np.max(np.linalg.norm(q-center,axis=-1)))
        local=radiation.local_spectrum(center,actual_radius)
        e[indices],h[indices]=local.evaluate(q,backend=backend)
        eb=max(eb,local.electric_error_bound); hb=max(hb,local.magnetic_error_bound)
    return RadiationEvaluation(e.reshape(p.shape),h.reshape(p.shape),float(eb),float(hb),len(groups))
