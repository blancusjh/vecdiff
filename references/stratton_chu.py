"""Independent historical direct-radiation implementation for comparison.

Prescribed-current radiation does not determine the correct dielectric
boundary currents. This distinction also applies to the new native method.
"""
import numpy as np


def franz_integral(points, sources, J, M, weights, k, n):
    """Independent E-only direct integral; exp(-i omega t), H normalized as Z0 H.

    All observation points must be separated from the source quadrature nodes.
    Uses explicit Hessian contraction, independently of the production kernel.
    """
    delta = np.asarray(points)[:, None, :]-np.asarray(sources)[None, :, :]
    radius = np.linalg.norm(delta, axis=-1)
    if np.any(radius == 0):
        raise ValueError("Observation at a source singularity")
    unit = delta/radius[..., None]
    green = np.exp(1j*k*radius)/(4*np.pi*radius)
    first = (1j*k-1/radius)*green
    second = ((1j*k-1/radius)**2+1/radius**2)*green
    jw = np.asarray(J)*np.asarray(weights)[:, None]
    mw = np.asarray(M)*np.asarray(weights)[:, None]
    radial = np.einsum("psi,si->ps", unit, jw)[..., None]*unit
    hessian = second[..., None]*radial+(first/radius)[..., None]*(jw-radial)
    electric = 1j*(k/n)*(green[..., None]*jw+hessian/k**2)
    magnetic = -np.cross(first[..., None]*unit, mw)
    return np.sum(electric+magnetic, axis=1)
