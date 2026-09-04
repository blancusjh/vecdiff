"""Vector Fresnel boundary solution for each incident wave and surface normal.

Normals point from medium 1 into medium 2; incidence must approach that side.
The p basis is k_hat cross s in each branch, including the reflected branch.
The complex transmitted direction retains the evanescent field under TIR.
Only the incident wave is required to be propagating. No total-field ray is
inferred and no global-x fallback substitutes for a transverse basis.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FresnelResult:
    reflected_k: np.ndarray
    transmitted_k: np.ndarray
    reflected_E: np.ndarray
    transmitted_E: np.ndarray
    s: np.ndarray
    incident_p: np.ndarray
    reflected_p: np.ndarray
    transmitted_p: np.ndarray
    rs: np.ndarray
    rp: np.ndarray
    ts: np.ndarray
    tp: np.ndarray


def solve(incident_k, incident_E, normal, medium1, medium2, wavelength=1.0):
    k, e, normal = np.broadcast_arrays(np.asarray(incident_k, complex), np.asarray(incident_E, complex), np.asarray(normal, float))
    if k.shape[-1] != 3 or not all(np.isfinite(x).all() for x in (k, e, normal)):
        raise ValueError("expected finite Cartesian vectors")
    if np.any(abs(k.imag) > 1e-12): raise ValueError("incident evanescent waves are not yet supported at an interface")
    k1, k2 = medium1.wavenumber(wavelength), medium2.wavenumber(wavelength)
    u = k.real/k1
    if not np.allclose(np.linalg.norm(u, axis=-1), 1, atol=1e-10, rtol=1e-10): raise ValueError("incident k violates dispersion")
    if not np.allclose(np.linalg.norm(normal, axis=-1), 1, atol=1e-12, rtol=1e-12): raise ValueError("normal must be unit length")
    if np.any(abs(np.sum(u*e, axis=-1)) > 1e-10*np.linalg.norm(e, axis=-1)+1e-14): raise ValueError("incident E is not transverse")
    ci = np.sum(u*normal, axis=-1)
    if np.any(ci <= 1e-12): raise ValueError("incident wave must point into the oriented interface, away from grazing")
    s0 = np.cross(u, normal)
    length = np.linalg.norm(s0, axis=-1)
    axis = np.eye(3)[np.argmin(abs(u), axis=-1)]
    fallback = np.cross(u, axis); fallback /= np.linalg.norm(fallback, axis=-1)[..., None]
    s = np.divide(s0, length[..., None], out=fallback.copy(), where=(length > 1e-12)[..., None])
    mu = medium1.n/medium2.n
    ct = np.sqrt((1-mu*mu*(1-ci*ci)).astype(complex))
    ur = u-2*ci[..., None]*normal
    ut = mu*u+(ct-mu*ci)[..., None]*normal
    pi, pr, pt = np.cross(u, s), np.cross(ur, s), np.cross(ut, s)
    ds = medium1.n*ci+medium2.n*ct
    dp = medium2.n*ci+medium1.n*ct
    rs, rp = (medium1.n*ci-medium2.n*ct)/ds, (medium2.n*ci-medium1.n*ct)/dp
    ts, tp = 2*medium1.n*ci/ds, 2*medium1.n*ci/dp
    es, ep = np.sum(e*s, axis=-1), np.sum(e*pi, axis=-1)
    er = (rs*es)[..., None]*s+(rp*ep)[..., None]*pr
    et = (ts*es)[..., None]*s+(tp*ep)[..., None]*pt
    return FresnelResult(k1*ur, k2*ut, er, et, s, pi, pr, pt, rs, rp, ts, tp)
