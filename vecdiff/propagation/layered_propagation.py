"""Coherent repeated reflection/refraction through parallel dielectric layers.

Recursively composed reflection/transmission matrices sum all round trips.
Only decaying propagation factors appear, including frustrated TIR. Interior
forward amplitudes are anchored at left boundaries and backward amplitudes at
right boundaries to avoid multiplying growing evanescent factors.
"""
from dataclasses import dataclass
import numpy as np
from ..fields.electric_spectrum import ElectricSpectrum
from ..interfaces.fresnel import tangential_coefficients


@dataclass(frozen=True)
class LayeredElectricField:
    incident: ElectricSpectrum
    stack: object
    kz: np.ndarray
    s: np.ndarray
    tangent_p: np.ndarray
    forward: tuple
    backward: tuple
    reflection: np.ndarray
    transmission: np.ndarray

    def __post_init__(self):
        def freeze(a):
            a = np.array(a, copy=True)
            a.setflags(write=False)
            return a
        for name in ("kz", "s", "tangent_p", "reflection", "transmission"):
            object.__setattr__(self, name, freeze(getattr(self, name)))
        for name in ("forward", "backward"):
            object.__setattr__(self, name, tuple(map(freeze, getattr(self, name))))

    def evaluate(self, points, *, region):
        """Evaluate E and normalized H in an explicitly selected physical region."""
        if not isinstance(region, int) or not 0 <= region < len(self.stack.media):
            raise ValueError("region index is out of range")
        p = np.asarray(points, float)
        if p.shape[-1:] != (3,) or not np.isfinite(p).all():
            raise ValueError("points must be finite (...,3) Cartesian coordinates")
        local = (p.reshape(-1, 3)-self.stack.frame.origin) @ self.stack.frame.rotation
        bounds = self.stack.boundaries
        left = bounds[max(0, region-1)]
        right = bounds[min(region, len(bounds)-1)]
        tol = 1e-12*max(1., bounds[-1])
        if (region > 0 and np.any(local[:, 2] < left-tol)) or (region < len(self.stack.media)-1 and np.any(local[:, 2] > right+tol)):
            raise ValueError("observation points lie outside the requested layer")
        incoming_k = self.incident.wavevectors @ self.stack.frame.rotation
        transverse = incoming_k[:, :2]
        kn = self.kz[:, region]
        e = np.zeros((len(local), 3), complex); h = e.copy()
        k0 = 2*np.pi/self.incident.wavelength
        for sign, amplitude, anchor in [(1, self.forward[region], left), (-1, self.backward[region], right)]:
            if not np.any(amplitude):
                continue  # Do not evaluate a growing exponential for an absent wave.
            kv = np.column_stack((transverse, sign*kn))
            ep = np.column_stack((self.tangent_p, -np.sum(transverse*self.tangent_p, axis=-1)/(sign*kn)))
            es = np.column_stack((self.s, np.zeros(len(kn))))
            a = amplitude[:, :1]*es+amplitude[:, 1:]*ep
            ha = np.cross(kv, a)/k0
            for start in range(0, len(local), 256):
                phase = np.exp(1j*(transverse @ local[start:start+256, :2].T+sign*kn[:, None]*(local[start:start+256, 2]-anchor)))
                e[start:start+256] += phase.T @ a
                h[start:start+256] += phase.T @ ha
        shape = p.shape
        return self.stack.frame.vectors(e).reshape(shape), self.stack.frame.vectors(h).reshape(shape)


def propagate_layers(incident, stack):
    """Sum all coherent interactions for every populated incident wavevector."""
    if incident.medium != stack.media[0]:
        raise ValueError("incident medium must match entrance half space")
    k = incident.wavevectors @ stack.frame.rotation
    if np.any(k.imag) or np.any(k[:, 2].real <= 0):
        raise ValueError("incident modes must propagate toward the stack in local +z")
    k0 = 2*np.pi/incident.wavelength
    transverse2 = np.sum(k[:, :2].real**2, axis=-1)
    kz = np.sqrt(np.array([m.n**2*k0*k0-transverse2 for m in stack.media], complex).T)
    if np.any(abs(kz) < 1e-13*k0):
        raise ValueError("a layer is exactly grazing/critical; choose a noncritical angle")
    mcount, regions = len(k), len(stack.media)
    rho = np.sqrt(transverse2)
    tangent = np.divide(k[:, :2].real, rho[:, None], out=np.tile([1., 0.], (mcount, 1)), where=rho[:, None] != 0)
    s = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    local_e = incident.amplitudes @ stack.frame.rotation
    phase_origin = np.exp(1j*incident.wavevectors @ stack.frame.origin)
    injection = np.column_stack((np.sum(local_e[:, :2]*s, axis=-1), np.sum(local_e[:, :2]*tangent, axis=-1)))*phase_origin[:, None]
    rs, ts = [], []
    for j in range(regions-1):
        r, t = tangential_coefficients(kz[:, j], kz[:, j+1], stack.media[j], stack.media[j+1], k0)
        rs.append(r); ts.append(t)
    factors = [np.ones((mcount, 1), complex)]
    factors += [np.exp(1j*kz[:, j]*stack.thicknesses[j-1])[:, None] for j in range(1, regions-1)]
    factors += [np.ones((mcount, 1), complex)]
    effective = [None]*(regions-1)
    effective[-1] = rs[-1]
    for j in range(regions-3, -1, -1):
        feedback = effective[j+1]*factors[j+1]**2
        effective[j] = rs[j]+ts[j]*(1-rs[j])*feedback/(1+rs[j]*feedback)
    forward = [injection]; backward = [effective[0]*injection]
    for j in range(regions-1):
        feedback = effective[j+1]*factors[j+1]**2 if j+1 < regions-1 else 0.
        following = ts[j]*forward[j]*factors[j]/(1+rs[j]*feedback)
        forward.append(following)
        backward.append(effective[j+1]*following*factors[j+1] if j+1 < regions-1 else np.zeros_like(injection))
    # Coefficients are defined relative to unit s/p tangential amplitudes.
    ttotal = np.ones_like(injection)
    for j in range(regions-1):
        feedback = effective[j+1]*factors[j+1]**2 if j+1 < regions-1 else 0.
        ttotal *= ts[j]*factors[j]/(1+rs[j]*feedback)
    return LayeredElectricField(incident, stack, kz, s, tangent, tuple(forward), tuple(backward), effective[0], ttotal)
