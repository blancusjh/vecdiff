"""Richards-Wolf aplanatic focusing reference, independent of interface physics.

Uses the shared ElectricSpectrum representation, not the production Fresnel
map. Uniform x polarization at the entrance pupil; sine-condition objective,
sqrt(cos(theta)) apodization; exp(-i omega t). This is an ideal-objective
reference and is not a dielectric-boundary solution.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from vecdiff import ElectricSpectrum, Medium


def spectrum(na, *, wavelength=1., medium=Medium(), focal_length=1., n_theta=80, n_phi=128,
             polarization="linear_x", vortex_charge=0):
    """Uniform pupil magnitude; choose linear_x, radial, or azimuthal polarization.

    A vortex multiplies the pupil by exp(i charge phi). All choices have equal
    pupil power for the same NA and focal length; no peak renormalization is used.
    """
    if not 0 < na < medium.n: raise ValueError("requires 0 < NA < n")
    if not np.isfinite(focal_length) or focal_length <= 0:
        raise ValueError("focal_length must be finite and positive")
    if not isinstance(vortex_charge, int) or n_theta < 2 or n_phi < 2:
        raise ValueError("integer vortex charge and at least two nodes per axis required")
    t, w = leggauss(n_theta); limit = np.arcsin(na/medium.n)
    t, w = (t+1)*limit/2, w*limit/2
    t, p = np.meshgrid(t, np.arange(n_phi)*2*np.pi/n_phi, indexing="ij")
    c, s, cp, sp = np.cos(t), np.sin(t), np.cos(p), np.sin(p)
    direction = np.stack((s*cp, s*sp, c), axis=-1)
    if polarization == "linear_x":
        e = np.stack((c*cp*cp+sp*sp, (c-1)*cp*sp, -s*cp), axis=-1)
    elif polarization == "radial":
        e = np.stack((c*cp, c*sp, -s), axis=-1)
    elif polarization == "azimuthal":
        e = np.stack((-sp, cp, 0*s), axis=-1)
    else:
        raise ValueError("polarization must be linear_x, radial, or azimuthal")
    k = medium.wavenumber(wavelength)
    weights = -1j*k*focal_length/(2*np.pi)*w[:, None]*(2*np.pi/n_phi)*s*np.sqrt(c)*np.exp(1j*vortex_charge*p)
    return ElectricSpectrum(k*direction.reshape(-1, 3), (weights[..., None]*e).reshape(-1, 3), wavelength, medium)
