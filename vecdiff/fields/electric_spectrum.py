"""A discrete superposition of homogeneous Maxwell plane waves.

Each amplitude already includes its spectral quadrature weight: E(r) is
sum_j amplitudes[j]*exp(i wavevectors[j].r). This accommodates FFT, arbitrary
directions and nonuniform spectra without confusing densities with weights.
"""
from dataclasses import dataclass
import numpy as np
from ..media.medium import Medium
from ..fourier.nufft import synthesize
from .electric_field import ElectricField


@dataclass(frozen=True, eq=False)
class ElectricSpectrum:
    wavevectors: np.ndarray
    amplitudes: np.ndarray
    wavelength: float = 1.0
    medium: Medium = Medium()

    def __post_init__(self):
        k, a = np.array(self.wavevectors, complex, copy=True), np.array(self.amplitudes, complex, copy=True)
        if k.ndim != 2 or k.shape[1] != 3 or a.shape != k.shape or not np.isfinite(k).all() or not np.isfinite(a).all():
            raise ValueError("wavevectors and amplitudes must be finite arrays (modes, 3)")
        kn = self.medium.wavenumber(self.wavelength)
        if not np.allclose(np.sum(k*k, axis=-1), kn**2, rtol=1e-9, atol=1e-10*kn**2):
            raise ValueError("wavevectors violate the medium dispersion relation")
        scale = np.linalg.norm(k, axis=-1)*np.linalg.norm(a, axis=-1)
        if np.any(abs(np.sum(k*a, axis=-1)) > 1e-9*scale + 1e-14*kn):
            raise ValueError("electric amplitudes violate k dot E = 0")
        k.setflags(write=False); a.setflags(write=False)
        object.__setattr__(self, "wavevectors", k); object.__setattr__(self, "amplitudes", a)

    @property
    def magnetic_amplitudes(self):
        """Normalized magnetic amplitudes Z0 H_SI."""
        return np.cross(self.wavevectors, self.amplitudes)/(2*np.pi/self.wavelength)

    def evaluate(self, points, *, backend="direct"):
        p = np.asarray(points, float)
        if p.shape[-1:] != (3,) or not np.isfinite(p).all():
            raise ValueError("points must be finite (..., 3) coordinates")
        shape = p.shape
        values = synthesize(self.wavevectors, np.concatenate((self.amplitudes.T, self.magnetic_amplitudes.T)),
                            p.reshape(-1, 3), backend=backend).T.reshape(shape[:-1]+(6,))
        return values[..., :3], values[..., 3:]

    def field(self, domain, sampling, *, backend="direct"):
        points = domain.points(sampling) if hasattr(domain, "points") else sampling.points
        e, _ = self.evaluate(points, backend=backend)
        return ElectricField(e[..., 0], e[..., 1], sampling, domain, self.wavelength, self.medium, e[..., 2])

    def translated(self, displacement):
        """Field evaluated at r+displacement, represented in the original coordinates."""
        return ElectricSpectrum(self.wavevectors, self.amplitudes*np.exp(1j*self.wavevectors @ displacement)[:, None],
                                self.wavelength, self.medium)


def plane_wave(direction=(0, 0, 1), polarization=(1, 0, 0), *, wavelength=1., medium=Medium()):
    u = np.asarray(direction, float)
    if u.shape != (3,) or not np.isfinite(u).all() or np.linalg.norm(u) == 0:
        raise ValueError("direction must be a finite nonzero vector")
    u = u/np.linalg.norm(u)
    return ElectricSpectrum(u[None]*medium.wavenumber(wavelength), np.asarray(polarization, complex)[None], wavelength, medium)
