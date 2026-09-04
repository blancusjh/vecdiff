"""Constitutive data for lossless, isotropic, nonmagnetic media.

Fields use exp(-i omega t). H is stored as Z0*H_SI, so B/mu0 = H/Z0
and D/epsilon0 = n**2 E. Lengths and vacuum wavelength share any one unit.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Medium:
    n: float = 1.0

    def __post_init__(self):
        if not np.isreal(self.n) or not np.isfinite(self.n) or self.n <= 0:
            raise ValueError("Medium currently requires a positive real refractive index")

    @property
    def epsilon_r(self):
        return self.n**2

    def wavenumber(self, wavelength):
        if not np.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("vacuum wavelength must be positive")
        return 2 * np.pi * self.n / wavelength
