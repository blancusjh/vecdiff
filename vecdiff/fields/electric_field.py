"""An electric field on a physical domain and a separate sampling.

Ez=None is missing information, never an implicit zero. All components are
global Cartesian phasors. The transverse subclass denotes specified Ex/Ey,
not s/TE polarization. Arrays are immutable to avoid stale derived state.
"""
from dataclasses import dataclass
import numpy as np
from ..media.medium import Medium


@dataclass(frozen=True, eq=False)
class ElectricField:
    Ex: np.ndarray
    Ey: np.ndarray
    sampling: object
    domain: object
    wavelength: float = 1.0
    medium: Medium = Medium()
    Ez: np.ndarray | None = None

    def __post_init__(self):
        self.medium.wavenumber(self.wavelength)
        for name in ("Ex", "Ey", "Ez"):
            value = getattr(self, name)
            if value is None and name == "Ez": continue
            a = np.array(value, complex, copy=True)
            if a.shape != self.sampling.shape or not np.isfinite(a).all():
                raise ValueError(f"{name} must be finite and match sampling.shape")
            a.setflags(write=False)
            object.__setattr__(self, name, a)

    @property
    def components(self):
        if self.Ez is None:
            raise ValueError("Ez is unspecified; complete the field before using all components")
        return np.stack((self.Ex, self.Ey, self.Ez), axis=-1)

    def transverse_norm2(self): return abs(self.Ex)**2 + abs(self.Ey)**2

    def norm2(self): return np.sum(abs(self.components)**2, axis=-1)

    def complete(self, *, direction=1):
        from ..propagation.propagation import propagate
        return propagate(self, 0, direction=direction)


class TransverseElectricField(ElectricField):
    def __init__(self, Ex, Ey, sampling, domain, wavelength=1.0, medium=Medium()):
        super().__init__(Ex, Ey, sampling, domain, wavelength, medium, Ez=None)
