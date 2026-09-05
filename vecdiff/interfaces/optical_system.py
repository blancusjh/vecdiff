"""Placed optical encounters; import formats are owned exclusively by IO/."""
from dataclasses import dataclass
import numpy as np
from .dielectric_interface import DielectricInterface
from .assembly import InterfaceAssembly


@dataclass(frozen=True)
class SurfaceEncounter:
    number: int
    surface: object
    interaction: str
    incident_medium: object
    transmitted_medium: object
    semidiameter: float | None
    thickness: float
    direction: int
    material_after: str

    def __post_init__(self):
        if self.interaction not in ("refract", "reflect", "stop") or self.direction not in (-1, 1):
            raise ValueError("invalid encounter interaction or direction")
        if not np.isfinite(self.thickness) or (self.semidiameter is not None and (not np.isfinite(self.semidiameter) or self.semidiameter <= 0)):
            raise ValueError("finite thickness and positive clear semidiameter required")
        if self.interaction != "refract" and self.incident_medium != self.transmitted_medium:
            raise ValueError("reflection and stops must preserve the propagation medium")

    def dielectric_interface(self):
        if self.interaction != "refract":
            raise ValueError(f"encounter {self.number} is {self.interaction}, not a dielectric interface")
        return DielectricInterface(self.surface, self.incident_medium,
                                   self.transmitted_medium, normal_sign=self.direction)


@dataclass(frozen=True)
class OpticalSystem:
    """Sequential geometry, media, clear apertures and signed spacings.

    Repeated visits to a physical surface remain distinct encounters. Importing
    a folded prescription does not imply that a selected propagator supports it.
    """
    encounters: tuple
    wavelength: float
    length_unit: str = "mm"
    name: str = ""

    def __post_init__(self):
        encounters = tuple(self.encounters)
        if not encounters or not all(isinstance(e, SurfaceEncounter) for e in encounters):
            raise ValueError("an optical system needs SurfaceEncounter records")
        if not np.isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValueError("wavelength must be finite and positive")
        for a,b in zip(encounters, encounters[1:]):
            if a.transmitted_medium != b.incident_medium:
                raise ValueError("adjacent encounters must agree on their medium")
        object.__setattr__(self, "encounters", encounters)

    @property
    def image_z(self):
        last = self.encounters[-1]
        return float(last.surface.frame.origin[2]+last.thickness)

    def dielectric_assembly(self):
        unsupported = [(e.number, e.interaction) for e in self.encounters if e.interaction != "refract"]
        if unsupported:
            raise ValueError(f"dielectric assembly cannot discard mirrors or stops: {unsupported}")
        return InterfaceAssembly(tuple(e.dielectric_interface() for e in self.encounters))
