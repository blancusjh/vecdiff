"""An oriented geometric boundary between two constitutive media."""
from dataclasses import dataclass
from ..media.medium import Medium
from ..surfaces.surface import Surface


@dataclass(frozen=True)
class DielectricInterface:
    surface: Surface
    incident_medium: Medium
    transmitted_medium: Medium
    normal_sign: int = 1

    def __post_init__(self):
        if self.normal_sign not in (-1, 1): raise ValueError("normal_sign must be +1 or -1")
