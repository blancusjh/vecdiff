"""Physical field domains, independent of their sampling."""
from dataclasses import dataclass, field
from .frames import Frame


@dataclass(frozen=True)
class PlaneDomain:
    frame: Frame = field(default_factory=Frame)

    def points(self, sampling):
        import numpy as np
        x, y = sampling.xy
        return self.frame.points(np.stack((x, y, np.zeros_like(x)), axis=-1))
