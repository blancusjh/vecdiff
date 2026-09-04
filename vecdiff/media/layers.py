"""An ordered physical stack of parallel, homogeneous dielectric regions."""
from dataclasses import dataclass
import numpy as np
from .medium import Medium
from ..geometry.frames import Frame


@dataclass(frozen=True)
class LayerStack:
    """Media include entrance/exit half spaces; thicknesses are interior layers.

The first boundary passes through frame.origin. Layers extend along its local
+z normal. No quadrature, propagation, or convergence settings belong here.
    """
    media: tuple
    thicknesses: tuple
    frame: Frame = None

    def __post_init__(self):
        media, thickness = tuple(self.media), tuple(self.thicknesses)
        if len(media) < 2 or not all(isinstance(m, Medium) for m in media):
            raise ValueError("media must include at least two Medium instances")
        if len(thickness) != len(media)-2 or any(not np.isfinite(d) or d <= 0 for d in thickness):
            raise ValueError("one positive finite thickness is required per interior medium")
        object.__setattr__(self, "media", media)
        object.__setattr__(self, "thicknesses", thickness)
        object.__setattr__(self, "frame", self.frame or Frame())

    @property
    def boundaries(self):
        return np.concatenate(([0.], np.cumsum(self.thicknesses)))
