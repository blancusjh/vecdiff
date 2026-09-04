"""Rigid placements; vector components are always global Cartesian components."""
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True, eq=False)
class Frame:
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        o, r = np.array(self.origin, float), np.array(self.rotation, float)
        if o.shape != (3,) or not np.isfinite(o).all():
            raise ValueError("origin must be a finite three-vector")
        if r.shape != (3, 3) or not np.allclose(r.T @ r, np.eye(3), atol=1e-12, rtol=0) or not np.isclose(np.linalg.det(r), 1, atol=1e-12):
            raise ValueError("rotation must be right-handed and orthonormal")
        o.setflags(write=False); r.setflags(write=False)
        object.__setattr__(self, "origin", o)
        object.__setattr__(self, "rotation", r)

    def points(self, local):
        return np.asarray(local) @ self.rotation.T + self.origin

    def vectors(self, local):
        return np.asarray(local) @ self.rotation.T
