"""Euclidean placement, independent of wavelength and material."""

from dataclasses import dataclass, field

import numpy as np

from vecdiff_general._arrays import frozen_array


@dataclass(frozen=True, eq=False)
class Frame:
    """Right-handed orthonormal frame; basis columns are u, v, normal.

    Coordinates and translations are in metres. Vector components elsewhere
    in the public API are global Cartesian components unless stated otherwise.
    """

    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    basis: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        origin = frozen_array(self.origin)
        basis = frozen_array(self.basis)

        if origin.shape != (3,) or basis.shape != (3, 3):
            raise ValueError("expected origin (3,) and basis (3, 3)")

        if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-12, rtol=0):
            raise ValueError("frame basis must be orthonormal")

        if np.linalg.det(basis) < 0:
            raise ValueError("frame basis must be right-handed")

        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "basis", basis)

    @property
    def u(self):
        return self.basis[:, 0]

    @property
    def v(self):
        return self.basis[:, 1]

    @property
    def normal(self):
        return self.basis[:, 2]

    def translated(self, displacement):
        displacement = frozen_array(displacement)

        if displacement.shape != (3,):
            raise ValueError("displacement must have shape (3,)")

        return Frame(self.origin + displacement, self.basis)
