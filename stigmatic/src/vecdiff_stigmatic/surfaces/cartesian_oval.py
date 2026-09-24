"""The vertex-connected Cartesian-oval branch for two real axial conjugates."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .._arrays import positive


@dataclass(frozen=True)
class CartesianOval:
    """Geometry designed for specified indices; all distances are in metres.

    The vertex is z=0 and object_z < 0 < image_z. Design indices determine
    shape; the StigmaticSystem assigns actual materials and checks consistency.
    Equal-index, virtual-conjugate and non-graph branches are not implemented.
    """

    object_z: float
    image_z: float
    n_object: float
    n_image: float

    def __post_init__(self):
        for name in ("n_object", "n_image"):
            object.__setattr__(self, name, positive(getattr(self, name), name))

        if not np.isfinite(self.object_z) or not np.isfinite(self.image_z):
            raise ValueError("conjugate positions must be finite")

        if not self.object_z < 0 < self.image_z:
            raise ValueError(
                "initial Cartesian-oval model requires object_z < 0 < image_z"
            )

        if self.n_object == self.n_image:
            raise ValueError(
                "equal indices do not define this vertex-connected refracting branch"
            )

    @property
    def object_point(self):
        return np.array([0.0, 0.0, self.object_z])

    @property
    def image_point(self):
        return np.array([0.0, 0.0, self.image_z])

    @property
    def optical_path(self):
        return -self.n_object * self.object_z + self.n_image * self.image_z

    def sag(self, radius):
        """Solve the constant-optical-path equation on its vertex branch.

        Dimensionless bracketing makes the tolerance independent of the chosen
        lens size. Difference-of-squares avoids subtracting nearly equal paths
        near the axis. No extrapolation or clipping past the branch limit.
        """
        radius = np.asarray(radius, dtype=float)

        if not np.isfinite(radius).all() or np.any(radius < 0):
            raise ValueError("radius must be finite and nonnegative")

        length = self.image_z - self.object_z
        z0, zi = self.object_z / length, self.image_z / length
        n0, ni = self.n_object, self.n_image

        def solve_one(r):
            if r == 0:
                return 0.0

            def residual(z):
                l0, li = np.hypot(r, z - z0), np.hypot(r, z - zi)

                return n0 * (r * r + z * z - 2 * z * z0) / (l0 - z0) + ni * (
                    r * r + z * z - 2 * z * zi
                ) / (li + zi)

            def derivative(z):
                return n0 * (z - z0) / np.hypot(r, z - z0) + ni * (z - zi) / np.hypot(
                    r, z - zi
                )

            minimum = brentq(derivative, z0, zi, xtol=1e-15)

            if residual(minimum) >= 0:
                raise ValueError(
                    "aperture exceeds the vertex-connected Cartesian-oval branch"
                )

            lower, upper = sorted((0.0, minimum))

            return brentq(residual, lower, upper, xtol=1e-15)

        unique, inverse = np.unique(radius / length, return_inverse=True)
        heights = np.array([solve_one(r) for r in unique]) * length

        return heights[inverse].reshape(radius.shape)
