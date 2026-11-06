"""Closed spatial volumes, independent of materials, fields and propagation."""

from dataclasses import dataclass

import numpy as np

from vecdiff_general._arrays import frozen_array, positive
from vecdiff_general._backend import bd


@dataclass(frozen=True)
class SphereVolume:
    center: np.ndarray
    radius: float

    def __post_init__(self):
        center = frozen_array(self.center)

        if center.shape != (3,):
            raise ValueError("sphere centre must be a Cartesian position")

        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", positive(self.radius, "sphere radius"))

    def contains(self, points):
        return bd.linalg.norm(bd.asarray(points) - self.center, axis=-1) <= self.radius


@dataclass(frozen=True)
class VolumeBetweenSurfaces:
    """Two coaxial graph faces closed by an explicit cylindrical rim.

    The two faces must be graph surfaces with the same frame orientation and
    coaxial vertices. Their local radii are bounded by rim_radius. Boundary
    points are included. Face ordering is checked at queried interior points;
    this membership calculation is not a global non-intersection certificate.

    The volume carries no material. Assigning a material and selecting which
    boundaries a propagation model treats are separate operations.
    """

    front: object
    back: object
    rim_radius: float

    def __post_init__(self):
        object.__setattr__(self, "rim_radius", positive(self.rim_radius, "rim radius"))

        if not bd.allclose(
            self.front.frame.basis, self.back.frame.basis, rtol=0, atol=1e-12
        ):
            raise ValueError("volume faces must share a frame orientation")

        displacement = (
            self.back.frame.origin - self.front.frame.origin
        ) @ self.front.frame.basis

        if bd.linalg.norm(displacement[:2]) > 1e-12 or displacement[2] <= 0:
            raise ValueError("back face must be coaxial and beyond the front vertex")

    def contains(self, points):
        local = (bd.asarray(points) - self.front.frame.origin) @ self.front.frame.basis
        radius = bd.linalg.norm(local[..., :2], axis=-1)
        inside = radius <= self.rim_radius
        result = bd.zeros(radius.shape, dtype=bool)

        if bd.any(inside):
            xy = local[inside, :2]
            front_height = self.front.height(xy)
            spacing = bd.dot(
                self.back.frame.origin - self.front.frame.origin,
                self.front.frame.normal,
            )
            back_height = spacing + self.back.height(xy)

            if bd.any(back_height <= front_height):
                raise ValueError("the volume faces intersect within the queried rim")

            result[inside] = (local[inside, 2] >= front_height) & (
                local[inside, 2] <= back_height
            )

        return result
