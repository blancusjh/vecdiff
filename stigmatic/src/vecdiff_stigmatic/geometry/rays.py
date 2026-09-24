"""Explicit ray directions, reference-sphere embeddings and integration measures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .._arrays import frozen_array
from ..sampling.grids import SurfaceGrid

if TYPE_CHECKING:
    from ..interfaces.optical_system import StigmaticSystem


def _ray_geometry(system, points):
    surface = system.surface
    incident = points - surface.object_point
    transmitted = surface.image_point - points
    object_distance, image_distance = (
        np.linalg.norm(incident, axis=-1),
        np.linalg.norm(transmitted, axis=-1),
    )
    incident_direction, transmitted_direction = (
        incident / object_distance[..., None],
        transmitted / image_distance[..., None],
    )
    gradient = (
        system.object_material.n * incident_direction
        - system.image_material.n * transmitted_direction
    )
    normal = np.sign(system.object_material.n - system.image_material.n) * gradient
    normal /= np.linalg.norm(normal, axis=-1)[..., None]
    cos_incidence, cos_transmission = (
        np.sum(incident_direction * normal, axis=-1),
        np.sum(transmitted_direction * normal, axis=-1),
    )

    if (
        np.any(cos_incidence <= 1e-10)
        or np.any(cos_transmission <= 1e-10)
        or np.any(normal[..., 2] <= 1e-10)
    ):
        raise ValueError(
            "aperture includes grazing or an unsupported refracting branch"
        )

    if np.any(incident_direction[..., 2] <= 0) or np.any(
        transmitted_direction[..., 2] <= 0
    ):
        raise ValueError("rays must propagate forward between real conjugates")

    return (
        incident_direction,
        transmitted_direction,
        normal,
        object_distance,
        image_distance,
        cos_incidence,
        cos_transmission,
    )


@dataclass(frozen=True, eq=False)
class RayBundle:
    system: StigmaticSystem
    surface_grid: SurfaceGrid
    entrance_grid: SurfaceGrid
    exit_grid: SurfaceGrid
    radius: np.ndarray
    azimuth: np.ndarray
    incident_direction: np.ndarray
    transmitted_direction: np.ndarray
    s: np.ndarray
    incident_p: np.ndarray
    transmitted_p: np.ndarray
    object_distance: np.ndarray
    image_distance: np.ndarray
    cos_incidence: np.ndarray
    cos_transmission: np.ndarray

    def __post_init__(self):
        names = (
            "radius",
            "azimuth",
            "incident_direction",
            "transmitted_direction",
            "s",
            "incident_p",
            "transmitted_p",
            "object_distance",
            "image_distance",
            "cos_incidence",
            "cos_transmission",
        )

        for name in names:
            object.__setattr__(self, name, frozen_array(getattr(self, name)))

    @property
    def geometric_amplitude(self):
        return (
            (-self.system.object_z / self.system.image_z)
            * self.image_distance
            / self.object_distance
        )

    @property
    def optical_path(self):
        return (
            self.system.object_material.n * self.object_distance
            + self.system.image_material.n * self.image_distance
        )

    @property
    def image_solid_angle_weights(self):
        return self.exit_grid.weights / self.system.image_z**2
