"""A stigmatic refracting element with assigned materials and real conjugates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.legendre import leggauss

from .._arrays import positive, positive_integer
from ..geometry.rays import RayBundle, _ray_geometry
from ..media.medium import Dielectric
from ..sampling.grids import SurfaceGrid
from ..surfaces.cartesian_oval import CartesianOval


@dataclass(frozen=True)
class StigmaticSystem:
    object_z: float
    image_z: float
    object_material: Dielectric
    image_material: Dielectric
    surface: CartesianOval = field(init=False)

    def __post_init__(self):
        if not all(
            isinstance(m, Dielectric)
            for m in (self.object_material, self.image_material)
        ):
            raise TypeError("stigmatic media must be Dielectric materials")

        surface = CartesianOval(
            self.object_z, self.image_z, self.object_material.n, self.image_material.n
        )
        object.__setattr__(self, "surface", surface)

    def rays(self, aperture, *, radial_order=48, azimuthal_order=96):
        """Gauss radial / periodic azimuth quadrature over the physical aperture."""
        aperture = positive(aperture, "aperture")
        radial_order = positive_integer(radial_order, "radial_order")
        azimuthal_order = positive_integer(azimuthal_order, "azimuthal_order")

        if azimuthal_order < 4:
            raise ValueError("azimuthal_order must be at least four")

        surface = self.surface
        # Check the requested edge as well as the interior Gauss nodes.
        edge_z = surface.sag(np.array([aperture]))[0]
        _ray_geometry(self, np.array([[aperture, 0.0, edge_z]]))
        nodes, radial_weights = leggauss(radial_order)
        radius = aperture * (nodes + 1) / 2
        radial_weights = aperture * radial_weights / 2
        phi = 2 * np.pi * np.arange(azimuthal_order) / azimuthal_order
        radial_grid, angle = np.meshgrid(radius, phi, indexing="ij")
        surface_height = np.broadcast_to(
            surface.sag(radius)[:, None], radial_grid.shape
        )
        points = np.stack(
            (radial_grid * np.cos(angle), radial_grid * np.sin(angle), surface_height),
            axis=-1,
        )
        (
            incident_direction,
            transmitted_direction,
            normal,
            object_distance,
            image_distance,
            cos_incidence,
            cos_transmission,
        ) = _ray_geometry(self, points)
        s_polarization = np.stack(
            (-np.sin(angle), np.cos(angle), np.zeros_like(angle)), axis=-1
        )
        incident_p, transmitted_p = (
            np.cross(s_polarization, incident_direction),
            np.cross(s_polarization, transmitted_direction),
        )
        azimuthal_weight = 2 * np.pi / azimuthal_order
        area = radial_grid * radial_weights[:, None] * azimuthal_weight / normal[..., 2]
        surface_grid = SurfaceGrid(points, normal, area)
        entrance_radius, exit_radius = -self.object_z, self.image_z
        entrance_area = entrance_radius**2 * cos_incidence * area / object_distance**2
        exit_area = exit_radius**2 * cos_transmission * area / image_distance**2
        entrance_grid = SurfaceGrid(
            surface.object_point + entrance_radius * incident_direction,
            incident_direction,
            entrance_area,
        )
        # Normals on the exit reference cap point inward, along propagation.
        exit_grid = SurfaceGrid(
            surface.image_point - exit_radius * transmitted_direction,
            transmitted_direction,
            exit_area,
        )

        return RayBundle(
            self,
            surface_grid,
            entrance_grid,
            exit_grid,
            radial_grid,
            angle,
            incident_direction,
            transmitted_direction,
            s_polarization,
            incident_p,
            transmitted_p,
            object_distance,
            image_distance,
            cos_incidence,
            cos_transmission,
        )
