"""Geometric propagation from entrance sphere through the interface to exit sphere."""

from __future__ import annotations

import numpy as np

from ..fields import ElectricField, ElectromagneticField
from ..fields.transfer_result import TransferResult
from ..geometry.rays import RayBundle
from ..interfaces.fresnel import (
    apply_fresnel,
    fresnel_reflection_matrix,
    fresnel_transmission_matrix,
    reflection_coefficients,
    transmission_coefficients,
)
from ..media.medium import Z0


def _geometric_field(grid, wavelength, values, direction, index):
    return ElectromagneticField(
        ElectricField(grid, wavelength, values),
        index * np.cross(direction, values) / Z0,
    )


def transfer(rays, source):
    """G -> local Fresnel encounter -> G', retaining reflected power.

    Optical path relative to the two reference spheres is zero on the oval.
    E_G' = A (tp E_p p_image + ts E_s s). H here is the local geometric H;
    focal E/H are obtained separately by the full vector Debye integral.
    """

    if not isinstance(rays, RayBundle) or not isinstance(source, ElectricField):
        raise TypeError("expected RayBundle and ElectricField")

    if source.grid is not rays.entrance_grid:
        raise ValueError(
            "source must be prescribed on this bundle's entrance reference grid"
        )

    error = np.abs(np.sum(source.values * rays.incident_direction, axis=-1))

    if np.any(error > 1e-10 * np.linalg.norm(source.values, axis=-1)):
        raise ValueError("source must be transverse to the incident rays")

    incident_index = rays.system.object_material.n
    transmitted_index = rays.system.image_material.n
    normal = rays.surface_grid.normals

    transmission = fresnel_transmission_matrix(
        rays.incident_direction,
        rays.transmitted_direction,
        normal,
        incident_index,
        transmitted_index,
    )
    refracted = apply_fresnel(source.values, transmission)
    exit_values = rays.geometric_amplitude[..., None] * refracted

    reflection = fresnel_reflection_matrix(
        rays.incident_direction,
        rays.transmitted_direction,
        normal,
        incident_index,
        transmitted_index,
    )
    reflected = apply_fresnel(source.values, reflection)

    to_surface = -rays.system.object_z / rays.object_distance
    reflected_values = reflected * to_surface[..., None]
    reflected_direction = (
        rays.incident_direction - 2 * rays.cos_incidence[..., None] * normal
    )

    # This complete transfer explicitly retains both sets for inspection.
    transmission_s, transmission_p = transmission_coefficients(
        incident_index, transmitted_index, rays.cos_incidence, rays.cos_transmission
    )
    reflection_s, reflection_p = reflection_coefficients(
        incident_index, transmitted_index, rays.cos_incidence, rays.cos_transmission
    )

    # Common incident phase on the physical surface is essential for traces.
    vacuum_wavenumber = 2 * np.pi / source.wavelength0
    surface_phase = np.exp(
        1j
        * vacuum_wavenumber
        * incident_index
        * (rays.object_distance + rays.system.object_z)
    )
    surface_factor = to_surface * surface_phase
    wavelength = source.wavelength0

    return TransferResult(
        rays,
        _geometric_field(
            rays.entrance_grid,
            wavelength,
            source.values,
            rays.incident_direction,
            incident_index,
        ),
        _geometric_field(
            rays.exit_grid,
            wavelength,
            exit_values,
            rays.transmitted_direction,
            transmitted_index,
        ),
        _geometric_field(
            rays.surface_grid,
            wavelength,
            source.values * surface_factor[..., None],
            rays.incident_direction,
            incident_index,
        ),
        _geometric_field(
            rays.surface_grid,
            wavelength,
            reflected_values * surface_phase[..., None],
            reflected_direction,
            incident_index,
        ),
        _geometric_field(
            rays.surface_grid,
            wavelength,
            refracted * surface_factor[..., None],
            rays.transmitted_direction,
            transmitted_index,
        ),
        reflection_s,
        reflection_p,
        transmission_s,
        transmission_p,
    )
