"""Local planar-interface scattering of each incident mode at a curved surface."""

from dataclasses import dataclass

import numpy as np

from vecdiff_general.comparisons.surface_radiation import SurfaceRadiation
from vecdiff_general.fields import ElectricField, ElectromagneticField
from vecdiff_general.fields.spectrum import PlaneSpectrum
from vecdiff_general.media.medium import Z0
from vecdiff_general.propagation._directions import (
    direction_from_wavevector,
    reflected_direction,
    transmitted_direction,
)
from vecdiff_general.propagation._fresnel import (
    apply_fresnel,
    fresnel_reflection_matrix,
    fresnel_transmission_matrix,
)


@dataclass(frozen=True)
class SurfaceInteraction:
    interface: object
    incident_trace: ElectromagneticField
    reflected_trace: ElectromagneticField
    transmitted_trace: ElectromagneticField
    reflected: SurfaceRadiation
    transmitted: SurfaceRadiation
    incident_side: str


def scatter_surface(incident, interface, grid, *, illuminated_only=False):
    """Apply Fresnel to every incident spectral mode before coherent summation.

    The returned traces are local physical-optics boundary data. Subsequent
    Green radiation solves homogeneous Maxwell equations for these currents,
    but need not reproduce an exact global dielectric boundary solution.
    """

    if isinstance(incident, PlaneSpectrum):
        incident = incident.expansion

    if incident.region == interface.minus:
        normal_sign, destination, incident_side = 1, interface.plus, "minus"
    elif incident.region == interface.plus:
        normal_sign, destination, incident_side = -1, interface.minus, "plus"
    else:
        raise ValueError("incident region is not adjacent to this interface")

    points = grid.points.reshape(-1, 3)
    local = (points - interface.surface.frame.origin) @ interface.surface.frame.basis
    expected = interface.surface.points(local[:, :2])
    tolerance = 64 * np.finfo(float).eps * max(1.0, np.max(np.abs(points)))

    if np.max(np.abs(expected - points)) > tolerance:
        raise ValueError("surface grid is not embedded on this interface")

    normals = normal_sign * grid.normals.reshape(-1, 3)
    reflected_electric = np.zeros_like(points, dtype=complex)
    transmitted_electric = np.zeros_like(points, dtype=complex)
    reflected_magnetic = np.zeros_like(points, dtype=complex)
    transmitted_magnetic = np.zeros_like(points, dtype=complex)
    vacuum_wavenumber = 2 * np.pi / incident.wavelength0

    for wavevector, amplitude in zip(incident.wavevectors, incident.amplitudes):
        if not np.any(amplitude):
            continue

        if np.any(np.abs(wavevector.imag) > 1e-12 * vacuum_wavenumber):
            raise ValueError(
                "curved local-interface scattering requires propagating incident modes"
            )

        illuminated = normals @ wavevector.real > 1e-12 * np.linalg.norm(wavevector)

        if not illuminated_only and not np.all(illuminated):
            raise ValueError(
                "surface includes back-facing/grazing points; choose a supported domain or an explicit shadow approximation"
            )

        if not np.any(illuminated):
            continue

        incident_index = incident.region.material.n
        transmitted_index = destination.material.n
        local_normals = normals[illuminated]

        direction_incident = direction_from_wavevector(
            wavevector, vacuum_wavenumber * incident_index
        )
        direction_transmitted = transmitted_direction(
            direction_incident, local_normals, incident_index, transmitted_index
        )
        direction_reflected = reflected_direction(direction_incident, local_normals)

        transmission = fresnel_transmission_matrix(
            direction_incident,
            direction_transmitted,
            local_normals,
            incident_index,
            transmitted_index,
        )
        reflection = fresnel_reflection_matrix(
            direction_incident,
            direction_transmitted,
            local_normals,
            incident_index,
            transmitted_index,
        )

        electric_transmitted = apply_fresnel(amplitude, transmission)
        electric_reflected = apply_fresnel(amplitude, reflection)

        wavevector_transmitted = (
            vacuum_wavenumber * transmitted_index * direction_transmitted
        )
        wavevector_reflected = vacuum_wavenumber * incident_index * direction_reflected

        phase = np.exp(1j * ((points[illuminated] - incident.origin) @ wavevector))[
            :, None
        ]
        reflected_electric[illuminated] += phase * electric_reflected
        transmitted_electric[illuminated] += phase * electric_transmitted
        reflected_magnetic[illuminated] += (
            phase
            * np.cross(wavevector_reflected, electric_reflected)
            / (vacuum_wavenumber * Z0)
        )
        transmitted_magnetic[illuminated] += (
            phase
            * np.cross(wavevector_transmitted, electric_transmitted)
            / (vacuum_wavenumber * Z0)
        )

    def trace(electric, magnetic):
        return ElectromagneticField(
            ElectricField(
                grid, incident.wavelength0, electric.reshape(grid.shape + (3,))
            ),
            magnetic.reshape(grid.shape + (3,)),
        )

    reflected_trace = trace(reflected_electric, reflected_magnetic)
    transmitted_trace = trace(transmitted_electric, transmitted_magnetic)

    return SurfaceInteraction(
        interface,
        incident.evaluate(grid),
        reflected_trace,
        transmitted_trace,
        SurfaceRadiation.from_field(
            reflected_trace, incident.region, normal_sign=-normal_sign
        ),
        SurfaceRadiation.from_field(
            transmitted_trace, destination, normal_sign=normal_sign
        ),
        incident_side,
    )
