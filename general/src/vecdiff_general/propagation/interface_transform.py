"""Coherent vector propagation through a planar dielectric interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..fields import ElectricField, ElectromagneticField
from ..fields.spectrum import PlaneSpectrum, PlaneWaveExpansion
from ..fourier.nufft import phase_factor
from ..interfaces.dielectric_interface import PlanarInterface
from ..interfaces.directions import (
    direction_from_wavevector,
    reflected_direction,
    transmitted_direction,
)
from ..interfaces.fresnel import (
    apply_fresnel,
    fresnel_reflection_matrix,
    fresnel_transmission_matrix,
)
from ..sampling.grids import Grid


@dataclass(frozen=True)
class InterfaceSolution:
    interface: PlanarInterface
    incident: PlaneWaveExpansion
    reflected: PlaneWaveExpansion
    transmitted: PlaneWaveExpansion
    incident_side: str

    def evaluate(self, grid: Grid, *, side):
        """Total E/H in one half-space. The side is required for boundary traces."""

        if side not in ("minus", "plus"):
            raise ValueError("side must be 'minus' or 'plus'")

        frame = self.interface.frame
        signed_distance = (grid.points - frame.origin) @ frame.normal
        coordinate_scale = max(
            1.0, float(np.max(np.abs(grid.points))), float(np.max(np.abs(frame.origin)))
        )
        tolerance = 32 * np.finfo(float).eps * coordinate_scale

        if (side == "minus" and np.any(signed_distance > tolerance)) or (
            side == "plus" and np.any(signed_distance < -tolerance)
        ):
            raise ValueError("observation points lie outside the requested half-space")

        if side != self.incident_side:
            return self.transmitted.evaluate(grid)

        incident = self.incident.evaluate(grid)
        reflected = self.reflected.evaluate(grid)

        return ElectromagneticField(
            ElectricField(
                grid,
                self.incident.wavelength0,
                incident.electric.values + reflected.electric.values,
            ),
            incident.magnetic + reflected.magnetic,
        )


def scatter_plane(incident, interface):
    """Scatter every mode, preserving coherent phase and both branches.

    Incident evanescent modes are outside this first implementation's domain.
    Transmitted evanescent waves under TIR are retained. Exact grazing and
    critical incidence require a separate limiting treatment and are rejected.
    """

    if isinstance(incident, PlaneSpectrum):
        incident = incident.expansion

    if not isinstance(incident, PlaneWaveExpansion) or not isinstance(
        interface, PlanarInterface
    ):
        raise TypeError("expected a plane spectrum and PlanarInterface")

    if incident.region == interface.minus:
        normal, destination, side = interface.frame.normal, interface.plus, "minus"
    elif incident.region == interface.plus:
        normal, destination, side = -interface.frame.normal, interface.minus, "plus"
    else:
        raise ValueError("incident region is not adjacent to the interface")

    occupied = np.any(incident.amplitudes != 0, axis=-1)
    k = incident.wavevectors[occupied]
    amplitudes = incident.amplitudes[occupied]

    phase = phase_factor(k, (interface.frame.origin - incident.origin)[None, :])[0]
    amplitudes = amplitudes * phase[:, None]
    vacuum_wavenumber = 2 * np.pi / incident.wavelength0
    incident_index = incident.region.material.n
    transmitted_index = destination.material.n

    direction_incident = direction_from_wavevector(
        k, vacuum_wavenumber * incident_index
    )
    direction_transmitted = transmitted_direction(
        direction_incident, normal, incident_index, transmitted_index
    )
    direction_reflected = reflected_direction(direction_incident, normal)

    transmission = fresnel_transmission_matrix(
        direction_incident,
        direction_transmitted,
        normal,
        incident_index,
        transmitted_index,
    )
    reflection = fresnel_reflection_matrix(
        direction_incident,
        direction_transmitted,
        normal,
        incident_index,
        transmitted_index,
    )

    electric_transmitted = apply_fresnel(amplitudes, transmission)
    electric_reflected = apply_fresnel(amplitudes, reflection)

    wavevector_transmitted = (
        vacuum_wavenumber * transmitted_index * direction_transmitted
    )
    wavevector_reflected = vacuum_wavenumber * incident_index * direction_reflected

    def make(vectors, values, region):
        return PlaneWaveExpansion(
            vectors, values, interface.frame.origin, incident.wavelength0, region
        )

    return InterfaceSolution(
        interface,
        make(k, amplitudes, incident.region),
        make(wavevector_reflected, electric_reflected, incident.region),
        make(wavevector_transmitted, electric_transmitted, destination),
        side,
    )
