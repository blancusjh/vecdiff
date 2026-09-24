"""Stable coherent scattering through parallel layers, including evanescent gaps.

Forward fields are anchored at left boundaries, backward fields at right
boundaries. Only decaying evanescent factors enter the scattering recursion.
"""

from dataclasses import dataclass

import numpy as np

from .._arrays import frozen_array
from ..fields import ElectricField, ElectromagneticField
from ..fields.spectrum import PlaneSpectrum, PlaneWaveExpansion
from ..interfaces.layer_admittance import tangential_coefficients
from ..media.layers import LayerStack
from ..media.medium import Z0


@dataclass(frozen=True)
class LayeredSolution:
    incident: PlaneWaveExpansion
    stack: LayerStack
    normal_wavenumbers: np.ndarray
    s_basis: np.ndarray
    tangential_p_basis: np.ndarray
    forward: tuple[np.ndarray, ...]
    backward: tuple[np.ndarray, ...]
    reflection: np.ndarray
    transmission: np.ndarray

    def __post_init__(self):
        for name in (
            "normal_wavenumbers",
            "s_basis",
            "tangential_p_basis",
            "reflection",
            "transmission",
        ):
            object.__setattr__(
                self, name, frozen_array(getattr(self, name), dtype=complex)
            )

        for name in ("forward", "backward"):
            object.__setattr__(
                self,
                name,
                tuple(
                    frozen_array(value, dtype=complex) for value in getattr(self, name)
                ),
            )

    def evaluate(self, grid, *, region: int):
        """Evaluate total SI E/H in one explicitly selected material region."""

        if not isinstance(region, int) or not 0 <= region < len(self.stack.regions):
            raise ValueError("region index is outside the layer stack")

        frame = self.stack.frame
        points = (grid.points.reshape(-1, 3) - frame.origin) @ frame.basis
        boundaries = self.stack.boundaries
        left_anchor = boundaries[max(0, region - 1)]
        right_anchor = boundaries[min(region, len(boundaries) - 1)]
        tolerance = 32 * np.finfo(float).eps * max(1.0, np.max(np.abs(grid.points)))

        if region > 0 and np.any(points[:, 2] < left_anchor - tolerance):
            raise ValueError("points lie before the selected layer")

        if region < len(self.stack.regions) - 1 and np.any(
            points[:, 2] > right_anchor + tolerance
        ):
            raise ValueError("points lie after the selected layer")

        incident_local = self.incident.wavevectors @ frame.basis
        transverse = incident_local[:, :2]
        normal = self.normal_wavenumbers[:, region]
        vacuum_wavenumber = 2 * np.pi / self.incident.wavelength0
        electric = np.zeros((len(points), 3), dtype=complex)
        magnetic = np.zeros_like(electric)

        branches = (
            (1, self.forward[region], left_anchor),
            (-1, self.backward[region], right_anchor),
        )

        for direction, coefficients, anchor in branches:
            if not np.any(coefficients):
                continue

            wavevectors = np.column_stack((transverse, direction * normal))
            normal_p = -np.sum(transverse * self.tangential_p_basis, axis=-1) / (
                direction * normal
            )
            p_basis = np.column_stack((self.tangential_p_basis, normal_p))
            s_basis = np.column_stack((self.s_basis, np.zeros(len(normal))))
            amplitudes = coefficients[:, :1] * s_basis + coefficients[:, 1:] * p_basis
            magnetic_amplitudes = np.cross(wavevectors, amplitudes) / (
                vacuum_wavenumber * Z0
            )

            for start in range(0, len(points), 256):
                stop = start + 256
                phase = np.exp(
                    1j
                    * (
                        points[start:stop, :2] @ transverse.T
                        + (points[start:stop, 2] - anchor)[:, None] * direction * normal
                    )
                )
                electric[start:stop] += phase @ amplitudes
                magnetic[start:stop] += phase @ magnetic_amplitudes

        shape = grid.shape + (3,)
        electric = (electric @ frame.basis.T).reshape(shape)
        magnetic = (magnetic @ frame.basis.T).reshape(shape)

        return ElectromagneticField(
            ElectricField(grid, self.incident.wavelength0, electric), magnetic
        )


def propagate_layers(incident, stack: LayerStack) -> LayeredSolution:
    """Sum all coherent round trips by recursively composing scattering data."""

    if isinstance(incident, PlaneSpectrum):
        incident = incident.expansion

    if incident.region != stack.regions[0]:
        raise ValueError("incident material region must match the entrance half-space")

    occupied = np.any(incident.amplitudes != 0, axis=-1)
    incident = PlaneWaveExpansion(
        incident.wavevectors[occupied],
        incident.amplitudes[occupied],
        incident.origin,
        incident.wavelength0,
        incident.region,
    )
    local_wavevectors = incident.wavevectors @ stack.frame.basis

    if np.any(np.abs(local_wavevectors.imag) > 0) or np.any(
        local_wavevectors[:, 2].real <= 0
    ):
        raise ValueError("incident modes must propagate toward the stack in local +z")

    vacuum_wavenumber = 2 * np.pi / incident.wavelength0
    transverse_squared = np.sum(local_wavevectors[:, :2].real ** 2, axis=-1)
    normal_wavenumbers = np.sqrt(
        np.array(
            [
                region.material.n**2 * vacuum_wavenumber**2 - transverse_squared
                for region in stack.regions
            ],
            dtype=complex,
        ).T
    )

    if np.any(np.abs(normal_wavenumbers) < 1e-13 * vacuum_wavenumber):
        raise ValueError(
            "exact grazing/critical layers require a limiting representation"
        )

    mode_count, region_count = normal_wavenumbers.shape
    transverse_norm = np.sqrt(transverse_squared)
    tangent_p = np.divide(
        local_wavevectors[:, :2].real,
        transverse_norm[:, None],
        out=np.tile([1.0, 0.0], (mode_count, 1)),
        where=transverse_norm[:, None] != 0,
    )
    s_basis = np.column_stack((-tangent_p[:, 1], tangent_p[:, 0]))
    local_electric = incident.amplitudes @ stack.frame.basis
    phase_to_boundary = np.exp(
        1j * (incident.wavevectors @ (stack.frame.origin - incident.origin))
    )
    injection = (
        np.column_stack(
            (
                np.sum(local_electric[:, :2] * s_basis, axis=-1),
                np.sum(local_electric[:, :2] * tangent_p, axis=-1),
            )
        )
        * phase_to_boundary[:, None]
    )

    reflections, transmissions = [], []

    for boundary in range(region_count - 1):
        reflection, transmission = tangential_coefficients(
            normal_wavenumbers[:, boundary],
            normal_wavenumbers[:, boundary + 1],
            stack.regions[boundary].material,
            stack.regions[boundary + 1].material,
            vacuum_wavenumber,
        )
        reflections.append(reflection)
        transmissions.append(transmission)

    propagation = [np.ones((mode_count, 1), dtype=complex)]
    propagation.extend(
        np.exp(1j * normal_wavenumbers[:, layer] * thickness)[:, None]
        for layer, thickness in enumerate(stack.thicknesses, start=1)
    )
    propagation.append(np.ones((mode_count, 1), dtype=complex))

    effective = [None] * (region_count - 1)
    effective[-1] = reflections[-1]

    for boundary in range(region_count - 3, -1, -1):
        returning = effective[boundary + 1] * propagation[boundary + 1] ** 2
        reflection = reflections[boundary]
        effective[boundary] = reflection + (
            transmissions[boundary]
            * (1 - reflection)
            * returning
            / (1 + reflection * returning)
        )

    forward = [injection]
    backward = [effective[0] * injection]
    total_transmission = np.ones_like(injection)

    for boundary in range(region_count - 1):
        returning = (
            effective[boundary + 1] * propagation[boundary + 1] ** 2
            if boundary + 1 < region_count - 1
            else 0.0
        )
        step_transmission = (
            transmissions[boundary]
            * propagation[boundary]
            / (1 + reflections[boundary] * returning)
        )
        following = step_transmission * forward[boundary]
        forward.append(following)
        backward.append(
            effective[boundary + 1] * following * propagation[boundary + 1]
            if boundary + 1 < region_count - 1
            else np.zeros_like(injection)
        )
        total_transmission *= step_transmission

    return LayeredSolution(
        incident,
        stack,
        normal_wavenumbers,
        s_basis,
        tangent_p,
        tuple(forward),
        tuple(backward),
        effective[0],
        total_transmission,
    )
