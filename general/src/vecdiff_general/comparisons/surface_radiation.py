"""Vector Green radiation of specified SI electric and magnetic surface currents.

This module propagates boundary data; it does not solve the dielectric boundary
problem that generated them. Normal orientation selects the intended region.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

from .._arrays import frozen_array, positive, positive_integer
from ..fields import ElectricField, ElectromagneticField
from ..fields.spectrum import HalfSpaceSpectrum, LocalSpectrum, PlaneWaveExpansion
from ..fourier.nufft import NUFFT
from ..media.medium import Z0, HomogeneousRegion
from ..sampling.grids import SurfaceGrid


@dataclass(frozen=True)
class SurfaceRadiation:
    grid: SurfaceGrid
    electric_current: np.ndarray
    magnetic_current: np.ndarray
    wavelength0: float
    region: HomogeneousRegion

    def __post_init__(self):
        if not isinstance(self.grid, SurfaceGrid):
            raise TypeError("surface currents require a SurfaceGrid with area weights")

        for name in ("electric_current", "magnetic_current"):
            value = frozen_array(getattr(self, name), dtype=complex)

            if value.shape != self.grid.shape + (3,):
                raise ValueError("surface currents must match their physical grid")

            object.__setattr__(self, name, value)

        object.__setattr__(
            self, "wavelength0", positive(self.wavelength0, "vacuum wavelength")
        )

        if not isinstance(self.region, HomogeneousRegion):
            raise TypeError("surface radiation requires homogeneous material context")

    @classmethod
    def from_field(cls, field: ElectromagneticField, region, *, normal_sign=1):
        if normal_sign not in (-1, 1):
            raise ValueError("normal_sign must be +1 or -1")

        normal = normal_sign * field.grid.normals
        electric_current = np.cross(normal, field.magnetic)
        magnetic_current = -np.cross(normal, field.electric.values)

        return cls(
            field.grid,
            electric_current,
            magnetic_current,
            field.electric.wavelength0,
            region,
        )

    def evaluate(self, grid, *, chunk_size=32):
        """Full dyadic Green integral, including reactive near-field terms.

        Observations at a quadrature node are singular and rejected. Near a
        surface, distance and quadrature must be converged independently.
        """
        chunk_size = positive_integer(chunk_size, "chunk_size")
        points = grid.points.reshape(-1, 3)
        sources = self.grid.points.reshape(-1, 3)
        weights = self.grid.weights.ravel()[:, None]
        electric_current = self.electric_current.reshape(-1, 3) * weights
        magnetic_current = self.magnetic_current.reshape(-1, 3) * weights
        vacuum_wavenumber = 2 * np.pi / self.wavelength0
        wavenumber = self.region.material.n * vacuum_wavenumber
        electric = np.empty_like(points, dtype=complex)
        magnetic = np.empty_like(points, dtype=complex)

        for start in range(0, len(points), chunk_size):
            stop = start + chunk_size
            displacement = points[start:stop, None, :] - sources[None, :, :]
            distance = np.linalg.norm(displacement, axis=-1)

            if np.any(distance == 0):
                raise ValueError("Green evaluation at a source node is singular")

            direction = displacement / distance[..., None]
            green = np.exp(1j * wavenumber * distance) / (4 * np.pi * distance)
            first_derivative = (1j * wavenumber - 1 / distance) * green
            second_derivative = (
                (1j * wavenumber - 1 / distance) ** 2 + 1 / distance**2
            ) * green

            def dyadic(current):
                radial_current = (
                    np.sum(direction * current, axis=-1)[..., None] * direction
                )
                tangential_current = current - radial_current
                hessian = second_derivative[..., None] * radial_current
                hessian += (first_derivative / distance)[..., None] * tangential_current

                return green[..., None] * current + hessian / wavenumber**2

            gradient = first_derivative[..., None] * direction
            electric[start:stop] = np.sum(
                1j * vacuum_wavenumber * Z0 * dyadic(electric_current)
                - np.cross(gradient, magnetic_current),
                axis=1,
            )
            magnetic[start:stop] = np.sum(
                np.cross(gradient, electric_current)
                + 1j
                * vacuum_wavenumber
                * self.region.material.n**2
                / Z0
                * dyadic(magnetic_current),
                axis=1,
            )

        shape = grid.shape + (3,)

        return ElectromagneticField(
            ElectricField(grid, self.wavelength0, electric.reshape(shape)),
            magnetic.reshape(shape),
        )

    def electric(self, grid, **kwargs):
        return self.evaluate(grid, **kwargs).electric

    def local_spectrum(self, center, radius):
        """Compress radiation on an observation ball with conservative absolute bounds."""
        center = np.asarray(center, dtype=float)
        radius = float(radius)

        if (
            center.shape != (3,)
            or not np.isfinite(center).all()
            or not np.isfinite(radius)
            or radius < 0
        ):
            raise ValueError(
                "finite center and nonnegative observation radius required"
            )

        displacement = center - self.grid.points.reshape(-1, 3)
        distance = np.linalg.norm(displacement, axis=-1)

        if np.any(distance <= radius):
            raise ValueError("observation ball intersects the source quadrature")

        direction = displacement / distance[:, None]
        vacuum_wavenumber = 2 * np.pi / self.wavelength0
        wavenumber = self.region.material.n * vacuum_wavenumber
        weights = self.grid.weights.ravel()[:, None]
        # Normalize currents only inside this derivation; the public H is SI.
        electric_current = Z0 * self.electric_current.reshape(-1, 3) * weights
        magnetic_current = self.magnetic_current.reshape(-1, 3) * weights
        transverse_current = (
            electric_current
            - direction * np.sum(direction * electric_current, axis=-1)[:, None]
        )
        green = np.exp(1j * wavenumber * distance) / (4 * np.pi * distance)
        amplitudes = (
            1j
            * green[:, None]
            * (
                vacuum_wavenumber * transverse_current
                - wavenumber * np.cross(direction, magnetic_current)
            )
        )
        expansion = PlaneWaveExpansion(
            wavenumber * direction, amplitudes, center, self.wavelength0, self.region
        )

        minimum_distance = distance - radius
        phase_bound = np.minimum(2.0, wavenumber * radius**2 / (2 * minimum_distance))
        green_bound = (
            radius / (distance * minimum_distance) + phase_bound / distance
        ) / (4 * np.pi)
        direction_bound = 2 * radius / minimum_distance
        electric_norm = np.linalg.norm(electric_current, axis=-1)
        magnetic_norm = np.linalg.norm(magnetic_current, axis=-1)
        reactive_dyadic = (
            2
            * (
                1 / (wavenumber * minimum_distance)
                + 1 / (wavenumber * minimum_distance) ** 2
            )
            / (4 * np.pi * minimum_distance)
        )
        reactive_curl = 1 / (4 * np.pi * minimum_distance**2)
        electric_bound = np.sum(
            green_bound
            * (vacuum_wavenumber * electric_norm + wavenumber * magnetic_norm)
            + direction_bound
            * (2 * vacuum_wavenumber * electric_norm + wavenumber * magnetic_norm)
            / (4 * np.pi * distance)
            + vacuum_wavenumber * reactive_dyadic * electric_norm
            + reactive_curl * magnetic_norm
        )
        index_squared = self.region.material.n**2
        magnetic_bound = (
            np.sum(
                green_bound
                * (
                    wavenumber * electric_norm
                    + vacuum_wavenumber * index_squared * magnetic_norm
                )
                + direction_bound
                * (
                    wavenumber * electric_norm
                    + 2 * vacuum_wavenumber * index_squared * magnetic_norm
                )
                / (4 * np.pi * distance)
                + reactive_curl * electric_norm
                + vacuum_wavenumber * index_squared * reactive_dyadic * magnetic_norm
            )
            / Z0
        )

        return LocalSpectrum(
            expansion, radius, float(electric_bound), float(magnetic_bound)
        )

    def angular_spectrum(
        self, *, direction=1, polar_order=48, azimuthal_order=96, backend="direct"
    ):
        """Propagating Weyl hemisphere beyond the source's global-z envelope.

        This approximation omits evanescent radiation. The angular Jacobian
        cancels the Weyl 1/kz singularity. It is not an FFT periodic cell.
        """

        if direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")

        polar_order = positive_integer(polar_order, "polar_order")
        azimuthal_order = positive_integer(azimuthal_order, "azimuthal_order")
        nodes, weights = leggauss(polar_order)
        theta = (nodes + 1) * np.pi / 4
        theta_weights = weights * np.pi / 4
        theta, phi = np.meshgrid(
            theta,
            2 * np.pi * np.arange(azimuthal_order) / azimuthal_order,
            indexing="ij",
        )
        vacuum_wavenumber = 2 * np.pi / self.wavelength0
        wavenumber = self.region.material.n * vacuum_wavenumber
        directions = np.stack(
            (
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                direction * np.cos(theta),
            ),
            axis=-1,
        )
        wavevectors = wavenumber * directions.reshape(-1, 3)
        sources = self.grid.points.reshape(-1, 3)
        origin = np.mean(sources, axis=0)
        currents = np.concatenate(
            (
                Z0 * self.electric_current.reshape(-1, 3),
                self.magnetic_current.reshape(-1, 3),
            ),
            axis=-1,
        )
        currents *= self.grid.weights.ravel()[:, None]
        transformed = NUFFT(
            currents, sources - origin, wavevectors, sign=-1, backend=backend
        )
        electric_current, magnetic_current = transformed[:, :3], transformed[:, 3:]
        transverse = (
            electric_current
            - wavevectors
            * np.sum(wavevectors * electric_current, axis=-1)[:, None]
            / wavenumber**2
        )
        measure = (
            wavenumber
            * np.sin(theta)
            * theta_weights[:, None]
            * (2 * np.pi / azimuthal_order)
            / (2 * (2 * np.pi) ** 2)
        ).ravel()
        amplitudes = (
            -vacuum_wavenumber * transverse + np.cross(wavevectors, magnetic_current)
        ) * measure[:, None]
        expansion = PlaneWaveExpansion(
            wavevectors, amplitudes, origin, self.wavelength0, self.region
        )
        boundary = np.max(sources[:, 2]) if direction == 1 else np.min(sources[:, 2])

        return HalfSpaceSpectrum(expansion, float(boundary), direction)
