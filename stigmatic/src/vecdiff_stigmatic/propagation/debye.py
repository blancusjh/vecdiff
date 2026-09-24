"""Direct vector Debye quadrature, with physical output positions and SI H."""

from dataclasses import dataclass

import numpy as np

from ..fields import ElectricField, ElectromagneticField
from ..fields.transfer_result import TransferResult
from ..media.medium import Z0


@dataclass(frozen=True)
class DebyeFocus:
    """Angular integral of the transmitted reference-sphere field.

    E(r) = -i k b exp(i k b)/(2 pi) integral E_G'(u)
           exp(i k u dot (r-image)) dOmega.

    It is a source-free Maxwell plane-wave superposition in the image medium.
    Its relation to the actual refracting surface is the Debye/geometric
    approximation. It is not a full curved dielectric boundary solution.
    """

    transfer: TransferResult

    def __post_init__(self):
        if not isinstance(self.transfer, TransferResult):
            raise TypeError("DebyeFocus requires a reference-sphere TransferResult")

    @property
    def origin(self):
        return self.transfer.rays.system.surface.image_point

    @property
    def wavevectors(self):
        result = self.transfer
        k = (
            2
            * np.pi
            * result.rays.system.image_material.n
            / result.exit.electric.wavelength0
        )

        return k * result.rays.transmitted_direction.reshape(-1, 3)

    @property
    def amplitudes(self):
        result = self.transfer
        radius = result.rays.system.image_z
        k = (
            2
            * np.pi
            * result.rays.system.image_material.n
            / result.exit.electric.wavelength0
        )
        prefactor = -1j * k * radius * np.exp(1j * k * radius) / (2 * np.pi)
        weighted = (
            result.exit.electric.values
            * result.rays.image_solid_angle_weights[..., None]
        )

        return prefactor * weighted.reshape(-1, 3)

    def evaluate(self, grid, *, chunk_size=256, backend="direct", tolerance=1e-12):
        """Evaluate with direct, NUFFT, or analytic Fourier–Bessel synthesis.

        The Hankel backend integrates azimuth analytically for the sampled
        harmonic representation. Radial quadrature and the physical model are
        unchanged; no axisymmetric-field assumption is made.
        """
        from ..fourier.nufft import synthesize
        from ..fourier.polar import azimuthal_coefficients, cylindrical_synthesize

        wavevectors = self.wavevectors
        amplitudes = self.amplitudes
        wavelength = self.transfer.exit.electric.wavelength0
        magnetic_amplitudes = np.cross(wavevectors, amplitudes) / (
            2 * np.pi / wavelength * Z0
        )
        coefficients = np.concatenate((amplitudes, magnetic_amplitudes), axis=-1)

        if backend == "hankel":
            radial_count, azimuthal_count = self.transfer.rays.radius.shape
            rings = coefficients.reshape(radial_count, azimuthal_count, 6)
            orders, harmonics = azimuthal_coefficients(rings, axis=1)
            harmonics *= azimuthal_count
            first_meridian = wavevectors.reshape(radial_count, azimuthal_count, 3)[:, 0]
            # Rays converge: the radial wavevector in the surface's phi chart is negative.
            values = cylindrical_synthesize(
                first_meridian[:, 0],
                first_meridian[:, 2],
                harmonics,
                orders,
                grid.points - self.origin,
            )
        else:
            values = synthesize(
                wavevectors,
                coefficients,
                grid.points - self.origin,
                backend=backend,
                tolerance=tolerance,
                chunk_size=chunk_size,
            )

        return ElectromagneticField(
            ElectricField(grid, wavelength, values[..., :3]), values[..., 3:]
        )

    def electric(self, grid, **kwargs):
        return self.evaluate(grid, **kwargs).electric
