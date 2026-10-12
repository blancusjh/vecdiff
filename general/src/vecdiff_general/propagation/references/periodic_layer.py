"""Independent Fourier–Maxwell reference for shallow 1D-periodic dielectric layers.

This Rayleigh representation matches both tangential E and H at both graphs.
Its field, bandwidth and independent-node boundary convergence must be tested.
It is not a reference for arbitrary deep profiles or finite three-dimensional
lenses. All electric/magnetic Cartesian components are retained (TE or TM).
"""

from dataclasses import dataclass

import numpy as np

from vecdiff_general.fields import ElectricField, ElectromagneticField
from vecdiff_general.media.medium import Z0


def modal_basis(kx, index, wavelength, sign, polarization):
    k0 = 2 * np.pi / wavelength
    q = np.sqrt(((index * k0) ** 2 - kx**2).astype(complex))
    vectors = np.column_stack((kx, np.zeros(len(kx)), sign * q))
    e = np.zeros_like(vectors)
    if polarization == "TE":
        e[:, 1] = 1
    elif polarization == "TM":
        e[:, 0], e[:, 2] = sign * q / (index**2 * k0), -kx / (index**2 * k0)
    else:
        raise ValueError("polarization must be TE or TM")
    return vectors, e, np.cross(vectors, e) / k0


def trace_matrices(kx, index, wavelength, sign, polarization, x, z, slope, origin):
    k, e, h = modal_basis(kx, index, wavelength, sign, polarization)
    phase = np.exp(1j * (x[:, None] * k[:, 0] + (z[:, None] - origin) * k[:, 2]))
    if polarization == "TE":
        electric = phase * e[None, :, 1]
        magnetic = phase * (h[None, :, 0] + slope[:, None] * h[None, :, 2])
    else:
        electric = phase * (e[None, :, 0] + slope[:, None] * e[None, :, 2])
        magnetic = phase * h[None, :, 1]
    return electric, magnetic


def evaluate_modes(
    kx, coefficients, index, wavelength, sign, polarization, grid, origin
):
    k, e, h = modal_basis(kx, index, wavelength, sign, polarization)
    p = grid.points.reshape(-1, 3)
    ev, hv = [], []
    for start in range(0, len(p), 512):
        block = p[start : start + 512]
        phase = np.exp(
            1j * (block[:, 0, None] * k[:, 0] + (block[:, 2, None] - origin) * k[:, 2])
        )
        ev.append(phase @ (e * coefficients[:, None]))
        hv.append(phase @ (h * coefficients[:, None]) / Z0)
    return ElectromagneticField(
        ElectricField(grid, wavelength, np.concatenate(ev).reshape(grid.shape + (3,))),
        np.concatenate(hv).reshape(grid.shape + (3,)),
    )


@dataclass
class PeriodicLayerReference:
    period: float
    thickness: float
    wavelength: float
    exterior_index: float
    interior_index: float
    lower_height: object
    lower_slope: object
    upper_height: object
    upper_slope: object
    polarization: str = "TE"
    half_modes: int = 16

    def __post_init__(self):
        if (
            min(
                self.period,
                self.thickness,
                self.wavelength,
                self.exterior_index,
                self.interior_index,
            )
            <= 0
        ):
            raise ValueError("geometry, indices and wavelength must be positive")
        if not isinstance(self.half_modes, int) or self.half_modes < 1:
            raise ValueError("half_modes must be a positive integer")
        self.kx = (
            2 * np.pi * np.arange(-self.half_modes, self.half_modes + 1) / self.period
        )
        count = 2 * len(self.kx) + 1
        x = np.arange(count) * self.period / count - self.period / 2
        matrices = []
        targets = []
        self.incident = np.zeros(len(self.kx), complex)
        self.incident[self.half_modes] = (
            1 if self.polarization == "TE" else self.exterior_index
        )
        for interface in (0, 1):
            z = (
                self.lower_height(x)
                if interface == 0
                else self.thickness + self.upper_height(x)
            )
            slope = self.lower_slope(x) if interface == 0 else self.upper_slope(x)
            traces = [
                trace_matrices(
                    self.kx,
                    n,
                    self.wavelength,
                    s,
                    self.polarization,
                    x,
                    z,
                    slope,
                    origin,
                )
                for n, s, origin in (
                    (self.exterior_index, -1, 0),
                    (self.interior_index, 1, 0),
                    (self.interior_index, -1, self.thickness),
                    (self.exterior_index, 1, self.thickness),
                )
            ]
            incoming = trace_matrices(
                self.kx,
                self.exterior_index,
                self.wavelength,
                1,
                self.polarization,
                x,
                z,
                slope,
                0,
            )
            for component in (0, 1):
                block = [a[component] for a in traces]
                if interface == 0:
                    block = [block[0], -block[1], -block[2], np.zeros_like(block[3])]
                    rhs = -incoming[component] @ self.incident
                else:
                    block = [np.zeros_like(block[0]), block[1], block[2], -block[3]]
                    rhs = np.zeros(count, complex)
                matrices.append(np.column_stack(block))
                targets.append(rhs)
        matrix = np.vstack(matrices)
        target = np.concatenate(targets)
        solution, residuals, rank, singular = np.linalg.lstsq(
            matrix, target, rcond=None
        )
        if rank < matrix.shape[1]:
            raise ValueError("periodic reference trace system is rank deficient")
        self.reflected, self.up, self.down, self.transmitted = np.split(solution, 4)
        self.condition = float(singular[0] / singular[-1])
        self.solve_residual = float(
            np.linalg.norm(matrix @ solution - target) / np.linalg.norm(target)
        )

    def internal(self, grid):
        from vecdiff_general.observables import coherent_sum

        return coherent_sum(
            [
                evaluate_modes(
                    self.kx,
                    a,
                    self.interior_index,
                    self.wavelength,
                    s,
                    self.polarization,
                    grid,
                    origin,
                )
                for a, s, origin in ((self.up, 1, 0), (self.down, -1, self.thickness))
            ]
        )
