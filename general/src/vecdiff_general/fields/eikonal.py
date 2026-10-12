"""An explicit high-frequency phase and transverse envelope, with material context."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from vecdiff_general._arrays import positive
from vecdiff_general.fields.electromagnetic_field import (
    ElectricField,
    ElectromagneticField,
)
from vecdiff_general.media.medium import Z0, HomogeneousRegion


@dataclass(frozen=True)
class EikonalField:
    """E(r) = envelope(r) exp(i k0 optical_path(r)).

    optical_path is in metres and its gradient has norm n. This is a declared
    WKB representation, not a phase inferred from an interfering total field.
    Its local H and ray transversality are leading-order approximations.
    """

    optical_path: Callable
    path_gradient: Callable
    envelope: Callable
    wavelength0: float
    region: HomogeneousRegion

    def __post_init__(self):
        object.__setattr__(
            self, "wavelength0", positive(self.wavelength0, "vacuum wavelength")
        )

        if not all(
            callable(value)
            for value in (self.optical_path, self.path_gradient, self.envelope)
        ):
            raise TypeError(
                "phase, gradient and envelope must be callable spatial functions"
            )

        if not isinstance(self.region, HomogeneousRegion):
            raise TypeError(
                "eikonal representation needs a homogeneous material region"
            )

    def directions(self, points):
        gradient = np.broadcast_to(
            np.asarray(self.path_gradient(points), dtype=float), np.shape(points)
        )

        if not np.isfinite(gradient).all() or not np.allclose(
            np.linalg.norm(gradient, axis=-1),
            self.region.material.n,
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError(
                "optical-path gradient violates the eikonal equation |grad L| = n"
            )

        return gradient / self.region.material.n

    def amplitudes(self, points):
        electric = np.broadcast_to(
            np.asarray(self.envelope(points), dtype=complex), np.shape(points)
        )
        direction = self.directions(points)

        if not np.isfinite(electric).all() or np.any(
            np.abs(np.sum(direction * electric, axis=-1))
            > 1e-10 * np.linalg.norm(electric, axis=-1)
        ):
            raise ValueError("eikonal envelope must be finite and locally transverse")

        return electric

    def evaluate(self, grid):
        path = np.broadcast_to(
            np.asarray(self.optical_path(grid.points), dtype=float), grid.shape
        )

        if not np.isfinite(path).all():
            raise ValueError("optical path must be finite")

        electric = (
            self.amplitudes(grid.points)
            * np.exp(2j * np.pi * path / self.wavelength0)[..., None]
        )
        magnetic = (
            self.region.material.n
            / Z0
            * np.cross(self.directions(grid.points), electric)
        )

        return ElectromagneticField(
            ElectricField(grid, self.wavelength0, electric), magnetic
        )
