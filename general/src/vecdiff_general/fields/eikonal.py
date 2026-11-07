"""An explicit high-frequency phase and transverse envelope, with material context."""

from collections.abc import Callable
from dataclasses import dataclass

from vecdiff_general._arrays import positive
from vecdiff_general._backend import bd
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
        gradient = bd.broadcast_to(
            bd.asarray(self.path_gradient(points), dtype=float), bd.shape(points)
        )

        if not bd.isfinite(gradient).all() or not bd.allclose(
            bd.linalg.norm(gradient, axis=-1),
            self.region.material.n,
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError(
                "optical-path gradient violates the eikonal equation |grad L| = n"
            )

        return gradient / self.region.material.n

    def amplitudes(self, points):
        electric = bd.broadcast_to(
            bd.asarray(self.envelope(points), dtype=complex), bd.shape(points)
        )
        direction = self.directions(points)

        if not bd.isfinite(electric).all() or bd.any(
            bd.abs(bd.sum(direction * electric, axis=-1))
            > 1e-10 * bd.linalg.norm(electric, axis=-1)
        ):
            raise ValueError("eikonal envelope must be finite and locally transverse")

        return electric

    def evaluate(self, grid):
        path = bd.broadcast_to(
            bd.asarray(self.optical_path(grid.points), dtype=float), grid.shape
        )

        if not bd.isfinite(path).all():
            raise ValueError("optical path must be finite")

        electric = (
            self.amplitudes(grid.points)
            * bd.exp(2j * bd.pi * path / self.wavelength0)[..., None]
        )
        magnetic = (
            self.region.material.n
            / Z0
            * bd.cross(self.directions(grid.points), electric)
        )

        return ElectromagneticField(
            ElectricField(grid, self.wavelength0, electric), magnetic
        )
