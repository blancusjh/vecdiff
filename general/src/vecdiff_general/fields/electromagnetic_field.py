"""Electric and electromagnetic fields on explicit physical grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._arrays import frozen_array, positive
from ..sampling.grids import CartesianGrid, Grid, PolarGrid


@dataclass(frozen=True, eq=False)
class ElectricField:
    """Complete global Cartesian electric phasor in V/m; exp(-i omega t)."""

    grid: Grid
    wavelength0: float
    values: np.ndarray

    def __post_init__(self):
        if not isinstance(self.grid, Grid):
            raise TypeError("field requires a physical Grid")

        values = frozen_array(self.values, dtype=complex)

        if values.shape != self.grid.shape + (3,):
            raise ValueError("electric values must have grid.shape + (3,)")

        object.__setattr__(self, "values", values)
        object.__setattr__(
            self, "wavelength0", positive(self.wavelength0, "vacuum wavelength")
        )

    @property
    def Ex(self):
        return self.values[..., 0]

    @property
    def Ey(self):
        return self.values[..., 1]

    @property
    def Ez(self):
        return self.values[..., 2]

    def norm2(self):
        return np.sum(np.abs(self.values) ** 2, axis=-1)


@dataclass(frozen=True, eq=False)
class TransverseElectricField:
    """Two prescribed E components along an embedded plane's local u and v axes.

    A dispersion relation and branch are required to complete this input.
    It does not mean TE polarization and does not prescribe normal E=0.
    """

    grid: CartesianGrid | PolarGrid
    wavelength0: float
    values: np.ndarray

    def __post_init__(self):
        if not isinstance(self.grid, (CartesianGrid, PolarGrid)):
            raise TypeError(
                "transverse input requires a CartesianGrid or PolarGrid on a plane"
            )

        values = frozen_array(self.values, dtype=complex)

        if values.shape != self.grid.shape + (2,):
            raise ValueError("transverse values must have grid.shape + (2,)")

        object.__setattr__(self, "values", values)
        object.__setattr__(
            self, "wavelength0", positive(self.wavelength0, "vacuum wavelength")
        )


@dataclass(frozen=True, eq=False)
class ElectromagneticField:
    """Consistent E and H evaluated on the same grid. H is in A/m, not Z0 H."""

    electric: ElectricField
    magnetic: np.ndarray

    def __post_init__(self):
        if not isinstance(self.electric, ElectricField):
            raise TypeError("electric must be an ElectricField")

        magnetic = frozen_array(self.magnetic, dtype=complex)

        if magnetic.shape != self.electric.values.shape:
            raise ValueError("magnetic and electric values must have equal shapes")

        object.__setattr__(self, "magnetic", magnetic)

    @property
    def grid(self):
        return self.electric.grid

    def poynting(self):
        """Time-averaged energy-flow density in W/m²."""
        from ..observables.electromagnetism import poynting

        return poynting(self)

    def power(self):
        """Signed time-averaged power in watts through the oriented grid."""
        from ..observables.electromagnetism import power

        return power(self)
