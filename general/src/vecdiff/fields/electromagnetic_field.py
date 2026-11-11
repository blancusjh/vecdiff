"""Electric and electromagnetic fields on explicit physical grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vecdiff._arrays import frozen_array, positive
from vecdiff._backend import bd, is_device_array
from vecdiff.sampling.grids import CartesianGrid, Grid, PolarGrid


@dataclass(frozen=True, eq=False)
class ElectricField:
    """Global Cartesian electric phasor in V/m; exp(-i omega t).

    ``valid`` marks computed samples. Uncomputed values are NaN, never zeros.
    Nonfinite values at valid samples are rejected.
    """

    grid: Grid
    wavelength0: float
    values: np.ndarray
    valid: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.grid, Grid):
            raise TypeError("field requires a physical Grid")

        valid = (
            bd.ones(self.grid.shape, dtype=bool)
            if self.valid is None
            else bd.asarray(self.valid, dtype=bool)
        )
        if valid.shape != self.grid.shape:
            raise ValueError("field validity mask must have the grid shape")
        values = bd.array(self.values, dtype=complex, copy=True)

        if values.shape != self.grid.shape + (3,):
            raise ValueError("electric values must have grid.shape + (3,)")

        if not bd.isfinite(values[valid]).all():
            raise ValueError("computed electric samples must be finite")
        values[~valid] = bd.nan + 1j * bd.nan
        if not is_device_array(values):
            values.setflags(write=False)
        object.__setattr__(self, "valid", frozen_array(valid, dtype=bool))
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self, "wavelength0", positive(self.wavelength0, "vacuum wavelength")
        )

    @classmethod
    def _computed(cls, grid, wavelength0, values):
        """Every sample computed by the library: values are kept, not copied.

        The caller owns values (grid.shape + (3,) complex) and a validated
        wavelength; only finiteness is checked, as the constructor would.
        """
        if not bd.isfinite(values).all():
            raise ValueError("computed electric samples must be finite")
        if not is_device_array(values):
            values.setflags(write=False)
        field = object.__new__(cls)
        object.__setattr__(field, "grid", grid)
        object.__setattr__(field, "wavelength0", wavelength0)
        object.__setattr__(field, "values", values)
        valid = frozen_array(bd.ones(grid.shape, dtype=bool), dtype=bool)
        object.__setattr__(field, "valid", valid)
        return field

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
        """|E|² in V²/m²: the electric energy density's shape, not the intensity.

        The power a detector receives is ElectromagneticField.irradiance().
        """
        return bd.sum(bd.abs(self.values) ** 2, axis=-1)


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
    """Consistent E and H evaluated on the same grid. H is in A/m, not Z0 H.

    The samples E marks invalid are uncomputed for H too, and NaN.
    """

    electric: ElectricField
    magnetic: np.ndarray

    def __post_init__(self):
        if not isinstance(self.electric, ElectricField):
            raise TypeError("electric must be an ElectricField")

        magnetic = bd.array(self.magnetic, dtype=complex, copy=True)

        if magnetic.shape != self.electric.values.shape:
            raise ValueError("magnetic and electric values must have equal shapes")

        valid = self.electric.valid
        if not bd.isfinite(magnetic[valid]).all():
            raise ValueError("computed magnetic samples must be finite")
        magnetic[~valid] = bd.nan + 1j * bd.nan
        if not is_device_array(magnetic):
            magnetic.setflags(write=False)
        object.__setattr__(self, "magnetic", magnetic)

    @classmethod
    def _computed(cls, electric, magnetic):
        """E from ElectricField._computed and H, or a function forming H.

        A function is called once, when H is first read, so a caller that
        reads only E never forms it; until then it holds what the function
        refers to. H is checked to be finite as the constructor would.
        """
        if callable(magnetic):
            field = object.__new__(_DeferredMagneticField)
            object.__setattr__(field, "_form_magnetic", magnetic)
        else:
            field = object.__new__(cls)
            object.__setattr__(field, "magnetic", _computed_magnetic(magnetic))
        object.__setattr__(field, "electric", electric)
        return field

    @property
    def grid(self):
        return self.electric.grid

    @property
    def valid(self):
        """The computed samples."""
        return self.electric.valid

    def poynting(self):
        """Time-averaged energy-flow density in W/m²."""
        from vecdiff.observables.electromagnetism import poynting

        return poynting(self)

    def irradiance(self, normal=None):
        """Time-averaged power per unit area through the grid's surface, in W/m²."""
        from vecdiff.observables.electromagnetism import irradiance

        return irradiance(self, normal)

    def power(self):
        """Signed time-averaged power in watts through the oriented grid."""
        from vecdiff.observables.electromagnetism import power

        return power(self)


def _computed_magnetic(values):
    if not bd.isfinite(values).all():
        raise ValueError("computed magnetic samples must be finite")
    if not is_device_array(values):
        values.setflags(write=False)
    return values


class _DeferredMagneticField(ElectromagneticField):
    """ElectromagneticField whose H is formed when it is first read."""

    def __getattr__(self, name):
        if name != "magnetic":
            raise AttributeError(name)
        state = object.__getattribute__(self, "__dict__")
        if "_form_magnetic" not in state:
            raise AttributeError(name)
        magnetic = _computed_magnetic(state["_form_magnetic"]())
        object.__setattr__(self, "magnetic", magnetic)
        del state["_form_magnetic"]
        return magnetic

    def __reduce__(self):
        return ElectromagneticField, (self.electric, self.magnetic)
