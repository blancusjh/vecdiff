"""Public observables operations for the independent general library."""

from vecdiff_general.observables.diagnostics import boundary_residuals
from vecdiff_general.observables.electromagnetism import (
    coherent_sum,
    energy_density,
    power,
    poynting,
)
from vecdiff_general.observables.imaging import detector_image
from vecdiff_general.observables.maxwell import ElectricFieldJet, plane_wave_jet
from vecdiff_general.observables.momentum import (
    canonical_momentum,
    force,
    helicity_density,
    orbital_angular_momentum,
    spin_density,
    spin_momentum,
    torque,
)

__all__ = [
    "coherent_sum",
    "energy_density",
    "power",
    "poynting",
    "spin_density",
    "helicity_density",
    "canonical_momentum",
    "spin_momentum",
    "orbital_angular_momentum",
    "force",
    "torque",
    "ElectricFieldJet",
    "plane_wave_jet",
    "detector_image",
    "boundary_residuals",
]
