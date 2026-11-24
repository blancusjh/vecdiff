"""What is measured on a field: energy flow, polarization, momentum, images."""

from vecdiff.observables.diagnostics import boundary_residuals
from vecdiff.observables.electromagnetism import (
    coherent_sum,
    energy_density,
    irradiance,
    power,
    poynting,
)
from vecdiff.observables.imaging import detector_image
from vecdiff.observables.maxwell import ElectricFieldJet, plane_wave_jet
from vecdiff.observables.momentum import (
    canonical_momentum,
    force,
    helicity_density,
    orbital_angular_momentum,
    spin_density,
    spin_momentum,
    torque,
)
from vecdiff.observables.polarization import (
    ellipse_parameters,
    ensemble_stokes,
    jones_components,
    stokes,
    stokes_components,
)

__all__ = [
    "coherent_sum",
    "energy_density",
    "power",
    "poynting",
    "irradiance",
    "stokes",
    "stokes_components",
    "ensemble_stokes",
    "ellipse_parameters",
    "jones_components",
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
