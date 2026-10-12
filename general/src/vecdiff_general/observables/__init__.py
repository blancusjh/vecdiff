"""Public observables operations for the independent general library."""

from vecdiff_general.observables.electromagnetism import (
    coherent_sum,
    energy_density,
    power,
    poynting,
)
from vecdiff_general.observables.imaging import detector_image

__all__ = ["coherent_sum", "energy_density", "power", "poynting", "detector_image"]
