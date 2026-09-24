"""Public observables operations for the independent general library."""

from .electromagnetism import coherent_sum, energy_density, power, poynting
from .imaging import detector_image

__all__ = ["coherent_sum", "energy_density", "power", "poynting", "detector_image"]
