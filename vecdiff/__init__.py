"""Electric-field spectral interface transformations and Maxwell propagation.

The public API is the only Python module at this package's root. Literature
solvers live in the repository's external references/ package, never here.
"""
from .fields.electric_field import ElectricField, TransverseElectricField
from .fields.electric_spectrum import ElectricSpectrum, plane_wave
from .media.medium import Medium
from .geometry.domains import PlaneDomain
from .geometry.frames import Frame
from .sampling.grids import CartesianGrid, PointSampling
from .sampling.surface_sampling import SurfaceSampling, sample_surface
from .surfaces.surface import Surface, Plane, FreeformSurface
from .surfaces.axisymmetric import AxisymmetricSurface, Sphere, SphericalCap
from .surfaces.cartesian_oval import CartesianOval
from .interfaces.dielectric_interface import DielectricInterface
from .propagation.propagation import propagate, spectrum_of
from .propagation.interface_transform import interface_transform

__version__ = "0.3.0"
__all__ = ["ElectricField", "TransverseElectricField", "ElectricSpectrum", "plane_wave", "Medium",
           "PlaneDomain", "Frame", "CartesianGrid", "PointSampling", "SurfaceSampling", "sample_surface",
           "Surface", "Plane", "FreeformSurface", "AxisymmetricSurface", "Sphere", "SphericalCap", "CartesianOval",
           "DielectricInterface", "propagate", "spectrum_of", "interface_transform"]
