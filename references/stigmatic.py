"""Preserved stigmatic comparison; imports only the external legacy package.

Historical 'exact' function names refer to that model's integral evaluation,
not proof of the global dielectric Maxwell boundary conditions.
"""
from .legacy.vecdiff.wave.stigmatic import exact_focal_cut, referee, oval_surface

__all__ = ["exact_focal_cut", "referee", "oval_surface"]
