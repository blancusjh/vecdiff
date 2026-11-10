"""Exact solutions of Maxwell's equations to check the transport against.

One homogeneous or layered body under any propagating incident field, with
its Debye series: the total field, or the waves that met a given number of
reflections. They are never used by the transport itself.
"""

from vecdiff.references.cylinder import InfiniteCylinder
from vecdiff.references.layered_sphere import LayeredSphere
from vecdiff.references.mie import MieSphere

__all__ = ["MieSphere", "LayeredSphere", "InfiniteCylinder"]
