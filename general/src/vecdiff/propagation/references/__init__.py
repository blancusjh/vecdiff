"""Independent full-wave comparisons, never selected by operational propagation."""

from vecdiff.propagation.references.bem import DielectricBEM
from vecdiff.propagation.references.boundary import boundary_meridian
from vecdiff.propagation.references.cylinder import InfiniteCylinder
from vecdiff.propagation.references.fem import DielectricFEM, TetrahedralMesh
from vecdiff.propagation.references.layered_sphere import LayeredSphere
from vecdiff.propagation.references.mesh import SurfaceMesh
from vecdiff.propagation.references.meshing import (
    surface_mesh,
    tetrahedral_mesh,
)
from vecdiff.propagation.references.mie import MieSphere

__all__ = [
    "DielectricBEM",
    "DielectricFEM",
    "SurfaceMesh",
    "TetrahedralMesh",
    "surface_mesh",
    "tetrahedral_mesh",
    "MieSphere",
    "LayeredSphere",
    "InfiniteCylinder",
    "boundary_meridian",
]
