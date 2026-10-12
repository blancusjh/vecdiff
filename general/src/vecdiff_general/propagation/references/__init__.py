"""Independent full-wave comparisons, never selected by operational propagation."""

from vecdiff_general.propagation.references.bem import DielectricBEM
from vecdiff_general.propagation.references.boundary import boundary_meridian
from vecdiff_general.propagation.references.fem import DielectricFEM, TetrahedralMesh
from vecdiff_general.propagation.references.mesh import SurfaceMesh
from vecdiff_general.propagation.references.meshing import (
    surface_mesh,
    tetrahedral_mesh,
)
from vecdiff_general.propagation.references.mie import MieSphere

__all__ = [
    "DielectricBEM",
    "DielectricFEM",
    "SurfaceMesh",
    "TetrahedralMesh",
    "surface_mesh",
    "tetrahedral_mesh",
    "MieSphere",
    "boundary_meridian",
]
