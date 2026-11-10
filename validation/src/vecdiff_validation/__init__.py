"""Independent full-wave solvers and comparisons that validate vecdiff.

Boundary elements, finite elements, the method of fundamental solutions,
prescribed-current comparisons and the benchmarks built on them. Not part of
the vecdiff library: installed separately, with a Fortran extension.
"""

from vecdiff_validation.bem import DielectricBEM
from vecdiff_validation.boundary import boundary_meridian
from vecdiff_validation.fem import DielectricFEM, TetrahedralMesh
from vecdiff_validation.mesh import SurfaceMesh
from vecdiff_validation.meshing import surface_mesh, tetrahedral_mesh

__all__ = [
    "DielectricBEM",
    "DielectricFEM",
    "SurfaceMesh",
    "TetrahedralMesh",
    "surface_mesh",
    "tetrahedral_mesh",
    "boundary_meridian",
]
