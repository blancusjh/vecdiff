"""Optional Gmsh mesh generation, independent of the numerical solvers.

Gmsh supplies geometry/meshes only. BEM/FEM assembly and field reconstruction
are our Fortran routines; neither SCUFF-EM nor FEniCS is imported.
"""

from contextlib import contextmanager
from uuid import uuid4

import numpy as np

from vecdiff_general.propagation.references.mesh import SurfaceMesh
from vecdiff_general.surfaces import Plane, SphericalCap
from vecdiff_general.volumes import SphereVolume, VolumeBetweenSurfaces


@contextmanager
def _model():
    try:
        import gmsh
    except ImportError as exc:
        raise ImportError(
            "mesh generation requires vecdiff-general[reference-mesh]; supplied meshes do not"
        ) from exc
    owned = not gmsh.isInitialized()
    if owned:
        gmsh.initialize()
    previous = gmsh.model.getCurrent()
    options = {
        k: gmsh.option.getNumber(k)
        for k in (
            "General.Terminal",
            "Mesh.MeshSizeMin",
            "Mesh.MeshSizeMax",
            "Mesh.MeshSizeFromCurvature",
        )
    }
    name = "logos-" + uuid4().hex
    gmsh.model.add(name)
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        yield gmsh
    finally:
        gmsh.model.setCurrent(name)
        gmsh.model.remove()
        for k, v in options.items():
            gmsh.option.setNumber(k, v)
        if owned:
            gmsh.finalize()
        elif previous:
            gmsh.model.setCurrent(previous)


def _body(gmsh, volume, scale):
    occ = gmsh.model.occ
    if isinstance(volume, SphereVolume):
        return [(3, occ.addSphere(*(volume.center / scale), volume.radius / scale))]
    if not isinstance(volume, VolumeBetweenSurfaces):
        raise TypeError(
            "automatic meshing supports spheres and coaxial spherical/planar lenses"
        )
    front, back = volume.front, volume.back
    if not isinstance(front, (Plane, SphericalCap)) or not isinstance(
        back, (Plane, SphericalCap)
    ):
        raise TypeError("provide a mesh for nonspherical lenses")
    for surface in (front, back):
        if not np.allclose(surface.frame.basis, np.eye(3)) or not np.allclose(
            surface.frame.origin[:2] / scale, 0
        ):
            raise ValueError("automatic lens meshing requires coaxial +z frames")
    z0, z1 = front.frame.origin[2] / scale, back.frame.origin[2] / scale
    radius = volume.rim_radius / scale
    if z1 <= z0:
        raise ValueError("lens back vertex must follow the front vertex")
    entities = [(3, occ.addCylinder(0, 0, z0, 0, 0, z1 - z0, radius))]
    for surface, z, sign in ((front, z0, 1), (back, z1, -1)):
        if isinstance(surface, SphericalCap):
            curvature = surface.radius / scale
            if sign * curvature <= 0:
                raise ValueError(
                    "automatic meshing currently supports convex lenses only"
                )
            ball = [(3, occ.addSphere(0, 0, z + curvature, abs(curvature)))]
            entities, _ = occ.intersect(entities, ball)
    return entities


def _nodes(gmsh, scale):
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(tags)
    return tags[order], coordinates.reshape(-1, 3)[order] * scale


def _elements(gmsh, dimension, entity, tags):
    kinds, _, nodes = gmsh.model.mesh.getElements(dimension, entity)
    expected, width = (2, 3) if dimension == 2 else (4, 4)
    chunks = [
        np.searchsorted(tags, n).reshape(-1, width)
        for k, n in zip(kinds, nodes)
        if k == expected
    ]
    if not chunks:
        raise ValueError("mesher produced no first-order elements")
    return np.concatenate(chunks)


def surface_mesh(volume, edge_size):
    """Closed triangular mesh; ``edge_size`` in metres, not an error tolerance."""
    if not np.isfinite(edge_size) or edge_size <= 0:
        raise ValueError("edge_size must be positive")
    with _model() as gmsh:
        _body(gmsh, volume, edge_size)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1)
        gmsh.model.mesh.generate(2)
        tags, vertices = _nodes(gmsh, edge_size)
        triangles = _elements(gmsh, 2, -1, tags)
    # OCC supports here are convex; normals point away from the interior center.
    corners = vertices[triangles]
    center = vertices.mean(axis=0)
    flip = (
        np.einsum(
            "ij,ij->i",
            corners.mean(axis=1) - center,
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        )
        < 0
    )
    triangles[flip] = triangles[flip][:, [0, 2, 1]]
    return SurfaceMesh(vertices, triangles)


def tetrahedral_mesh(volume, *, edge_size, physical_padding, pml_width):
    """Conforming inclusion/air/PML tetrahedra for vector FEM; all lengths in m."""
    from vecdiff_general.propagation.references.fem import TetrahedralMesh

    if any(
        not np.isfinite(x) or x <= 0 for x in (edge_size, physical_padding, pml_width)
    ):
        raise ValueError("edge_size, physical_padding and pml_width must be positive")
    scale = edge_size
    with _model() as gmsh:
        body = _body(gmsh, volume, scale)
        gmsh.model.occ.synchronize()
        bounds = np.array(gmsh.model.getBoundingBox(*body[0]))
        center = (bounds[:3] + bounds[3:]) / 2
        half = (bounds[3:] - bounds[:3]) / 2 + physical_padding / scale
        outer = half + pml_width / scale
        occ = gmsh.model.occ
        physical = [(3, occ.addBox(*(center - half), *(2 * half)))]
        box = [(3, occ.addBox(*(center - outer), *(2 * outer)))]
        _, maps = occ.fragment(box, physical + body)
        body_entities = {tag for dim, tag in maps[-1] if dim == 3}
        occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1)
        gmsh.model.mesh.generate(3)
        tags, vertices = _nodes(gmsh, scale)
        tetrahedra, interior = [], []
        for _, tag in gmsh.model.getEntities(3):
            cells = _elements(gmsh, 3, tag, tags)
            tetrahedra.append(cells)
            interior.extend([tag in body_entities] * len(cells))
    return TetrahedralMesh(
        vertices,
        np.concatenate(tetrahedra),
        np.array(interior),
        center * scale,
        half * scale,
        pml_width,
    )
