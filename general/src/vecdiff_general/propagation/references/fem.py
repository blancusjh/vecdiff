"""Independent vector edge-element FEM with Cartesian absorbing PML."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from vecdiff_general.fields import ElectricField, ElectromagneticField
from vecdiff_general.fields.spectrum import PlaneWaveExpansion
from vecdiff_general.media.medium import Z0
from vecdiff_general.sampling import PointGrid, SurfaceGrid

_EDGES = np.array(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
_FACES = np.array(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)))
_BARY = np.full((4, 4), 0.1381966011250105)
np.fill_diagonal(_BARY, 0.5854101966249685)


@dataclass
class TetrahedralMesh:
    """Conforming first-order tetrahedra, explicit material labels and PML box.

    ``interior`` labels inclusion cells. ``halfwidth`` bounds the physical box
    about ``center``; a PML shell of ``pml_width`` surrounds that box. All lengths
    are metres. Material and PML boundaries must be conforming mesh faces.
    """

    vertices: np.ndarray
    tetrahedra: np.ndarray
    interior: np.ndarray
    center: np.ndarray
    halfwidth: np.ndarray
    pml_width: float

    def __post_init__(self):
        self.vertices = np.array(self.vertices, float, copy=True)
        connectivity = np.asarray(self.tetrahedra)
        if not np.issubdtype(connectivity.dtype, np.integer):
            raise ValueError("tetrahedra must contain integer vertex indices")
        self.tetrahedra = np.array(connectivity, np.int32, copy=True)
        self.interior = np.array(self.interior, bool, copy=True)
        self.center = np.array(self.center, float, copy=True)
        self.halfwidth = np.array(self.halfwidth, float, copy=True)
        if (
            self.vertices.ndim != 2
            or self.vertices.shape[1] != 3
            or not np.isfinite(self.vertices).all()
        ):
            raise ValueError("vertices must be finite (N,3) coordinates")
        if (
            self.tetrahedra.ndim != 2
            or self.tetrahedra.shape[1] != 4
            or not len(self.tetrahedra)
        ):
            raise ValueError("tetrahedra must have shape (M,4)")
        if self.tetrahedra.min() < 0 or self.tetrahedra.max() >= len(self.vertices):
            raise ValueError("tetrahedron vertex index out of range")
        if (
            self.interior.shape != (len(self.tetrahedra),)
            or not self.interior.any()
            or self.interior.all()
        ):
            raise ValueError(
                "provide inclusion and exterior cells with one label per cell"
            )
        if (
            self.center.shape != (3,)
            or self.halfwidth.shape != (3,)
            or not np.isfinite(self.center).all()
            or not np.isfinite(self.halfwidth).all()
            or np.any(self.halfwidth <= 0)
            or not np.isfinite(self.pml_width)
            or self.pml_width <= 0
        ):
            raise ValueError("invalid physical/PML box")
        v = self.vertices[self.tetrahedra]
        jac = (v[:, 1:] - v[:, :1]).transpose(0, 2, 1)
        det = np.linalg.det(jac)
        if np.any(
            abs(det) < np.finfo(float).eps * np.linalg.norm(jac, axis=(1, 2)) ** 3
        ):
            raise ValueError("degenerate tetrahedron")
        self.volumes = abs(det) / 6
        inverse = np.linalg.inv(jac)
        self.gradients = np.concatenate(
            (-inverse.sum(axis=1)[:, None, :], inverse), axis=1
        )
        edge_nodes = self.tetrahedra[:, _EDGES]
        self.signs = np.where(edge_nodes[:, :, 0] < edge_nodes[:, :, 1], 1, -1).astype(
            np.int32
        )
        self.edges, edge_ids = np.unique(
            np.sort(edge_nodes, axis=2).reshape(-1, 2), axis=0, return_inverse=True
        )
        self.edge_ids = edge_ids.reshape(-1, 6)
        self._tree = cKDTree(v.mean(axis=1))
        self._lower, self._upper = v.min(axis=1), v.max(axis=1)
        face_nodes = self.tetrahedra[:, _FACES].reshape(-1, 3)
        unique, first, inverse, counts = np.unique(
            np.sort(face_nodes, axis=1),
            axis=0,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        if np.any(counts > 2):
            raise ValueError("nonmanifold tetrahedral faces")
        self.outer_faces = unique[counts == 1]
        outer = self.halfwidth + self.pml_width
        xyz = self.vertices[self.outer_faces] - self.center
        atwall = np.any(
            np.all(
                np.isclose(abs(xyz), outer, rtol=1e-8, atol=1e-10 * outer.max()), axis=1
            )
            & (np.all(xyz >= 0, axis=1) | np.all(xyz <= 0, axis=1)),
            axis=1,
        )
        if not atwall.all():
            raise ValueError("external tetrahedron faces must lie on the outer PML box")
        if np.any(abs(v[self.interior] - self.center) > self.halfwidth * (1 + 1e-8)):
            raise ValueError("dielectric inclusion must lie inside the physical box")
        label_sum = np.bincount(
            inverse, weights=np.repeat(self.interior, 4), minlength=len(unique)
        )
        interface = (counts == 2) & (label_sum == 1)
        self.interface_faces = unique[interface]
        # Outward normals are oriented using the incident inclusion tetrahedron.
        owners = np.repeat(np.arange(len(v)), 4)
        interior_owner = np.full(len(unique), -1, int)
        selected = np.repeat(self.interior, 4)
        interior_owner[inverse[selected]] = owners[selected]
        self._interface_owners = interior_owner[interface]

    def boundary_grid(self):
        v = self.vertices[self.interface_faces]
        normals = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        area = np.linalg.norm(normals, axis=1) / 2
        centers = v.mean(axis=1)
        owners = self.vertices[self.tetrahedra[self._interface_owners]].mean(axis=1)
        inward = np.einsum("ij,ij->i", normals, centers - owners) < 0
        normals[inward] *= -1
        return SurfaceGrid(centers, normals / (2 * area[:, None]), area)

    def locate(self, points, side=None):
        points = np.asarray(points).reshape(-1, 3)
        if np.any(abs(points - self.center) > self.halfwidth * (1 + 1e-10)):
            raise ValueError(
                "FEM evaluation is restricted to the physical box, excluding PML"
            )
        candidates = self._tree.query(points, k=min(40, len(self.tetrahedra)))[1]
        if candidates.ndim == 1:
            candidates = candidates[:, None]
        cells = np.empty(len(points), np.int32)
        bary = np.empty((len(points), 4))
        origin = self.vertices[self.tetrahedra[:, 0]]
        for i, p in enumerate(points):
            for trial in (candidates[i], None):
                if trial is None:
                    tol = 1e-10 * np.max(self.halfwidth)
                    trial = np.flatnonzero(
                        np.all(
                            (p >= self._lower - tol) & (p <= self._upper + tol), axis=1
                        )
                    )
                b = np.einsum("tac,tc->ta", self.gradients[trial], p - origin[trial])
                b[:, 0] += 1
                valid = np.all(b >= -1e-9, axis=1) & np.all(b <= 1 + 1e-9, axis=1)
                if side is not None:
                    valid &= self.interior[trial] == (side == "interior")
                hits = np.flatnonzero(valid)
                if len(hits):
                    if side is None and len(np.unique(self.interior[trial[hits]])) > 1:
                        raise ValueError("specify side for dielectric interface traces")
                    cells[i], bary[i] = trial[hits[0]], b[hits[0]]
                    break
            else:
                raise ValueError(
                    "evaluation point has no tetrahedron in the requested region"
                )
        return cells, bary


def _fingerprint(mesh, source, interior, pml_strength):
    """Hash of everything that determines the solved edge coefficients."""
    digest = hashlib.sha256()
    for array in (
        mesh.vertices,
        mesh.tetrahedra,
        mesh.interior,
        mesh.center,
        source.wavevectors,
        source.amplitudes,
        source.origin,
    ):
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    digest.update(
        repr(
            (
                source.wavelength0,
                source.region.material.n,
                interior.material.n,
                float(mesh.halfwidth)
                if np.ndim(mesh.halfwidth) == 0
                else tuple(np.ravel(mesh.halfwidth)),
                float(mesh.pml_width),
                pml_strength,
            )
        ).encode()
    )
    native = Path(__file__).resolve().parents[4] / "native"
    for path in (Path(__file__), native / "fem.f90", native / "reference_math.f90"):
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


class DielectricFEM:
    """First-family Nédélec FEM with quadratic Cartesian complex stretching.

    Solves a global scattered-field curl-curl problem. ``algebraic_residual``
    measures the weak discrete equations, not continuum accuracy. The lowest
    order H is cellwise constant: pointwise curl-H/Helmholtz finite differences
    are not a meaningful convergence test. Use field, flux, mesh and PML studies.

    cache: a directory (or the VECDIFF_FEM_CACHE environment variable) where
    solved coefficients are stored under a hash of the mesh, source, indices,
    PML settings and solver sources, and reloaded instead of solving again.
    """

    def __init__(
        self,
        source,
        mesh,
        interior,
        *,
        pml_strength=3.0,
        max_unknowns=40000,
        cache=None,
    ):
        from vecdiff_general.propagation.references import _reference_core

        if not isinstance(source, PlaneWaveExpansion) or not isinstance(
            mesh, TetrahedralMesh
        ):
            raise TypeError("FEM requires PlaneWaveExpansion and TetrahedralMesh")
        if not np.isfinite(pml_strength) or pml_strength <= 0:
            raise ValueError("pml_strength must be positive")
        self.source, self.mesh, self.interior = source, mesh, interior
        self.exterior, self.wavelength0 = source.region, source.wavelength0
        self.unknowns = len(mesh.edges)
        if self.unknowns > max_unknowns:
            raise ValueError(
                f"FEM needs {self.unknowns} edge unknowns (limit {max_unknowns})"
            )
        scale = self.wavelength0
        self._gradients = np.asfortranarray((mesh.gradients * scale).transpose(2, 1, 0))
        self._signs = np.asfortranarray(mesh.signs.T)
        cache = cache or os.environ.get("VECDIFF_FEM_CACHE")
        stored = None
        if cache:
            stored = Path(cache) / (
                "fem_" + _fingerprint(mesh, source, interior, pml_strength) + ".npz"
            )
            if stored.exists():
                with np.load(stored) as data:
                    self.coefficients = data["coefficients"]
                    self.free = data["free"]
                    self.algebraic_residual = float(data["algebraic_residual"])
                    self.assembly_seconds = float(data["assembly_seconds"])
                    self.solve_seconds = float(data["solve_seconds"])
                self._local = np.asfortranarray(self.coefficients[mesh.edge_ids].T)
                self.loaded_from = stored
                return
        self.loaded_from = None
        start = perf_counter()
        corners = mesh.vertices[mesh.tetrahedra] / scale
        points = np.einsum("qa,tac->tqc", _BARY, corners) * scale
        incident = source.evaluate(PointGrid(points)).electric.values
        indices = np.where(mesh.interior, interior.material.n, self.exterior.material.n)
        matrices, rhs = _reference_core.fem_assemble(
            corners.transpose(2, 1, 0),
            self._gradients,
            mesh.volumes / scale**3,
            self._signs,
            indices,
            self.exterior.material.n,
            incident.transpose(2, 1, 0),
            mesh.center / scale,
            mesh.halfwidth / scale,
            mesh.pml_width / scale,
            pml_strength,
        )
        ids = mesh.edge_ids
        rows = np.broadcast_to(ids[:, :, None], (len(ids), 6, 6)).ravel()
        columns = np.broadcast_to(ids[:, None, :], (len(ids), 6, 6)).ravel()
        matrix = coo_matrix(
            (matrices.transpose(2, 0, 1).ravel(), (rows, columns)),
            shape=(self.unknowns, self.unknowns),
        ).tocsc()
        vector = np.zeros(self.unknowns, complex)
        np.add.at(vector, ids.ravel(), rhs.T.ravel())
        boundary_edges = np.unique(
            np.sort(mesh.outer_faces[:, [[0, 1], [0, 2], [1, 2]]], axis=-1).reshape(
                -1, 2
            ),
            axis=0,
        )
        boundary_set = {tuple(e) for e in boundary_edges}
        self.free = np.array([tuple(e) not in boundary_set for e in mesh.edges])
        matrix = matrix[self.free][:, self.free]
        vector = vector[self.free]
        self.assembly_seconds = perf_counter() - start
        start = perf_counter()
        self.coefficients = np.zeros(self.unknowns, complex)
        self.coefficients[self.free] = spsolve(matrix, vector)
        self.solve_seconds = perf_counter() - start
        if not np.isfinite(self.coefficients).all():
            raise RuntimeError("FEM solve returned nonfinite coefficients")
        self.algebraic_residual = float(
            np.linalg.norm(matrix @ self.coefficients[self.free] - vector)
            / max(np.linalg.norm(vector), np.finfo(float).tiny)
        )
        self._local = np.asfortranarray(self.coefficients[ids].T)
        if stored is not None:
            stored.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                stored,
                coefficients=self.coefficients,
                free=self.free,
                algebraic_residual=self.algebraic_residual,
                assembly_seconds=self.assembly_seconds,
                solve_seconds=self.solve_seconds,
            )

    def evaluate(self, grid, *, side=None):
        from vecdiff_general.propagation.references import _reference_core

        if side not in (None, "interior", "exterior"):
            raise ValueError("side must be interior, exterior, or None")
        cells, bary = self.mesh.locate(grid.points, side)
        e, h = _reference_core.fem_fields(
            self._gradients, self._signs, self._local, cells + 1, bary.T
        )
        incident = self.source.evaluate(grid)
        shape = grid.shape + (3,)
        return ElectromagneticField(
            ElectricField(
                grid, self.wavelength0, e.T.reshape(shape) + incident.electric.values
            ),
            h.T.reshape(shape) / Z0 + incident.magnetic,
        )
