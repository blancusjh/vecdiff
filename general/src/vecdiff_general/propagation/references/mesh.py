"""Explicit meshes in metres; no solver or incident field hidden in geometry."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from ...sampling import SurfaceGrid


def gauss_rule(order):
    if not isinstance(order, (int, np.integer)) or order < 2:
        raise ValueError("quadrature order must be an integer >= 2")
    q, w = np.polynomial.legendre.leggauss(order)
    return (q + 1) / 2, w / 2


@dataclass
class SurfaceMesh:
    """Closed, consistently outward oriented triangular dielectric boundary.

    Coordinates have SI units. Linear triangles approximate curved boundaries;
    refining quadrature alone does not remove this geometric error.
    """

    vertices: np.ndarray
    triangles: np.ndarray

    def __post_init__(self):
        self.vertices = np.array(self.vertices, dtype=float, copy=True)
        original = np.asarray(self.triangles)
        if not np.issubdtype(original.dtype, np.integer):
            raise ValueError("triangle connectivity must contain integer indices")
        self.triangles = np.array(original, dtype=np.int32, copy=True)
        if (
            self.vertices.ndim != 2
            or self.vertices.shape[1] != 3
            or not np.isfinite(self.vertices).all()
        ):
            raise ValueError("vertices must be finite (N, 3) coordinates")
        if (
            self.triangles.ndim != 2
            or self.triangles.shape[1] != 3
            or len(self.triangles) < 4
        ):
            raise ValueError("triangles must have shape (M, 3)")
        if self.triangles.min() < 0 or self.triangles.max() >= len(self.vertices):
            raise ValueError("triangle vertex index out of range")
        v = self.corners
        cross = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        self.areas = np.linalg.norm(cross, axis=1) / 2
        if np.any(self.areas <= 0):
            raise ValueError("degenerate triangle")
        self.normals = cross / (2 * self.areas[:, None])
        center = self.vertices.mean(axis=0)
        signed_volume = np.einsum("ij,ij->i", v[:, 0] - center, cross).sum() / 6
        if signed_volume <= 0:
            raise ValueError("mesh must be oriented outward")
        edges = {}
        self.edge_ids = np.empty_like(self.triangles)
        self.factors = np.empty(self.triangles.shape)
        for t, triangle in enumerate(self.triangles):
            for a in range(3):
                u, w = triangle[(a + 1) % 3], triangle[(a + 2) % 3]
                key = tuple(sorted((u, w)))
                sign = 1 if u < w else -1
                if key not in edges:
                    edges[key] = (len(edges), [])
                identifier, incidences = edges[key]
                incidences.append(sign)
                self.edge_ids[t, a] = identifier
                self.factors[t, a] = (
                    sign
                    * np.linalg.norm(self.vertices[u] - self.vertices[w])
                    / (2 * self.areas[t])
                )
        if any(sorted(incidences) != [-1, 1] for _, incidences in edges.values()):
            raise ValueError(
                "each mesh edge needs exactly two oppositely oriented faces"
            )
        self.edges = np.array(list(edges), dtype=np.int32)
        for array in (
            self.vertices,
            self.triangles,
            self.areas,
            self.normals,
            self.edge_ids,
            self.factors,
            self.edges,
        ):
            array.setflags(write=False)

    @property
    def corners(self):
        return self.vertices[self.triangles]

    def quadrature(self, order=4):
        q, w = gauss_rule(order)
        u, v = np.meshgrid(q, q, indexing="ij")
        weights = (w[:, None] * w[None, :] * (1 - u)).ravel()
        bary = np.stack(((1 - u) * (1 - v), u, (1 - u) * v), -1).reshape(-1, 3)
        points = np.einsum("qa,tac->tqc", bary, self.corners)
        return points, 2 * self.areas[:, None] * weights

    def boundary_grid(self):
        """Face-centre traces, away from edges where traces are not unique."""
        return SurfaceGrid(self.corners.mean(axis=1), self.normals, self.areas)

    def locate(self, points):
        """Return inside flags and containing face (-1 for off-boundary points).

        Solid-angle winding works for nonconvex closed meshes as well. Exact
        edge/vertex traces are rejected: a unique face normal is required.
        """
        points = np.asarray(points).reshape(-1, 3)
        inside = np.empty(len(points), bool)
        faces = np.full(len(points), -1, dtype=np.int32)
        scale = np.linalg.norm(np.ptp(self.vertices, axis=0))
        tolerance = 2e-11 * scale
        v = self.corners
        a, b = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
        aa, ab, bb = (np.einsum("ij,ij->i", x, y) for x, y in ((a, a), (a, b), (b, b)))
        determinant = aa * bb - ab * ab
        for j, p in enumerate(points):
            d = p - v[:, 0]
            da, db = np.einsum("ij,ij->i", d, a), np.einsum("ij,ij->i", d, b)
            u, w = (bb * da - ab * db) / determinant, (aa * db - ab * da) / determinant
            on = (
                (abs(np.einsum("ij,ij->i", d, self.normals)) < tolerance)
                & (u >= -1e-10)
                & (w >= -1e-10)
                & (u + w <= 1 + 1e-10)
            )
            hits = np.flatnonzero(on)
            if len(hits):
                if (
                    len(hits) != 1
                    or min(u[hits[0]], w[hits[0]], 1 - u[hits[0]] - w[hits[0]]) < 1e-9
                ):
                    raise ValueError(
                        "evaluate boundary traces within faces, not on mesh edges/vertices"
                    )
                faces[j] = hits[0]
                inside[j] = False
                continue
            r = v - p
            length = np.linalg.norm(r, axis=-1)
            numerator = np.einsum("ij,ij->i", r[:, 0], np.cross(r[:, 1], r[:, 2]))
            denominator = length.prod(axis=1)
            for k in range(3):
                denominator += (
                    np.einsum("ij,ij->i", r[:, k], r[:, (k + 1) % 3])
                    * length[:, (k + 2) % 3]
                )
            inside[j] = abs(np.arctan2(numerator, denominator).sum()) > np.pi
        return inside, faces

    @classmethod
    def sphere(cls, volume, subdivisions=1):
        """Icosahedral sphere mesh with 20*4**subdivisions flat triangles."""
        if not isinstance(subdivisions, int) or not 0 <= subdivisions <= 6:
            raise ValueError("sphere subdivisions must be in [0, 6]")
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array(
            [(0, a, b * phi) for a in (-1, 1) for b in (-1, 1)]
            + [(a, b * phi, 0) for a in (-1, 1) for b in (-1, 1)]
            + [(b * phi, 0, a) for a in (-1, 1) for b in (-1, 1)],
            float,
        )
        vertices /= np.linalg.norm(vertices, axis=1)[:, None]
        triangles = ConvexHull(vertices).simplices.copy()
        for _ in range(subdivisions):
            new = list(vertices)
            mids = {}

            def midpoint(a, b):
                key = tuple(sorted((a, b)))
                if key not in mids:
                    p = vertices[a] + vertices[b]
                    mids[key] = len(new)
                    new.append(p / np.linalg.norm(p))
                return mids[key]

            faces = []
            for a, b, c in triangles:
                ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
                faces.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
            vertices, triangles = np.array(new), np.array(faces)
        v = vertices[triangles]
        inward = (
            np.einsum(
                "ij,ij->i", v[:, 0], np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
            )
            < 0
        )
        triangles[inward] = triangles[inward][:, [0, 2, 1]]
        return cls(volume.center + volume.radius * vertices, triangles)
