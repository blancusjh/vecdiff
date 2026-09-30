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


def triangle_quadrature(corners, order=4):
    """Collapsed Gauss product rule on triangles (T, 3, 3): points and weights.

    Exact for polynomials of total degree 2*order - 2 on each flat triangle.
    """
    q, w = gauss_rule(order)
    u, v = np.meshgrid(q, q, indexing="ij")
    weights = (w[:, None] * w[None, :] * (1 - u)).ravel()
    bary = np.stack(((1 - u) * (1 - v), u, (1 - u) * v), -1).reshape(-1, 3)
    points = np.einsum("qa,tac->tqc", bary, corners)
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return points, np.linalg.norm(cross, axis=1)[:, None] * weights


@dataclass(frozen=True)
class DualBasis:
    """Buffa–Christiansen functions on the barycentric refinement of a SurfaceMesh.

    Every coarse triangle is split into six through its centroid and edge
    midpoints; sub-triangle ``6 t + 2 i + s`` has the coarse corner ``i`` of
    triangle ``t`` as its first vertex. On a sub-triangle with corners P_k the
    dual function of coarse edge ``rows[s, j]`` is the div-conforming linear field

        b(x) = sum_k flux[s, j, k] (x - P_k) / (2 area[s]),

    i.e. ``flux[s, j, k]`` is its outward flux through the side opposite P_k.
    ``rows`` is padded with -1. Each b is scaled by its coarse edge length.
    ``apex`` is the coarse vertex at which each sub-triangle sits.
    """

    corners: np.ndarray
    parents: np.ndarray
    apex: np.ndarray
    areas: np.ndarray
    normals: np.ndarray
    rows: np.ndarray
    flux: np.ndarray

    def sample(self, order=3):
        """Quadrature points, weights and every function value per sub-triangle.

        Returns points (S, Q, 3), weights (S, Q) and values (S, J, Q, 3) of the
        functions ``rows[s, j]`` (zero in padded slots).
        """
        points, weights = triangle_quadrature(self.corners, order)
        half = (points[:, None, :, :] - self.corners[:, :, None, :]) / (
            2 * self.areas[:, None, None, None]
        )
        return points, weights, np.einsum("sjk,skqc->sjqc", self.flux, half)

    def gram(self, mesh):
        """Dense mixed Gram matrix G[a, b] = <n x b_a, f_b> with RWG f_b.

        The integrand is quadratic on each flat sub-triangle, so the degree-4
        rule used here is exact.
        """
        points, weights, values = self.sample(3)
        parent = self.parents
        rwg = mesh.factors[parent][:, :, None, None] * (
            points[:, None, :, :] - mesh.corners[parent][:, :, None, :]
        )
        rotated = np.cross(self.normals[:, None, None, :], values)
        local = np.einsum("sjqc,saqc,sq->sja", rotated, rwg, weights)
        used = np.broadcast_to((self.rows >= 0)[:, :, None], local.shape)
        rows = np.broadcast_to(self.rows[:, :, None], local.shape)
        columns = np.broadcast_to(mesh.edge_ids[parent][:, None, :], local.shape)
        gram = np.zeros((len(mesh.edges), len(mesh.edges)))
        np.add.at(gram, (rows[used], columns[used]), local[used])
        return gram


def buffa_christiansen(mesh):
    """Dual (BC) basis of a closed manifold SurfaceMesh, one function per edge.

    Construction of A. Buffa and S. H. Christiansen, "A dual finite element
    complex on the barycentric refinement", Math. Comp. 76 (2007) 1743-1769,
    with the explicit coefficients of F. P. Andriulli et al., "A multiplicative
    Calderon preconditioner for the electric field integral equation", IEEE
    TAP 56 (2008) 2398-2412. For coarse edge m from vertex v+ to v- the
    function carries unit flux (times |m|) from the dual cell of v+ to that of
    v-, half through each barycentric segment centroid-midpoint(m)-centroid.
    Inside the 2N sub-triangles of a vertex cell every sub-triangle takes the
    same share 1/(2N) of the divergence; the remaining circulation is fixed by
    zero flux through both halves of m itself (the BC gauge). The rotated
    functions n x b pair stably (uniform inf-sup) with RWG functions under
    <n x b, f>, which is what a second-kind (identity + compact) formulation
    needs for its identity term.
    """
    triangles, vertices = mesh.triangles, mesh.vertices
    nt = len(triangles)
    centroids = mesh.corners.mean(axis=1)
    midpoints = vertices[mesh.edges].mean(axis=1)
    lengths = np.linalg.norm(
        vertices[mesh.edges[:, 0]] - vertices[mesh.edges[:, 1]], axis=1
    )
    # Coarse corner i of triangle t: first edge (V_i, V_i+1) then second edge
    # (V_i, V_i-1), counter-clockwise about the outward normal.
    first = mesh.edge_ids[:, [2, 0, 1]]
    second = mesh.edge_ids[:, [1, 2, 0]]
    corners = np.empty((6 * nt, 3, 3))
    for i in range(3):
        v = vertices[triangles[:, i]]
        corners[2 * i :: 6] = np.stack((v, midpoints[first[:, i]], centroids), 1)
        corners[2 * i + 1 :: 6] = np.stack((v, centroids, midpoints[second[:, i]]), 1)
    # Ordered cells: the counter-clockwise ring of sub-triangles around a vertex.
    incident = [[] for _ in vertices]
    for t in range(nt):
        for i in range(3):
            incident[triangles[t, i]].append((t, i))
    rings = []
    for v, members in enumerate(incident):
        by_first = {first[t, i]: (t, i) for t, i in members}
        if len(by_first) != len(members) or len(members) < 3:
            raise ValueError(f"mesh is not manifold at vertex {v}")
        ring, (t, i) = [], members[0]
        for _ in members:
            ring.append((t, i))
            following = by_first.get(second[t, i])
            if following is None:
                raise ValueError(f"mesh is not manifold at vertex {v}")
            t, i = following
        if (t, i) != members[0] or len({x for x in ring}) != len(members):
            raise ValueError(f"mesh is not manifold at vertex {v}")
        rings.append(ring)
    width = max(len(ring) for ring in rings)
    rows = np.full((6 * nt, width), -1, dtype=np.int32)
    flux = np.zeros((6 * nt, width, 3))
    for v, ring in enumerate(rings):
        n = len(ring)
        cell = [6 * t + 2 * i + s for t, i in ring for s in (0, 1)]
        # Radial flux F_r from cell[r-1] to cell[r] (r = 1..2n-1) when the
        # ring starts just after the half of edge m; F_0 = F_2n = 0.
        radial = np.concatenate(([0.0], (np.arange(1, 2 * n) - n) / (2 * n), [0.0]))
        for j, (t, i) in enumerate(ring):
            m = first[t, i]
            sign = lengths[m] * (1.0 if mesh.edges[m, 0] == v else -1.0)
            for r in range(1, 2 * n + 1):
                s = cell[(2 * j + r - 1) % (2 * n)]
                outer = 0.5 if r in (1, 2 * n) else 0.0
                # Local sides: opposite v (outer), opposite X (next radial),
                # opposite Y (previous radial), for the sub-triangle (v, X, Y).
                rows[s, j] = m
                flux[s, j] = sign * np.array([outer, radial[r], -radial[r - 1]])
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    areas = np.linalg.norm(cross, axis=1) / 2
    for array in (corners, rows, flux, areas):
        array.setflags(write=False)
    return DualBasis(
        corners,
        np.repeat(np.arange(nt, dtype=np.int32), 6),
        np.repeat(triangles, 2, axis=1).ravel(),
        areas,
        cross / (2 * areas[:, None]),
        rows,
        flux,
    )


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
        return triangle_quadrature(self.corners, order)

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
