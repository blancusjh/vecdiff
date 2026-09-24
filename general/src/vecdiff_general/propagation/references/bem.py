"""Own Fortran PMCHWT boundary-element solver; independent of SCUFF-EM."""

import hashlib
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import solve

from ...fields import ElectricField, ElectromagneticField
from ...fields.spectrum import PlaneWaveExpansion
from ...media.medium import Z0
from ...sampling import PointGrid
from .mesh import SurfaceMesh, gauss_rule


class DielectricBEM:
    """Full-vector PMCHWT with Rao–Wilton–Glisson surface elements.

    One closed homogeneous, lossless, nonmagnetic inclusion. Coordinates are
    metres. Dense assembly/solve is deliberately resource limited. Mesh and
    quadrature convergence must be checked before treating results as a reference.
    No local Fresnel data, fitted sources, or operational-method fields enter it.

    Well-separated panel pairs (centroid distance above far_ratio times the
    sum of their sizes) are assembled with far_quadrature; field points farther
    than field_far_ratio panel sizes use field_far_quadrature. The kernel is
    smooth there; tests compare against the uniform rules.

    cache: a directory (or the VECDIFF_BEM_CACHE environment variable) where
    solved surface currents are stored under a fingerprint of the mesh, the
    illumination, both media, every quadrature setting and the solver sources
    (bem.py, bem.f90, reference_math.f90). A matching solve is reloaded instead
    of assembled and factorized; field evaluation is always recomputed.
    """

    def __init__(
        self,
        source,
        mesh,
        interior,
        *,
        quadrature=4,
        near_quadrature=7,
        field_quadrature=10,
        far_quadrature=3,
        far_ratio=2.0,
        field_far_quadrature=4,
        field_far_ratio=3.0,
        max_unknowns=5000,
        cache=None,
    ):
        from . import _reference_core

        if not isinstance(source, PlaneWaveExpansion):
            raise TypeError("BEM illumination must be an analytic PlaneWaveExpansion")
        if not isinstance(mesh, SurfaceMesh):
            raise TypeError("BEM requires a closed SurfaceMesh")
        self.source, self.mesh, self.interior = source, mesh, interior
        self.exterior, self.wavelength0 = source.region, source.wavelength0
        self.unknowns = 2 * len(mesh.edges)
        if self.unknowns > max_unknowns:
            raise ValueError(
                f"BEM needs {self.unknowns} unknowns (limit {max_unknowns}); dense storage grows quadratically"
            )
        q, w = gauss_rule(quadrature)
        qn, wn = gauss_rule(near_quadrature)
        qf, wf = gauss_rule(far_quadrature)
        self._q, self._w = gauss_rule(field_quadrature)
        self._qf, self._wf = gauss_rule(field_far_quadrature)
        self._field_far_ratio = float(field_far_ratio)
        self._mesh = SurfaceMesh(mesh.vertices / self.wavelength0, mesh.triangles)
        m = self._mesh
        self._corners = np.asfortranarray(m.corners.transpose(2, 1, 0))
        self._edges = np.asfortranarray(m.edge_ids.T + 1)
        self._factors = np.asfortranarray(m.factors.T)
        cache = cache or os.environ.get("VECDIFF_BEM_CACHE")
        stored = None
        if cache:
            key = _fingerprint(
                m,
                source,
                self.exterior.material.n,
                interior.material.n,
                (quadrature, near_quadrature, far_quadrature, far_ratio),
            )
            stored = Path(cache) / f"bem_{key}.npz"
            if stored.exists():
                with np.load(stored) as data:
                    self.coefficients = data["coefficients"]
                    self.algebraic_residual = float(data["algebraic_residual"])
                    self.assembly_seconds = float(data["assembly_seconds"])
                    self.solve_seconds = float(data["solve_seconds"])
                self.loaded_from = stored
                return
        self.loaded_from = None
        started = perf_counter()
        matrix = _reference_core.bem_assemble(
            len(m.edges),
            self._corners,
            m.triangles.T + 1,
            self._edges,
            self._factors,
            self.exterior.material.n,
            interior.material.n,
            q,
            w,
            qn,
            wn,
            qf,
            wf,
            float(far_ratio),
        )
        points, weights = m.quadrature(near_quadrature)
        incident = source.evaluate(PointGrid(points * self.wavelength0))
        basis = m.factors[:, None, :, None] * (
            points[:, :, None, :] - m.corners[:, None, :, :]
        )
        rhs = []
        for field in (incident.electric.values, Z0 * incident.magnetic):
            local = -np.einsum("tqac,tqc,tq->ta", basis, field, weights)
            global_rhs = np.zeros(len(m.edges), complex)
            np.add.at(global_rhs, m.edge_ids.ravel(), local.ravel())
            rhs.append(global_rhs)
        rhs = np.concatenate(rhs)
        self.assembly_seconds = perf_counter() - started
        started = perf_counter()
        self.coefficients = solve(matrix, rhs, check_finite=True)
        self.solve_seconds = perf_counter() - started
        self.algebraic_residual = float(
            np.linalg.norm(matrix @ self.coefficients - rhs)
            / max(np.linalg.norm(rhs), np.finfo(float).tiny)
        )
        if stored is not None:
            stored.parent.mkdir(parents=True, exist_ok=True)
            partial = stored.with_name(stored.stem + ".partial.npz")
            np.savez(
                partial,
                coefficients=self.coefficients,
                algebraic_residual=self.algebraic_residual,
                assembly_seconds=self.assembly_seconds,
                solve_seconds=self.solve_seconds,
            )
            partial.replace(stored)

    def evaluate(self, grid, *, side=None):
        """Total SI fields, including exact one-sided jumps at face interiors.

        ``side`` selects the interior or exterior trace on the actual flat mesh.
        It cannot request a continuation through a different medium. Off-surface
        close evaluation also requires increasing ``field_quadrature``.
        """
        from . import _reference_core

        if side not in (None, "interior", "exterior"):
            raise ValueError("side must be interior, exterior, or None")
        points = grid.points.reshape(-1, 3) / self.wavelength0
        inside, faces = self._mesh.locate(points)
        on = faces >= 0
        if np.any(on) and side is None:
            raise ValueError("specify side for boundary traces")
        if side is not None:
            requested = side == "interior"
            if np.any((inside != requested) & ~on):
                raise ValueError(
                    "side disagrees with the mesh region of an evaluation point"
                )
            inside[on] = requested
        e, h = np.empty(points.shape, complex), np.empty(points.shape, complex)
        for interior in (False, True):
            mask = inside == interior
            if not mask.any():
                continue
            index = (self.interior if interior else self.exterior).material.n
            ev, hv = _reference_core.bem_fields(
                self._corners,
                self._edges,
                self._factors,
                self.coefficients,
                points[mask].T,
                faces[mask] + 1,
                index,
                -1 if interior else 1,
                self._q,
                self._w,
                self._qf,
                self._wf,
                self._field_far_ratio,
            )
            e[mask], h[mask] = ev.T, hv.T / Z0
            if not interior:
                incident = self.source.evaluate(
                    PointGrid(points[mask] * self.wavelength0)
                )
                e[mask] += incident.electric.values
                h[mask] += incident.magnetic
        shape = grid.shape + (3,)
        return ElectromagneticField(
            ElectricField(grid, self.wavelength0, e.reshape(shape)), h.reshape(shape)
        )


def _fingerprint(mesh, source, n_exterior, n_interior, rules):
    """Hash of everything that determines the solved surface currents."""
    digest = hashlib.sha256()
    for array in (
        mesh.vertices,
        mesh.triangles,
        source.wavevectors,
        source.amplitudes,
        source.origin,
    ):
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    digest.update(repr((source.wavelength0, n_exterior, n_interior, rules)).encode())
    native = Path(__file__).resolve().parents[4] / "native"
    for path in (
        Path(__file__),
        native / "bem.f90",
        native / "reference_math.f90",
    ):
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:20]
