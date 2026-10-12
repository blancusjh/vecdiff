"""Own Fortran PMCHWT and Müller boundary-element solvers; independent of SCUFF-EM."""

import hashlib
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import solve

from vecdiff_general.fields import ElectricField, ElectromagneticField
from vecdiff_general.fields.spectrum import PlaneWaveExpansion
from vecdiff_general.media.medium import Z0
from vecdiff_general.propagation.references.mesh import (
    SurfaceMesh,
    buffa_christiansen,
    gauss_rule,
)
from vecdiff_general.sampling import PointGrid

FORMULATIONS = ("pmchwt", "muller")


class DielectricBEM:
    """Full-vector PMCHWT or Müller BEM with Rao–Wilton–Glisson surface currents.

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
    (bem.py, bem.f90, muller.f90, reference_math.f90). A matching solve is
    reloaded instead of assembled and factorized; field evaluation is always
    recomputed.

    formulation: "pmchwt" (default) or "muller". Both solve for the same
    unknowns j = Z0 n x H and m = E x n in the same RWG basis, so the field
    representation, evaluate(), caching and limits are shared; only the tested
    boundary equations differ, hence one class with a switch rather than a
    sibling class.

    PMCHWT tests the tangential E and H continuity equations with RWG
    functions. It is a first-kind system: its hypersingular T operators make
    the condition number grow like h^-2 under refinement.

    Müller (C. Müller, Foundations of the Mathematical Theory of
    Electromagnetic Waves, 1969) combines the rotated traces n x E and n x H
    of both sides weighted by n^2 (permittivity) and 1 (permeability), so the
    hypersingular parts cancel:

        j - n x (K_e - K_i) j - n x (n_e^2 T_e - n_i^2 T_i) m = n x Z0 H_inc
        m - n x (n_e^2 K_e - n_i^2 K_i) m / c
          + n x (n_e^2 T_e - n_i^2 T_i) j / c = -(n_e^2 / c) n x E_inc

    with c = (n_e^2 + n_i^2)/2: identity plus compact on smooth surfaces,
    free of interior resonances. The identity must be discretized with a stable
    pairing: the equations are tested with rotated Buffa–Christiansen
    functions n x b on the barycentric refinement (A. Buffa and
    S. H. Christiansen, Math. Comp. 76 (2007) 1743–1769), whose Gram matrix
    <n x b, f> with RWG is uniformly inf-sup stable (cond 2.2, 2.9, 3.1, 3.3 on
    icosphere levels 0–3). This is the mixed RWG/BC Müller discretization of
    S. Yan, J.-M. Jin and Z. Nie, IEEE Trans. Antennas Propag. 59 (2011)
    1299–1310. Rotated-RWG testing is not used: <n x f, f> is numerically
    singular on closed meshes (cond ~1e17 on the same icospheres).

    Measured on a 0.3-wavelength n = 1.5 sphere, icosphere levels 0–3 (60,
    240, 960, 3840 unknowns): condition number PMCHWT 6.1, 11, 44, 186 and
    Müller 5.0, 6.0, 6.7, 7.0; unrestarted GMRES iterations to 1e-6 PMCHWT
    35, 69, 122, 212 and Müller 12, 13, 15, 16 (docs/VALIDATION.md#bem has the
    ka = 5 case). Müller is not more accurate at equal mesh (E error vs Mie
    0.0044 against PMCHWT's 0.0035 at level 3); it is an independently derived
    system with a bounded spectrum. Its matrix is non-symmetric and assembled
    from 6x more (sub-triangle, panel) pairs than PMCHWT's symmetric half:
    about 12x the PMCHWT assembly time on 10 threads: 1.6 s at 960, ~20 s at
    3840 and 33 s at 4980 unknowns (near the default limit; 0.4 GB dense).

    keep_system: retain ``matrix`` and ``rhs`` (dense) for conditioning studies.
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
        formulation="pmchwt",
        keep_system=False,
    ):
        from vecdiff_general.propagation.references import _reference_core

        if not isinstance(source, PlaneWaveExpansion):
            raise TypeError("BEM illumination must be an analytic PlaneWaveExpansion")
        if not isinstance(mesh, SurfaceMesh):
            raise TypeError("BEM requires a closed SurfaceMesh")
        if formulation not in FORMULATIONS:
            raise ValueError(f"formulation must be one of {FORMULATIONS}")
        self.formulation = formulation
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
                formulation,
            )
            stored = Path(cache) / f"bem_{key}.npz"
            if stored.exists() and not keep_system:
                with np.load(stored) as data:
                    self.coefficients = data["coefficients"]
                    self.algebraic_residual = float(data["algebraic_residual"])
                    self.assembly_seconds = float(data["assembly_seconds"])
                    self.solve_seconds = float(data["solve_seconds"])
                self.loaded_from = stored
                return
        self.loaded_from = None
        started = perf_counter()
        rules = (q, w, qn, wn, qf, wf, float(far_ratio))
        assemble = _muller_system if formulation == "muller" else _pmchwt_system
        matrix, rhs = assemble(self, _reference_core, rules, near_quadrature)
        self.assembly_seconds = perf_counter() - started
        started = perf_counter()
        self.coefficients = solve(matrix, rhs, check_finite=True)
        self.solve_seconds = perf_counter() - started
        self.algebraic_residual = float(
            np.linalg.norm(matrix @ self.coefficients - rhs)
            / max(np.linalg.norm(rhs), np.finfo(float).tiny)
        )
        if keep_system:
            self.matrix, self.rhs = matrix, rhs
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
        from vecdiff_general.propagation.references import _reference_core

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


def _incident_tested(solution, points, weights, basis):
    """Tested incident traces <basis, E_inc> and <basis, Z0 H_inc> per element."""
    shape = points.shape
    incident = solution.source.evaluate(
        PointGrid(points.reshape(-1, 3) * solution.wavelength0)
    )
    electric = incident.electric.values.reshape(shape)
    magnetic = (Z0 * incident.magnetic).reshape(shape)
    return [
        np.einsum("t...qc,tqc,tq->t...", basis, field, weights)
        for field in (electric, magnetic)
    ]


def _pmchwt_system(solution, core, rules, order):
    """RWG-tested PMCHWT matrix and right-hand side (see class docstring)."""
    m = solution._mesh
    q, w, qn, wn, qf, wf, far_ratio = rules
    matrix = core.bem_assemble(
        len(m.edges),
        solution._corners,
        m.triangles.T + 1,
        solution._edges,
        solution._factors,
        solution.exterior.material.n,
        solution.interior.material.n,
        q,
        w,
        qn,
        wn,
        qf,
        wf,
        far_ratio,
    )
    points, weights = m.quadrature(order)
    basis = m.factors[:, :, None, None] * (
        points[:, None, :, :] - m.corners[:, :, None, :]
    )
    rhs = []
    for local in _incident_tested(solution, points, weights, basis):
        global_rhs = np.zeros(len(m.edges), complex)
        np.add.at(global_rhs, m.edge_ids.ravel(), -local.ravel())
        rhs.append(global_rhs)
    return matrix, np.concatenate(rhs)


def _muller_system(solution, core, rules, order):
    """BC-tested Müller matrix and right-hand side (see class docstring).

    Rows 0..ne-1 are the H (permeability-weighted) equation, rows ne.. the E
    (permittivity-weighted) equation divided by c; columns are (j, m) as for
    PMCHWT. The sparse Gram <n x b_a, f_b> is integrated exactly (quadratic on
    each barycentric sub-triangle) and added to both diagonal blocks.
    """
    m = solution._mesh
    ne = len(m.edges)
    q, w, qn, wn, qf, wf, far_ratio = rules
    n0, n1 = solution.exterior.material.n, solution.interior.material.n
    dual = buffa_christiansen(m)
    matrix = core.bem_assemble_muller(
        ne,
        solution._corners,
        m.triangles.T + 1,
        solution._edges,
        solution._factors,
        np.asfortranarray(dual.corners.transpose(2, 1, 0)),
        dual.parents + 1,
        dual.apex + 1,
        np.asfortranarray(dual.rows.T + 1),
        np.asfortranarray(dual.flux.transpose(2, 1, 0)),
        n0,
        n1,
        q,
        w,
        qn,
        wn,
        qf,
        wf,
        far_ratio,
    )
    gram = dual.gram(m)
    matrix[:ne, :ne] += gram
    matrix[ne:, ne:] += gram
    rows, used = dual.rows, dual.rows >= 0
    points, weights, dual_values = dual.sample(max(order, 2))
    electric, magnetic = _incident_tested(solution, points, weights, dual_values)
    c = (n0**2 + n1**2) / 2
    rhs = np.zeros(2 * ne, complex)
    np.add.at(rhs, rows[used], magnetic[used])
    np.add.at(rhs, ne + rows[used], -(n0**2 / c) * electric[used])
    return matrix, rhs


def _fingerprint(mesh, source, n_exterior, n_interior, rules, formulation):
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
    digest.update(
        repr((source.wavelength0, n_exterior, n_interior, rules, formulation)).encode()
    )
    native = Path(__file__).resolve().parents[4] / "native"
    for path in (
        Path(__file__),
        native / "bem.f90",
        native / "muller.f90",
        native / "reference_math.f90",
    ):
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:20]
