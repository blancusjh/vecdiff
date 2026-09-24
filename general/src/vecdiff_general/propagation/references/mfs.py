"""Axisymmetric Maxwell transmission by the method of fundamental solutions.

Historical experimental solver, retained for its focused regression tests.
The recommended reference API exposes DielectricBEM and DielectricFEM instead;
this auxiliary-source method is not used by the comparison notebook.

Electric dipole rings carry azimuthal Fourier mode m=1. Sources strictly inside
the body generate the outgoing exterior field; sources strictly outside generate
the regular interior field. The coupled solve matches both tangential E and H
on the ENTIRE closed boundary. This is auxiliary-source boundary collocation,
not a PMCHWT discretization or a prescribed-current physical-optics integral.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import lstsq

from ...fields import ElectricField, ElectromagneticField
from ...fields.spectrum import PlaneWaveExpansion
from ...media.medium import Z0
from ...volumes import SphereVolume
from .boundary import boundary_meridian


def _ring_fields(points, rings, index, azimuths, *, chunk_size=8):
    """Dipole-ring basis E and Z0 H on the phi=0 meridian, in lambda units.

    Pair-dependent sinh quadrature resolves the narrow near-source peak.
    Uniform azimuthal sampling would require O(radius / separation) nodes
    when an auxiliary surface is placed close to the dielectric boundary.
    """
    from numpy.polynomial.legendre import leggauss

    nodes, weights = leggauss((azimuths + 1) // 2)
    nodes, weights = (nodes + 1) / 2, weights / 2
    k = 2 * np.pi * index
    electric, magnetic = [], []
    for start in range(0, len(points), chunk_size):
        targets = points[start : start + chunk_size]
        nearest = np.hypot(
            targets[:, None, 0] - rings[None, :, 0],
            targets[:, None, 2] - rings[None, :, 1],
        )
        if np.any(nearest < 1e-13):
            raise ValueError("observation intersects an auxiliary source")
        scale = np.minimum(
            1.0,
            nearest
            / np.sqrt(np.maximum(targets[:, None, 0] * rings[None, :, 0], 1e-300)),
        )
        extent = np.arcsinh(np.pi / (2 * scale))
        parameter = extent[..., None] * nodes
        positive = 2 * scale[..., None] * np.sinh(parameter)
        jacobian = (
            2
            * scale[..., None]
            * np.cosh(parameter)
            * extent[..., None]
            * weights
            / (2 * np.pi)
        )
        phi = np.concatenate((-positive, positive), -1)
        phase = np.exp(1j * phi) * np.concatenate((jacobian, jacobian), -1)
        cp, sp = np.cos(phi), np.sin(phi)
        rotation = np.zeros(phi.shape + (3, 3))
        rotation[..., 0, 0], rotation[..., 1, 0] = cp, sp
        rotation[..., 0, 1], rotation[..., 1, 1] = -sp, cp
        rotation[..., 2, 2] = 1
        positions = np.stack(
            np.broadcast_arrays(
                rings[None, :, 0, None] * cp,
                rings[None, :, 0, None] * sp,
                rings[None, :, 1, None],
            ),
            -1,
        )
        displacement = targets[:, None, None, :] - positions
        r = np.linalg.norm(displacement, axis=-1)
        u = displacement / r[..., None]
        g = np.exp(1j * k * r) / (4 * np.pi * r)
        ur = np.einsum("psai,psaij->psaj", u, rotation)
        A = k * k + 1j * k / r - 1 / r**2
        B = -k * k - 3j * k / r + 3 / r**2
        e = g[..., None, None] * (
            A[..., None, None] * rotation
            + B[..., None, None] * u[..., :, None] * ur[..., None, :]
        )
        electric.append(np.einsum("psaij,psa->pisj", e, phase).reshape(len(r), 3, -1))
        cross = np.cross(u[..., :, None], rotation, axisa=-2, axisb=-2, axisc=-2)
        h = (index * g * (k * k + 1j * k / r))[..., None, None] * cross
        magnetic.append(np.einsum("psaij,psa->pisj", h, phase).reshape(len(r), 3, -1))
    return np.concatenate(electric), np.concatenate(magnetic)


@dataclass(frozen=True)
class BoundarySolveReport:
    unknowns: int
    equations: int
    rank: int
    condition: float
    fitted_residual: float
    assembly_seconds: float
    solve_seconds: float
    azimuths: int


class AxisymmetricMFS:
    """Full-wave reference for one closed axisymmetric dielectric body.

    Source: one normally incident +z plane wave, arbitrary complex polarization.
    ``source_count`` nodes per lens face/rim (total three times that for lenses).
    Accuracy MUST be measured on independent nodes while varying source_count,
    source_offset and azimuths; a small fitted residual alone is insufficient.
    Sharp rims and large electrical sizes can require prohibitively many nodes.
    """

    def __init__(
        self,
        source,
        volume,
        interior,
        *,
        source_count=24,
        collocation_count=None,
        source_offset=None,
        azimuths=None,
        rcond=1e-12,
        max_unknowns=2400,
    ):
        if not isinstance(source, PlaneWaveExpansion) or len(source.amplitudes) != 1:
            raise TypeError("boundary reference requires one analytic plane wave")
        k = source.wavevectors[0]
        if (
            np.any(k.imag)
            or np.linalg.norm(k[:2]) > 1e-12 * abs(k[2])
            or k[2].real <= 0
        ):
            raise ValueError("axisymmetric solver requires incidence along global +z")
        if not 0 < rcond < 1:
            raise ValueError("rcond must lie in (0,1)")
        self.source, self.volume, self.interior = source, volume, interior
        self.exterior, self.wavelength0 = source.region, source.wavelength0
        boundary = boundary_meridian(volume, source_count)
        unknowns = 6 * len(boundary.points)
        if unknowns > max_unknowns:
            raise ValueError(
                f"reference needs {unknowns} unknowns, exceeds budget {max_unknowns}"
            )
        if collocation_count is None:
            collocation_count = 2 * source_count + 3
        collocation = boundary_meridian(volume, collocation_count)
        if 6 * len(collocation.points) < unknowns:
            raise ValueError("need at least as many boundary equations as unknowns")
        # Homothetic proxy surfaces keep rim sources away from BOTH faces.
        # Offsetting each panel normal independently creates near-singular
        # sources at the adjacent panel as quadrature approaches a corner.
        if isinstance(volume, SphereVolume):
            center = np.array([0.0, volume.center[2]])
            length = volume.radius
        else:
            center = np.array(
                [0.0, (volume.front.frame.origin[2] + volume.back.frame.origin[2]) / 2]
            )
            length = min(
                volume.rim_radius,
                (volume.back.frame.origin[2] - volume.front.frame.origin[2]) / 2,
            )
        if source_offset is None:
            source_offset = min(
                0.3 * length, 0.3 * self.wavelength0 / interior.material.n
            )
        if not np.isfinite(source_offset) or source_offset <= 0:
            raise ValueError("source_offset must be positive")
        fraction = source_offset / length
        if fraction >= 0.9:
            raise ValueError(
                "source_offset must be less than 90% of the body half-size"
            )
        inside = center + (1 - fraction) * (boundary.points - center)
        outside = center + (1 + fraction) * (boundary.points - center)

        def xyz(rz):
            return np.stack((rz[:, 0], np.zeros(len(rz)), rz[:, 1]), -1)

        if (
            np.any(inside[:, 0] <= 0)
            or np.any(outside[:, 0] <= 0)
            or not np.all(volume.contains(xyz(inside)))
            or np.any(volume.contains(xyz(outside)))
        ):
            raise ValueError(
                "auxiliary rings must stay strictly on their own side; reduce source_offset"
            )
        self.inside_sources, self.outside_sources = (
            inside / self.wavelength0,
            outside / self.wavelength0,
        )
        if azimuths is None:
            electrical_radius = (
                np.max(outside[:, 0])
                / self.wavelength0
                * max(interior.material.n, self.exterior.material.n)
            )
            azimuths = max(
                128,
                int(np.ceil(16 * electrical_radius + 32)),
            )
        if not isinstance(azimuths, (int, np.integer)) or azimuths < 8:
            raise ValueError("azimuths must be an integer >= 8")
        azimuths = 2 * ((azimuths + 1) // 2)
        self.azimuths = azimuths
        started = perf_counter()
        points = xyz(collocation.points) / self.wavelength0
        Ee, He = _ring_fields(
            points, self.inside_sources, self.exterior.material.n, azimuths
        )
        Ei, Hi = _ring_fields(
            points, self.outside_sources, interior.material.n, azimuths
        )
        # Tangents in the meridional and azimuthal directions, at phi=0.
        tangent = np.stack(
            (
                collocation.normals[:, 1],
                np.zeros(len(points)),
                -collocation.normals[:, 0],
            ),
            -1,
        )
        normal = np.stack(
            (
                collocation.normals[:, 0],
                np.zeros(len(points)),
                collocation.normals[:, 1],
            ),
            -1,
        )

        def traces(E, H, index):
            # The normal equations are redundant in the continuum but suppress
            # unresolved, large normal traces in finite boundary collocation.
            return np.stack(
                (
                    np.einsum("pi,pij->pj", tangent, E),
                    E[:, 1, :],
                    np.einsum("pi,pij->pj", tangent, H) / self.exterior.material.n,
                    H[:, 1, :] / self.exterior.material.n,
                    np.einsum("pi,pij->pj", normal, E)
                    * (index / self.exterior.material.n) ** 2,
                    np.einsum("pi,pij->pj", normal, H) / self.exterior.material.n,
                ),
                1,
            )

        matrix = np.concatenate(
            (
                traces(Ee, He, self.exterior.material.n),
                -traces(Ei, Hi, interior.material.n),
            ),
            -1,
        )
        phase = np.exp(2j * np.pi * self.exterior.material.n * points[:, 2])
        # Unit circular incident E=(x+i y) exp(ikz); other helicity by reflection.
        Einc = phase[:, None] * np.array([1, 1j, 0])
        Hinc = self.exterior.material.n * phase[:, None] * np.array([-1j, 1, 0])
        internal_phase = np.exp(2j * np.pi * interior.material.n * points[:, 2])
        Ebackground = internal_phase[:, None] * np.array([1, 1j, 0])
        Hbackground = (
            interior.material.n * internal_phase[:, None] * np.array([-1j, 1, 0])
        )
        rhs = (
            traces(Ebackground[..., None], Hbackground[..., None], interior.material.n)
            - traces(Einc[..., None], Hinc[..., None], self.exterior.material.n)
        )[..., 0]
        weight = np.sqrt(collocation.weights / np.sum(collocation.weights))[:, None]
        A = (matrix * weight[..., None]).reshape(-1, unknowns)
        b = (rhs * weight).ravel()
        scale = np.linalg.norm(A, axis=0)
        if np.any(scale == 0) or not np.isfinite(A).all():
            raise ValueError("degenerate auxiliary source matrix")
        assembled = perf_counter()
        result, _, rank, singular = lstsq(
            A / scale, b, cond=rcond, lapack_driver="gelsd"
        )
        self.coefficients = result / scale
        solved = perf_counter()
        self.report = BoundarySolveReport(
            unknowns,
            len(b),
            int(rank),
            float(singular[0] / singular[-1]),
            float(
                np.linalg.norm(A @ self.coefficients - b)
                / max(np.linalg.norm(b), 1e-300)
            ),
            assembled - started,
            solved - assembled,
            azimuths,
        )

    def evaluate(self, grid, *, side=None):
        """Total SI fields; side selects an identical-point boundary trace."""
        if side not in (None, "interior", "exterior"):
            raise ValueError("side must be interior, exterior or None")
        points = grid.points.reshape(-1, 3)
        if side is not None:
            if isinstance(self.volume, SphereVolume):
                center = self.volume.center
            else:
                center = (
                    self.volume.front.frame.origin + self.volume.back.frame.origin
                ) / 2
            # The proxy construction requires a star-shaped body. Allow
            # rounding at a trace, reject continuation to the wrong side.
            test = center + (points - center) * (
                1 - 1e-10 if side == "interior" else 1 + 1e-10
            )
            contained = self.volume.contains(test)
            if (side == "interior" and not np.all(contained)) or (
                side == "exterior" and np.any(contained)
            ):
                raise ValueError(
                    "requested trace side is inconsistent with the observation region"
                )
        mask = (
            self.volume.contains(points)
            if side is None
            else np.full(len(points), side == "interior")
        )
        radius = np.linalg.norm(points[:, :2], axis=-1)
        phi = np.arctan2(points[:, 1], points[:, 0])
        meridian = (
            np.stack((radius, np.zeros(len(points)), points[:, 2]), -1)
            / self.wavelength0
        )
        E, H = np.zeros_like(points, complex), np.zeros_like(points, complex)
        split = len(self.coefficients) // 2
        origin_phase = np.exp(-1j * self.source.wavevectors[0] @ self.source.origin)
        ax, ay = self.source.amplitudes[0, :2] * origin_phase
        plus, minus = (ax - 1j * ay) / 2, (ax + 1j * ay) / 2
        for inside in (False, True):
            ids = np.flatnonzero(mask == inside)
            if not len(ids):
                continue
            rings = self.outside_sources if inside else self.inside_sources
            index = self.interior.material.n if inside else self.exterior.material.n
            coeff = self.coefficients[split:] if inside else self.coefficients[:split]
            e, h = _ring_fields(meridian[ids], rings, index, self.azimuths)
            e, h = e @ coeff, h @ coeff
            if inside:
                phase = np.exp(2j * np.pi * index * meridian[ids, 2])
                e += phase[:, None] * np.array([1, 1j, 0])
                h += index * phase[:, None] * np.array([-1j, 1, 0])
            pp = plus * np.exp(1j * phi[ids])
            pm = minus * np.exp(-1j * phi[ids])
            e = pp[:, None] * e + pm[:, None] * e * [1, -1, 1]
            h = pp[:, None] * h - pm[:, None] * h * [1, -1, 1]
            cp, sp = np.cos(phi[ids]), np.sin(phi[ids])

            def rotate(v):
                return np.stack(
                    (cp * v[:, 0] - sp * v[:, 1], sp * v[:, 0] + cp * v[:, 1], v[:, 2]),
                    -1,
                )

            E[ids], H[ids] = rotate(e), rotate(h) / Z0
        if np.any(~mask):
            from ...sampling import PointGrid

            incident = self.source.evaluate(PointGrid(points[~mask]))
            E[~mask] += incident.electric.values
            H[~mask] += incident.magnetic
        return ElectromagneticField(
            ElectricField(grid, self.wavelength0, E.reshape(grid.shape + (3,))),
            H.reshape(grid.shape + (3,)),
        )
