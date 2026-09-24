"""Order-N interface interaction for spherical and plane surfaces, per plane wave.

``OrderInteraction`` replaces the production eikonal Fresnel step T when passed
as ``propagate_high_frequency(..., interaction=OrderInteraction(order))``. The
incident field is taken as its plane-wave spectrum, not as one local ray:

* order 0 is the exact Fresnel interaction of every plane wave on the local
  tangent plane of every surface point, including total internal reflection;
* orders 1..N add the inverse-wavenumber curvature corrections of the local
  recurrence (``taylor_interface``), tabulated by ``SphericalExitTable``, but
  only where the series still converges at that incidence angle (optimal
  truncation). Elsewhere the retained field is order 0;
* beyond a surface edge the chart is flat at rim height, where the flat
  Fresnel interaction is exact at every order; it is synthesized by FFT.

Edges, stops and light passing outside an element are handled by
``propagate_high_frequency`` exactly as for its default T. A plane surface has
no curvature corrections, so all orders coincide there.
"""

import numpy as np

from ...fields import ElectricField, ElectromagneticField
from ...fourier.cartesian import FFT2, IFFT2, frequency_grid
from ...media.medium import Z0
from ...surfaces import Plane, SphericalCap
from ..operators import FresnelEncounter
from .polar_lens import SphericalExitTable, convergent_incidence


def local_fresnel(directions, amplitudes, normals, n1, n2, critical_width=0.0):
    """Exact plane-wave Fresnel on local tangent planes, broadcast over (..., 3).

    directions are real unit incident directions in n1, normals point into n2.
    Returns {"reflected"|"transmitted": (E, Z0 H)} at unit phase. Beyond the
    critical angle the transmitted wave is evanescent (complex direction).
    critical_width w > 0 uses sqrt(z + i w exp(-z²/w²)) for the transmitted
    normal cosine sqrt(z), z = cos² of the transmission angle: the kink at the
    critical angle is spread over |z| ~ w and the coefficients are unchanged
    (to exp(-z²/w²)) elsewhere. A curved interface spreads it physically over
    w ~ (kR)^(-2/3).
    """
    d, a, n = np.broadcast_arrays(directions, amplitudes, normals)
    if n1 == n2:
        # No interface: every mode continues unchanged, whatever its incidence.
        return {
            "reflected": (np.zeros_like(a, complex), np.zeros_like(a, complex)),
            "transmitted": (a + 0j, n1 * np.cross(d, a)),
        }
    c1 = np.sum(d * n, -1).real
    s = np.cross(d, n)
    size = np.linalg.norm(s, axis=-1, keepdims=True)
    fallback = np.cross(n, np.where(abs(n[..., :1]) < 0.9, [1.0, 0, 0], [0, 1.0, 0]))
    s = np.where(size > 1e-12, s / np.maximum(size, 1e-300), fallback)
    s = s / np.linalg.norm(s, axis=-1, keepdims=True)
    tangential = d - c1[..., None] * n
    # A mode meeting a steep chart point from behind its tangent plane (c1 < 0)
    # keeps travelling on its own side; the coefficients use |c1|.
    side = np.where(c1 < 0, -1.0, 1.0)
    c1 = abs(c1)
    z = 1 - (n1 / n2) ** 2 * (1 - c1**2)
    if critical_width:
        # A smooth continuation of sqrt(z) through its branch point at z = 0,
        # equal to it up to exp(-z²/w²) away from the critical angle.
        z = z + 1j * critical_width * np.exp(-((z / critical_width) ** 2))
    c2 = np.sqrt(z + 0j)
    dt = (n1 / n2) * tangential + (side * c2)[..., None] * n
    dr = tangential - (side * c1)[..., None] * n
    a_s, a_p = np.sum(a * s, -1), np.sum(a * np.cross(s, d), -1)
    ts = 2 * n1 * c1 / (n1 * c1 + n2 * c2)
    tp = 2 * n1 * c1 / (n2 * c1 + n1 * c2)
    rs = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
    rp = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
    et = (ts * a_s)[..., None] * s + (tp * a_p)[..., None] * np.cross(s, dt)
    er = (rs * a_s)[..., None] * s + (rp * a_p)[..., None] * np.cross(s, dr)
    return {
        "reflected": (er, n1 * np.cross(dr, er)),
        "transmitted": (et, n2 * np.cross(dt, et)),
    }


class OrderInteraction:
    """T^[N] for Plane and SphericalCap surfaces, applied per incident plane wave.

    order: highest retained inverse-wavenumber order (0..3).
    threshold: series-convergence limit, see ``polar_lens.series_ratio``.
    taper: fraction of the convergent angular range, ending at its limit, over
    which the corrections are blended smoothly to zero (see ``weight``).
    graded: each correction term m keeps its own convergent limit (see
    ``_stages``); False keeps all of them up to the limit of the highest order.
    window: (low, high) incident envelope amplitudes relative to the chart
    maximum; the corrections are kept where the envelope exceeds high, dropped
    below low and blended smoothly between (see ``presence``). None keeps them
    everywhere, as before this option existed.
    critical: on a sphere of radius R, the order-0 Fresnel kink at the critical
    angle is spread over cos² ~ critical * (n1 k0 R)^(-2/3) (see
    ``local_fresnel``); 0 keeps the flat-interface kink.
    envelope: width, in vacuum wavelengths, of the Gaussian that smooths the
    incident intensity into its envelope, so interference nulls do not count
    as darkness.
    chunk: surface points per vectorized block.
    """

    def __init__(
        self,
        order=3,
        *,
        threshold=0.5,
        taper=0.2,
        graded=True,
        window=(1e-3, 1e-1),
        envelope=0.5,
        critical=0.0,
        nodes=96,
        chunk=16,
        complete_curl=False,
    ):
        if not isinstance(order, int) or not 0 <= order <= 3:
            raise ValueError("order must be an integer from zero through three")
        if not 0 <= taper < 1:
            raise ValueError("taper must lie in [0, 1)")
        self.taper, self.graded = taper, graded
        if window is not None and not 0 < window[0] < window[1] <= 1:
            raise ValueError("window must be (low, high) with 0 < low < high <= 1")
        self.window = window
        if envelope < 0:
            raise ValueError("envelope must be nonnegative")
        self.envelope = envelope
        if critical < 0:
            raise ValueError("critical must be nonnegative")
        self.critical = critical
        # Orders >= 1 always use the complete curl of the truncated E for H.
        # complete_curl=True also gives order 0 that H (for boundary audits);
        # E is unchanged. The convergent range is then that of order 3.
        self.complete_curl = complete_curl
        self.order, self.threshold, self.nodes, self.chunk = (
            order,
            threshold,
            nodes,
            chunk,
        )
        self._tables = {}
        self.limits = {}
        self.term_limits = {}

    def weight(self, incidence, limit):
        """Share of the curvature corrections kept at each incidence angle.

        One inside the first (1 - taper) of the convergent range, zero at its
        limit, with an infinitely smooth blend between. A hard switch makes the
        retained field discontinuous in the direction of each plane wave;
        summed over a beam's plane waves that discontinuity no longer cancels
        where the beam is dark and radiates a spurious field. The smoother the
        blend (in angle, against the beam's angular width), the smaller it is.
        """
        if self.taper == 0:
            return (incidence <= limit).astype(float)
        start = (1 - self.taper) * limit
        s = np.clip((incidence - start) / (limit - start), 0, 1)
        with np.errstate(divide="ignore", over="ignore"):
            rise = np.where(s > 0, np.exp(-1 / s), 0.0)
            fall = np.where(s < 1, np.exp(-1 / (1 - s)), 0.0)
        return fall / (rise + fall)

    def presence(self, incident, plane):
        """Share of the corrections kept at each chart point (1 without window).

        The corrections are local: each plane wave's is exact for that wave,
        but where the incident field is dark they must cancel between waves.
        Waves beyond the convergent limit carry none, so there they do not
        cancel and would radiate a spurious field. Where the light is, that
        imbalance is only the (small) share of such waves.
        """
        intensity = np.sum(abs(incident.electric.values) ** 2, axis=-1)
        if self.window is None or not intensity.max():
            return np.ones(intensity.size)
        if self.envelope:
            width = self.envelope * incident.electric.wavelength0
            fx, fy = frequency_grid(plane.coordinates)
            kernel = np.exp(-2 * (np.pi * width) ** 2 * (fx**2 + fy**2))
            intensity = np.maximum(IFFT2(FFT2(intensity) * kernel).real, 0)
        size = np.sqrt(intensity).ravel()
        level = np.log(np.maximum(size / size.max(), 1e-300))
        low, high = np.log(self.window[0]), np.log(self.window[1])
        s = np.clip((level - low) / (high - low), 0, 1)
        with np.errstate(divide="ignore", over="ignore"):
            rise = np.where(s > 0, np.exp(-1 / s), 0.0)
            fall = np.where(s < 1, np.exp(-1 / (1 - s)), 0.0)
        return rise / (rise + fall)

    def _stages(self, radius, wavelength0, n1, n2, curvature):
        """(first term, last term, limit, table) per blended group of terms.

        Graded (default): one stage per term m = 1..order, each kept up to its
        own convergent limit (series_ratio of the terms through m), which is
        wider for lower m. This is optimal truncation term by term, so the
        first correction still acts where the third no longer converges.
        Otherwise a single stage keeps every term up to the limit of the
        highest order, as before the option existed.
        """
        key = (float(radius), float(wavelength0), n1, n2, curvature)
        if key not in self._tables:
            tabulated = max(self.order, 3 if self.complete_curl else 0)
            if self.graded and self.order > 0:
                groups = [(m, m) for m in range(1, self.order + 1)]
            else:
                groups = [(1, tabulated)]
            stages = []
            for first, last in groups:
                limit = convergent_incidence(
                    radius,
                    wavelength0,
                    n1,
                    n2,
                    last,
                    self.threshold,
                    curvature=curvature,
                )
                table = None
                if limit > 1e-6:
                    table = SphericalExitTable(
                        radius,
                        wavelength0,
                        n1,
                        n2,
                        limit,
                        order=last,
                        nodes=self.nodes,
                        curvature=curvature,
                    )
                stages.append((first, last, float(limit), table))
            self._tables[key] = stages
            self.limits[key] = stages[-1][2]
            self.term_limits[key] = tuple(stage[2] for stage in stages)
        return self._tables[key]

    def __call__(self, interface, modes, transport, incident, mask, inside):
        """FresnelEncounter on transport.grid for the plane waves in modes.

        modes: PlaneWaveExpansion of the incident field in interface.minus.
        incident: the incident trace already restricted to mask.
        mask: where a field exists on the chart (stops, element entrance).
        inside: chart points on the material face (within its edge).
        """
        grid, wavelength0 = transport.grid, modes.wavelength0
        n1, n2 = interface.minus.material.n, interface.plus.material.n
        occupied = np.flatnonzero(np.any(modes.amplitudes != 0, axis=-1))
        k = modes.wavevectors[occupied]
        if len(k) and np.any(abs(k.imag) > 1e-9 * np.max(abs(k))):
            raise ValueError("order interaction requires propagating incident modes")
        k = k.real
        directions = k / np.linalg.norm(k, axis=-1, keepdims=True)
        amplitudes = modes.amplitudes[occupied]
        surface = interface.surface
        order = self.order if n1 != n2 else 0
        tabulate = order > 0 or (self.complete_curl and n1 != n2)
        stages = []
        if tabulate and isinstance(surface, SphericalCap):
            curvature = 1 if surface.radius < 0 else -1
            stages = self._stages(abs(surface.radius), wavelength0, n1, n2, curvature)
        elif tabulate and not isinstance(surface, Plane):
            raise NotImplementedError(
                "curvature orders are tabulated for plane and spherical surfaces only"
            )
        width = 0.0
        if self.critical and isinstance(surface, SphericalCap) and n1 > n2:
            size = 2 * np.pi * n1 * abs(surface.radius) / wavelength0
            width = self.critical * size ** (-2 / 3)
        points = grid.points.reshape(-1, 3)
        normals = grid.normals.reshape(-1, 3)
        values = {
            branch: np.zeros((2, len(points), 3), complex)
            for branch in ("reflected", "transmitted")
        }
        face = np.flatnonzero((mask & inside).ravel())
        presence = self.presence(incident, transport.plane)
        for first in range(0, len(face), self.chunk):
            ids = face[first : first + self.chunk]
            n = normals[ids][:, None]
            phase = np.exp(1j * (points[ids] - modes.origin) @ k.T)  # (P, M)
            local = local_fresnel(directions[None], amplitudes[None], n, n1, n2, width)
            incidence = np.arccos(np.clip(np.sum(directions[None] * n, -1), -1, 1))
            zeroth = {b: tuple(v.copy() for v in local[b]) for b in local}
            for first_term, last_term, limit, table in stages:
                if table is None:
                    continue
                weight = self.weight(incidence, limit) * presence[ids, None]
                valid = weight > 0
                if not np.any(valid):
                    continue
                p_ids, m_ids = np.nonzero(valid)
                w = weight[p_ids, m_ids, None]
                for branch in local:
                    # Table entry N is the whole field retained through order N,
                    # with H the complete curl of that truncated E; entry 0's H
                    # already carries curl(A_0)/(ik), so the first stage is taken
                    # against the plane-wave Fresnel H of the local order 0.
                    fields = table.apply(
                        directions[m_ids], amplitudes[m_ids], n[p_ids, 0], branch
                    )
                    top = min(last_term, order) if order else 0
                    for f in (0, 1):
                        below = (
                            zeroth[branch][f][p_ids, m_ids]
                            if first_term == 1
                            else fields[:, first_term - 1, f]
                        )
                        local[branch][f][p_ids, m_ids] += w * (
                            fields[:, top, f] - below
                        )
            for branch, (e, h) in local.items():
                values[branch][0, ids] = np.einsum("pm,pmj->pj", phase, e)
                values[branch][1, ids] = np.einsum("pm,pmj->pj", phase, h)
        beyond = np.flatnonzero((mask & ~inside).ravel())
        if len(beyond):
            # Flat extension at rim height: exact flat Fresnel for every mode.
            frame_normal = transport.plane.frame.normal
            flat = local_fresnel(directions, amplitudes, frame_normal, n1, n2)
            for branch, (e, h) in flat.items():
                for f, vector in enumerate((e, h)):
                    if len(modes.amplitudes) == transport.size:
                        coefficients = np.zeros((transport.size, 3), complex)
                        coefficients[occupied] = vector
                        field = transport._synthesize(
                            coefficients.reshape(transport.plane.shape + (3,))
                            * transport.size
                        ).reshape(-1, 3)
                        values[branch][f, beyond] = field[beyond]
                    else:
                        phase = np.exp(1j * (points[beyond] - modes.origin) @ k.T)
                        values[branch][f, beyond] = phase @ vector
        shape = grid.shape + (3,)

        def trace(branch):
            e, h = values[branch]
            return ElectromagneticField(
                ElectricField(grid, wavelength0, e.reshape(shape)),
                h.reshape(shape) / Z0,
            )

        return FresnelEncounter(
            grid,
            incident,
            trace("reflected"),
            trace("transmitted"),
            grid.normals,
            grid.normals,
            0.0,
        )
