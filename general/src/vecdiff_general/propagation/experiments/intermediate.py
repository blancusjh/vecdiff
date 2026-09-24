"""A sphere's field in all space, carried through intermediate surfaces.

Q carries a field between a plane and a surface. To reach the space beside a
closed body the transport is made continuous: the intermediate surface S_t is
the plane z = t outside the sphere joined to the part of the sphere beyond
that plane. As t runs from one pole to the other, S_t moves continuously from
the sphere to a plane, and the family sweeps the whole space around it.

One step S_t -> S_{t+dt} has three parts:

* on the plane, the exact homogeneous step (ASM) over dt;
* inside the sphere section the family carries nothing: light that leaves the
  sphere outward does not re-enter it, so the field of the surface already
  passed vanishes there;
* next to the surface, where the plane meets the sphere, every outgoing plane
  wave of the interface interaction is carried from its surface point along
  its own wavevector (the local step), and replaces the plane field in a
  shell a fraction of a wavelength thick, blending into it smoothly.

Light going up (Re k_z >= 0) is marched upward, light going down downward.
The planes are the rows of the field map, so the plane is filled everywhere.

The sphere is iterated as in ``returns.ElementReturns`` (the operational
Debye series), but on the closed surface: an internal state is a set of plane
waves in all directions, and after each encounter it is the least-squares fit
(Q^-1) of the inward trace in which every wave is matched only where it leaves
the surface into the sphere (d . n < 0) - where it arrives, it belongs to the
next encounter. Light that enters near grazing and runs round the rim is thus
carried, with no assumption on the order in which it meets the surface.
Encounter 0 is the incident beam on the sphere: waves arriving at a point
(d . n < 0) are reflected and transmitted; waves leaving it outward (d . n > 0)
are blocked, which is the term -incident (the shadow). At grazing incidence the
reflected wave tends to -incident (r -> -1), so the two join continuously.
Encounter p >= 1 is internal state p meeting the whole surface; what it
transmits leaves the sphere.

Limits, reported rather than hidden: the interaction is local, so light that
tunnels from beyond the geometric edge into modes trapped by total internal
reflection (whispering-gallery resonances) is absent; the local step takes each
wave's amplitude at the nearest surface sample and carries it with a plane
wavefront; the iteration is truncated after a given number of passes; the
lattice is periodic (an absorber removes what reaches its edge).
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.fft
from scipy.sparse.linalg import LinearOperator, lsmr
from scipy.spatial import cKDTree

from ...fields.spectrum import PlaneWaveExpansion
from .orders import local_fresnel


def sphere_samples(radius, spacing):
    """Points, outward normals and areas on a sphere centred at the origin.

    Rings of equal polar step, each with points about ``spacing`` apart; the
    number of rings is even, so no ring lies on the equator. Points are
    ordered from the +z pole to the -z pole.
    """
    rings = 2 * int(np.ceil(np.pi * radius / spacing / 2))
    step = np.pi / rings
    normals, areas = [], []
    for j in range(rings):
        theta = (j + 0.5) * step
        count = max(3, int(round(2 * np.pi * radius * np.sin(theta) / spacing)))
        phi = (np.arange(count) + 0.5 * (j % 2)) * 2 * np.pi / count
        band = 2 * np.pi * radius**2 * (np.cos(theta - step / 2) - np.cos(theta + step / 2))
        normals.append(
            np.stack(
                (
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.full(count, np.cos(theta)),
                ),
                -1,
            )
        )
        areas.append(np.full(count, band / count))
    normals = np.concatenate(normals)
    return radius * normals, normals, np.concatenate(areas)


def sphere_directions(count):
    """Quasi-uniform unit vectors (Fibonacci lattice) and two transverse ones.

    Returns d (M, 3) and basis (M, 2, 3), with basis[:, 0] along z x d
    (x where d is on the axis) and basis[:, 1] = d x basis[:, 0].
    """
    j = np.arange(count) + 0.5
    z = 1 - 2 * j / count
    phi = np.pi * (1 + np.sqrt(5)) * j
    s = np.sqrt(1 - z * z)
    d = np.stack((s * np.cos(phi), s * np.sin(phi), z), -1)
    first = np.cross([0.0, 0.0, 1.0], d)
    size = np.linalg.norm(first, axis=-1, keepdims=True)
    first = np.where(size > 1e-9, first / np.maximum(size, 1e-300), [1.0, 0.0, 0.0])
    return d, np.stack((first, np.cross(d, first)), 1)


@dataclass(frozen=True)
class FitReport:
    """Relative L2(dS) residual of the tangential E fit, and LSMR data."""

    relative_residual: float
    iterations: int
    condition_estimate: float


class ClosedTransport:
    """Q and Q^-1 between plane waves in all directions and a closed surface.

    The field inside is sum_j A_j exp(i k d_j . (r - origin)), with A_j
    transverse to d_j. A wave belongs to the trace generated at a surface
    point only where it leaves the surface inward (d_j . n < 0, n outward);
    where it arrives it was generated elsewhere. Q^-1 therefore fits, at every
    point, the waves leaving it to the tangential E of the trace, in L2(dS)
    (weights: point areas), with LSMR as ``operators.SurfaceTransport`` does.
    """

    def __init__(self, count, region, wavelength0, points, normals, weights, origin):
        self.region, self.wavelength0 = region, wavelength0
        self.origin = np.asarray(origin, float)
        directions, self.basis = sphere_directions(count)
        self.wavevectors = 2 * np.pi * region.material.n / wavelength0 * directions
        self.points, self.normals = np.asarray(points, float), np.asarray(normals, float)
        self.root = np.sqrt(np.asarray(weights, float))
        # Phases of the waves leaving each point; the others are zero here.
        self.matrix = np.exp(1j * (self.points - self.origin) @ self.wavevectors.T)
        self.matrix *= self.normals @ directions.T < 0

    def _tangential(self, values):
        return values - self.normals * np.sum(self.normals * values, -1)[:, None]

    def expansion(self, amplitudes):
        return PlaneWaveExpansion(
            self.wavevectors, amplitudes, self.origin, self.wavelength0, self.region
        )

    def inverse(self, target, *, tolerance=1e-9, max_iterations=1000):
        """Plane waves whose leaving parts fit the tangential ``target`` (P, 3)."""
        modes, count = len(self.wavevectors), len(self.points)

        def matvec(vector):
            amplitudes = np.einsum("mi,mij->mj", vector.reshape(modes, 2), self.basis)
            values = self.matrix @ amplitudes
            return (self.root[:, None] * self._tangential(values)).ravel()

        def rmatvec(vector):
            values = self._tangential(self.root[:, None] * vector.reshape(count, 3))
            b = self.matrix.conj().T @ values
            return np.einsum("mj,mij->mi", b, self.basis).ravel()

        operator = LinearOperator(
            (3 * count, 2 * modes), matvec=matvec, rmatvec=rmatvec, dtype=complex
        )
        right = (self.root[:, None] * self._tangential(target)).ravel()
        result = lsmr(operator, right, atol=tolerance, btol=tolerance, maxiter=max_iterations)
        amplitudes = np.einsum("mi,mij->mj", result[0].reshape(modes, 2), self.basis)
        residual = np.linalg.norm(matvec(result[0]) - right) / max(np.linalg.norm(right), 1e-300)
        return self.expansion(amplitudes), FitReport(
            float(residual), int(result[2]), float(result[6])
        )


def local_fields(interaction, stages, order, directions, amplitudes, normals, n1, n2, presence):
    """Per-mode local fields of ``OrderInteraction`` at points, unit phase.

    The same order-0 Fresnel interaction and graded curvature corrections as
    ``OrderInteraction.__call__``, but kept per (point, mode) instead of summed,
    so that each outgoing wave can be carried along its own wavevector.
    Returns {"reflected"|"transmitted": (E, Z0 H)}, each (P, M, 3).
    """
    n = normals[:, None]
    local = local_fresnel(directions[None], amplitudes[None], n, n1, n2)
    incidence = np.arccos(np.clip(np.sum(directions[None] * n, -1), -1, 1))
    zeroth = {b: tuple(v.copy() for v in local[b]) for b in local}
    for first_term, last_term, limit, table in stages:
        if table is None:
            continue
        weight = interaction.weight(incidence, limit) * presence[:, None]
        valid = weight > 0
        if not np.any(valid):
            continue
        p_ids, m_ids = np.nonzero(valid)
        w = weight[p_ids, m_ids, None]
        for branch in local:
            fields = table.apply(directions[m_ids], amplitudes[m_ids], n[p_ids, 0], branch)
            top = min(last_term, order) if order else 0
            for f in (0, 1):
                below = (
                    zeroth[branch][f][p_ids, m_ids]
                    if first_term == 1
                    else fields[:, first_term - 1, f]
                )
                local[branch][f][p_ids, m_ids] += w * (fields[:, top, f] - below)
    return local


def outgoing_wavevectors(directions, normals, n1, n2, wavelength0, branch):
    """Order-0 wavevectors (P, M, 3) of the reflected or transmitted waves.

    normals (P, 3) point into n2. Transmitted beyond the critical angle the
    normal component is imaginary and positive (a wave decaying into n2).
    """
    k0 = 2 * np.pi / wavelength0
    n = normals[:, None]
    c1 = np.sum(directions[None] * n, -1)[..., None]
    if branch == "reflected":
        return n1 * k0 * (directions[None] - 2 * c1 * n)
    c2 = np.sqrt(1 - (n1 / n2) ** 2 * (1 - c1**2) + 0j)
    return n2 * k0 * ((n1 / n2) * (directions[None] - c1 * n) + c2 * n)


def _smooth_step(s):
    """1 at s <= 0, 0 at s >= 1, infinitely smooth between."""
    s = np.clip(s, 0, 1)
    with np.errstate(divide="ignore", over="ignore"):
        rise = np.where(s > 0, np.exp(-1 / s), 0.0)
        fall = np.where(s < 1, np.exp(-1 / (1 - s)), 0.0)
    return fall / (rise + fall)


class PlaneMarch:
    """The intermediate surfaces: planes z[k] over a periodic square lattice.

    On every plane, the lattice points of the band around the sphere,
    shell[0] < h < shell[3] with h the distance outside the sphere, take the
    local step of the surface waves (carried outward for h > 0, and inward
    for h < 0, where they continue the outgoing waves so that the plane data
    has no jump at the surface). The weight is one on [shell[1], shell[2]]
    and falls smoothly to zero at both ends, where the marched plane field
    takes over. ``half`` is the lattice half width; beyond ``absorber`` * half
    the field is absorbed. ``march`` returns the y = 0 row of every plane.
    """

    def __init__(
        self,
        z,
        half,
        step,
        center,
        radius,
        wavelength0,
        index=1.0,
        *,
        shell=(-0.25, -0.1, 0.05, 0.15),
        absorber=(0.5, 0.97),
    ):
        self.z = np.asarray(z, float)
        dz = np.diff(self.z)
        if not np.allclose(dz, dz[0]):
            raise ValueError("planes must be equally spaced")
        self.dz = float(dz[0])
        count = 2 * int(round(half / step))
        self.x = (np.arange(count) - count // 2) * step
        self.count, self.center, self.radius = count, np.asarray(center, float), radius
        k = 2 * np.pi * index / wavelength0
        f = 2 * np.pi * np.fft.fftfreq(count, step)
        kz = np.sqrt(k**2 - f[:, None] ** 2 - f[None] ** 2 + 0j)
        self.propagator = np.exp(1j * kz * self.dz)
        a0, a1 = absorber[0] * half, absorber[1] * half
        profile = _smooth_step((abs(self.x) - a0) / (a1 - a0))
        self.absorber = profile[:, None] * profile[None]
        self.shell = shell = tuple(float(s) * wavelength0 for s in shell)
        indices, weights, slices, points, depth = [], [], [], [], []
        start = 0
        for zk in self.z:
            dzc = zk - self.center[2]
            reach = (radius + shell[3]) ** 2 - dzc**2
            if reach <= 0:
                indices.append(np.zeros(0, int))
                weights.append(np.zeros(0))
                slices.append(slice(start, start))
                continue
            near_x = np.flatnonzero(abs(self.x - self.center[0]) <= np.sqrt(reach))
            near_y = np.flatnonzero(abs(self.x - self.center[1]) <= np.sqrt(reach))
            iy, ix = np.meshgrid(near_y, near_x, indexing="ij")
            iy, ix = iy.ravel(), ix.ravel()
            xy = np.stack((self.x[ix], self.x[iy]), -1)
            h = np.sqrt(np.sum((xy - self.center[:2]) ** 2, -1) + dzc**2) - radius
            band = (h > shell[0]) & (h < shell[3])
            h = h[band]
            indices.append((iy * count + ix)[band])
            weights.append(
                np.where(
                    h < shell[1],
                    _smooth_step((shell[1] - h) / (shell[1] - shell[0])),
                    _smooth_step((h - shell[2]) / (shell[3] - shell[2])),
                )
            )
            slices.append(slice(start, start + len(h)))
            start += len(h)
            points.append(np.column_stack((xy[band], np.full(len(h), zk))))
            depth.append(h)
        self.indices, self.weights, self.slices = indices, weights, slices
        self.shell_points = np.concatenate(points) if points else np.zeros((0, 3))
        self.shell_depth = np.concatenate(depth) if depth else np.zeros(0)

    def march(self, values, direction, *, workers=-1):
        """Rows y = 0 of every plane for band values (S, 3) going up (+1) or down."""
        count = self.count
        u = np.zeros((count, count, 3), complex)
        rows = np.zeros((len(self.z), count, 3), complex)
        order = range(len(self.z)) if direction > 0 else range(len(self.z) - 1, -1, -1)
        started = False
        for k in order:
            if started:
                u = scipy.fft.ifft2(
                    scipy.fft.fft2(u, axes=(0, 1), workers=workers)
                    * self.propagator[..., None],
                    axes=(0, 1),
                    workers=workers,
                )
                u *= self.absorber[..., None]
            if len(self.indices[k]):
                started = True
                flat = u.reshape(-1, 3)
                w = self.weights[k][:, None]
                ids = self.indices[k]
                flat[ids] = w * values[self.slices[k]] + (1 - w) * flat[ids]
            if started:
                rows[k] = u[count // 2]
        return rows


@dataclass(frozen=True)
class IntermediateSolution:
    """One run: internal states and what leaves the sphere, per encounter.

    internal[p - 1] is internal state p (a ``PlaneWaveExpansion`` in the
    sphere). emissions[p] = (up, down): the local step of encounter p's
    outgoing waves at the march's band points, (S, 3) each, for p = 0 (with
    the shadow -incident) up to ``terms``; emissions["rest"] is that of the
    later states together. reports: the Q^-1 fit of every internal state.
    """

    internal: tuple
    emissions: dict
    reports: tuple
    seconds: float


class IntermediateSphere:
    """A dielectric sphere under a beam, with every field carried by our operators.

    beam: PlaneWaveExpansion of the incident field in ``outside``.
    center, radius: the sphere, of material ``inside``. interaction: an
    ``OrderInteraction`` (its order, graded corrections and presence window
    are used as in its own call). spacing: distance between surface samples;
    directions: number of plane-wave directions of an internal state.
    """

    def __init__(
        self,
        beam,
        center,
        radius,
        inside,
        outside,
        interaction,
        *,
        spacing=1 / 6,
        directions=7000,
        inverse_tolerance=1e-9,
        max_iterations=1000,
        chunk=32,
    ):
        self.beam, self.interaction = beam, interaction
        self.center, self.radius = np.asarray(center, float), float(radius)
        self.inside, self.outside = inside, outside
        self.wavelength0 = beam.wavelength0
        self.chunk = chunk
        self.inverse = dict(tolerance=inverse_tolerance, max_iterations=max_iterations)
        points, normals, areas = sphere_samples(radius, spacing * beam.wavelength0)
        self.points, self.normals, self.areas = points + self.center, normals, areas
        self.transport = ClosedTransport(
            directions, inside, self.wavelength0, self.points, normals, areas, self.center
        )
        self.tree = cKDTree(self.points)

    def _presence(self, intensity):
        """``OrderInteraction.presence`` with its smoothing done on the sphere."""
        interaction = self.interaction
        if interaction.window is None or not intensity.max():
            return np.ones_like(intensity)
        if interaction.envelope:
            width = interaction.envelope * self.wavelength0
            if getattr(self, "_pairs", None) is None:
                self._pairs = self.tree.sparse_distance_matrix(
                    self.tree, 3 * width, output_type="ndarray"
                )
            pairs, areas = self._pairs, self.areas
            i, j = pairs["i"], pairs["j"]
            g = np.exp(-(pairs["v"] ** 2) / (2 * width**2)) * areas[j]
            off = i != j
            numerator = np.bincount(i[off], g[off] * intensity[j[off]], len(intensity))
            denominator = np.bincount(i[off], g[off], len(intensity))
            intensity = (numerator + areas * intensity) / (denominator + areas)
        size = np.sqrt(intensity)
        level = np.log(np.maximum(size / size.max(), 1e-300))
        low, high = np.log(interaction.window[0]), np.log(interaction.window[1])
        return 1 - _smooth_step((level - low) / (high - low))

    def _encounter(self, modes, n1, n2, into, curvature, fit, march=None, shadow=False):
        """One encounter on the whole surface: inward trace and outward local steps.

        into: +1 if the normals into n2 are the outward sphere normals, -1 if
        they point into the sphere. fit: the branch that stays inside
        ("transmitted" at the entrance, "reflected" otherwise); the other
        leaves, and with a ``march`` its local steps to the band points are
        returned, split by direction. shadow: add -incident for waves leaving
        the surface outward. Returns (inward E trace (P, 3), up, down).
        """
        interaction, wavelength0 = self.interaction, self.wavelength0
        k = modes.wavevectors.real
        directions = k / np.linalg.norm(k, axis=-1, keepdims=True)
        amplitudes = modes.amplitudes
        order = interaction.order if n1 != n2 else 0
        stages = []
        if order > 0:
            stages = interaction._stages(self.radius, wavelength0, n1, n2, curvature)
        leave = "reflected" if fit == "transmitted" else "transmitted"
        # Incident intensity at the samples, for the presence window.
        intensity = np.zeros(len(self.points))
        for first in range(0, len(self.points), 256):
            phase = np.exp(1j * (self.points[first : first + 256] - modes.origin) @ k.T)
            intensity[first : first + 256] = np.sum(abs(phase @ amplitudes) ** 2, -1)
        presence = self._presence(intensity)
        trace = np.zeros((len(self.points), 3), complex)
        up = down = None
        if march is not None:
            up = np.zeros((len(march.shell_points), 3), complex)
            down = np.zeros_like(up)
            foot = self._feet(march)
            order_by_foot = np.argsort(foot, kind="stable")
            sorted_foot = foot[order_by_foot]
        for first in range(0, len(self.points), self.chunk):
            block = np.arange(first, min(first + self.chunk, len(self.points)))
            p = self.points[block]
            n = into * self.normals[block]
            phase = np.exp(1j * (p - modes.origin) @ k.T)  # (P, M)
            arriving = n @ directions.T > 0  # (P, M)
            local = local_fields(
                interaction, stages, order, directions, amplitudes, n, n1, n2, presence[block]
            )
            e_in = local[fit][0] * arriving[..., None]
            trace[block] = np.einsum("pm,pmj->pj", phase, e_in)
            if march is None:
                continue
            # Band points whose nearest surface sample is in this block.
            lo, hi = np.searchsorted(sorted_foot, [block[0], block[-1] + 1])
            if hi == lo:
                continue
            shell = order_by_foot[lo:hi]
            where = foot[shell] - block[0]
            delta = march.shell_points[shell] - p[where]
            k_out = outgoing_wavevectors(directions, n, n1, n2, wavelength0, leave)[where]
            real = k_out.real
            step = np.exp(1j * np.sum(real * delta[:, None], -1)) * np.exp(
                -abs(np.sum(k_out.imag * delta[:, None], -1))
            )
            weight = phase[where] * step * arriving[where]
            e_out = local[leave][0][where]
            rising = real[..., 2] >= 0
            up[shell] += np.einsum("sm,smj->sj", weight * rising, e_out)
            down[shell] += np.einsum("sm,smj->sj", weight * ~rising, e_out)
            if shadow:
                blocked = ~arriving[where]
                wave = np.exp(1j * (march.shell_points[shell] - modes.origin) @ k.T) * blocked
                rising = k[:, 2] >= 0
                up[shell] -= (wave * rising) @ amplitudes
                down[shell] -= (wave * ~rising) @ amplitudes
        return trace, up, down

    def _feet(self, march):
        """Nearest surface sample of every band point (radial projection)."""
        key = id(march)
        if getattr(self, "_feet_key", None) != key:
            relative = march.shell_points - self.center
            radial = relative / np.linalg.norm(relative, axis=-1, keepdims=True)
            _, self._feet_cache = self.tree.query(self.center + self.radius * radial)
            self._feet_key = key
        return self._feet_cache

    def run(self, passes, march, *, terms=3, verbose=False):
        """Internal states 1..passes and the emissions (see IntermediateSolution)."""
        start = perf_counter()
        n_out, n_in = self.outside.material.n, self.inside.material.n

        def report(text):
            if verbose:
                print(f"    {text}: {perf_counter() - start:.0f} s", flush=True)

        trace, up, down = self._encounter(
            self.beam, n_out, n_in, -1, -1, "transmitted", march, shadow=True
        )
        emissions = {0: (up, down)}
        state, fit = self.transport.inverse(trace, **self.inverse)
        internal, reports = [state], [fit]
        report(f"encounter 0, fit 1 (residual {fit.relative_residual:.2e})")
        for p in range(1, passes + 1):
            emit = p <= terms
            if not emit and p == passes:
                break
            trace, up, down = self._encounter(
                state, n_in, n_out, 1, 1, "reflected", march if emit else None
            )
            if emit:
                emissions[p] = (up, down)
            if p < passes:
                state, fit = self.transport.inverse(trace, **self.inverse)
                internal.append(state)
                reports.append(fit)
                report(f"encounter {p}, fit {p + 1} (residual {fit.relative_residual:.2e})")
        if passes > terms:
            rest = self.transport.expansion(
                sum(s.amplitudes for s in internal[terms:])
            )
            _, up, down = self._encounter(rest, n_in, n_out, 1, 1, "reflected", march)
            emissions["rest"] = (up, down)
            report(f"encounters {terms + 1}-{passes} together")
        return IntermediateSolution(
            tuple(internal), emissions, tuple(reports), perf_counter() - start
        )
