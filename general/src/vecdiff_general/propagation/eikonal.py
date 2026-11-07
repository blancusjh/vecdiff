"""Eikonal transport of a field across one curved interface between reference planes.

The plane-surface-plane map of the eikonal-angular-spectrum formulation: a
point P of the input reference plane follows its carrier ray to the interface
Sigma, refracts by the vector Snell law and is extrapolated to P' on the same
geometric plane in the transmitted medium. Signed legs l_i and l_t give the
optical path increment Delta S = n_i l_i + n_t l_t, and the field is carried by

    E'(Phi(rho)) = g exp(i k0 Delta S) [t_s E_s s + t_p E_p p_t + t_p E_par s_t],

with g = sqrt(s_iz c_t / (c_i s_tz J)) and J = det D Phi. The Jacobian follows
analytically from the surface Hessian and the carrier's direction Jacobian;
no spherical focus is assumed. The output lattice is filled through the
inverse map, so the incident field is evaluated exactly at the preimages and
never interpolated. The transmitted plane components are re-expanded in the
transmitted medium, which restores the longitudinal component spectrally.

Rays that miss the surface, meet it from behind, are totally reflected, leave
backward or fold the map carry no field; the incident power they hold is
reported, not hidden. The transfer is the stated leading-order model: it is
exact for a plane wave on a plane interface and the identity for index-matched
media, not a solution of the curved-boundary Maxwell problem.
"""

import math
from dataclasses import dataclass, fields, replace

import numpy as np

from vecdiff_general._backend import bd, divide, to_numpy
from vecdiff_general.fields import TransverseElectricField
from vecdiff_general.fields.angular_spectrum import angular_spectrum
from vecdiff_general.fields.spectrum import PlaneSpectrum
from vecdiff_general.propagation._lattice import (
    READ,
    LatticeSpectrum,
    PlaneReader,
    default_block,
    in_blocks,
    lattice_points,
    lattice_spectrum,
    plane_blocks,
    row_blocks,
)
from vecdiff_general.sampling import CartesianGrid
from vecdiff_general.surfaces.surface import ImplicitSurface

_EPS = np.finfo(float).eps
# Transverse band of the lattice leg's re-expansion, in units of n k0.
_SPECTRAL_CUTOFF = 0.995


def _masked(function, xy):
    """function(xy) on (N, 2) points with NaN where the surface refuses them."""
    try:
        return function(xy)
    except ValueError:
        if len(xy) == 1:
            return None

    half = len(xy) // 2
    pieces = (xy[:half], xy[half:])
    parts = [_masked(function, piece) for piece in pieces]
    present = [part for part in parts if part is not None]

    if not present:
        return None

    filled = [
        part if part is not None else bd.full((len(x),) + present[0].shape[1:], bd.nan)
        for part, x in zip(parts, pieces)
    ]

    return bd.concatenate(filled)


def _surface(surface, name, xy, tail):
    """surface.name(xy) inside the surface's domain, NaN outside."""
    out = bd.full(xy.shape[:-1] + tail, bd.nan)
    inside = bd.isfinite(xy).all(axis=-1)
    inside[inside] = surface.domain(xy[inside])
    if bool(inside.any()):
        values = _masked(getattr(surface, name), xy[inside])
        if values is not None:
            out[inside] = values

    return out


@dataclass(frozen=True)
class RayMap:
    """Plane-surface-plane map quantities of labelled rays, surface-local axes.

    A ray is labelled by its point on the first reference plane of its
    family; for a carrier that is the point P on this map's own input plane.
    ``start`` is P (where the incident field is read), ``output`` Phi(P),
    ``path`` Delta S in metres, ``flux`` the factor g. ``jacobian`` is
    d output / d label with output components first, ``start_jacobian`` the
    same for P (the identity for a carrier). ``surface_point`` is Q and
    ``surface_derivative``, ``transmitted_derivative`` are dQ / d label and
    d s_t / d label, shape (..., 3, 2), so the refracted rays can be carried
    on (Appendix B); ``label`` is the label itself. Directions are unit vectors
    in the surface frame;
    ``valid`` marks rays the chart supports.
    """

    valid: np.ndarray
    output: np.ndarray
    path: np.ndarray
    flux: np.ndarray
    jacobian: np.ndarray
    incident: np.ndarray
    transmitted: np.ndarray
    normal: np.ndarray
    cos_incident: np.ndarray
    cos_transmitted: np.ndarray
    surface_height: np.ndarray
    start: np.ndarray
    start_jacobian: np.ndarray
    surface_point: np.ndarray
    surface_derivative: np.ndarray
    transmitted_derivative: np.ndarray
    label: np.ndarray

    def subset(self, mask):
        """The same quantities at the points selected by ``mask``."""
        return RayMap(*(getattr(self, f.name)[mask] for f in fields(self)))


class InterfaceMap:
    """Phi, Delta S and g for one surface, carrier and pair of indices.

    ``offset`` places the reference plane along the surface normal from the
    vertex, in metres (0 is the vertex plane; positive lies in the direction
    of propagation). ``carrier`` is a fields.Carrier, whose rays are labelled
    by their surface-local points on this plane (..., 2), or a RefractedRays
    family from the previous surface, labelled on its first plane.
    """

    def __init__(self, surface, carrier, n_incident, n_transmitted, offset=0.0):
        if not isinstance(carrier, RefractedRays) and not all(
            callable(getattr(carrier, name, None))
            for name in ("directions", "jacobians")
        ):
            raise TypeError("eikonal transport needs a fields.Carrier")

        self.surface, self.carrier = surface, carrier
        self.n_incident, self.n_transmitted = float(n_incident), float(n_transmitted)
        self.offset = float(offset)
        self.basis = surface.frame.basis
        surface.hessian(bd.zeros((1, 2)))  # fail early when no Hessian is declared

    def _global(self, local):
        return self.surface.frame.origin + local @ self.basis.T

    @property
    def family(self):
        """The label lattice's map for a refracted family, None for a carrier."""
        return self.carrier if isinstance(self.carrier, RefractedRays) else None

    def forward(self, xy):
        """RayMap of the rays labelled xy (..., 2): plane points for a carrier."""
        xy = bd.asarray(xy, float)
        shape = xy.shape[:-1]
        xy = xy.reshape(-1, 2)
        finite = bd.isfinite(xy).all(axis=-1)
        xy = bd.where(finite[:, None], xy, 0.0)

        if self.family is not None:
            p, s, dp, ds, alive = self.family.rays(xy, self.surface.frame, self.offset)
            finite = finite & alive
        else:
            p = bd.concatenate((xy, bd.full((len(xy), 1), self.offset)), axis=-1)
            world = self._global(p)
            s = self.carrier.directions(world) @ self.basis
            jacobian = bd.einsum(
                "ji,njk,kl->nil", self.basis, self.carrier.jacobians(world), self.basis
            )
            ds = jacobian[:, :, :2]
            dp = bd.zeros((len(xy), 3, 2))
            dp[:, 0, 0] = dp[:, 1, 1] = 1.0

        result = list(self._trace(bd.where(finite[:, None], p, 0.0), s, dp, ds))
        result[0] = result[0] & finite
        result[11] = bd.where(finite[:, None], result[11], bd.nan)  # start
        result.append(xy)

        return RayMap(*(value.reshape(shape + value.shape[1:]) for value in result))

    def _intersect(self, p, s):
        """Signed incident leg to the surface: a starting leg, then Newton on the graph.

        Starts are tried in turn for the rays Newton has not yet found: the
        height under P where the graph has one there, the crossing with the
        osculating sphere at the vertex (the surface itself for a cap), and a
        march along the ray. Behind the surface, as from a rim-depth plane, the
        height under P can start Newton off the graph's domain.
        """
        height = _surface(self.surface, "height", p[:, :2], ())
        leg = (height - p[:, 2]) / s[:, 2]
        scale = bd.linalg.norm(p, axis=-1) + 1e-300
        leg = self._newton(p, s, leg, scale)

        for start in ("osculating", "march"):
            retry = bd.flatnonzero(~bd.isfinite(leg))
            if not len(retry):
                break
            if start == "osculating":
                first = self._osculating(p[retry], s[retry])
            else:
                reach = 4 * bd.maximum(scale[retry], abs(self.offset))
                first = self._bracket(p[retry], s[retry], reach)
            leg[retry] = self._newton(p[retry], s[retry], first, scale[retry])

        return leg

    def _osculating(self, p, s):
        """Crossing with the sphere osculating the graph at the vertex (mean curvature)."""
        apex = bd.zeros((1, 2))
        h0 = float(_surface(self.surface, "height", apex, ())[0])
        curvature = (
            float(bd.trace(_surface(self.surface, "hessian", apex, (2, 2))[0])) / 2
        )

        if not np.isfinite(curvature) or abs(curvature) < 1e-300:
            return (h0 - p[:, 2]) / s[:, 2]

        center = bd.asarray([0.0, 0.0, h0 + 1 / curvature])
        offset = p - center
        b = bd.sum(s * offset, axis=-1)
        d = bd.sum(offset**2, axis=-1) - 1 / curvature**2
        root = bd.sqrt(bd.where(b * b - d >= 0, b * b - d, bd.nan))
        # The vertex side of the sphere: its near crossing when convex toward
        # the incident side (curvature > 0), its far crossing otherwise.
        return -b - root if curvature > 0 else -b + root

    def _newton(self, p, s, leg, scale):
        leg = bd.array(leg, dtype=float)
        active = bd.isfinite(leg)

        for _ in range(60):
            if not bool(active.any()):
                break

            q = p[active] + leg[active, None] * s[active]
            residual = q[:, 2] - _surface(self.surface, "height", q[:, :2], ())
            slope = s[active, 2] - bd.sum(
                _surface(self.surface, "gradient", q[:, :2], (2,)) * s[active, :2],
                axis=-1,
            )
            step = residual / slope
            leg[active] = leg[active] - step
            done = ~bd.isfinite(step) | (
                bd.abs(step) <= 8 * _EPS * (bd.abs(leg[active]) + scale[active])
            )
            index = bd.flatnonzero(active)
            leg[index[~bd.isfinite(step)]] = bd.nan
            active[index[done]] = False

        leg[active] = bd.nan

        return leg

    def _bracket(self, p, s, reach, steps=48):
        """Starting legs where the graph has no height under P: march along the ray.

        The first sign change of z - h(x, y), forward then backward within each
        ray's ``reach``, brackets the crossing; linear interpolation starts
        Newton. All march steps are evaluated in one surface call per direction.
        """
        start = bd.full(len(p), bd.nan)

        for direction in (1.0, -1.0):
            todo = bd.flatnonzero(bd.isnan(start))
            if len(todo) == 0:
                break
            legs = (
                direction * reach[todo][None] * (bd.arange(steps + 1) / steps)[:, None]
            )
            q = p[todo][None] + legs[..., None] * s[todo][None]
            height = _surface(self.surface, "height", q[..., :2].reshape(-1, 2), ())
            f = q[..., 2] - height.reshape(q.shape[:2])
            change = (
                bd.isfinite(f[:-1])
                & bd.isfinite(f[1:])
                & (bd.sign(f[:-1]) != bd.sign(f[1:]))
            )
            first = bd.argmax(change, axis=0)
            found = change.any(axis=0)
            column = bd.arange(len(todo))
            f0, f1 = f[first, column], f[first + 1, column]
            l0, l1 = legs[first, column], legs[first + 1, column]
            guess = l0 + (l1 - l0) * f0 / bd.where(f0 != f1, f0 - f1, 1.0)
            start[todo[found]] = guess[found]

        return start

    def _trace(self, p, s, dp, ds):
        """Rays from points p with directions s; dp, ds their (3, 2) label derivatives."""
        n_i, n_t = self.n_incident, self.n_transmitted
        eta = n_i / n_t
        if isinstance(self.surface, ImplicitSurface):
            # The whole surface F = 0, beyond its graph chart.
            leg_i = self.surface.crossing(p, s)
            q = p + leg_i[:, None] * s
            _, grad_f, hess_f = self.surface.implicit(q)
        else:
            # F = z - h(x, y) on the graph.
            leg_i = self._intersect(p, s)
            q = p + leg_i[:, None] * s
            slope = _surface(self.surface, "gradient", q[:, :2], (2,))
            curvature = _surface(self.surface, "hessian", q[:, :2], (2, 2))
            grad_f = bd.concatenate((-slope, bd.ones((len(q), 1))), axis=-1)
            hess_f = bd.zeros((len(q), 3, 3))
            hess_f[:, :2, :2] = -curvature
        size = bd.linalg.norm(grad_f, axis=-1)
        normal = grad_f / size[:, None]
        c_i = bd.sum(s * normal, axis=-1)
        tangent = eta * (s - c_i[:, None] * normal)
        radicand = 1 - bd.sum(tangent**2, axis=-1)
        c_t = bd.sqrt(bd.where(radicand > 0, radicand, bd.nan))
        s_t = tangent + c_t[:, None] * normal
        valid = (
            bd.isfinite(leg_i)
            & bd.isfinite(size)
            & (s[:, 2] > 0)
            & (c_i > 1e-12)
            & (radicand > 1e-24)
            & (s_t[:, 2] > 1e-12)
        )
        leg_t = (self.offset - q[:, 2]) / s_t[:, 2]
        out = q + leg_t[:, None] * s_t
        path = n_i * leg_i + n_t * leg_t

        # Appendix A: derivatives along the labels a = x, y (axis 1).
        ds_a = bd.swapaxes(ds, 1, 2)  # (n, a, component)
        w = bd.swapaxes(dp, 1, 2) + leg_i[:, None, None] * ds_a
        w_n = bd.sum(w * normal[:, None], axis=-1)
        dq = w - s[:, None] * (w_n / c_i[:, None])[..., None]
        t = bd.einsum("nij,naj->nai", hess_f, dq)
        dn = (
            t - normal[:, None] * bd.sum(t * normal[:, None], axis=-1)[..., None]
        ) / size[:, None, None]
        dc_i = bd.sum(ds_a * normal[:, None], axis=-1) + bd.sum(
            s[:, None] * dn, axis=-1
        )
        dc_t = eta**2 * (c_i / c_t)[:, None] * dc_i
        ds_t = (
            eta * ds_a
            + (c_t - eta * c_i)[:, None, None] * dn
            + (dc_t - eta * dc_i)[..., None] * normal[:, None]
        )
        moved = dq + leg_t[:, None, None] * ds_t
        dp_out = moved - s_t[:, None] * (moved[..., 2] / s_t[:, 2:3])[..., None]
        jacobian = bd.swapaxes(dp_out[..., :2], 1, 2)
        det = (
            jacobian[:, 0, 0] * jacobian[:, 1, 1]
            - jacobian[:, 0, 1] * jacobian[:, 1, 0]
        )
        # Area ratio between the planes: per label, output over input.
        det_in = dp[:, 0, 0] * dp[:, 1, 1] - dp[:, 0, 1] * dp[:, 1, 0]
        ratio = divide(det, det_in, det_in != 0)
        valid &= bd.isfinite(ratio) & (det_in > 0) & (ratio > 0)
        flux = bd.sqrt(divide(s[:, 2] * c_t, c_i * s_t[:, 2] * ratio, valid))

        return (
            valid,
            out[:, :2],
            path,
            bd.where(valid, flux, 0.0),
            jacobian,
            s,
            s_t,
            normal,
            c_i,
            c_t,
            q[:, 2],
            p[:, :2],
            dp[:, :2],
            q,
            bd.swapaxes(dq, 1, 2),
            bd.swapaxes(ds_t, 1, 2),
        )

    def inverse(self, targets, samples, *, tolerance=1e-12):
        """Preimages on the input plane of output points, and where they exist.

        ``samples`` (M, 2) seed a forward table; each target starts from the
        linear interpolation of that table and finishes by damped Newton steps
        with the analytic D Phi. Targets outside the table's image or without a
        converged preimage are marked missing.
        """
        targets = bd.asarray(targets, float).reshape(-1, 2)
        scale = float(bd.max(bd.abs(targets))) + 1e-300

        return self.invert(targets, self.seed_table(samples), tolerance * scale)

    def seed_table(self, samples):
        """The forward table of seed labels that starts every inversion."""
        samples = bd.asarray(samples, float).reshape(-1, 2)
        table = self.forward(samples)
        keep = to_numpy(table.valid)

        return _SeedTable(to_numpy(table.output)[keep], to_numpy(samples)[keep])

    def invert(self, targets, table, limit):
        """inverse() of ``targets`` from a seed table, converged to ``limit`` in metres.

        Each target is inverted on its own, so blocks of targets give the
        preimages of the whole lattice.
        """
        targets = bd.asarray(targets, float).reshape(-1, 2)
        guess = table.guess(to_numpy(targets))
        found = bd.asarray(np.isfinite(guess).all(axis=-1))
        rho = bd.asarray(np.nan_to_num(guess))
        residual = bd.full(len(targets), bd.inf)
        active = found.copy()

        for _ in range(40):
            index = bd.flatnonzero(active)
            if len(index) == 0:
                break
            m = self.forward(rho[index])
            error = m.output - targets[index]
            norm = bd.where(m.valid, bd.linalg.norm(error, axis=-1), bd.inf)
            residual[index] = norm
            converged = norm <= limit
            active[index[converged]] = False
            index, error = index[~converged], error[~converged]
            jac = m.jacobian[~converged]
            det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
            step = (
                -bd.stack(
                    (
                        jac[:, 1, 1] * error[:, 0] - jac[:, 0, 1] * error[:, 1],
                        -jac[:, 1, 0] * error[:, 0] + jac[:, 0, 0] * error[:, 1],
                    ),
                    axis=-1,
                )
                / det[:, None]
            )
            current = residual[index]
            pending = bd.isfinite(step).all(axis=-1)
            # A target without a finite step (an unsupported ray) keeps it:
            # nothing moves it, so it stops now.
            stuck = ~pending
            step = bd.where(pending[:, None], step, 0.0)

            # The full step first; where it does not improve, the first of
            # eleven halvings that does, all traced in one call.
            for factors in (bd.ones(1), 0.5 ** bd.arange(1.0, 12.0)):
                trying = bd.flatnonzero(pending)
                if not len(trying):
                    break
                trial = (
                    rho[index[trying]][None]
                    + factors[:, None, None] * step[trying][None]
                )
                mt = self.forward(trial.reshape(-1, 2))
                norm_t = bd.where(
                    mt.valid,
                    bd.linalg.norm(
                        mt.output - bd.tile(targets[index[trying]], (len(factors), 1)),
                        axis=-1,
                    ),
                    bd.inf,
                ).reshape(len(factors), -1)
                better = norm_t < current[trying][None]
                first = bd.argmax(better, axis=0)
                improved = bd.flatnonzero(better.any(axis=0))
                rho[index[trying[improved]]] = trial[first[improved], improved]
                residual[index[trying[improved]]] = norm_t[first[improved], improved]
                pending[trying[improved]] = False

            # No improving step (or none at all): this target stops here.
            active[index[pending | stuck]] = False

        converged = found & (residual <= limit)

        return rho, converged, residual


class _SeedTable:
    """Images and labels of the seed rays, with the interpolants that start Newton."""

    def __init__(self, images, sources):
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import cKDTree

        self.images, self.sources = images, sources
        # griddata(method="linear") is this interpolant, built once here; one
        # evaluation builds its lazy parts before blocks share it.
        self._linear = LinearNDInterpolator(images, sources)
        self._linear(images[:1])
        self._tree = None
        if len(images) > 1:
            tree = cKDTree(images)
            self._tree = tree, np.median(tree.query(images, k=2)[0][:, 1])

    def guess(self, wanted):
        """Linear interpolation of the labels at ``wanted`` (host), NaN off the table."""
        guess = self._linear(wanted)
        missing = np.flatnonzero(~np.isfinite(guess).all(axis=-1))

        if len(missing) and self._tree is not None:
            # Targets on the hull boundary lose the linear guess to roundoff:
            # start them from the nearest table ray if it lies close enough.
            tree, spacing = self._tree
            distance, nearest = tree.query(wanted[missing])
            close = distance <= 2 * spacing
            guess[missing[close]] = self.sources[nearest[close]]

        return guess


class RefractedRays:
    """The rays a surface refracted, carried on to a later reference plane (Appendix B).

    ``previous`` is that surface's InterfaceMap; the labels are its own (the
    points of the family's first plane, local to the first surface) and
    ``axes`` = (u, v) are that plane's lattice axes, which seed inverse maps.
    The refracted direction (Eq. 3.6) and the Appendix A derivatives of the
    surface point and direction are continued in a straight line to the
    plane: no carrier is evaluated off the rays, so nothing is inverted.
    """

    def __init__(self, previous, axes=None):
        self.previous, self.axes = previous, axes

    @property
    def labels(self):
        u, v = self.axes
        return bd.stack(bd.meshgrid(u, v), axis=-1).reshape(-1, 2)

    def rays(self, labels, frame, offset):
        """Points on the plane z = offset of ``frame``, directions, (3, 2) label
        derivatives of both, and which rays live, in that frame's local axes."""
        r = self.previous.forward(labels)
        old = self.previous.surface.frame
        turn = old.basis.T @ frame.basis
        q = (old.origin + r.surface_point @ old.basis.T - frame.origin) @ frame.basis
        s = r.transmitted @ turn
        dq = bd.einsum("nil,ij->njl", r.surface_derivative, turn)
        ds = bd.einsum("nil,ij->njl", r.transmitted_derivative, turn)
        leg = (offset - q[:, 2]) / s[:, 2]
        p = q + leg[:, None] * s
        moved = dq + leg[:, None, None] * ds
        dp = moved - s[:, :, None] * (moved[:, 2:3, :] / s[:, 2, None, None])
        alive = r.valid & (s[:, 2] > 1e-12) & bd.isfinite(p).all(axis=-1)

        return p, s, dp, ds, alive


@dataclass(frozen=True)
class EikonalReport:
    """What the eikonal leg could not carry, and how well its map was inverted.

    ``azimuthal_orders`` lists the orders the axisymmetric leg carried along
    its meridian; it is empty for the lattice leg.
    """

    unsupported_relative_amplitude: float
    dark_samples: int
    newton_residual: float
    azimuthal_orders: tuple = ()


def carried(incident, rays, s, n_incident, n_transmitted):
    """t_s E_s s + t_p E_p p_t + t_p E_par s_t for unit TE axes s, rows of vectors.

    The continued dyadic of Eq. 4.3 without the factor g exp(i k0 Delta S).
    It does not depend on the sign of s.
    """
    p_i, p_t = bd.cross(s, rays.incident), bd.cross(s, rays.transmitted)
    c_i, c_t = rays.cos_incident, rays.cos_transmitted
    t_s = 2 * n_incident * c_i / (n_incident * c_i + n_transmitted * c_t)
    t_p = 2 * n_incident * c_i / (n_transmitted * c_i + n_incident * c_t)
    e_s = bd.sum(incident * s, axis=-1)
    e_p = bd.sum(incident * p_i, axis=-1)
    e_par = bd.sum(incident * rays.incident, axis=-1)

    return (
        (t_s * e_s)[:, None] * s
        + (t_p * e_p)[:, None] * p_t
        + (t_p * e_par)[:, None] * rays.transmitted
    )


def ray_weights(window, rays):
    """Ray weights: one, the window's callable, or a raised-cosine incidence roll-off."""
    cosine = rays.cos_incident
    if window is None:
        return bd.ones_like(cosine)
    if callable(window):
        return bd.asarray(window(rays), dtype=float)
    start, end = (np.radians(angle) for angle in window)
    fraction = bd.clip(
        (bd.arccos(bd.clip(cosine, -1, 1)) - start) / (end - start), 0, 1
    )
    return 0.5 * (1 + bd.cos(bd.pi * fraction))


def check_window(window):
    """None, a callable of the RayMap, or (start, end) incidence degrees as floats."""
    if window is None or callable(window):
        return window
    window = tuple(float(angle) for angle in window)
    if not (len(window) == 2 and 0 <= window[0] < window[1] <= 90):
        raise ValueError(
            "window is (start, end) incidence in degrees with "
            "0 <= start < end <= 90, or a callable of the RayMap"
        )
    return window


class EikonalTransport:
    """Eikonal leg across one interface between copies of a reference plane.

    ``plane`` is the CartesianGrid of the incident field on the reference
    plane (its frame shares the surface orientation). The output lattice in
    the transmitted medium keeps that spacing and grows, up to ``max_growth``
    times the input size per axis, until it holds the image of every
    supported ray: refraction can spread the image beyond the input
    footprint, and cutting it would diffract spuriously. ``window`` = (start,
    end) rolls the carried field off with a raised cosine between those
    incidence angles in degrees; a callable window receives the RayMap of
    carried rays and returns their weights ray by ray (a reference fan).
    Hard-edged inputs are sensitive to this choice, which belongs with any
    result. The map and its inverse are prepared once per geometry and
    carrier and reused for every incident spectrum.

    The lattices are streamed: rays are traced, inverted and carried
    ``block_size`` points at a time, and only the preimages, the lit and
    supported masks and the carried field are kept, in host memory. A beam
    is read exactly at the preimages, and its leg returns a LatticeSpectrum,
    which holds lattices a PlaneSpectrum cannot.
    """

    def __init__(
        self,
        plane,
        interface,
        carrier,
        wavelength0,
        *,
        window=None,
        max_growth=4,
        seed_stride=None,
        block_size=None,
    ):
        if not isinstance(plane, CartesianGrid):
            raise TypeError("eikonal transport needs a CartesianGrid reference plane")

        surface = interface.surface
        frame = surface.frame

        if not bd.allclose(plane.frame.basis, frame.basis, atol=1e-12, rtol=0):
            raise ValueError("reference plane and surface must share an orientation")

        shift = (plane.frame.origin - frame.origin) @ frame.basis

        if bd.linalg.norm(shift[:2]) > 1e-12 * max(1.0, float(bd.max(bd.abs(plane.u)))):
            raise ValueError(
                "reference plane and surface must share a transverse origin"
            )

        block_size = default_block() if block_size is None else block_size
        if (
            isinstance(block_size, bool)
            or int(block_size) != block_size
            or block_size < 1
        ):
            raise ValueError("block_size must be a positive integer")

        self.window = check_window(window)
        self.plane, self.interface, self.carrier = plane, interface, carrier
        self.wavelength0 = float(wavelength0)
        self.block_size = int(block_size)
        self.map = InterfaceMap(
            surface,
            carrier,
            interface.minus.material.n,
            interface.plus.material.n,
            float(shift[2]),
        )
        family = self.map.family
        # A carrier's rays are labelled on this plane; refracted ones on the first.
        axes = (plane.u, plane.v) if family is None else family.axes
        width = len(axes[0])
        count = width * len(axes[1])
        valid = np.zeros(count, bool)
        image = np.full((count, 2), np.nan)
        starts = None if family is None else np.full((count, 2), np.nan)
        need = [0.0, 0.0]

        def label(rows):
            incoming = self.map.forward(lattice_points(axes[0], axes[1], rows))
            # Rays the window zeroes carry nothing and do not widen the lattice.
            carried = bd.flatnonzero(incoming.valid)
            if self.window is not None and len(carried):
                weight = ray_weights(self.window, incoming.subset(incoming.valid))
                carried = carried[weight > 0]
            reach = (
                to_numpy(bd.max(bd.abs(incoming.output[carried]), axis=0))
                if len(carried)
                else (0.0, 0.0)
            )
            start = None if starts is None else to_numpy(incoming.start)
            return (
                rows,
                to_numpy(incoming.valid),
                to_numpy(incoming.output),
                start,
                reach,
            )

        blocks = row_blocks(len(axes[1]), width, self.block_size)
        for rows, ok, output, start, reach in in_blocks(label, blocks):
            flat = slice(rows.start * width, rows.stop * width)
            valid[flat], image[flat] = ok, output
            if starts is not None:
                starts[flat] = start
            need = [max(a, float(b)) for a, b in zip(need, reach)]

        self.output_plane = self._cover(plane, need, max_growth)
        out = self.output_plane
        slack = 1e-6 * float(min(plane.u[1] - plane.u[0], plane.v[1] - plane.v[0]))
        bounds = [float(out.u[0]), float(out.u[-1]), float(out.v[0]), float(out.v[-1])]
        with np.errstate(invalid="ignore"):
            imaged = (
                (image[:, 0] >= bounds[0] - slack)
                & (image[:, 0] <= bounds[1] + slack)
                & (image[:, 1] >= bounds[2] - slack)
                & (image[:, 1] <= bounds[3] + slack)
            )
        del image
        # Supported rays of this plane's lattice, in host memory.
        self.support = valid & imaged
        if family is not None:
            self.support = self._covered(starts, valid)

        stride = seed_stride or max(1, int(np.ceil(max(len(a) for a in axes) / 256)))
        seeds = bd.stack(bd.meshgrid(axes[0][::stride], axes[1][::stride]), axis=-1)
        table = self.map.seed_table(seeds.reshape(-1, 2))
        scale = max(float(bd.max(bd.abs(out.u))), float(bd.max(bd.abs(out.v))))
        limit = 1e-12 * (scale + 1e-300)
        bounds = [
            float(plane.u[0]),
            float(plane.u[-1]),
            float(plane.v[0]),
            float(plane.v[-1]),
        ]
        size = math.prod(out.shape)
        lit = np.zeros(size, bool)
        preimage = np.zeros((size, 2))
        self.residual, top = 0.0, None

        def invert(rows):
            rho, found, residual = self.map.invert(
                lattice_points(out.u, out.v, rows), table, limit
            )
            rays = self.map.forward(rho[found])
            # Starts beyond the sampled input plane would read the periodic
            # continuation of the incident spectrum; they carry no field.
            start = rays.start
            inside = (
                rays.valid
                & (start[:, 0] >= bounds[0] - slack)
                & (start[:, 0] <= bounds[1] + slack)
                & (start[:, 1] >= bounds[2] - slack)
                & (start[:, 1] <= bounds[3] + slack)
            )
            found[bd.flatnonzero(found)[~inside]] = False
            worst = height = None
            if bool(found.any()):
                worst = float(bd.max(residual[found]))
                height = float(bd.max(rays.surface_height[inside]))
            return rows, to_numpy(found), to_numpy(rho), worst, height

        blocks = row_blocks(len(out.v), len(out.u), self.block_size)
        for rows, found, rho, worst, height in in_blocks(invert, blocks):
            flat = slice(rows.start * len(out.u), rows.stop * len(out.u))
            lit[flat], preimage[flat] = found, rho
            if worst is not None:
                self.residual = max(self.residual, worst)
                top = height if top is None else max(top, height)

        self.lit, self.preimage = lit, preimage
        self.boundary = 0.0 if top is None else max(0.0, top - self.map.offset)

    @property
    def rays(self):
        """RayMap of the carried rays, at the lit output points in order."""
        return self.map.forward(bd.asarray(self.preimage[self.lit]))

    def _covered(self, starts, valid):
        """Input-lattice points that a carried ray of the family starts near.

        The ray-validity of the label lattice, interpolated linearly over its
        starts, must be one: lattice points between rays that fail here, or
        that no ray of the family reaches, are unsupported.
        """
        from scipy.interpolate import griddata

        alive = np.isfinite(starts).all(axis=-1)
        lattice = np.stack(
            np.meshgrid(to_numpy(self.plane.u), to_numpy(self.plane.v)), axis=-1
        ).reshape(-1, 2)
        if alive.sum() < 3:
            return np.zeros(len(lattice), bool)
        value = griddata(
            starts[alive], valid[alive].astype(float), lattice, method="linear"
        )

        return np.nan_to_num(value) > 1 - 1e-9

    @staticmethod
    def _cover(plane, need, max_growth):
        """The input lattice, or one with its spacing and frame grown to reach ``need``.

        ``need`` is the largest |u| and |v| of the image to hold.
        """
        sizes = []
        for axis, reach in zip((plane.u, plane.v), need):
            step = float(axis[1] - axis[0])
            extent = float(bd.max(bd.abs(axis)))
            count = len(axis) if reach <= extent else 2 * int(np.ceil(reach / step)) + 4
            sizes.append(min(count, int(max_growth * len(axis))))
        if sizes == [len(plane.u), len(plane.v)]:
            return plane
        u = (bd.arange(sizes[0]) - sizes[0] // 2) * float(plane.u[1] - plane.u[0])
        v = (bd.arange(sizes[1]) - sizes[1] // 2) * float(plane.v[1] - plane.v[0])
        return CartesianGrid(u, v, plane.frame)

    def _reader(self, spectrum, method, **kwargs):
        """The incident field at points (x, y) of this plane, E (n, 3) global."""
        frame = self.map.surface.frame
        origin = frame.origin + self.map.offset * frame.normal
        axes = self.map.basis[:, :2]
        if isinstance(spectrum, LatticeSpectrum):
            blocks = spectrum.blocks
        elif hasattr(spectrum, "beam"):
            # The beam exactly, from plane waves that resolve the whole plane.
            plane = self.plane
            corners = bd.stack(
                [
                    origin + a * axes[:, 0] + b * axes[:, 1]
                    for a in (plane.u[0], plane.u[-1])
                    for b in (plane.v[0], plane.v[-1])
                ]
            )
            beam = spectrum.beam
            step = float(max(plane.u[1] - plane.u[0], plane.v[1] - plane.v[0]))
            reach = float(bd.max(bd.linalg.norm(corners - beam.frame.origin, axis=-1)))
            blocks = plane_blocks(beam.plane_waves(reach + step))
        else:
            blocks = plane_blocks(spectrum.expansion)

        return PlaneReader(blocks, origin, axes, method=method, **kwargs)

    def _transfer(self, r):
        """Per ray, the matrix from incident E (local axes) to the carried u, v components.

        g exp(i k0 Delta S) times the window and t_s s s + t_p p_t p_i + t_p s_t s_i
        (Eq. 4.3), of shape (n, 2, 3).
        """
        s = bd.cross(r.normal, r.incident)
        size = bd.linalg.norm(s, axis=-1, keepdims=True)
        fallback = bd.cross(r.incident, bd.asarray([0.0, 1.0, 0.0]))
        fallback = bd.where(
            bd.linalg.norm(fallback, axis=-1, keepdims=True) > 1e-6,
            fallback,
            bd.cross(r.incident, bd.asarray([1.0, 0.0, 0.0])),
        )
        s = bd.where(size > 1e-12, s, fallback)
        s = s / bd.linalg.norm(s, axis=-1, keepdims=True)
        n_i, n_t = self.map.n_incident, self.map.n_transmitted
        p_i, p_t = bd.cross(s, r.incident), bd.cross(s, r.transmitted)
        c_i, c_t = r.cos_incident, r.cos_transmitted
        t_s = 2 * n_i * c_i / (n_i * c_i + n_t * c_t)
        t_p = 2 * n_i * c_i / (n_t * c_i + n_i * c_t)
        k0 = 2 * bd.pi / self.wavelength0
        factor = self._window(r) * r.flux * bd.exp(1j * k0 * r.path)
        dyadic = (
            t_s[:, None, None] * s[:, :2, None] * s[:, None, :]
            + t_p[:, None, None] * p_t[:, :2, None] * p_i[:, None, :]
            + t_p[:, None, None] * r.transmitted[:, :2, None] * r.incident[:, None, :]
        )

        return factor[:, None, None] * dyadic

    def _carried(self, spectrum, method, **kwargs):
        """Carried u, v components on the output lattice: a host array (n, 2)."""
        if spectrum.expansion.region != self.interface.minus:
            raise ValueError(
                "incident spectrum must live in the interface's incident medium"
            )
        if method == "auto":
            import importlib.util

            method = "nufft" if importlib.util.find_spec("finufft") else "direct"

        read = self._reader(spectrum, method, **kwargs)
        basis = self.map.basis
        values = np.zeros((self.lit.size, 2), complex)
        lit = np.flatnonzero(self.lit)

        def geometry(index):
            rays = self.map.forward(bd.asarray(self.preimage[index]))
            return self._transfer(rays), rays.start

        # Incident sums cost most per call, so each reads many blocks' starts.
        for begin in range(0, len(lit), READ):
            index = lit[begin : begin + READ]
            parts = [
                index[part : part + self.block_size]
                for part in range(0, len(index), self.block_size)
            ]
            transfers, starts = zip(*in_blocks(geometry, parts))
            incident = read(bd.concatenate(starts)) @ basis
            carried = bd.einsum("nij,nj->ni", bd.concatenate(transfers), incident)
            values[index] = to_numpy(carried)

        return values

    def _lost(self, spectrum, method, **kwargs):
        """Relative amplitude of the incident field on unsupported lattice points."""
        if isinstance(spectrum, PlaneSpectrum):
            lattice = spectrum.propagate(
                float(
                    (self.plane.frame.origin - spectrum.grid.frame.origin)
                    @ spectrum.grid.frame.normal
                )
            ).electric.values.reshape(-1, 3)
            power = to_numpy(bd.sum(bd.abs(lattice) ** 2, axis=-1))
            lost, total = power[~self.support].sum(), power.sum()
        else:
            if method == "auto":
                import importlib.util

                method = "nufft" if importlib.util.find_spec("finufft") else "direct"
            read = self._reader(spectrum, method, **kwargs)
            plane, lost, total = self.plane, 0.0, 0.0
            rows = row_blocks(len(plane.v), len(plane.u), READ)
            for block in rows:
                flat = slice(block.start * len(plane.u), block.stop * len(plane.u))
                field = read(lattice_points(plane.u, plane.v, block))
                power = to_numpy(bd.sum(bd.abs(field) ** 2, axis=-1))
                lost += power[~self.support[flat]].sum()
                total += power.sum()

        return float(np.sqrt(lost / max(float(total), 1e-300)))

    def carry(self, spectrum, *, method="auto", **kwargs):
        """Carried plane components on the output lattice and the EikonalReport.

        The TransverseElectricField holds g exp(i k0 Delta S) T E at every
        lattice point with a supported preimage and zero elsewhere, before the
        spectral re-expansion that ``transmit`` performs.
        """
        values = self._carried(spectrum, method, **kwargs)
        field = TransverseElectricField(
            self.output_plane,
            self.wavelength0,
            bd.asarray(values.reshape(self.output_plane.shape + (2,))),
        )

        return field, self._report(spectrum, method, **kwargs)

    def _report(self, spectrum, method, **kwargs):
        return EikonalReport(
            self._lost(spectrum, method, **kwargs),
            int((~self.lit).sum()),
            self.residual,
        )

    def _window(self, rays):
        """Ray weights: one, the window's callable, or a raised-cosine incidence roll-off."""
        return ray_weights(self.window, rays)

    def transmit(self, spectrum, *, method="auto", **kwargs):
        """(Spectrum in the transmitted medium on the plane, EikonalReport).

        The carried components are re-expanded within the transverse band
        sin < _SPECTRAL_CUTOFF of the transmitted medium, as the manuscript
        does: nearer the light circle E_z = -kappa . E / k_z amplifies what the
        lattice leaves there (the edge of the carried support, for a hard
        pupil) a hundredfold and more. A sampled incident field (a
        PlaneSpectrum) gives a PlaneSpectrum; a beam or a LatticeSpectrum
        gives a LatticeSpectrum, transformed in place.
        """
        values = self._carried(spectrum, method, **kwargs)
        report = self._report(spectrum, method, **kwargs)
        plus = self.interface.plus

        if not isinstance(spectrum, PlaneSpectrum):
            outgoing = lattice_spectrum(
                self.output_plane,
                plus,
                self.wavelength0,
                values,
                cutoff=_SPECTRAL_CUTOFF,
                block=self.block_size,
            )
            return outgoing, report

        field = TransverseElectricField(
            self.output_plane,
            self.wavelength0,
            bd.asarray(values.reshape(self.output_plane.shape + (2,))),
        )
        outgoing = angular_spectrum(field, plus, evanescent="discard")
        k = 2 * bd.pi * plus.material.n / self.wavelength0
        transverse = (
            outgoing.expansion.wavevectors.real @ outgoing.grid.frame.basis[:, :2]
        )
        inside = bd.linalg.norm(transverse, axis=-1) <= _SPECTRAL_CUTOFF * k
        amplitudes = bd.where(inside[:, None], outgoing.expansion.amplitudes, 0)
        outgoing = PlaneSpectrum(
            outgoing.grid, replace(outgoing.expansion, amplitudes=amplitudes)
        )

        return outgoing, report
