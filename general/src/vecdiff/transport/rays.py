"""The plane-surface-plane ray map that carries a field across one interface.

The plane-surface-plane map of the eikonal-angular-spectrum formulation: a
point P of the input reference plane follows its carrier ray to the interface
Sigma, refracts by the vector Snell law and is extrapolated to P' on the same
geometric plane in the transmitted medium. Signed legs l_i and l_t give the
optical path increment Delta S = n_i l_i + n_t l_t, and the field is carried by

    E'(Phi(rho)) = g exp(i k0 Delta S) [t_s E_s s + t_p E_p p_t + t_p E_par s_t],

with g = sqrt(s_iz c_t / (c_i s_tz J)) and J = det D Phi. The Jacobian follows
analytically from the surface Hessian and the carrier's direction Jacobian;
no spherical focus is assumed. Where the ray tube has collapsed between the
planes J has passed through zero, and the root continues through it as
1 / sqrt(J) -> exp(-i pi kappa / 2) / sqrt(|J|): kappa counts the collapsed
transverse directions met along the ray (one at a focal line, two at a point
focus), forward on a real segment and backward on a virtual one. The output lattice is filled through the
inverse map, so the incident field is evaluated exactly at the preimages and
never interpolated. The transmitted plane components are re-expanded in the
transmitted medium, which restores the longitudinal component spectrally.

The reflected branch is the same map with the law of reflection,
s_r = s - 2 c_i n, back to the same geometric plane in the incident medium:
Delta S = n_i (l_i + l_r) and g = sqrt(s_iz / (|s_rz| J)) (c_r = c_i), where
power conservation along the ray tube, R I_i c_i d sigma = I_r c_i d sigma
at the surface and I |s_z| dA constant along each segment, leaves |r|^2 to the
dyadic and the area ratio to g.

Rays that miss the surface, meet it from behind, are totally reflected or
leave backward carry no field; the incident power they hold is reported, not
hidden. Where the map folds, rays from both sides of a caustic reach the same points
and one ray per point cannot carry the field: the rays beyond the fold carry
nothing, their share is reported, and a leg that would lose more than a
hundredth of its amplitude that way is refused. The transfer is the stated leading-order model: it is
exact for a plane wave on a plane interface and the identity for index-matched
media, not a solution of the curved-boundary Maxwell problem.
"""

from dataclasses import dataclass, fields

import numpy as np

from vecdiff._backend import bd, divide, to_numpy
from vecdiff.surfaces.surface import ImplicitSurface

_EPS = np.finfo(float).eps
# exp(-i pi kappa / 2) for kappa mod 4, exactly.
_QUARTER_TURNS = (1.0, -1j, -1.0, 1j)


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


def _collapses(s, a1, a2, b1, b2, leg):
    """Signed count of the collapses of a straight ray tube along its leg.

    Neighbouring rays of the tube lie at a_j + tau b_j from the ray (rows of
    3-vectors, j = 1, 2 the label axes), and s is its unit direction. The
    tube's cross-section s . (a_1 + tau b_1) x (a_2 + tau b_2) is a quadratic
    in tau; each root strictly inside the leg is a transverse direction in
    which the tube collapses there (a double root is a point focus). Roots of
    a leg run forward count +1, of a leg continued backward (leg < 0) -1.
    The rays of an eikonal are normal to its wavefronts, so the roots are
    real; a discriminant that roundoff left negative is a double root.
    """
    c0 = bd.sum(s * bd.cross(a1, a2), axis=-1)
    c1 = bd.sum(s * (bd.cross(a1, b2) + bd.cross(b1, a2)), axis=-1)
    c2 = bd.sum(s * bd.cross(b1, b2), axis=-1)
    root = bd.sqrt(bd.maximum(c1 * c1 - 4 * c2 * c0, 0.0))
    # The stable pair of roots: q / c2 and c0 / q.
    q = -0.5 * (c1 + bd.where(c1 < 0, -root, root))
    lo, hi = bd.minimum(leg, 0.0), bd.maximum(leg, 0.0)
    count = bd.zeros(len(leg), dtype=int)
    for numerator, denominator in ((q, c2), (c0, q)):
        defined = denominator != 0
        tau = divide(numerator, denominator, defined)
        count = count + (defined & (tau > lo) & (tau < hi))

    return bd.where(leg < 0, -count, count)


@dataclass(frozen=True)
class RayMap:
    """Plane-surface-plane map quantities of labelled rays, surface-local axes.

    A ray is labelled by its point on the first reference plane of its
    family; for a carrier that is the point P on this map's own input plane.
    ``start`` is P (where the incident field is read), ``output`` Phi(P),
    ``path`` Delta S in metres, ``flux`` the factor g, complex: its phase is
    exp(-i pi caustics / 2). ``caustics`` is the signed count kappa of
    transverse directions in which the ray tube collapsed between the planes:
    crossings on a real segment add, crossings on a virtual one (a segment
    continued backward) subtract. ``jacobian`` is
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
    caustics: np.ndarray

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
    ``branch`` "reflected" follows the reflected rays back to the plane in
    the incident medium instead of the refracted ones into the next: the
    RayMap's ``transmitted`` and ``cos_transmitted`` then hold s_r and
    c_r = c_i, its rays head back (s_rz < 0), and n_transmitted is not used.
    """

    def __init__(
        self,
        surface,
        carrier,
        n_incident,
        n_transmitted,
        offset=0.0,
        *,
        branch="transmitted",
    ):
        if branch not in ("transmitted", "reflected"):
            raise ValueError("branch must be 'transmitted' or 'reflected'")
        if not isinstance(carrier, RefractedRays) and not all(
            callable(getattr(carrier, name, None))
            for name in ("directions", "jacobians")
        ):
            raise TypeError("eikonal transport needs a fields.Carrier")

        self.surface, self.carrier = surface, carrier
        self.n_incident, self.n_transmitted = float(n_incident), float(n_transmitted)
        self.offset = float(offset)
        self.branch = branch
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
        caustics = result.pop()
        result.extend((xy, caustics))

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
        reflected = self.branch == "reflected"
        if reflected:
            # s_r = s - 2 c_i n heads back to the plane, in the incident medium;
            # its cosine against -n is c_i itself.
            c_t, n_t = c_i, n_i
            s_t = s - 2 * c_i[:, None] * normal
            heading = s_t[:, 2] < -1e-12
        else:
            tangent = eta * (s - c_i[:, None] * normal)
            radicand = 1 - bd.sum(tangent**2, axis=-1)
            c_t = bd.sqrt(bd.where(radicand > 0, radicand, bd.nan))
            s_t = tangent + c_t[:, None] * normal
            heading = (radicand > 1e-24) & (s_t[:, 2] > 1e-12)
        valid = (
            bd.isfinite(leg_i)
            & bd.isfinite(size)
            & (s[:, 2] > 0)
            & (c_i > 1e-12)
            & heading
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
        if reflected:
            ds_t = ds_a - 2 * (
                dc_i[..., None] * normal[:, None] + c_i[:, None, None] * dn
            )
        else:
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
        # Area ratio between the planes: per label, output over input. Both
        # are measured in the plane's own (x, y) axes, so a plane mirror at
        # normal incidence has ratio +1: the mirror reverses the ray tube's
        # handedness about its direction (s_r . n = -c_i) and the output
        # direction's s_z, together, and the sign of the ratio still changes
        # only where the tube collapses.
        det_in = dp[:, 0, 0] * dp[:, 1, 1] - dp[:, 0, 1] * dp[:, 1, 0]
        ratio = divide(det, det_in, det_in != 0)
        # Where the tube collapsed on the way the ratio changed sign, and the
        # root of g continues through each zero by exp(-i pi / 2).
        caustics = _collapses(
            s, dp[:, :, 0], dp[:, :, 1], ds[:, :, 0], ds[:, :, 1], leg_i
        ) + _collapses(s_t, dq[:, 0], dq[:, 1], ds_t[:, 0], ds_t[:, 1], leg_t)
        odd = caustics % 2 == 1
        valid &= bd.isfinite(ratio) & (ratio != 0) & ((ratio < 0) == odd)
        flux = (
            bd.sqrt(bd.abs(divide(s[:, 2] * c_t, c_i * s_t[:, 2] * ratio, valid)))
            * bd.asarray(_QUARTER_TURNS)[caustics % 4]
        )
        caustics = bd.where(valid, caustics, 0)

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
            caustics,
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
