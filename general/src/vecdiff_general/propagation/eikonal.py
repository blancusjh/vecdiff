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

from dataclasses import dataclass, fields

import numpy as np

from vecdiff_general._backend import bd, divide, to_numpy
from vecdiff_general.fields import TransverseElectricField
from vecdiff_general.fields.angular_spectrum import angular_spectrum
from vecdiff_general.sampling import CartesianGrid, PointGrid

_EPS = np.finfo(float).eps


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
    """Plane-surface-plane map quantities at input-plane points, surface-local axes.

    ``output`` is Phi(rho), ``path`` Delta S in metres, ``flux`` the factor g,
    ``jacobian`` D Phi with output components first; directions are unit
    vectors in the surface frame. ``valid`` marks rays the chart supports.
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

    def subset(self, mask):
        """The same quantities at the points selected by ``mask``."""
        return RayMap(*(getattr(self, f.name)[mask] for f in fields(self)))


class InterfaceMap:
    """Phi, Delta S and g for one surface, carrier and pair of indices.

    ``offset`` places the reference plane along the surface normal from the
    vertex, in metres (0 is the vertex plane; positive lies in the direction
    of propagation). Points are surface-local plane coordinates (..., 2).
    """

    def __init__(self, surface, carrier, n_incident, n_transmitted, offset=0.0):
        if not all(
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

    def forward(self, xy):
        """RayMap at input-plane points xy (..., 2)."""
        xy = bd.asarray(xy, float)
        shape = xy.shape[:-1]
        xy = xy.reshape(-1, 2)
        finite = bd.isfinite(xy).all(axis=-1)
        xy = bd.where(finite[:, None], xy, 0.0)
        p = bd.concatenate((xy, bd.full((len(xy), 1), self.offset)), axis=-1)
        world = self._global(p)
        s = self.carrier.directions(world) @ self.basis
        ds = bd.einsum(
            "ji,njk,kl->nil", self.basis, self.carrier.jacobians(world), self.basis
        )
        result = list(self._trace(p, s, ds))
        result[0] = result[0] & finite

        return RayMap(*(value.reshape(shape + value.shape[1:]) for value in result))

    def _intersect(self, p, s):
        """Signed incident leg to the surface: a starting leg, then Newton on the graph.

        The start is the height under P where the graph has one there, else the
        crossing with the osculating sphere at the vertex, else a march along
        the ray; a ray Newton loses from a geometric start is retried from the
        march.
        """
        height = _surface(self.surface, "height", p[:, :2], ())
        leg = (height - p[:, 2]) / s[:, 2]
        scale = bd.linalg.norm(p, axis=-1) + 1e-300
        lost = bd.flatnonzero(~bd.isfinite(leg))

        if len(lost):
            leg[lost] = self._osculating(p[lost], s[lost])

        leg = self._newton(p, s, leg, scale)
        retry = bd.flatnonzero(~bd.isfinite(leg))
        retry = retry[bd.isin(retry, lost)] if len(lost) else retry[:0]

        if len(retry):
            reach = 4 * bd.maximum(scale[retry], abs(self.offset))
            start = self._bracket(p[retry], s[retry], reach)
            leg[retry] = self._newton(p[retry], s[retry], start, scale[retry])

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

    def _trace(self, p, s, ds):
        n_i, n_t = self.n_incident, self.n_transmitted
        eta = n_i / n_t
        leg_i = self._intersect(p, s)
        q = p + leg_i[:, None] * s
        slope = _surface(self.surface, "gradient", q[:, :2], (2,))
        curvature = _surface(self.surface, "hessian", q[:, :2], (2, 2))
        grad_f = bd.concatenate((-slope, bd.ones((len(q), 1))), axis=-1)
        size = bd.linalg.norm(grad_f, axis=-1)
        normal = grad_f / size[:, None]
        hess_f = bd.zeros((len(q), 3, 3))
        hess_f[:, :2, :2] = -curvature
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

        # Appendix A: derivatives along the plane axes a = x, y (axis 1).
        unit = bd.zeros((1, 2, 3))
        unit[0, 0, 0] = unit[0, 1, 1] = 1.0
        ds_a = bd.swapaxes(ds[:, :, :2], 1, 2)  # (n, a, component)
        w = unit + leg_i[:, None, None] * ds_a
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
        dp = moved - s_t[:, None] * (moved[..., 2] / s_t[:, 2:3])[..., None]
        jacobian = bd.swapaxes(dp[..., :2], 1, 2)
        det = (
            jacobian[:, 0, 0] * jacobian[:, 1, 1]
            - jacobian[:, 0, 1] * jacobian[:, 1, 0]
        )
        valid &= bd.isfinite(det) & (det > 0)
        flux = bd.sqrt(divide(s[:, 2] * c_t, c_i * s_t[:, 2] * det, valid))

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
        )

    def inverse(self, targets, samples, *, tolerance=1e-12):
        """Preimages on the input plane of output points, and where they exist.

        ``samples`` (M, 2) seed a forward table; each target starts from the
        linear interpolation of that table and finishes by damped Newton steps
        with the analytic D Phi. Targets outside the table's image or without a
        converged preimage are marked missing.
        """
        from scipy.interpolate import griddata
        from scipy.spatial import cKDTree

        targets = bd.asarray(targets, float).reshape(-1, 2)
        table = self.forward(bd.asarray(samples, float).reshape(-1, 2))
        keep = to_numpy(table.valid)
        images = to_numpy(table.output)[keep]
        sources = to_numpy(bd.asarray(samples, float).reshape(-1, 2))[keep]
        wanted = to_numpy(targets)
        guess = griddata(images, sources, wanted, method="linear")
        missing = np.flatnonzero(~np.isfinite(guess).all(axis=-1))

        if len(missing) and len(images) > 1:
            # Targets on the hull boundary lose the linear guess to roundoff:
            # start them from the nearest table ray if it lies close enough.
            tree = cKDTree(images)
            spacing = np.median(tree.query(images, k=2)[0][:, 1])
            distance, nearest = tree.query(wanted[missing])
            close = distance <= 2 * spacing
            guess[missing[close]] = sources[nearest[close]]

        found = bd.asarray(np.isfinite(guess).all(axis=-1))
        rho = bd.asarray(np.nan_to_num(guess))
        scale = float(bd.max(bd.abs(targets))) + 1e-300
        limit = tolerance * scale
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
            index, m, error = index[~converged], m, error[~converged]
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
            factor = bd.ones(len(index))
            pending = bd.isfinite(step).all(axis=-1)
            step = bd.where(pending[:, None], step, 0.0)

            for _ in range(12):
                if not bool(pending.any()):
                    break
                trial = rho[index] + factor[:, None] * step
                mt = self.forward(trial)
                norm_t = bd.where(
                    mt.valid,
                    bd.linalg.norm(mt.output - targets[index], axis=-1),
                    bd.inf,
                )
                better = pending & (norm_t < current)
                rho[index[better]] = trial[better]
                residual[index[better]] = norm_t[better]
                pending &= ~better
                factor = bd.where(pending, factor / 2, factor)

            # No improving step (or none at all): this target stops here.
            active[index[pending | ~bd.isfinite(step).all(axis=-1)]] = False

        converged = found & (residual <= limit)

        return rho, converged, residual


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
    incidence angles in degrees; a callable window receives the RayMap of the
    carried rays and returns their weights (a reference fan). Hard-edged
    inputs are sensitive to this choice, which belongs with any result.
    The map and its inverse are prepared once per
    geometry and carrier and reused for every incident spectrum.
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

        self.window = check_window(window)
        self.plane, self.interface, self.carrier = plane, interface, carrier
        self.wavelength0 = float(wavelength0)
        self.map = InterfaceMap(
            surface,
            carrier,
            interface.minus.material.n,
            interface.plus.material.n,
            float(shift[2]),
        )
        x, y = bd.meshgrid(plane.u, plane.v)
        lattice = bd.stack((x, y), axis=-1).reshape(-1, 2)
        incoming = self.map.forward(lattice)
        self.support = incoming.valid
        self.output_plane = self._cover(
            plane, incoming.output[self.support], max_growth
        )
        out = self.output_plane
        slack = 1e-6 * float(min(plane.u[1] - plane.u[0], plane.v[1] - plane.v[0]))
        imaged = (
            (incoming.output[:, 0] >= out.u[0] - slack)
            & (incoming.output[:, 0] <= out.u[-1] + slack)
            & (incoming.output[:, 1] >= out.v[0] - slack)
            & (incoming.output[:, 1] <= out.v[-1] + slack)
        )
        self.support = self.support & imaged
        targets = bd.stack(bd.meshgrid(out.u, out.v), axis=-1).reshape(-1, 2)
        stride = seed_stride or max(1, int(np.ceil(max(plane.shape) / 256)))
        seeds = bd.stack(bd.meshgrid(plane.u[::stride], plane.v[::stride]), axis=-1)
        rho, found, residual = self.map.inverse(targets, seeds.reshape(-1, 2))
        # Preimages beyond the sampled input plane would read the periodic
        # continuation of the incident spectrum; they carry no field.
        inside = (
            (rho[:, 0] >= plane.u[0] - slack)
            & (rho[:, 0] <= plane.u[-1] + slack)
            & (rho[:, 1] >= plane.v[0] - slack)
            & (rho[:, 1] <= plane.v[-1] + slack)
        )
        self.lit = found & inside
        self.preimage = rho
        rays = self.map.forward(rho[self.lit])
        self.lit[bd.flatnonzero(self.lit)[~rays.valid]] = False
        self.rays = rays.subset(rays.valid)
        self.residual = (
            float(bd.max(residual[self.lit])) if bool(self.lit.any()) else 0.0
        )
        heights = self.rays.surface_height - self.map.offset
        self.boundary = float(max(0.0, bd.max(heights))) if heights.size else 0.0

    @staticmethod
    def _cover(plane, image, max_growth):
        """The input lattice, or one with its spacing and frame grown to hold ``image``."""
        sizes = []
        for axis, column in ((plane.u, 0), (plane.v, 1)):
            step = float(axis[1] - axis[0])
            extent = float(bd.max(bd.abs(axis)))
            need = float(bd.max(bd.abs(image[:, column]))) if len(image) else 0.0
            count = len(axis) if need <= extent else 2 * int(np.ceil(need / step)) + 4
            sizes.append(min(count, int(max_growth * len(axis))))
        if sizes == [len(plane.u), len(plane.v)]:
            return plane
        u = (bd.arange(sizes[0]) - sizes[0] // 2) * float(plane.u[1] - plane.u[0])
        v = (bd.arange(sizes[1]) - sizes[1] // 2) * float(plane.v[1] - plane.v[0])
        return CartesianGrid(u, v, plane.frame)

    def carry(self, spectrum, *, method="auto", **kwargs):
        """Carried plane components on the output lattice and the EikonalReport.

        The TransverseElectricField holds g exp(i k0 Delta S) T E at every
        lattice point with a supported preimage and zero elsewhere, before the
        spectral re-expansion that ``transmit`` performs.
        """
        import importlib.util

        if spectrum.expansion.region != self.interface.minus:
            raise ValueError(
                "incident spectrum must live in the interface's incident medium"
            )

        if method == "auto":
            method = "nufft" if importlib.util.find_spec("finufft") else "direct"

        basis = self.map.basis
        points = self.map._global(
            bd.concatenate(
                (
                    self.preimage[self.lit],
                    bd.full((int(self.lit.sum()), 1), self.map.offset),
                ),
                axis=-1,
            )
        )
        incident = (
            spectrum.evaluate(
                PointGrid(points), method=method, **kwargs
            ).electric.values
            @ basis
        )
        r = self.rays
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
        k0 = 2 * bd.pi / self.wavelength0
        weight = self._window(r)
        transfer = carried(incident, r, s, self.map.n_incident, self.map.n_transmitted)
        values = bd.zeros((self.lit.size, 2), complex)
        values[self.lit] = (
            (weight * r.flux * bd.exp(1j * k0 * r.path))[:, None] * transfer
        )[:, :2]
        field = TransverseElectricField(
            self.output_plane,
            self.wavelength0,
            values.reshape(self.output_plane.shape + (2,)),
        )
        lattice = spectrum.propagate(
            float(
                (self.plane.frame.origin - spectrum.grid.frame.origin)
                @ spectrum.grid.frame.normal
            )
        ).electric.values.reshape(-1, 3)
        power = bd.sum(bd.abs(lattice) ** 2, axis=-1)
        lost = float(
            bd.sqrt(bd.sum(power[~self.support]) / max(float(bd.sum(power)), 1e-300))
        )
        report = EikonalReport(lost, int((~self.lit).sum()), self.residual)

        return field, report

    def _window(self, rays):
        """Ray weights: one, the window's callable, or a raised-cosine incidence roll-off."""
        return ray_weights(self.window, rays)

    def transmit(self, spectrum, *, method="auto", **kwargs):
        """(PlaneSpectrum in the transmitted medium on the plane, EikonalReport)."""
        field, report = self.carry(spectrum, method=method, **kwargs)
        outgoing = angular_spectrum(field, self.interface.plus, evanescent="discard")

        return outgoing, report
