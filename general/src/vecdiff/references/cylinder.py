"""Infinite circular cylinder: exact cylindrical-wave solution and its Debye series.

Local frame: origin at the volume's centre, axis a, and u, v across it;
rho and phi about the axis (phi from u toward v), z along it. A field of
axial wavenumber h and azimuthal order n varies as Z_n(kappa rho) e^{i n phi}
e^{i h z}, kappa² = (k0 n_r)² - h², and follows from its axial components
E_z = a Z_n and Z0 H_z = b Z_n (eta = Z0 H):

    E_rho = (i/kappa²) [h d_rho E_z + (k0/rho) d_phi eta_z]
    E_phi = (i/kappa²) [(h/rho) d_phi E_z - k0 d_rho eta_z]
    eta_rho = (i/kappa²) [h d_rho eta_z - (k0 n_r²/rho) d_phi E_z]
    eta_phi = (i/kappa²) [(h/rho) d_phi eta_z + k0 n_r² d_rho E_z]

On rho = R a wave of surface values v = (E_z, eta_z) and logarithmic
derivative L = Z'/Z has azimuthal components (alpha I + L P) v, with
alpha = (i/kappa²) h i n / R and P = (i/kappa²) k0 kappa [[0, -1], [n_r², 0]]:
for h != 0 the interface couples the two polarizations, a 2 x 2 matrix per
(h, n).

Debye series, Wronskian-reduced (prototype cyl_exact): with L1, L2 the
logarithmic derivatives of H1 and H2 = conj(H1), for unit surface values,

    K = (alpha_o - alpha_i) I + L1_o P_o - L2_i P_i,
    T_in = K^-1 (L1_o - L2_o) P_o,   R_out = T_in - I,
    T_out = K^-1 (L1_i - L2_i) P_i,  R_in = T_out - I,

where L1 - L2 = 4i / (pi x |H1|²) is the Wronskian: no nearly equal Hankel
functions are subtracted (the direct Hankel-basis solve is ill-conditioned;
the prototype's index-matched test gave 2.6e5 with it). G = H1_i / H2_i
continues the inward wave through the axis. Inside, term p is
(G R_in)^(p-1) T_in applied to the incoming half of the incident wave; the
first entry is p = 1. Outside, term p >= 1 is T_out G (R_in G)^(p-1) T_in
and term 0 the external reflection with the diffraction half, also by the
Wronskian so that it keeps relative accuracy. The total field is solved
directly in the J (inside) and J, H1 (outside) bases.
"""

import numpy as np

from vecdiff._backend import cpu_reference, leggauss
from vecdiff._printing import block, guarded, number, region_text, shorten, vector
from vecdiff.fields import ElectricField, ElectromagneticField, FocusedBeam
from vecdiff.fields.spectrum import PlaneWaveExpansion
from vecdiff.media.absorption import REFERENCE, region_of, require_transparent
from vecdiff.media.medium import Z0
from vecdiff.references import _bessel
from vecdiff.references.layered_sphere import _check_term
from vecdiff.sampling import PointGrid
from vecdiff.volumes import CylinderVolume

# Points per evaluation block: (orders x axial nodes x points) tables stay small.
_BLOCK = 512


def _signed(count, x):
    """Indices and parity of n = -count..count: f_-n = (-1)^n f_n for J and H1."""
    n = np.arange(-count, count + 1)
    odd = ((n < 0) & (np.abs(n) % 2 == 1)).reshape((-1,) + (1,) * np.ndim(x))
    return np.abs(n), odd


def _regular(count, x):
    """J_n, J_n' mantissas and their log scale, n = -count..count."""
    value, derivative, scale = _bessel.regular(count, x, spherical=False)
    index, odd = _signed(count, x)
    sign = np.where(odd, -1.0, 1.0)
    return sign * value[index], sign * derivative[index], scale[index]


def _outgoing(count, x):
    """log H1_n and H1_n' / H1_n, n = -count..count."""
    log, dlog = _bessel.outgoing(count, x, spherical=False)
    index, odd = _signed(count, x)
    return log[index] + np.where(odd, 1j * np.pi, 0), dlog[index]


def _azimuthal_sums(alpha, values, count):
    """sum_j values[j] e^{-i n alpha_j}, n = -count..count: shape (2 count + 1, columns)."""
    columns = values.shape[1]
    stacked = np.concatenate([values, np.conj(values)], axis=1)
    block = 256
    steps = np.exp(-1j * np.outer(np.arange(block), alpha))
    out = np.empty((count + 1, 2 * columns), complex)
    for start in range(0, count + 1, block):
        stop = min(start + block, count + 1)
        out[start:stop] = (
            steps[: stop - start] * np.exp(-1j * start * alpha)
        ) @ stacked
    # sum v e^{+i m alpha} = conj(sum conj(v) e^{-i m alpha})
    return np.concatenate([np.conj(out[1:, columns:])[::-1], out[:, :columns]])


def _focused_density(beam, s):
    """Plane-wave density of a FocusedBeam at arbitrary unit directions s (..., 3).

    The pointwise form of FocusedBeam._density, which works on product grids.
    """
    local = s @ beam.frame.basis
    sine = np.hypot(local[..., 0], local[..., 1])
    cosine = local[..., 2]
    phi = np.arctan2(local[..., 1], local[..., 0])
    f, k = beam.focal_length, beam.wavenumber
    pupil = beam._transverse(-f * local[..., 0], -f * local[..., 1])
    cp, sp = np.cos(phi), np.sin(phi)
    meridional = pupil[..., 0] * cp + pupil[..., 1] * sp
    azimuthal = -pupil[..., 0] * sp + pupil[..., 1] * cp
    apodization = np.sqrt(cosine) if beam.apodization == "aplanatic" else cosine
    scale = -1j * k * f / (2 * np.pi) * apodization
    u, v, n = beam.frame.u, beam.frame.v, beam.frame.normal
    theta_hat = (
        (cosine * cp)[..., None] * u
        + (cosine * sp)[..., None] * v
        - sine[..., None] * n
    )
    phi_hat = -sp[..., None] * u + cp[..., None] * v
    density = (scale * meridional)[..., None] * theta_hat + (scale * azimuthal)[
        ..., None
    ] * phi_hat
    inside = np.arctan2(sine, cosine) <= beam._theta_max()
    return np.where(inside[..., None], density, 0)


# Angle between the beam and cylinder axes below which they count as aligned.
_ALIGNED = 1e-14
# Smallest angle to the axis direction that is resolved: the pupil edge passing
# closer is not seen, and nodes closer are dropped (< 1e-14 of the solid angle).
_EDGE_FLOOR = 1e-7
# n clustered nodes resolve a singular point at the fraction r of the panel's
# width beyond its end when n sqrt(r) exceeds about 15 (measured, to 1e-12).
_EDGE_RESOLVED = 20.0
# Largest width of a graded panel over its distance to the axis.
_EDGE_GROWTH = 4.0


def _polar_cuts(gamma, top, count):
    """Panels in the polar angle about +a for a cap of half-angle top at gamma.

    count(lo, hi) is the number of nodes a panel between two polar angles
    would get. Returns the increasing cut angles and, per panel, whether it
    lies close to an axis direction and is laid out in the polar angle.
    """

    def from_pole(gamma, count):
        # Cuts measured from the pole at angle gamma from the cap's centre.
        eps = abs(gamma - top)
        inside = gamma < top
        if eps < _EDGE_FLOOR:
            return [0.0 if inside else eps], 0
        cuts = [0.0, eps] if inside else [eps]
        end = min(gamma + top, np.pi)
        if end <= eps or count(eps, end) * np.sqrt(eps / (end - eps)) >= _EDGE_RESOLVED:
            return cuts, 0
        # Graded panels stay in this pole's hemisphere.
        end = min(end, np.pi / 2)
        while True:
            cuts.append(cuts[-1] * (1 + _EDGE_GROWTH))
            if cuts[-1] * _EDGE_GROWTH >= end - cuts[-1]:
                return cuts, len(cuts) - 1

    north, north_near = from_pole(gamma, count)
    south, south_near = from_pole(
        np.pi - gamma, lambda lo, hi: count(np.pi - hi, np.pi - lo)
    )
    cuts = north + [np.pi - x for x in reversed(south)]
    middle = len(cuts) - 1 - north_near - south_near
    return cuts, [True] * north_near + [False] * middle + [True] * south_near


def _zoned_cuts(gamma, angles, count, half_width):
    """Panels in the polar angle about +a for a cap with edge cones inside it.

    angles are the half-angles of the cones about the beam axis where the
    density jumps, the cap's own last. Each cone cuts the cone of a polar
    angle about a in an arc of half-width half_width(angle, cone), with
    branch points at |gamma -/+ cone|: all of them are cuts, and each cone
    near an axis direction brings the graded panels of _polar_cuts. A branch
    point just beyond a panel on which its arc is partial (two edges close to
    each other, a thin ring) is not resolved by that panel's nodes; the panel
    is then graded toward that end, each piece no wider than _EDGE_GROWTH
    times its distance to the point.
    """
    if len(angles) == 1:
        return _polar_cuts(gamma, angles[0], count)
    # The merged panels are shorter than a cone's own: take the least count.
    families = [_polar_cuts(gamma, cone, lambda lo, hi: 24) for cone in angles]
    low, high = families[-1][0][0], families[-1][0][-1]
    points = sorted(x for cuts, _ in families for x in cuts if low <= x <= high)
    polar = [
        (lo, hi)
        for cuts, near in families
        for lo, hi, flag in zip(cuts[:-1], cuts[1:], near)
        if flag
    ]
    branches = []
    for cone in angles:
        far = gamma + cone
        branches.append((abs(gamma - cone), cone))
        branches.append((far if far <= np.pi else 2 * np.pi - far, cone))
    tiny = 1e-13
    cuts = [points[0]]
    for x in points[1:]:
        if x - cuts[-1] > tiny:
            cuts.append(x)
    graded = [cuts[0]]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        middle = (lo + hi) / 2
        seen = [
            point
            for point, cone in branches
            if 0 < half_width(middle, cone) < np.pi
            and (point < lo - tiny or point > hi + tiny)
        ]
        below = max((x for x in seen if x < lo), default=None)
        above = min((x for x in seen if x > hi), default=None)
        nodes = count(lo, hi)

        def unresolved(distance):
            return nodes * np.sqrt(distance / (hi - lo)) < _EDGE_RESOLVED

        below = below if below is not None and unresolved(lo - below) else None
        above = above if above is not None and unresolved(above - hi) else None
        inner_lo, inner_hi = [], []
        limit = middle if below is not None and above is not None else None
        if below is not None:
            x, stop = lo, hi if limit is None else limit
            while (x - below) * _EDGE_GROWTH < stop - x:
                x = below + (x - below) * (1 + _EDGE_GROWTH)
                inner_lo.append(x)
        if above is not None:
            x, stop = hi, lo if limit is None else limit
            while (above - x) * _EDGE_GROWTH < x - stop:
                x = above - (above - x) * (1 + _EDGE_GROWTH)
                inner_hi.append(x)
        if limit is not None:
            inner_lo.append(limit)
        inner = [x for x in inner_lo + inner_hi[::-1] if lo + tiny < x < hi - tiny]
        graded += sorted(set(inner)) + [hi]
    near = [
        any(lo <= (x + y) / 2 <= hi for lo, hi in polar)
        for x, y in zip(graded[:-1], graded[1:])
    ]
    return graded, near


def _basis(axis):
    """Right-handed columns (u, v, axis)."""
    seed = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(seed, axis)
    u /= np.linalg.norm(u)
    return np.stack((u, np.cross(axis, u), axis), -1)


def _solve(K00, K01, K10, K11, x0, x1):
    det = K00 * K11 - K01 * K10
    return (K11 * x0 - K01 * x1) / det, (K00 * x1 - K10 * x0) / det


@cpu_reference
class InfiniteCylinder:
    """Exact reference for an infinite circular cylinder, ``order`` azimuthal orders.

    ``incident`` is any propagating field of the exterior region: a
    PlaneWaveExpansion (normal or oblique incidence, any number of waves) or
    a beam from fields. A FocusedBeam is expanded on its own quadrature in
    axial wavenumber and azimuth about the axis; other beams through their
    plane waves. ``reach`` (default the radius) is the axial distance beyond
    the beam's focus over which that quadrature is exact; a PlaneWaveExpansion
    has no such limit. ``axial_nodes`` overrides the number of axial nodes,
    shared between the panels of that quadrature.
    ``evaluate`` takes ``side`` and ``term`` as MieSphere does.

    The Debye terms are smooth in the axial wavenumber and converge fast. The
    total field inside also holds the cylinder's internal resonances, sharp in
    h, so its axial quadrature converges slowly: refine axial_nodes and
    compare before relying on it. Increase order to verify convergence.
    """

    @guarded
    def __repr__(self):
        """Its cylinder, indices, size parameter, order and incident field."""
        n_in, n_out = self.interior.material.n, self.exterior.material.n
        return shorten(
            f"InfiniteCylinder(radius={number(self.volume.radius)}, "
            f"axis={vector(self.volume.axis)}, n {number(n_in)} in {number(n_out)}, "
            f"x={number(self._size())}, order={self.order}, incident={self.incident!r})"
        )

    def _size(self):
        return self.k0 * self.exterior.material.n * self.volume.radius

    @guarded
    def __str__(self):
        """Its cylinder, media, size parameter, truncation, reach and incident field."""
        n_in, n_out = self.interior.material.n, self.exterior.material.n
        limit = (
            "none" if not np.isfinite(self.axial_limit) else number(self.axial_limit)
        )
        return block(
            "InfiniteCylinder: exact cylinder reference",
            [
                ("cylinder", repr(self.volume)),
                ("interior", region_text(self.interior)),
                ("exterior", region_text(self.exterior)),
                ("relative index", number(n_in / n_out)),
                (
                    "size parameter",
                    f"x = k0 n_exterior radius = {number(self._size())}",
                ),
                ("order", f"azimuthal orders |m| <= {self.order}"),
                ("axial limit", limit),
                ("wavelength0", number(self.wavelength0)),
                ("incident", repr(self.incident)),
            ],
        )

    def __init__(
        self, incident, volume, interior, *, order=None, reach=None, axial_nodes=None
    ):
        if not isinstance(volume, CylinderVolume):
            raise TypeError("InfiniteCylinder requires a CylinderVolume")
        interior = region_of(interior, "the cylinder's interior")
        if isinstance(incident, PlaneWaveExpansion):
            if np.any(incident.wavevectors[np.any(incident.amplitudes != 0, -1)].imag):
                raise ValueError("InfiniteCylinder illumination must be propagating")
        elif not hasattr(incident, "plane_waves"):
            raise TypeError(
                "InfiniteCylinder requires a PlaneWaveExpansion or a beam from fields"
            )
        for region in (incident.region, interior):
            require_transparent(region, "InfiniteCylinder", REFERENCE)
        self.incident, self.volume, self.interior = incident, volume, interior
        self.exterior = incident.region
        self.wavelength0 = incident.wavelength0
        self.k0 = 2 * np.pi / self.wavelength0
        self.basis = _basis(np.asarray(volume.axis))
        x = self.k0 * self.exterior.material.n * volume.radius
        if order is None:
            order = int(np.ceil(x + 6 * x ** (1 / 3) + 12))
        if not isinstance(order, (int, np.integer)) or order < 1:
            raise ValueError("azimuthal order must be a positive integer")
        self.order = order
        self.reach = float(volume.radius if reach is None else reach)
        if not np.isfinite(self.reach) or self.reach < 0:
            raise ValueError("reach must be finite and nonnegative")
        if axial_nodes is not None and (
            not isinstance(axial_nodes, (int, np.integer)) or axial_nodes < 1
        ):
            raise ValueError("axial_nodes must be a positive integer")
        self._axial_nodes = axial_nodes
        self.axial_limit = np.inf
        # sin of the polar angle of axial nodes laid out in it, NaN elsewhere.
        self._sine = None
        if isinstance(incident, PlaneWaveExpansion):
            self.axial, self.coefficients = self._waves(incident)
        elif getattr(incident, "is_plane_wave", False):
            self.axial, self.coefficients = self._waves(incident.plane_waves(0.0))
        elif isinstance(incident, FocusedBeam):
            self.axial, self.coefficients = self._focused(incident)
        else:
            offset = np.linalg.norm(volume.center - incident.frame.origin)
            radius = offset + volume.radius + self.reach
            self.axial, self.coefficients = self._waves(incident.plane_waves(radius))
            self.axial_limit = self.reach

    def _waves(self, waves):
        """Axial wavenumbers and regular coefficients (2, 2N+1, H) of discrete waves."""
        keep = np.any(waves.amplitudes != 0, axis=-1)
        k, amplitude = waves.wavevectors[keep].real, waves.amplitudes[keep]
        amplitude = (
            amplitude * np.exp(1j * k @ (self.volume.center - waves.origin))[:, None]
        )
        u, v, a = self.basis.T
        h = k @ a
        if np.any(np.hypot(k @ u, k @ v) < 1e-12 * np.linalg.norm(k, axis=-1)):
            raise ValueError(
                "a plane wave along the cylinder axis has no cylindrical expansion"
            )
        alpha = np.arctan2(k @ v, k @ u)
        axial = np.stack((amplitude @ a, np.cross(k, amplitude) @ a / self.k0), -1)
        scale = np.linalg.norm(k, axis=-1)
        keys, inverse = np.unique(np.round(h / scale, 12), return_inverse=True)
        n = np.arange(-self.order, self.order + 1)
        nodes, coefficients = [], []
        for group in range(len(keys)):
            members = inverse == group
            nodes.append(h[members].mean())
            sums = _azimuthal_sums(alpha[members], axial[members], self.order)
            coefficients.append((1j ** (n % 4))[:, None] * sums)
        return np.array(nodes), np.stack(coefficients, -1).transpose(1, 0, 2)

    def _focused(self, beam):
        """Product quadrature in axial wavenumber and azimuth about the axis.

        On each h the arc where the beam's cap is cut by the cone of that h is
        integrated by Gauss-Legendre, with its ends at the pupil edge; where
        the cone lies wholly in the cap, by the periodic rule. In the polar
        angle of the cone about the axis the arc's half-width has square-root
        branch points where the cap touches an extreme angle (it opens from
        zero) and, when the cap holds the axis direction, where the cone first
        fits inside it (it closes to pi). The range is cut at the closing
        angle into panels, and on each h = h_mid - (dh / 2) cos t clusters the
        nodes at both ends, which removes the branch points there and that of
        the transverse wavenumber at h = +/-k.

        When the pupil edge passes the axis direction at a small angle eps,
        on either side, the half-width varies on the scale eps (it has further
        singular points at polar angles 0 and -eps). The panels then grow
        geometrically from eps, each no wider than four times its distance to
        the axis, and are laid out in the polar angle, which resolves
        directions that h = k cos(angle) cannot tell from the axis.

        A pupil with interior edges (stops, rings, phase belts) has one such
        cone per edge. The range is cut at the branch points of every cone,
        and on each h the azimuthal rule runs zone by zone, between the arcs
        of consecutive cones: one arc, or one on either side of the centre.
        """
        k, N = beam.wavenumber, self.order
        u, v, a = self.basis.T
        normal = beam.frame.normal
        along = float(normal @ a)
        across = normal - along * a
        spread = float(np.linalg.norm(across))
        centre_azimuth = np.arctan2(across @ v, across @ u)
        top = beam._theta_max()
        # arccos(along) loses half the digits of a small angle.
        gamma = np.arctan2(spread, along)
        offset = self.volume.center - beam.frame.origin
        offset_a = float(offset @ a)
        offset_u, offset_v = float(offset @ u), float(offset @ v)

        # Cones about the beam axis where the density jumps; the cap's own last.
        cones = (*beam._polar_edges(top), top)

        def half_width(angle, cone=top):
            # Of the arc that the cone at this polar angle cuts from a cap.
            if spread <= _ALIGNED:
                return np.pi
            if np.sin(angle) <= 0:
                return np.pi if np.cos(angle) * along >= np.cos(cone) else 0.0
            c = (np.cos(cone) - along * np.cos(angle)) / (np.sin(angle) * spread)
            return float(np.arccos(np.clip(c, -1, 1)))

        def default_count(lo, hi):
            low, high = k * np.cos(hi), k * np.cos(lo)
            kappa_low, kappa_high = k * np.sin(hi), k * np.sin(lo)
            kappa_range = abs(kappa_low - kappa_high)
            if low < 0 < high:
                kappa_range = max(kappa_range, k - min(kappa_low, kappa_high))
            across = np.hypot(offset_u, offset_v) + 2 * self.volume.radius
            excursion = (high - low) * (abs(offset_a) + self.reach) + kappa_range * (
                across
            )
            # Where the arc closes over the panel its ends sweep the cone, and
            # the phase there runs at up to kappa * across per unit of t.
            sweep = max(abs(half_width(hi, e) - half_width(lo, e)) for e in cones)
            sweep /= np.pi
            excursion += sweep * max(kappa_low, kappa_high) * across
            # excursion is the one-way range of a phase quadratic in h
            # (defocus), which runs out and back over the interval; in t its
            # rate peaks at pi / 2 times the mean. The uniform rule
            # excursion / 2 thus becomes pi / 2 excursion.
            return int(np.ceil(np.pi / 2 * excursion + 8 * excursion ** (1 / 3) + 24))

        # Polar angles, from +a, between which the arc is one analytic piece;
        # near[i] marks a piece laid out in the polar angle.
        if spread > _ALIGNED:
            cuts, near = _zoned_cuts(gamma, cones, default_count, half_width)
        else:
            cuts = [0.0, *cones] if along > 0 else [np.pi - e for e in cones[::-1]]
            cuts = cuts if along > 0 else [*cuts, np.pi]
            near = [False] * (len(cuts) - 1)
        counts = [default_count(lo, hi) for lo, hi in zip(cuts[:-1], cuts[1:])]
        if self._axial_nodes:
            if self._axial_nodes < len(counts):
                raise ValueError(
                    f"axial_nodes must be at least {len(counts)} for this beam"
                )
            # Share the requested total between the panels as the default does.
            share = (
                np.cumsum(counts) / np.sum(counts) * (self._axial_nodes - len(counts))
            )
            counts = 1 + np.diff(np.r_[0, np.round(share).astype(int)])
        h, kappa_h, weight_h, sine_h = [], [], [], []
        for lo, hi, polar, panel in zip(cuts[:-1], cuts[1:], near, counts):
            x, w = leggauss(int(panel))
            t, w = np.pi / 2 * (x + 1), np.pi / 2 * w
            if polar:
                if min(lo, np.pi - hi) == 0:
                    # Full circles about the axis are analytic in the angle.
                    angle = (hi + lo) / 2 + (hi - lo) / 2 * x
                    w = (hi - lo) / np.pi * w
                else:
                    angle = (hi + lo) / 2 - (hi - lo) / 2 * np.cos(t)
                    w = (hi - lo) / 2 * np.sin(t) * w
                sine = np.sin(angle)
                # The interface system is singular along the axis; directions
                # this close to it carry < 1e-14 of the solid angle.
                keep = sine >= _EDGE_FLOOR
                sine, angle, w = sine[keep], angle[keep], w[keep]
                h.append(k * np.cos(angle))
                kappa_h.append(k * sine)
                weight_h.append(k * sine * w)
                sine_h.append(sine)
            else:
                low, high = k * np.cos(hi), k * np.cos(lo)
                mid, half = (high + low) / 2, (high - low) / 2
                h.append(mid - half * np.cos(t))
                kappa_h.append(np.sqrt(k**2 - h[-1] ** 2))
                weight_h.append(half * np.sin(t) * w)
                sine_h.append(np.full(len(t), np.nan))
        h, kappa_h = np.concatenate(h), np.concatenate(kappa_h)
        weight_h, self._sine = np.concatenate(weight_h), np.concatenate(sine_h)
        count = len(h)
        n = np.arange(-N, N + 1)
        coefficients = np.zeros((2, 2 * N + 1, count), complex)
        for q in range(count):
            kappa = kappa_h[q]
            # Half-width of the arc inside each cone: zone i lies between
            # those of cones i - 1 and i, on both sides of the centre.
            pieces, inner = [], 0.0
            for cone in cones:
                if spread * kappa > 0:
                    c = (np.cos(cone) - along * h[q] / k) / (kappa / k * spread)
                    outer = np.pi if c <= -1 else 0.0 if c >= 1 else np.arccos(c)
                else:
                    outer = np.pi if along * h[q] / k >= np.cos(cone) else 0.0
                if outer > inner:
                    pieces.append((inner, outer))
                    inner = outer
            alpha, weight = [], []
            for inner, outer in pieces:
                if inner == 0 and outer == np.pi:
                    nodes = int(2 * (N + kappa * np.hypot(offset_u, offset_v)) + 32)
                    alpha.append(2 * np.pi * np.arange(nodes) / nodes)
                    weight.append(np.full(nodes, 2 * np.pi / nodes))
                    continue
                # Arcs as (centre, half-width) about the cap's azimuth; the
                # nodes stay inside the zone, where alone its density is known.
                if inner == 0:
                    arcs = [(0.0, outer)]
                elif outer == np.pi:
                    arcs = [(np.pi, np.pi - inner)]
                else:
                    middle, width = (outer + inner) / 2, (outer - inner) / 2
                    arcs = [(-middle, width), (middle, width)]
                for middle, width in arcs:
                    start = centre_azimuth + middle
                    probe = start + width * np.linspace(-1, 1, 65)
                    slope = kappa * np.max(
                        np.abs(-np.sin(probe) * offset_u + np.cos(probe) * offset_v)
                    )
                    phase = (N + slope) * 2 * width
                    nodes = int(np.ceil(phase / 2 + 8 * phase ** (1 / 3) + 24))
                    x, w = leggauss(nodes)
                    alpha.append(start + width * x)
                    weight.append(width * w)
            if not alpha:
                continue
            alpha, weight = np.concatenate(alpha), np.concatenate(weight)
            s = (h[q] / k) * a + (kappa / k) * (
                np.cos(alpha)[:, None] * u + np.sin(alpha)[:, None] * v
            )
            amplitude = (
                _focused_density(beam, s)
                * (weight * weight_h[q] / k * np.exp(1j * k * s @ offset))[:, None]
            )
            axial = np.stack(
                (amplitude @ a, (k / self.k0) * np.cross(s, amplitude) @ a), -1
            )
            coefficients[:, :, q] = (
                (1j ** (n % 4))[:, None] * _azimuthal_sums(alpha, axial, N)
            ).T
        self.axial_limit = abs(offset_a) + self.reach
        return h, coefficients

    def _transverse(self, index, nodes):
        """Transverse wavenumber in a medium of this index at axial nodes."""
        h = self.axial[nodes]
        k = self.k0 * index
        if self._sine is None:
            return np.sqrt(k**2 - h**2)
        # Nodes laid out in the polar angle keep their own sine: h = k cos
        # cannot hold a direction closer to the axis than sqrt(eps).
        ko = self.k0 * self.exterior.material.n
        polar = (k**2 - ko**2) + (ko * self._sine[nodes]) ** 2
        return np.sqrt(np.where(np.isnan(polar), k**2 - h**2, polar))

    def _response(self, h, incident, term, kappa=None):
        """Coefficients of the selected field for axial nodes h.

        incident (2, 2N+1, H) are the regular coefficients of (E_z, eta_z).
        kappa optionally gives the transverse wavenumbers outside and inside
        where h does not determine them to working accuracy.
        Returns {"interior": (J mantissa (2, 2N+1, H), log scale), "exterior":
        (H1 mantissa, log scale)}, either None when the term has no part there.
        """
        N, R = self.order, self.volume.radius
        n = np.arange(-N, N + 1)[:, None]
        no, ni = self.exterior.material.n, self.interior.material.n
        ko, ki = self.k0 * no, self.k0 * ni
        if kappa is None:
            kappa = np.sqrt(ko**2 - h**2), np.sqrt(ki**2 - h**2)
        kappa_o, kappa_i = kappa
        xo, xi = kappa_o * R, kappa_i * R
        (vo, dvo, so), (lo, Do) = _regular(N, xo), _outgoing(N, xo)
        (vi, dvi, si), (li, Di) = _regular(N, xi), _outgoing(N, xi)
        co, ci = 1j / kappa_o**2, 1j / kappa_i**2
        alpha = (co - ci) * h * 1j * n / R
        bo, go = co * self.k0 * kappa_o, co * self.k0 * no**2 * kappa_o
        bi, gi = ci * self.k0 * kappa_i, ci * self.k0 * ni**2 * kappa_i
        a, b = incident
        Po = (-bo * b, go * a)
        Pi = (-bi * b, gi * a)
        out = {"interior": None, "exterior": None}
        log_w = np.log(2j / (np.pi * xo)) - lo  # 2i / (pi x_o H1_o)
        if term is None or isinstance(term, tuple):
            # Total field: M (c, d) = 2i/(pi x_o H1_o) P_o (a, b), J values scaled.
            M00 = alpha * vi
            M01 = -(Do * vi * bo - dvi * bi)
            M10 = Do * vi * go - dvi * gi
            interior = np.stack(_solve(M00, M01, M10, M00, *Po))
            out["interior"] = (interior, log_w - si)
            # c J_i - (a, b) J_o, by the Wronskian, over H1_o.
            Y0 = vo * dvi * Pi[0] - dvo * vi * Po[0] - alpha * vo * vi * a
            Y1 = vo * dvi * Pi[1] - dvo * vi * Po[1] - alpha * vo * vi * b
            out["exterior"] = (np.stack(_solve(M00, M01, M10, M00, Y0, Y1)), so - lo)
            if term is None:
                return out
        K00 = alpha
        K01 = -(Do * bo - np.conj(Di) * bi)
        K10 = Do * go - np.conj(Di) * gi

        def through(y0, y1):
            # T_out y = K^-1 (L1_i - L2_i) P_i y
            wi = 2j * Di.imag
            return _solve(K00, K01, K10, K00, -wi * bi * y1, wi * gi * y0)

        G = np.exp(2j * li.imag)
        first = np.stack(_solve(K00, K01, K10, K00, *Po))  # times e^{log_w}
        terms = (
            range(1, term + 1)
            if isinstance(term, (int, np.integer))
            else range(1, term[1] + 1)
        )
        wave, interior, exterior = first, [], []
        for p in terms:
            interior.append(wave)
            exterior.append(np.stack(through(*(G * wave))))
            # next pass: G R_in applied to the inward surface value
            wave = G * (np.stack(through(*wave)) - wave)
        scale_in = log_w + np.log(2) - np.conj(li)
        scale_out = log_w - lo
        # term 0: -K^-1 [alpha J_o + J_o' P_o - L2_i J_o P_i] (a, b) / H1_o
        X0 = alpha * vo * a + dvo * Po[0] - np.conj(Di) * vo * Pi[0]
        X1 = alpha * vo * b + dvo * Po[1] - np.conj(Di) * vo * Pi[1]
        zeroth = -np.stack(_solve(K00, K01, K10, K00, X0, X1))
        if isinstance(term, (int, np.integer)):
            if term == 0:
                out["exterior"] = (zeroth, so - lo)
            else:
                out["interior"] = (interior[-1], scale_in)
                out["exterior"] = (exterior[-1], scale_out)
            return out
        # ("after", P): total minus terms 0..P, carried on a common scale.
        (ci_m, ci_s), (ce_m, ce_s) = out["interior"], out["exterior"]
        rest_in = ci_m - np.exp(scale_in - ci_s) * sum(interior, 0 * ci_m)
        rest_out = ce_m - np.exp(so - lo - ce_s) * zeroth
        rest_out = rest_out - np.exp(scale_out - ce_s) * sum(exterior, 0 * ce_m)
        out["interior"], out["exterior"] = (rest_in, ci_s), (rest_out, ce_s)
        return out

    def _fields(self, h, coefficients, scale, kappa, index, rho, phi, regular):
        """(Hc, U, 6) cylindrical components (E_rho, E_phi, E_z, eta_rho, eta_phi, eta_z)."""
        N = self.order
        n = np.arange(-N, N + 1)[:, None]
        rho = np.maximum(rho, 1e-12 * self.volume.radius)
        x = kappa[:, None] * rho[None]
        if regular:
            v, dv, s = _regular(N, x)
        else:
            lg, D = _outgoing(N, x)
        angular = np.exp(1j * n * phi[None])
        out = np.empty((len(h), len(rho), 6), complex)
        for q in range(len(h)):
            with np.errstate(over="ignore", invalid="ignore"):
                if regular:
                    w = np.exp(scale[:, q, None] + s[:, q])
                    Z, dZ = w * v[:, q], w * dv[:, q]
                else:
                    w = np.exp(scale[:, q, None] + lg[:, q])
                    Z, dZ = w, w * D[:, q]
            Ze, dZe = Z * angular, dZ * angular
            mE, mH = coefficients[0, :, q], coefficients[1, :, q]
            Ez, Hz = mE @ Ze, mH @ Ze
            dE, dH = kappa[q] * (mE @ dZe), kappa[q] * (mH @ dZe)
            nE, nH = (1j * n[:, 0] * mE) @ Ze / rho, (1j * n[:, 0] * mH) @ Ze / rho
            c = 1j / kappa[q] ** 2
            out[q] = np.stack(
                (
                    c * (h[q] * dE + self.k0 * nH),
                    c * (h[q] * nE - self.k0 * dH),
                    Ez,
                    c * (h[q] * dH - self.k0 * index**2 * nE),
                    c * (h[q] * nH + self.k0 * index**2 * dE),
                    Hz,
                ),
                -1,
            )
        return out

    def evaluate(self, grid, *, side=None, term=None, chunk_size=8, method="auto"):
        """Total field, or one Debye term, inside or outside the cylinder.

        side and term as for MieSphere: term None is the total field (incident
        plus scattered outside), term p >= 1 the waves that entered and met
        p - 1 internal reflections (p = 1 is the first entry), term 0 the
        external reflection and diffraction outside, ("after", P) the rest.
        chunk_size axial nodes are summed at a time. Points that share their
        position across the axis share the cylindrical-wave sums, so planes
        and lines containing the axis direction are cheap. method selects how the incident field is summed outside ("auto",
        "direct" or "nufft", as for PlaneWaveExpansion.evaluate).
        """
        _check_term(term)
        if side not in (None, "interior", "exterior"):
            raise ValueError("side must be interior, exterior or None")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        points = grid.points.reshape(-1, 3)
        local = (points - self.volume.center) @ self.basis
        rho = np.hypot(local[:, 0], local[:, 1])
        phi, z = np.arctan2(local[:, 1], local[:, 0]), local[:, 2]
        R = self.volume.radius
        eps = 1e-10 * R
        if side == "interior" and np.any(rho > R + eps):
            raise ValueError("interior evaluation outside the cylinder")
        if side == "exterior" and np.any(rho < R - eps):
            raise ValueError("exterior evaluation inside the cylinder")
        if np.any(np.abs(z) > self.axial_limit * (1 + 1e-12) + eps):
            raise ValueError(
                "evaluation beyond the axial reach of the beam's expansion; pass reach="
            )
        inside = rho < R if side is None else np.full(len(points), side == "interior")
        values = np.zeros((len(points), 6), complex)
        for flag, key in ((True, "interior"), (False, "exterior")):
            ids = np.flatnonzero(inside == flag)
            if not len(ids) or (flag and term == 0):
                continue
            index = (self.interior if flag else self.exterior).material.n
            transverse = np.round(np.c_[rho[ids], phi[ids]] / [R, 1], 12)
            unique, where = np.unique(transverse, axis=0, return_inverse=True)
            where = where.ravel()
            urho, uphi = unique[:, 0] * R, unique[:, 1]
            block = max(1, _BLOCK // max(1, chunk_size))
            for start in range(0, len(self.axial), chunk_size):
                stop = min(start + chunk_size, len(self.axial))
                nodes = slice(start, stop)
                h = self.axial[nodes]
                no, ni = self.exterior.material.n, self.interior.material.n
                part = self._response(
                    h,
                    self.coefficients[:, :, nodes],
                    term,
                    (self._transverse(no, nodes), self._transverse(ni, nodes)),
                )[key]
                if part is None:
                    continue
                kappa = self._transverse(index, nodes)
                for first in range(0, len(urho), block):
                    sel = slice(first, first + block)
                    fields = self._fields(
                        h, part[0], part[1], kappa, index, urho[sel], uphi[sel], flag
                    )
                    members = np.flatnonzero((where >= first) & (where < first + block))
                    phase = np.exp(1j * np.outer(z[ids[members]], h))
                    values[ids[members]] += np.einsum(
                        "ph,hpc->pc", phase, fields[:, where[members] - first]
                    )
        cp, sp = np.cos(phi), np.sin(phi)
        u, v, a = self.basis.T

        def cartesian(r, t, axial):
            return (
                (r * cp - t * sp)[:, None] * u
                + (r * sp + t * cp)[:, None] * v
                + axial[:, None] * a
            )

        E = cartesian(*values[:, :3].T)
        H = cartesian(*values[:, 3:].T) / Z0
        outside = ~inside
        if term is None and np.any(outside):
            incident = self.incident.evaluate(PointGrid(points[outside]), method=method)
            E[outside] += incident.electric.values
            H[outside] += incident.magnetic
        if not np.isfinite(E).all() or not np.isfinite(H).all():
            raise ValueError(
                "nonfinite cylinder field; check order and observation radii"
            )
        return ElectromagneticField(
            ElectricField(grid, self.wavelength0, E.reshape(grid.shape + (3,))),
            H.reshape(grid.shape + (3,)),
        )
