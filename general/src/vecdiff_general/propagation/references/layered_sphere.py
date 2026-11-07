"""Concentric spherical layers: exact multipole solution and its Debye series.

Regions j = 0..L are the incident region and then the layers inward; layer
j fills volumes[j - 1] minus volumes[j]. In region j a multipole (n, m) of
polarization TE (A, M waves) or TM (B, N waves) has the Riccati amplitude

    X(r) = c_psi psi_n(k_j r) + c_xi xi_n(k_j r) + c_zeta zeta_n(k_j r),

psi = x j_n, xi = x h1_n (outgoing), zeta = x h2_n (incoming), with the field
coefficients of MieSphere (E ~ A z_n X_nm + B W_nm, z_n = X / x). Across the
interface between a (outside) and b (inside), m = n_b / n_a,

    TE: X_a = X_b / m,  X_a' = X_b',      TM: X_a = X_b,  X_a' = X_b' / m.

Total field. Region j holds C_j [psi - sigma_j xi]: sigma_j scatters for
everything inside it, sigma_L = 0 in the core and sigma_0 = b_n (TE), a_n (TM)
outside. Inside out, sigma = (psi X' - psi' X) / (xi X' - xi' X) from the
Cauchy data (X, X') of the field within; outside in, C_j from the continuity
of X and X'. Regular and Hankel functions carry their own logarithmic scales,
so a layer in which a multipole tunnels neither overflows nor loses its
decaying field to cancellation.

Debye series. For unit surface values of the incoming (zeta_a) or outgoing
(xi_b) wave, the one-way coefficients of an interface follow from the
logarithmic derivatives D of xi, whose imaginary part is the Wronskian 1/|xi|²:

    Delta = D_a - nu conj(D_b),   nu = m (TE), 1 / m (TM),
    r_out = (nu conj(D_b) - conj(D_a)) / Delta,   t_in = 2i / (mu |xi_a|² Delta),
    r_in = (nu D_b - D_a) / Delta,                t_out = 2i mu nu / (|xi_b|² Delta),

mu = 1/m (TE), 1 (TM). The response of everything inside a radius, outgoing
over incoming surface value, has unit modulus and crosses a layer as a phase;
in the core it is xi / zeta, the inward wave continued through the centre.
With each reflection weighted by lambda every field is a power series in
lambda. Debye term p >= 1 is its coefficient of lambda^(p - 1): the waves that
entered and met p - 1 reflections, at any interface and from either side.
Term 0 is the external reflection with the diffraction that forms the shadow,
(r_out zeta_0 / xi_0 - 1) / 2 outside. Term 1 is the first entry: transmitted
inward through every interface, never reflected, continued through the
centre. For one layer the terms are those of MieSphere.
"""

import numpy as np

from vecdiff_general._backend import cpu_reference
from vecdiff_general.fields import ElectricField, ElectromagneticField
from vecdiff_general.media.medium import Z0, HomogeneousRegion
from vecdiff_general.propagation.references import _bessel
from vecdiff_general.propagation.references.mie import (
    _spherical_sum,
    _SphericalExpansion,
)
from vecdiff_general.sampling import PointGrid
from vecdiff_general.volumes import SphereVolume

_POLARIZATIONS = ("TE", "TM")


def _product(a, b):
    """Product of power series in lambda along axis 0, truncated at their degree."""
    out = np.zeros(np.broadcast_shapes(a.shape, b.shape), complex)
    for k in range(len(out)):
        for i in range(k + 1):
            out[k] += a[i] * b[k - i]
    return out


def _geometric(x):
    """(1 - lambda x)^-1 for a power series x, truncated at its degree."""
    w = np.zeros_like(x)
    w[0] = 1
    for k in range(1, len(x)):
        for i in range(k):
            w[k] += x[i] * w[k - 1 - i]
    return w


def _check_term(term):
    if term is not None and not (
        (isinstance(term, (int, np.integer)) and term >= 0)
        or (
            isinstance(term, tuple)
            and len(term) == 2
            and term[0] == "after"
            and isinstance(term[1], (int, np.integer))
            and term[1] >= 0
        )
    ):
        raise ValueError("term must be None, an integer p >= 0 or ('after', P)")


@cpu_reference
class LayeredSphere(_SphericalExpansion):
    """Exact reference for concentric spherical layers, truncated at ``order``.

    ``volumes`` are concentric SphereVolumes, outermost first, and
    ``interiors`` the regions filling them: interiors[i] occupies volumes[i]
    minus volumes[i + 1]. ``incident`` is any propagating field of the
    exterior region, as for MieSphere: a PlaneWaveExpansion or a beam from
    fields. Region j = 0 is the exterior, j = len(volumes) the core.

    ``evaluate`` returns the total field or one Debye term (``term``), in the
    region of each point or, with ``side``, in one region for every point so
    that both traces of an interface can be taken at identical points.
    Increase order to verify convergence; the automatic order is a starting
    estimate, not an accuracy certificate.
    """

    def __init__(self, incident, volumes, interiors, *, order=None):
        volumes, interiors = tuple(volumes), tuple(interiors)
        if not volumes or len(volumes) != len(interiors):
            raise ValueError("give one interior region for every volume")
        if not all(isinstance(v, SphereVolume) for v in volumes):
            raise TypeError("LayeredSphere requires SphereVolumes")
        if not all(isinstance(r, HomogeneousRegion) for r in interiors):
            raise TypeError("interiors must be HomogeneousRegions")
        centre, outer = volumes[0].center, volumes[0].radius
        if any(np.max(np.abs(v.center - centre)) > 1e-12 * outer for v in volumes):
            raise ValueError("layers must be concentric")
        radii = [float(v.radius) for v in volumes]
        if any(inner >= r for r, inner in zip(radii, radii[1:])):
            raise ValueError("radii must decrease inward")
        self._check_incident(incident, "LayeredSphere")
        self.incident, self.volumes, self.interiors = incident, volumes, interiors
        self.volume, self.exterior = volumes[0], incident.region
        self.regions = (self.exterior,) + interiors
        self.radii = tuple(radii)
        self.wavelength0 = incident.wavelength0
        self.k0 = 2 * np.pi / self.wavelength0
        x = self.k0 * self.exterior.material.n * outer
        if order is None:
            order = int(np.ceil(x + 6 * x ** (1 / 3) + 12))
        if not isinstance(order, (int, np.integer)) or order < 1:
            raise ValueError("multipole order must be a positive integer")
        self.order = order
        self._outside, self._inside = [], []
        for i, radius in enumerate(radii):
            pair = self.k0 * radius * np.array([self._index(i), self._index(i + 1)])
            value, derivative, scale = (t[1:] for t in _bessel.regular(order, pair))
            log, dlog = (t[1:] for t in _bessel.outgoing(order, pair))
            for side, store in ((0, self._outside), (1, self._inside)):
                store.append(
                    dict(
                        v=value[:, side],
                        dv=derivative[:, side],
                        s=scale[:, side],
                        lx=log[:, side],
                        D=dlog[:, side],
                    )
                )
        self._total = {p: self._standing(p) for p in _POLARIZATIONS}
        sigma = {p: np.exp(self._total[p][1][0]) for p in _POLARIZATIONS}
        self.a, self.b = sigma["TM"], sigma["TE"]
        if not all(
            np.isfinite(c).all() for p in _POLARIZATIONS for c in self._total[p][0]
        ) or not all(np.isfinite(c).all() for c in (self.a, self.b)):
            raise ValueError("nonfinite layered-sphere coefficients")
        self._debye = {}
        self.basis, self.coefficients = self._project()

    def _index(self, region):
        return self.regions[region].material.n

    def _ratios(self, interface, polarization):
        """(mu_v, mu_d) of X_a = mu_v X_b, X_a' = mu_d X_b' at an interface."""
        m = self._index(interface) / self._index(interface + 1)
        return (m, 1.0) if polarization == "TE" else (1.0, m)

    def _standing(self, polarization):
        """log C_j (j = 0..L) and log sigma_j (j = 0..L-1), orders 1..order."""
        layers = len(self.interiors)
        log_sigma, outer, inner = [None] * layers, [None] * layers, [None] * layers
        core = self._inside[-1]
        u, du = core["v"].astype(complex), core["dv"].astype(complex)
        inner[-1] = (u, du, core["s"])
        for i in range(layers - 1, -1, -1):
            mv, md = self._ratios(i, polarization)
            u, du = mv * u, md * du
            size = np.maximum(np.abs(u), np.abs(du))
            u, du = u / size, du / size
            a = self._outside[i]
            # sigma xi_a / e^{s_a}: bounded, no division by psi or its zeros.
            ratio = (a["v"] * du - a["dv"] * u) / (du - a["D"] * u)
            with np.errstate(divide="ignore"):  # sigma = 0 if index-matched
                log_sigma[i] = a["s"] - a["lx"] + np.log(ratio)
            outer[i] = (a["v"] - ratio, a["dv"] - ratio * a["D"], a["s"])
            if i:
                # Region i at its outer radius: psi - sigma xi with own scales.
                c = self._inside[i - 1]
                term = log_sigma[i] + c["lx"]
                top = np.maximum(c["s"], term.real)
                w, p = np.exp(term - top), np.exp(c["s"] - top)
                u, du = c["v"] * p - w, c["dv"] * p - c["D"] * w
                inner[i - 1] = (u, du, top)
        log_C = [np.zeros(self.order, complex)]
        for i in range(layers):
            mv, md = self._ratios(i, polarization)
            (Fa, dFa, Sa), (Fb, dFb, Sb) = outer[i], inner[i]
            numerator = mv * np.conj(Fb) * Fa + md * np.conj(dFb) * dFa
            denominator = mv**2 * np.abs(Fb) ** 2 + md**2 * np.abs(dFb) ** 2
            log_C.append(log_C[-1] + Sa - Sb + np.log(numerator / denominator))
        return log_C, log_sigma

    def _interface(self, i, polarization):
        """r_out, log t_in, r_in, log t_out of interface i for unit surface values.

        Also log of (R - 1) / 2 for the coefficient ratio R = r_out zeta_a / xi_a,
        the exterior term 0, from 2 (nu conj(D_b) psi_a - psi_a') / (Delta xi_a):
        the subtraction is done by the Wronskian, so it keeps relative accuracy
        where R is within roundoff of 1.
        """
        zero = np.zeros(self.order, complex)
        if self._index(i) == self._index(i + 1):
            return zero, zero, zero, zero, zero + np.log(0.5) + 1j * np.pi
        mv, md = self._ratios(i, polarization)
        nu = md / mv
        a, b = self._outside[i], self._inside[i]
        Da, Db = a["D"], b["D"]
        delta = Da - nu * np.conj(Db)
        return (
            (nu * np.conj(Db) - np.conj(Da)) / delta,
            np.log(2j / (mv * delta)) - 2 * a["lx"].real,
            (nu * Db - Da) / delta,
            np.log(2j * md / delta) - 2 * b["lx"].real,
            a["s"] - a["lx"] + np.log((nu * np.conj(Db) * a["v"] - a["dv"]) / delta),
        )

    def _series(self, degree):
        """Debye amplitudes up to lambda^degree: {region: [(kind, scale, series)]}."""
        if degree in self._debye:
            return self._debye[degree]
        layers, shape = len(self.interiors), (degree + 1, self.order)
        result = {}
        for polarization in _POLARIZATIONS:
            one_way = [self._interface(i, polarization) for i in range(layers)]
            response, resolvent = [None] * layers, [None] * layers
            G = np.zeros(shape, complex)
            G[0] = np.exp(2j * self._inside[-1]["lx"].imag)
            for i in range(layers - 1, -1, -1):
                r_out, log_t_in, r_in, log_t_out, _ = one_way[i]
                response[i], resolvent[i] = G, _geometric(r_in * G)
                if i:
                    outward = np.exp(log_t_in + log_t_out) * _product(G, resolvent[i])
                    if degree:
                        outward[1] += r_out
                    phase = self._inside[i - 1]["lx"].imag - self._outside[i]["lx"].imag
                    G = outward * np.exp(2j * phase)
            regions = {}
            scale = np.log(0.5) + np.conj(self._outside[0]["lx"])
            series = np.zeros(shape, complex)
            series[0] = 1
            for i in range(layers):
                r_out, log_t_in, r_in, log_t_out, _ = one_way[i]
                scale, series = scale + log_t_in, _product(series, resolvent[i])
                if i == 0:
                    entered = (scale, series)
                b = self._inside[i]
                if i == layers - 1:
                    regions[i + 1] = [
                        ("psi", np.log(2) + scale - np.conj(b["lx"]), series)
                    ]
                else:
                    outgoing = _product(response[i], series)
                    regions[i + 1] = [
                        ("zeta", scale - np.conj(b["lx"]), series),
                        ("xi", scale - b["lx"], outgoing),
                    ]
                    scale = scale + np.conj(self._outside[i + 1]["lx"] - b["lx"])
            *_, log_t_out, log_reflection = one_way[0]
            regions[0] = [
                (
                    "xi",
                    log_t_out + entered[0] - self._outside[0]["lx"],
                    _product(response[0], entered[1]),
                )
            ]
            result[polarization] = regions, log_reflection
        self._debye[degree] = result
        return result

    def _entries(self, region, term):
        """[(kind, log c_TE, log c_TM)] of the selected field in one region."""
        if term is None or isinstance(term, tuple):
            entries = []
            for p in _POLARIZATIONS:
                log_C, log_sigma = self._total[p]
                if region == 0:
                    pairs = [("xi", log_sigma[0] + 1j * np.pi)]
                elif region == len(self.interiors):
                    pairs = [("psi", log_C[region])]
                else:
                    pairs = [
                        ("psi", log_C[region]),
                        ("xi", log_C[region] + log_sigma[region] + 1j * np.pi),
                    ]
                entries.append(pairs)
            total = [(k, te, tm) for (k, te), (_, tm) in zip(*entries)]
            if term is None:
                return total
            terms = range(0 if region == 0 else 1, term[1] + 1)
            return total + [
                (kind, te + 1j * np.pi, tm + 1j * np.pi)
                for p in terms
                for kind, te, tm in self._entries(region, p)
            ]
        if term == 0 and region:
            return []
        series = self._series(max(term - 1, 0))
        entries = []
        for p in _POLARIZATIONS:
            regions, log_reflection = series[p]
            if term == 0:
                entries.append([("xi", log_reflection)])
            else:
                entries.append(
                    [(k, s + np.log(c[term - 1]), None) for k, s, c in regions[region]]
                )
        if term == 0:
            return [("xi", entries[0][0][1], entries[1][0][1])]
        return [(k, te, tm) for (k, te, _), (_, tm, _) in zip(*entries)]

    def efficiencies(self):
        """Scattering/extinction cross sections divided by pi radius² (one plane wave)."""
        from vecdiff_general.fields.spectrum import PlaneWaveExpansion

        if not (
            isinstance(self.incident, PlaneWaveExpansion)
            and len(self.incident.amplitudes) == 1
        ):
            raise TypeError("efficiencies are defined for one incident plane wave")
        ell = np.arange(1, self.order + 1)
        x = self.k0 * self.exterior.material.n * self.radii[0]
        return {
            "scattering": float(
                2 / x**2 * np.sum((2 * ell + 1) * (abs(self.a) ** 2 + abs(self.b) ** 2))
            ),
            "extinction": float(
                2 / x**2 * np.sum((2 * ell + 1) * (self.a + self.b).real)
            ),
        }

    def _fields(self, local, region, term):
        """Fields of the selected term at local points of one region."""
        index = self._index(region)
        radius = np.linalg.norm(local, axis=-1)
        x = self.k0 * index * np.maximum(radius, 1e-12 * self.radii[-1])
        count = self.order
        tables, radial = {}, [np.zeros((count, len(x)), complex) for _ in range(4)]
        for kind, log_te, log_tm in self._entries(region, term):
            if kind == "psi":
                if "psi" not in tables:
                    tables["psi"] = [
                        t[1 : count + 1] for t in _bessel.regular(count, x)
                    ]
                value, derivative, scale = tables["psi"]
            else:
                if "xi" not in tables:
                    tables["xi"] = [
                        t[1 : count + 1] for t in _bessel.outgoing(count, x)
                    ]
                scale, derivative = tables["xi"]
                if kind == "zeta":
                    scale, derivative = np.conj(scale), np.conj(derivative)
                value = 1.0
            for log_c, (Z, D) in ((log_te, radial[:2]), (log_tm, radial[2:])):
                with np.errstate(over="ignore", invalid="ignore"):
                    w = np.exp(log_c[:count, None] + scale) / x
                Z += w * value
                D += w * derivative
        ZA, DA, ZB, DB = radial
        return _spherical_sum(
            self.coefficients, self.basis, local, x, ZA, DA, ZB, DB, index
        )

    def evaluate(self, grid, *, side=None, term=None, chunk_size=128):
        """Total field, or one Debye term, in each point's region or in ``side``.

        side None takes the region containing each point (a point on an
        interface belongs to the region outside it); "exterior", "interior"
        (the core) or a region index j = 0..L evaluates that region's solution
        at every point, which must lie within its closure. term None is the
        total field: incident plus scattered outside. Term p >= 1 contains the
        waves that entered and met p - 1 reflections; inside it is regular in
        the core and made of incoming and outgoing waves in a layer, outside
        it is what leaves. Term 0 exists outside only. ("after", P) is every
        term beyond P, so terms 0..P plus it recover the total minus the
        incident field outside and the total inside.

        Inside a layer a Debye term of a multipole that turns within it grows
        like the irregular function beyond its turning point and cancels only
        in the sum, as the exterior Debye parts do beyond the incident tail.
        """
        _check_term(term)
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        layers = len(self.interiors)
        side = {"exterior": 0, "interior": layers}.get(side, side)
        if side is not None and side not in range(layers + 1):
            raise ValueError("side must be exterior, interior, a region index or None")
        points = grid.points.reshape(-1, 3)
        distance = np.linalg.norm(points - self.volume.center, axis=-1)
        radii = np.array(self.radii)
        region = np.sum(distance[:, None] < radii, axis=1)
        if side is not None:
            eps = 1e-10 * self.radii[0]
            bounds = np.r_[np.inf, radii, 0.0]
            if np.any(distance > bounds[side] + eps) or np.any(
                distance < bounds[side + 1] - eps
            ):
                raise ValueError(f"evaluation outside region {side}")
            region = np.full(len(points), side)
        E, H = np.zeros_like(points, complex), np.zeros_like(points, complex)
        local = (points - self.volume.center) @ self.basis
        for j in range(layers + 1):
            if j and term == 0:
                continue
            indices = np.flatnonzero(region == j)
            for start in range(0, len(indices), chunk_size):
                ids = indices[start : start + chunk_size]
                electric, magnetic = self._fields(local[ids], j, term)
                E[ids] += electric
                H[ids] += magnetic / Z0
        outside = region == 0
        if term is None and np.any(outside):
            incident = self.incident.evaluate(PointGrid(points[outside]))
            E[outside] += incident.electric.values
            H[outside] += incident.magnetic
        if not np.isfinite(E).all() or not np.isfinite(H).all():
            raise ValueError(
                "nonfinite layered-sphere field; check order, observation radii "
                "and whether a Debye term is asked where it does not cancel"
            )
        return ElectromagneticField(
            ElectricField(grid, self.wavelength0, E.reshape(grid.shape + (3,))),
            H.reshape(grid.shape + (3,)),
        )
