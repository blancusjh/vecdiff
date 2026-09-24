"""Local inverse-wavenumber dielectric expansion, with complete normal jets.

Independent implementation of the supplied eikonal/transport recurrence. The
ordinary non-grazing, propagating expansion is experimental; it is not a uniform
grazing/critical approximation or a closed-body scattering solver. Order zero
is Fresnel. No reference fields or fitted amplitude enter the coefficients.
"""

from dataclasses import dataclass, field

import numpy as np

from ...fields import ElectricField, ElectromagneticField
from ...media.medium import Z0
from ...sampling import PointGrid, SurfaceGrid
from ._taylor import TaylorSpace, cross, dot, solve_trace
from .asymptotic import wkb_electric_jet


class _GraphCalculus:
    def __init__(self, height):
        self.h = height
        self.g = [height.derivative(i) for i in range(2)]

    def derivative(self, a, axis):
        if axis == 2:
            return a.derivative(2)
        return a.derivative(axis) - self.g[axis] * a.derivative(2)

    def gradient(self, a):
        return [self.derivative(a, i) for i in range(3)]

    def divergence(self, a):
        return sum(self.derivative(a[i], i) for i in range(3))

    def curl(self, a):
        return [
            self.derivative(a[2], 1) - self.derivative(a[1], 2),
            self.derivative(a[0], 2) - self.derivative(a[2], 0),
            self.derivative(a[1], 0) - self.derivative(a[0], 1),
        ]

    def laplacian(self, a):
        return sum(self.derivative(self.derivative(a, i), i) for i in range(3))

    def phase(self, boundary, index, sign):
        s = boundary + 0
        g = self.g
        metric = 1 + dot(g, g)
        metric_inv = metric.reciprocal()
        for power in range(1, s.space.degree + 1):
            tangent = [s.derivative(i) for i in range(2)]
            gd = dot(g, tangent)
            root = (gd * gd + metric * (index**2 - dot(tangent, tangent))) ** 0.5
            rhs = (gd + sign * root) * metric_inv
            s.set_normal_coefficient(rhs, power, s.space.degree)
        return s

    def transport(self, boundary, phase, previous, degree):
        w = self.gradient(phase)
        divw = self.laplacian(phase)
        inverse = (w[2] - dot(w[:2], self.g)).reciprocal()
        output = [v.truncate(degree) for v in boundary]
        forcing = [self.laplacian(v) for v in previous]
        for power in range(1, degree + 1):
            for i, a in enumerate(output):
                rhs = (
                    -(
                        w[0] * a.derivative(0)
                        + w[1] * a.derivative(1)
                        + 0.5 * divw * a
                        + 0.5 * forcing[i]
                    )
                    * inverse
                )
                a.set_normal_coefficient(rhs, power, degree)
        return output


@dataclass
class InterfaceExpansion:
    """Dimensionless jets with physical length_scale and Cartesian vector axes."""

    calculus: _GraphCalculus
    phases: tuple
    amplitudes: tuple
    points: np.ndarray
    normals: np.ndarray
    length_scale: float
    indices: tuple
    order: int
    _derivatives: dict = field(default_factory=dict, repr=False)
    _traces: dict = field(default_factory=dict, repr=False)

    def trace(self, branch, wavelength0, *, order=None, grid=None):
        """Surface E and complete-curl H without constructing second derivatives."""
        which = {"incident": 0, "reflected": 1, "transmitted": 2}[branch]
        order = self.order if order is None else order
        if not isinstance(order, (int, np.integer)) or not 0 <= order <= self.order:
            raise ValueError("retained order must lie within the computed expansion")
        if not np.isfinite(wavelength0) or wavelength0 <= 0:
            raise ValueError("wavelength must be positive and finite")
        grid = PointGrid(self.points) if grid is None else grid
        if not np.array_equal(grid.points, self.points):
            raise ValueError("trace grid must use the expansion points")
        ik = 2j * np.pi * self.length_scale / wavelength0
        c = self.calculus
        s = self.phases[which]
        if which not in self._traces:
            self._traces[which] = (
                np.stack([v.c[0] for v in c.gradient(s)], -1),
                [
                    (
                        np.stack([v.c[0] for v in a], -1),
                        np.stack([v.c[0] for v in c.curl(a)], -1),
                    )
                    for a in self.amplitudes[which]
                ],
            )
        w, coefficients = self._traces[which]
        electric = np.zeros_like(w)
        magnetic = np.zeros_like(w)
        for m, (av, ca) in enumerate(coefficients[: order + 1]):
            electric += av / ik**m
            magnetic += np.cross(w, av) / ik**m + ca / ik ** (m + 1)
        phase = np.exp(ik * s.c[0])[:, None]
        return ElectromagneticField(
            ElectricField(grid, wavelength0, phase * electric), phase * magnetic / Z0
        )

    def jet(self, branch, wavelength0, *, order=None, grid=None):
        """Physical E and full derivatives for a retained local branch field.

        branch is 'incident', 'reflected', or 'transmitted'. Normal derivatives
        are from transport, not surface finite differences. Grid weights affect
        diagnostics only and never enter the material coefficient recurrence.
        """
        which = {"incident": 0, "reflected": 1, "transmitted": 2}[branch]
        order = self.order if order is None else order
        if not isinstance(order, (int, np.integer)) or not 0 <= order <= self.order:
            raise ValueError("retained order must lie within the computed expansion")
        if grid is None:
            grid = PointGrid(self.points)
        if not np.allclose(
            grid.points, self.points, rtol=0, atol=1e-14 * self.length_scale
        ):
            raise ValueError("jet grid must use the expansion's physical points")
        if which in self._derivatives:
            return wkb_electric_jet(
                grid, wavelength0, *self._derivatives[which], order=order
            )
        c, scale = self.calculus, self.length_scale
        phase = self.phases[which]
        phase_gradient = c.gradient(phase)
        s = phase.c[0].real * scale
        w = np.stack([v.c[0] for v in phase_gradient], -1)
        ds = (
            np.stack(
                [
                    np.stack([c.derivative(v, i).c[0] for v in phase_gradient], -1)
                    for i in range(3)
                ],
                -2,
            )
            / scale
        )
        a, da, dda = [], [], []
        for m, coeff in enumerate(self.amplitudes[which]):
            grad = [[c.derivative(v, i) for v in coeff] for i in range(3)]
            a.append(np.stack([v.c[0] for v in coeff], -1) / scale**m)
            da.append(
                np.stack([np.stack([v.c[0] for v in row], -1) for row in grad], -2)
                / scale ** (m + 1)
            )
            dda.append(
                np.stack(
                    [
                        np.stack(
                            [
                                np.stack([c.derivative(v, j).c[0] for v in row], -1)
                                for row in grad
                            ],
                            -2,
                        )
                        for j in range(3)
                    ],
                    -3,
                )
                / scale ** (m + 2)
            )
        self._derivatives[which] = (s, w, ds, np.array(a), np.array(da), np.array(dda))
        return wkb_electric_jet(
            grid,
            wavelength0,
            s,
            w,
            ds,
            np.array(a),
            np.array(da),
            np.array(dda),
            order=order,
        )

    def surface_grid(self, projected_weights):
        """Use caller-supplied physical dx dy quadrature, converting it to dS."""
        g = self.calculus.g
        jac = np.sqrt(1 + sum(v.c[0].real ** 2 for v in g))
        return SurfaceGrid(
            self.points, self.normals, np.asarray(projected_weights) * jac
        )


def interface_expansion(
    height,
    xy,
    *,
    n_incident,
    n_transmitted,
    direction=(0, 0, 1),
    amplitude=(1, 0, 0),
    order=3,
    length_scale=1.0,
    derivative_depth=2,
):
    """Generate local A_0…A_N for a plane wave and an analytic graph.

    ``xy`` are physical chart positions, shape (points,2). ``height(u,v)`` is a
    Taylor-compatible callable returning h/L from dimensionless u=x/L,v=y/L,
    where L=length_scale. It uses +,-,*,/, nonnegative integer powers and sqrt
    expressed as **0.5, and .cos(). No particular surface family enters the recurrence.

    At least two derivatives of A_N are retained for full Maxwell residuals.
    Additional Taylor depth does not increase the retained optical order.
    Incident illumination is one real-direction plane wave; apply each spectral
    component separately and sum coherently. Critical/TIR/grazing cases raise.
    """
    if isinstance(order, bool) or not isinstance(order, int) or not 0 <= order <= 3:
        raise ValueError("order must be an integer from zero through three")
    if not isinstance(derivative_depth, int) or derivative_depth < 2:
        raise ValueError("at least two amplitude derivatives are required")
    if (
        not np.isfinite([length_scale, n_incident, n_transmitted]).all()
        or min(length_scale, n_incident, n_transmitted) <= 0
    ):
        raise ValueError("length scale and indices must be finite and positive")
    xy = np.asarray(xy, float)
    u = np.asarray(direction, float)
    incident_a = np.asarray(amplitude, complex)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError("xy must be finite (points,2) coordinates")
    if u.shape != (3,) or not np.isfinite(u).all() or abs(np.dot(u, u) - 1) > 1e-12:
        raise ValueError("incident direction must be a real unit vector")
    if (
        incident_a.shape != (3,)
        or not np.isfinite(incident_a).all()
        or abs(np.dot(u, incident_a)) > 1e-12 * np.linalg.norm(incident_a)
    ):
        raise ValueError(
            "incident electric amplitude must be a finite transverse vector"
        )
    degree = 2 * order + derivative_depth + 2
    space = TaylorSpace(degree, len(xy))
    x = space.variable(0, xy[:, 0] / length_scale)
    y = space.variable(1, xy[:, 1] / length_scale)
    h = space.constant(0) + height(x, y)
    calc = _GraphCalculus(h)
    g = calc.g
    normal = np.column_stack((-g[0].c[0].real, -g[1].c[0].real, np.ones(len(xy))))
    normal /= np.linalg.norm(normal, axis=-1)[:, None]
    if np.any(normal @ u <= 1e-8):
        raise ValueError(
            "incident field must point into the plus medium away from grazing"
        )
    wi = n_incident * u
    boundary_phase = wi[0] * x + wi[1] * y + wi[2] * h
    phases = (
        boundary_phase + wi[2] * space.variable(2),
        calc.phase(boundary_phase, n_incident, -1),
        calc.phase(boundary_phase, n_transmitted, 1),
    )
    w = [[v.trace() for v in calc.gradient(s)] for s in phases]
    zero = [space.constant(0) for _ in range(3)]
    amplitudes = [[], [], []]
    basis = [[space.constant(float(i == j)) for i in range(3)] for j in range(3)]

    # Four tangent conditions and the two branch divergence recurrences.
    def tangent(v):
        return [v[i] + g[i] * v[2] for i in range(2)]

    columns = []
    for branch, sign in ((1, 1), (2, -1)):
        for e in basis:
            columns.append(
                [sign * v for v in tangent(e)]
                + [sign * v for v in tangent(cross(w[branch], e))]
                + [
                    dot(w[branch], e) if branch == 1 else space.constant(0),
                    dot(w[branch], e) if branch == 2 else space.constant(0),
                ]
            )
    matrix = [list(row) for row in zip(*columns)]
    for m in range(order + 1):
        remaining = 2 * (order - m) + derivative_depth
        ai = [space.constant(v if m == 0 else 0) for v in incident_a]
        amplitudes[0].append(ai)
        previous = [a[m - 1] if m else zero for a in amplitudes]
        curls = [calc.curl(a) for a in previous]
        magnetic_rhs = [-v for v in cross(w[0], ai)]
        magnetic_rhs = [v - ci - cr + ct for v, ci, cr, ct in zip(magnetic_rhs, *curls)]
        rhs = [-v for v in tangent(ai)] + tangent(magnetic_rhs)
        rhs += [
            -calc.divergence(previous[1]).trace(),
            -calc.divergence(previous[2]).trace(),
        ]
        rhs = [v.trace() for v in rhs]
        solved = solve_trace(matrix, rhs, remaining)
        for branch in (1, 2):
            boundary = solved[3 * (branch - 1) : 3 * branch]
            amplitudes[branch].append(
                calc.transport(boundary, phases[branch], previous[branch], remaining)
            )
    points = np.column_stack((xy, h.c[0].real * length_scale))
    return InterfaceExpansion(
        calc,
        phases,
        tuple(amplitudes),
        points,
        normal,
        length_scale,
        (n_incident, n_incident, n_transmitted),
        order,
    )
