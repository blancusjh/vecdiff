"""A cone with a hyperbolic tip, as a vecdiff surface, and a doughnut beam profile.

``HyperbolicCone`` is z = g (sqrt(r² + a²) - a): far from the axis the cone of
slope g, on the axis a cap of curvature g / a. It is a surface of revolution
(an ``AxisymmetricSurface``, carried on the meridian) that also declares its
implicit form, so rays meet it through the exact quadratic of the hyperboloid
sheet (z / g + a)² = r² + a². A graph-only cone is met by Newton from the
height under the plane point; for a cone of slope 2 that start lies across the
axis for the inner surface of a shell, and Newton converges to the far
crossing. The exact crossing removes that.

``Doughnut`` is the LG_0^1 amplitude (rho / w) exp(-rho² / w²) without its
vortex, with its support and bandwidth declared for the quadratures.
"""

from dataclasses import dataclass, field

import numpy as np
from vecdiff._backend import bd
from vecdiff.fields.profiles import ROUNDOFF_EXPONENT, _Profile
from vecdiff.geometry.frames import Frame
from vecdiff.surfaces import AxisymmetricSurface, ImplicitSurface


@dataclass(frozen=True, repr=False)
class Doughnut(_Profile):
    """(rho / w) exp(-rho² / w²)."""

    waist: float

    def _arguments(self):
        return [f"waist={self.waist:g}"]

    def __call__(self, x, y):
        r2 = bd.asarray(x, float) ** 2 + bd.asarray(y, float) ** 2
        return bd.sqrt(r2) / self.waist * bd.exp(-r2 / self.waist**2)

    @property
    def support(self):
        return self.waist * float(np.sqrt(ROUNDOFF_EXPONENT))

    @property
    def bandwidth(self):
        return 3 * float(np.sqrt(ROUNDOFF_EXPONENT)) / self.waist

    @property
    def edges(self):
        return ()


@dataclass(frozen=True, repr=False)
class HyperbolicCone(AxisymmetricSurface, ImplicitSurface):
    """z = g (sqrt(r² + a²) - a), slope g = ``cone_slope``, tip a = ``tip``."""

    sag_function: object = field(init=False, repr=False)
    slope_function: object = field(init=False, repr=False)
    frame: Frame = field(default_factory=Frame)
    curvature_function: object = field(init=False, repr=False)
    cone_slope: float = 1.0
    tip: float = 1e-6

    def __post_init__(self):
        g, a = float(self.cone_slope), float(self.tip)
        if g == 0 or not a > 0:
            raise ValueError("a cone needs a nonzero slope and a positive tip")
        object.__setattr__(self, "sag_function", lambda r: g * (bd.sqrt(r * r + a * a) - a))
        object.__setattr__(self, "slope_function", lambda r: g * r / bd.sqrt(r * r + a * a))
        object.__setattr__(
            self, "curvature_function", lambda r: g * a * a / (r * r + a * a) ** 1.5
        )

    def _parameters(self):
        return [("slope", f"{self.cone_slope:g}"), ("tip", f"{self.tip:g}")]

    def implicit(self, points):
        q = bd.asarray(points, dtype=float)
        xy = q[..., :2]
        value = q[..., 2] - self.height(xy)
        ones = bd.ones(q.shape[:-1] + (1,))
        gradient = bd.concatenate((-self.gradient(xy), ones), axis=-1)
        hessian = bd.zeros(q.shape[:-1] + (3, 3))
        hessian[..., :2, :2] = -self.hessian(xy)
        return value, gradient, hessian

    def crossing(self, points, directions, *, leaving=False):
        g, a = self.cone_slope, self.tip
        p = bd.asarray(points, dtype=float)
        s = bd.asarray(directions, dtype=float)
        w0 = p[..., 2] / g + a
        A = s[..., 2] ** 2 / g**2 - bd.sum(s[..., :2] ** 2, -1)
        B = 2 * (w0 * s[..., 2] / g - bd.sum(p[..., :2] * s[..., :2], -1))
        C = w0**2 - bd.sum(p[..., :2] ** 2, -1) - a * a
        disc = B * B - 4 * A * C
        root = bd.sqrt(bd.where(disc >= 0, disc, bd.nan))
        q = -0.5 * (B + bd.where(B >= 0, 1.0, -1.0) * root)
        with np.errstate(divide="ignore", invalid="ignore"):
            roots = (bd.where(q != 0, C / q, bd.nan), bd.where(A != 0, q / A, bd.nan))
        best = bd.full(p.shape[:-1], bd.nan)
        for t in roots:
            point = p + t[..., None] * s
            # The sheet z = h(r), not its mirror z/g + a < 0.
            sheet = point[..., 2] / g + a > 0
            ones = bd.ones(point.shape[:-1] + (1,))
            grad = bd.concatenate((-self.gradient(point[..., :2]), ones), -1)
            along = bd.sum(grad * s, -1)
            wanted = (along < 0) if leaving else (along > 0)
            ok = bd.isfinite(t) & sheet & wanted
            take = ok & (~bd.isfinite(best) | (abs(t) < abs(best)))
            best = bd.where(take, t, best)
        return best

    def reversed(self):
        return HyperbolicCone(
            frame=self.frame.reversed(), cone_slope=-self.cone_slope, tip=self.tip
        )
