"""Circular cylinders: the graph of the front half and the closed cylinder."""

from dataclasses import dataclass, field

from vecdiff_general._backend import bd, divide
from vecdiff_general.geometry.frames import Frame
from vecdiff_general.surfaces.surface import GraphSurface, ImplicitSurface


@dataclass(frozen=True)
class CylindricalCap(GraphSurface):
    """z = R - sqrt(R² - x²): the front half of a circular cylinder along the frame's v.

    The axis is parallel to v through (0, 0, R); R is signed as for
    SphericalCap. The graph is defined for |x| < |R| and any y.
    """

    radius: float
    frame: Frame = field(default_factory=Frame)

    def __post_init__(self):
        if not bd.isfinite(self.radius) or self.radius == 0:
            raise ValueError("signed curvature radius must be finite and nonzero")

    def domain(self, xy):
        return bd.asarray(xy, dtype=float)[..., 0] ** 2 < self.radius**2

    def _root(self, xy):
        x2 = bd.asarray(xy, dtype=float)[..., 0] ** 2

        if bd.any(x2 >= self.radius**2):
            raise ValueError("cylinder coordinates reach or exceed its widest line")

        return bd.sqrt(self.radius**2 - x2)

    def height(self, xy):
        x = bd.asarray(xy, dtype=float)[..., 0]

        return bd.sign(self.radius) * x**2 / (abs(self.radius) + self._root(xy))

    def gradient(self, xy):
        xy = bd.asarray(xy, dtype=float)
        slope = bd.sign(self.radius) * xy[..., 0] / self._root(xy)

        return bd.stack((slope, bd.zeros_like(slope)), axis=-1)

    def hessian(self, xy):
        root = self._root(xy)
        out = bd.zeros(bd.asarray(xy).shape + (2,))
        out[..., 0, 0] = bd.sign(self.radius) * self.radius**2 / root**3

        return out


@dataclass(frozen=True)
class Cylinder(CylindricalCap, ImplicitSurface):
    """The closed circular cylinder whose front half is the CylindricalCap.

    Regions, charts and targets use the half-cylinder's graph; eikonal
    transport meets the whole cylinder, F = (R² - x² - (z - R)²) / (2R), with
    grad F = (-x, 0, R - z) / R from the minus into the plus side.
    """

    def implicit(self, points):
        q = bd.asarray(points, dtype=float)
        R = float(self.radius)
        x, z = q[..., 0], q[..., 2] - R
        value = (R**2 - x**2 - z**2) / (2 * R)
        gradient = bd.stack((-x, bd.zeros_like(x), -z), axis=-1) / R
        hessian = bd.broadcast_to(
            -bd.asarray([[1.0, 0, 0], [0, 0, 0], [0, 0, 1.0]]) / R, q.shape + (3,)
        ).copy()

        return value, gradient, hessian

    def crossing(self, points, directions):
        q, s = bd.asarray(points, dtype=float), bd.asarray(directions, dtype=float)
        x, z = q[..., 0], q[..., 2] - float(self.radius)
        a = s[..., 0] ** 2 + s[..., 2] ** 2
        b = x * s[..., 0] + z * s[..., 2]
        c = x**2 + z**2 - float(self.radius) ** 2
        disc = b * b - a * c
        root = bd.sqrt(bd.where((disc >= 0) & (a > 0), disc, bd.nan))
        # Into the plus side: the near root for R > 0, the far one for R < 0.
        if self.radius > 0:
            return divide(c, -b + root, (-b + root) != 0)
        return (-b + root) / a
