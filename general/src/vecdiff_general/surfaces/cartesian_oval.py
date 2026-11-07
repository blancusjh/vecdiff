"""The vertex-connected Cartesian-oval branch for two real axial conjugates."""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from vecdiff_general._arrays import positive
from vecdiff_general._backend import bd, to_numpy
from vecdiff_general.geometry.frames import Frame
from vecdiff_general.surfaces.surface import GraphSurface, ImplicitSurface


def _largest(values):
    """Largest magnitude of an array, 0 when empty (CuPy's max takes no initial)."""
    values = bd.abs(bd.asarray(values))
    return float(bd.max(values)) if values.size else 0.0


def gots_parameters(no, z0, ni, zi):
    """Return (G, O, T, S) for real conjugates z0 < 0 < zi.

    The vertex is the local origin. Indices are positive and unequal;
    coordinates are metres. Units are 1, 1/m, 1/m³ and 1/m² respectively.
    These are shape-design coefficients, not material assignments.
    See the thesis, Appendix D, equations D.5–D.8. z0=-inf selects
    their analytic collimated-input limit: G=-(no/ni)², O=ni/[zi(ni-no)],
    T=S=0.
    """
    design = OvalDesign(z0, zi, no, ni)
    no, ni = design.n_object, design.n_image
    if bd.isneginf(z0):
        return -((no / ni) ** 2), ni / (zi * (ni - no)), 0.0, 0.0

    length = zi - z0
    a, b = z0 / length, zi / length

    path = ni * b - no * a
    difference = ni * a - no * b
    weighted_position = ni**2 * b - no**2 * a

    G = weighted_position**2 / (ni * no * path * difference)
    O = difference / (a * b * (ni - no)) / length  # noqa: E741 - GOTS notation
    T = (ni - no) * (ni + no) ** 2 / (4 * ni * no * a * b * path) / length**3
    S = (ni + no) * weighted_position / (2 * ni * no * a * b * path) / length**2

    return G, O, T, S


@dataclass(frozen=True)
class OvalDesign:
    """Geometry designed for specified indices; all distances are in metres.

    The vertex is z=0. A real object lies in front (object_z < 0) and a real
    image behind (image_z > 0); object_z > 0 is a virtual object, toward which
    the incident rays converge, and image_z < 0 a virtual image, from which
    the refracted rays diverge. The surface keeps sigma_o n_o |P - O| +
    sigma_i n_i |P - I| constant, sigma = +1 real and -1 virtual. Design
    indices determine shape; dielectric interfaces assign the actual
    materials separately. object_z=-inf denotes collimated incidence along
    local +z. Equal-index and non-graph branches are not implemented.
    """

    object_z: float
    image_z: float
    n_object: float
    n_image: float

    def __post_init__(self):
        for name in ("n_object", "n_image"):
            object.__setattr__(self, name, positive(getattr(self, name), name))

        if not (
            bd.isfinite(self.object_z) or bd.isneginf(self.object_z)
        ) or not bd.isfinite(self.image_z):
            raise ValueError("object_z must be finite or -inf; image_z must be finite")

        if self.object_z == 0 or self.image_z == 0:
            raise ValueError("a conjugate at the vertex defines no oval")

        if self.object_z == self.image_z:
            raise ValueError("object and image must be distinct points")

        if self.n_object == self.n_image:
            raise ValueError(
                "equal indices do not define this vertex-connected refracting branch"
            )

    @property
    def real(self):
        """Both conjugates real: object in front, image behind."""
        return self.object_z < 0 < self.image_z

    @property
    def signs(self):
        """(sigma_o, sigma_i): +1 for a real conjugate, -1 for a virtual one."""
        return (1.0 if self.object_z < 0 else -1.0, 1.0 if self.image_z > 0 else -1.0)

    @property
    def vertex_curvature(self):
        """Paraxial curvature from n_i / z_i - n_o / z_o = (n_i - n_o) c."""
        no, ni = self.n_object, self.n_image
        inbound = 0.0 if bd.isneginf(self.object_z) else no / self.object_z
        return (ni / self.image_z - inbound) / (ni - no)

    def implicit(self, points):
        """G - C over (n_o - n_i), its gradient and Hessian: unit normal +z at the vertex.

        G = sigma_o n_o |P - O| + sigma_i n_i |P - I| (n_o z for collimated
        input); the gradient points from the object side into the image side.
        """
        q = bd.asarray(points, dtype=float)
        so, si = self.signs
        scale = self.n_object - self.n_image
        value = bd.zeros(q.shape[:-1]) - self.optical_path
        gradient = bd.zeros(q.shape)
        hessian = bd.zeros(q.shape + (3,))
        terms = [(self.image_z, si * self.n_image)]
        if bd.isneginf(self.object_z):
            value = value + self.n_object * q[..., 2]
            gradient[..., 2] += self.n_object
        else:
            terms.append((self.object_z, so * self.n_object))
        for z, weight in terms:
            offset = q - bd.asarray([0.0, 0.0, z])
            distance = bd.linalg.norm(offset, axis=-1)
            unit = offset / distance[..., None]
            value = value + weight * distance
            gradient = gradient + weight * unit
            hessian = (
                hessian
                + weight
                * (bd.eye(3) - unit[..., :, None] * unit[..., None, :])
                / distance[..., None, None]
            )

        return value / scale, gradient / scale, hessian / scale

    def _virtual_sag(self, radius):
        """Vertex-branch height for virtual conjugates: Newton from the vertex sphere."""
        radius = bd.asarray(radius, dtype=float)
        c = self.vertex_curvature
        rc = bd.clip(c * radius, -0.999, 0.999)
        z = c * radius**2 / (1 + bd.sqrt(1 - rc**2))
        xy = bd.stack((radius, bd.zeros_like(radius)), axis=-1)
        for _ in range(60):
            value, gradient, _ = self.implicit(bd.concatenate((xy, z[..., None]), -1))
            step = value / gradient[..., 2]
            z = z - step
            if _largest(step) <= 1e-15 * max(1.0, _largest(radius)):
                break
        value, gradient, _ = self.implicit(bd.concatenate((xy, z[..., None]), -1))
        scale = abs(self.optical_path) + _largest(radius)
        if bd.any(gradient[..., 2] <= 0) or bd.any(bd.abs(value) > 1e-10 * scale):
            raise ValueError(
                "aperture exceeds the vertex-connected Cartesian-oval branch"
            )
        return z

    @property
    def object_point(self):
        if bd.isneginf(self.object_z):
            raise ValueError("collimated incidence has no finite object point")

        return bd.array([0.0, 0.0, self.object_z])

    @property
    def image_point(self):
        return bd.array([0.0, 0.0, self.image_z])

    @property
    def optical_path(self):
        """Vertex optical path; for collimated input, referenced to z=0."""
        if bd.isneginf(self.object_z):
            return self.n_image * self.image_z

        return -self.n_object * self.object_z + self.n_image * self.image_z

    @property
    def gots(self):
        """Shape coefficients (G, O, T, S), in SI units (real conjugates)."""
        if not self.real:
            raise NotImplementedError("GOTS coefficients are given for real conjugates")
        return gots_parameters(self.n_object, self.object_z, self.n_image, self.image_z)

    def z(self, rho):
        """Axial coordinate z(rho) from the explicit GOTS expression.

        rho is distance from the vertex, NOT cylindrical radius. Only the
        vertex-connected graph branch is accepted; there is no clipping.
        """
        rho = bd.asarray(rho, dtype=float)

        if not bd.isfinite(rho).all() or bd.any(rho < 0):
            raise ValueError("rho must be finite and nonnegative")

        G, O, T, S = self.gots  # noqa: E741 - GOTS notation
        rho2 = rho**2
        discriminant = 1 + (2 * S - O**2 * G) * rho2

        if bd.any(discriminant <= 0):
            raise ValueError("rho reaches or exceeds the GOTS branch limit")

        denominator = 1 + S * rho2 + bd.sqrt(discriminant)

        if bd.any(denominator <= 0):
            raise ValueError("rho is outside the vertex-connected GOTS branch")

        z = (O + T * rho2) * rho2 / denominator
        radius2 = rho2 - z**2

        if bd.any(radius2 < 0):
            raise ValueError("GOTS parameter has no real cylindrical radius")

        li = bd.sqrt(radius2 + (z - self.image_z) ** 2)

        if bd.isneginf(self.object_z):
            incident_path = self.n_object * z
            incident_derivative = self.n_object
        else:
            l0 = bd.sqrt(radius2 + (z - self.object_z) ** 2)
            incident_path = self.n_object * l0
            incident_derivative = self.n_object * (z - self.object_z) / l0

        derivative = incident_derivative + self.n_image * (z - self.image_z) / li

        if bd.any(derivative * (self.n_object - self.n_image) <= 0):
            raise ValueError("rho reaches or passes the Cartesian-oval graph fold")

        if not bd.allclose(
            incident_path + self.n_image * li,
            self.optical_path,
            rtol=1e-12,
            atol=0,
        ):
            raise ValueError("GOTS point is outside the physical optical-path branch")

        return z

    def r(self, rho):
        """Cylindrical radius r(rho), in metres, on the same GOTS branch."""
        z = self.z(rho)

        return bd.sqrt(bd.asarray(rho, dtype=float) ** 2 - z**2)

    def rho_from_r(self, radius):
        """Invert the parameter using the unsquared optical-path equation.

        Solving the physical graph avoids selecting extraneous quartic roots.
        """
        return bd.hypot(bd.asarray(radius, dtype=float), self.sag(radius))

    def sag(self, radius):
        """Solve the constant-optical-path equation on its vertex branch.

        Dimensionless bracketing makes the tolerance independent of the chosen
        lens size. Difference-of-squares avoids subtracting nearly equal paths
        near the axis. No extrapolation or clipping past the branch limit.
        """
        radius = bd.asarray(radius, dtype=float)

        if not bd.isfinite(radius).all() or bd.any(radius < 0):
            raise ValueError("radius must be finite and nonnegative")

        if not self.real:
            return self._virtual_sag(radius)

        if bd.isneginf(self.object_z):
            no, ni, zi = self.n_object, self.n_image, self.image_z
            scaled_radius = radius / zi
            discriminant = 1 + (no + ni) / (no - ni) * scaled_radius**2

            if bd.any(discriminant <= 0):
                raise ValueError("aperture reaches the collimated oval graph fold")

            return ni / (ni - no) * radius * scaled_radius / (1 + bd.sqrt(discriminant))

        length = self.image_z - self.object_z
        z0, zi = self.object_z / length, self.image_z / length
        n0, ni = self.n_object, self.n_image

        def solve_one(r):
            if r == 0:
                return 0.0

            # Scalar root finding stays on the host for either backend.
            def residual(z):
                l0, li = np.hypot(r, z - z0), np.hypot(r, z - zi)

                return n0 * (r * r + z * z - 2 * z * z0) / (l0 - z0) + ni * (
                    r * r + z * z - 2 * z * zi
                ) / (li + zi)

            def derivative(z):
                return n0 * (z - z0) / np.hypot(r, z - z0) + ni * (z - zi) / np.hypot(
                    r, z - zi
                )

            minimum = brentq(derivative, z0, zi, xtol=1e-15)

            if residual(minimum) >= 0:
                raise ValueError(
                    "aperture exceeds the vertex-connected Cartesian-oval branch"
                )

            lower, upper = sorted((0.0, minimum))

            return brentq(residual, lower, upper, xtol=1e-15)

        unique, inverse = np.unique(to_numpy(radius) / length, return_inverse=True)
        heights = np.array([solve_one(float(r)) for r in unique]) * length

        return bd.asarray(heights[inverse].reshape(radius.shape))


@dataclass(frozen=True)
class CartesianOval(GraphSurface):
    """Placed Cartesian oval; refractive indices are immutable shape-design inputs."""

    object_z: float
    image_z: float
    n_object: float
    n_image: float
    frame: Frame = field(default_factory=Frame)

    def __post_init__(self):
        # Validate the design at construction, before sampling the surface.
        self.design

    @classmethod
    def from_conjugates(cls, no, z0, ni, zi, *, frame=None):
        """Construct from (no, z0, ni, zi), relative to the local vertex."""
        return cls(z0, zi, no, ni, Frame() if frame is None else frame)

    @property
    def gots(self):
        """Return (G, O, T, S) for this shape design."""
        return self.design.gots

    def z(self, rho):
        """Local axial coordinate in metres; rho is distance from the vertex."""
        return self.design.z(rho)

    def r(self, rho):
        """Cylindrical radius in metres from the GOTS vertex-distance parameter."""
        return self.design.r(rho)

    def rho_from_r(self, radius):
        """Convert physical aperture radius to the GOTS parameter, in metres."""
        return self.design.rho_from_r(radius)

    def parametric_points(self, rho, phi):
        """Revolve (r(rho), z(rho)) and embed in the surface frame.

        rho is in metres and phi in radians; inputs follow NumPy broadcasting.
        """
        rho, phi = bd.broadcast_arrays(bd.asarray(rho), bd.asarray(phi))

        if not bd.isfinite(phi).all():
            raise ValueError("azimuth must be finite")

        z = self.z(rho)
        r = bd.sqrt(rho**2 - z**2)
        local = bd.stack((r * bd.cos(phi), r * bd.sin(phi), z), axis=-1)

        return self.frame.origin + local @ self.frame.basis.T

    @property
    def design(self):
        return OvalDesign(self.object_z, self.image_z, self.n_object, self.n_image)

    def height(self, xy):
        return self.design.sag(bd.linalg.norm(bd.asarray(xy, dtype=float), axis=-1))

    def _graph_derivatives(self, xy):
        """Slope and Hessian of the graph from the implicit form (virtual conjugates)."""
        xy = bd.asarray(xy, dtype=float)
        local = bd.concatenate((xy, self.height(xy)[..., None]), axis=-1)
        _, gradient, second = self.design.implicit(local)
        axial = gradient[..., 2]
        slope = -gradient[..., :2] / axial[..., None]
        mixed = second[..., :2, 2]
        hessian = (
            second[..., :2, :2]
            + mixed[..., :, None] * slope[..., None, :]
            + slope[..., :, None] * mixed[..., None, :]
            + second[..., 2, 2][..., None, None]
            * slope[..., :, None]
            * slope[..., None, :]
        )
        return slope, -hessian / axial[..., None, None]

    def gradient(self, xy):
        if not self.design.real:
            return self._graph_derivatives(xy)[0]
        xy = bd.asarray(xy, dtype=float)
        height = self.height(xy)
        radius2 = bd.sum(xy**2, axis=-1)
        image_distance = bd.sqrt(radius2 + (height - self.image_z) ** 2)

        if bd.isneginf(self.object_z):
            object_curvature = 0.0
            object_axial = self.n_object
        else:
            object_distance = bd.sqrt(radius2 + (height - self.object_z) ** 2)
            object_curvature = self.n_object / object_distance
            object_axial = object_curvature * (height - self.object_z)

        numerator = (object_curvature + self.n_image / image_distance)[..., None] * xy
        denominator = (
            object_axial + self.n_image * (height - self.image_z) / image_distance
        )

        if bd.any(bd.abs(denominator) < 1e-12):
            raise ValueError("Cartesian-oval graph reaches a singular branch")

        return -numerator / denominator[..., None]

    def hessian(self, xy):
        """Implicit second derivatives of G = n_o |r - O| + n_i |r - I| = const."""
        if not self.design.real:
            return self._graph_derivatives(xy)[1]
        xy = bd.asarray(xy, dtype=float)
        height = self.height(xy)
        slope = self.gradient(xy)
        local = bd.concatenate((xy, height[..., None]), axis=-1)
        second = bd.zeros(xy.shape[:-1] + (3, 3))
        axial = bd.zeros(xy.shape[:-1])
        conjugates = [(self.image_z, self.n_image)]

        if bd.isneginf(self.object_z):
            axial = axial + self.n_object
        else:
            conjugates.append((self.object_z, self.n_object))

        for z, index in conjugates:
            offset = local - bd.asarray([0.0, 0.0, z])
            distance = bd.linalg.norm(offset, axis=-1)
            unit = offset / distance[..., None]
            axial = axial + index * unit[..., 2]
            second = (
                second
                + index
                * (bd.eye(3) - unit[..., :, None] * unit[..., None, :])
                / distance[..., None, None]
            )

        mixed = second[..., :2, 2]
        hessian = (
            second[..., :2, :2]
            + mixed[..., :, None] * slope[..., None, :]
            + slope[..., :, None] * mixed[..., None, :]
            + second[..., 2, 2][..., None, None]
            * slope[..., :, None]
            * slope[..., None, :]
        )

        return -hessian / axial[..., None, None]


@dataclass(frozen=True)
class CartesianOvoid(CartesianOval, ImplicitSurface):
    """The closed Cartesian ovoid whose vertex branch is the CartesianOval.

    Regions, charts and targets use the oval's graph; eikonal transport meets
    the whole ovoid G = const, with F = (G - C) / (n_o - n_i) so that grad F
    is the unit normal +z at the vertex. A ray is followed by Newton steps
    on F from its crossing with the vertex's osculating sphere.
    """

    def implicit(self, points):
        return self.design.implicit(points)

    def crossing(self, points, directions):
        p, s = bd.asarray(points, dtype=float), bd.asarray(directions, dtype=float)
        c = self.design.vertex_curvature
        if abs(c) > 1e-300:
            centre = bd.asarray([0.0, 0.0, 1 / c])
            offset = p - centre
            b = bd.sum(s * offset, axis=-1)
            disc = b * b - (bd.sum(offset**2, axis=-1) - 1 / c**2)
            root = bd.sqrt(bd.where(disc >= 0, disc, 0.0))
            leg = -b - root if c > 0 else -b + root
        else:
            leg = -p[..., 2] / s[..., 2]
        scale = abs(self.design.optical_path) + bd.linalg.norm(p, axis=-1)
        for _ in range(60):
            value, gradient, _ = self.implicit(p + leg[..., None] * s)
            slope = bd.sum(s * gradient, axis=-1)
            step = value / slope
            leg = leg - step
            if _largest(step) <= 1e-14 * max(1.0, _largest(scale)):
                break
        value, gradient, _ = self.implicit(p + leg[..., None] * s)
        good = (bd.abs(value) <= 1e-9 * scale) & (bd.sum(s * gradient, axis=-1) > 0)
        return bd.where(good, leg, bd.nan)
