"""The vertex-connected Cartesian-oval branch for two real axial conjugates."""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .._arrays import positive
from ..geometry.frames import Frame
from .surface import GraphSurface


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
    if np.isneginf(z0):
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

    The vertex is z=0 and object_z < 0 < image_z. Design indices determine
    shape; dielectric interfaces assign the actual materials separately.
    object_z=-inf denotes collimated incidence along local +z.
    Equal-index, virtual-conjugate and non-graph branches are not implemented.
    """

    object_z: float
    image_z: float
    n_object: float
    n_image: float

    def __post_init__(self):
        for name in ("n_object", "n_image"):
            object.__setattr__(self, name, positive(getattr(self, name), name))

        if not (
            np.isfinite(self.object_z) or np.isneginf(self.object_z)
        ) or not np.isfinite(self.image_z):
            raise ValueError("object_z must be finite or -inf; image_z must be finite")

        if not self.object_z < 0 < self.image_z:
            raise ValueError(
                "initial Cartesian-oval model requires object_z < 0 < image_z"
            )

        if self.n_object == self.n_image:
            raise ValueError(
                "equal indices do not define this vertex-connected refracting branch"
            )

    @property
    def object_point(self):
        if np.isneginf(self.object_z):
            raise ValueError("collimated incidence has no finite object point")

        return np.array([0.0, 0.0, self.object_z])

    @property
    def image_point(self):
        return np.array([0.0, 0.0, self.image_z])

    @property
    def optical_path(self):
        """Vertex optical path; for collimated input, referenced to z=0."""
        if np.isneginf(self.object_z):
            return self.n_image * self.image_z

        return -self.n_object * self.object_z + self.n_image * self.image_z

    @property
    def gots(self):
        """Shape coefficients (G, O, T, S), in SI units."""
        return gots_parameters(self.n_object, self.object_z, self.n_image, self.image_z)

    def z(self, rho):
        """Axial coordinate z(rho) from the explicit GOTS expression.

        rho is distance from the vertex, NOT cylindrical radius. Only the
        vertex-connected graph branch is accepted; there is no clipping.
        """
        rho = np.asarray(rho, dtype=float)

        if not np.isfinite(rho).all() or np.any(rho < 0):
            raise ValueError("rho must be finite and nonnegative")

        G, O, T, S = self.gots  # noqa: E741 - GOTS notation
        rho2 = rho**2
        discriminant = 1 + (2 * S - O**2 * G) * rho2

        if np.any(discriminant <= 0):
            raise ValueError("rho reaches or exceeds the GOTS branch limit")

        denominator = 1 + S * rho2 + np.sqrt(discriminant)

        if np.any(denominator <= 0):
            raise ValueError("rho is outside the vertex-connected GOTS branch")

        z = (O + T * rho2) * rho2 / denominator
        radius2 = rho2 - z**2

        if np.any(radius2 < 0):
            raise ValueError("GOTS parameter has no real cylindrical radius")

        li = np.sqrt(radius2 + (z - self.image_z) ** 2)

        if np.isneginf(self.object_z):
            incident_path = self.n_object * z
            incident_derivative = self.n_object
        else:
            l0 = np.sqrt(radius2 + (z - self.object_z) ** 2)
            incident_path = self.n_object * l0
            incident_derivative = self.n_object * (z - self.object_z) / l0

        derivative = incident_derivative + self.n_image * (z - self.image_z) / li

        if np.any(derivative * (self.n_object - self.n_image) <= 0):
            raise ValueError("rho reaches or passes the Cartesian-oval graph fold")

        if not np.allclose(
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

        return np.sqrt(np.asarray(rho, dtype=float) ** 2 - z**2)

    def rho_from_r(self, radius):
        """Invert the parameter using the unsquared optical-path equation.

        Solving the physical graph avoids selecting extraneous quartic roots.
        """
        return np.hypot(radius, self.sag(radius))

    def sag(self, radius):
        """Solve the constant-optical-path equation on its vertex branch.

        Dimensionless bracketing makes the tolerance independent of the chosen
        lens size. Difference-of-squares avoids subtracting nearly equal paths
        near the axis. No extrapolation or clipping past the branch limit.
        """
        radius = np.asarray(radius, dtype=float)

        if not np.isfinite(radius).all() or np.any(radius < 0):
            raise ValueError("radius must be finite and nonnegative")

        if np.isneginf(self.object_z):
            no, ni, zi = self.n_object, self.n_image, self.image_z
            scaled_radius = radius / zi
            discriminant = 1 + (no + ni) / (no - ni) * scaled_radius**2

            if np.any(discriminant <= 0):
                raise ValueError("aperture reaches the collimated oval graph fold")

            return ni / (ni - no) * radius * scaled_radius / (1 + np.sqrt(discriminant))

        length = self.image_z - self.object_z
        z0, zi = self.object_z / length, self.image_z / length
        n0, ni = self.n_object, self.n_image

        def solve_one(r):
            if r == 0:
                return 0.0

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

        unique, inverse = np.unique(radius / length, return_inverse=True)
        heights = np.array([solve_one(r) for r in unique]) * length

        return heights[inverse].reshape(radius.shape)


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
        rho, phi = np.broadcast_arrays(rho, phi)

        if not np.isfinite(phi).all():
            raise ValueError("azimuth must be finite")

        z = self.z(rho)
        r = np.sqrt(rho**2 - z**2)
        local = np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=-1)

        return self.frame.origin + local @ self.frame.basis.T

    @property
    def design(self):
        return OvalDesign(self.object_z, self.image_z, self.n_object, self.n_image)

    def height(self, xy):
        return self.design.sag(np.linalg.norm(xy, axis=-1))

    def gradient(self, xy):
        xy = np.asarray(xy, dtype=float)
        height = self.height(xy)
        radius2 = np.sum(xy**2, axis=-1)
        image_distance = np.sqrt(radius2 + (height - self.image_z) ** 2)

        if np.isneginf(self.object_z):
            object_curvature = 0.0
            object_axial = self.n_object
        else:
            object_distance = np.sqrt(radius2 + (height - self.object_z) ** 2)
            object_curvature = self.n_object / object_distance
            object_axial = object_curvature * (height - self.object_z)

        numerator = (object_curvature + self.n_image / image_distance)[..., None] * xy
        denominator = (
            object_axial + self.n_image * (height - self.image_z) / image_distance
        )

        if np.any(np.abs(denominator) < 1e-12):
            raise ValueError("Cartesian-oval graph reaches a singular branch")

        return -numerator / denominator[..., None]
