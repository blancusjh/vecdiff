"""Target-centred surface quadrature for controlled one-sided Green limits."""

import numpy as np
from numpy.polynomial.legendre import leggauss

from vecdiff_general._arrays import positive, positive_integer
from vecdiff_general.sampling.grids import SurfaceGrid


def near_surface_grid(
    surface,
    center_xy,
    aperture,
    distance_scale,
    *,
    radial_order=96,
    azimuthal_order=128,
):
    """Resolve a small target distance using r = distance_scale*sinh(t).

    The polar chart is centred at center_xy, while the circular integration
    patch remains centred on the surface's local axis. Integrate the complex
    fields at several nonzero offsets before extrapolating boundary traces.
    """
    center = np.asarray(center_xy, dtype=float)
    aperture = positive(aperture, "aperture")
    distance_scale = positive(distance_scale, "distance_scale")
    radial_order = positive_integer(radial_order, "radial_order")
    azimuthal_order = positive_integer(azimuthal_order, "azimuthal_order")

    if (
        center.shape != (2,)
        or not np.isfinite(center).all()
        or np.linalg.norm(center) >= aperture
    ):
        raise ValueError("quadrature centre must lie strictly inside the surface patch")

    angle = 2 * np.pi * np.arange(azimuthal_order) / azimuthal_order
    direction = np.column_stack((np.cos(angle), np.sin(angle)))
    projection = direction @ center
    radial_limit = -projection + np.sqrt(
        projection**2 + aperture**2 - np.dot(center, center)
    )
    maximum_parameter = np.arcsinh(radial_limit / distance_scale)
    nodes, weights = leggauss(radial_order)
    parameter = (nodes[:, None] + 1) * maximum_parameter[None, :] / 2
    radius = distance_scale * np.sinh(parameter)
    derivative = distance_scale * np.cosh(parameter)
    xy = center + radius[..., None] * direction[None, :, :]
    area = radius * derivative * weights[:, None] * maximum_parameter[None, :] / 2
    area *= 2 * np.pi / azimuthal_order * surface.area_jacobian(xy)

    return SurfaceGrid(surface.points(xy), surface.normals(xy), area)
