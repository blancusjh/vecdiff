"""Physical quadrature rules for placed graph surfaces."""

import numpy as np
from numpy.polynomial.legendre import leggauss

from .._arrays import positive, positive_integer
from .grids import SurfaceGrid


def sample_surface(surface, aperture: float, *, radial_order=48, azimuthal_order=96):
    """Gauss radial and periodic azimuth quadrature on a circular surface patch.

    aperture is a physical radius in the surface's local chart. This function
    sets integration support; it does not declare an absorbing optical stop.
    """
    aperture = positive(aperture, "integration aperture")
    radial_order = positive_integer(radial_order, "radial_order")
    azimuthal_order = positive_integer(azimuthal_order, "azimuthal_order")
    nodes, weights = leggauss(radial_order)
    radius = aperture * (nodes + 1) / 2
    radial_weights = aperture * weights / 2
    azimuth = 2 * np.pi * np.arange(azimuthal_order) / azimuthal_order
    radial, angle = np.meshgrid(radius, azimuth, indexing="ij")
    xy = np.stack((radial * np.cos(angle), radial * np.sin(angle)), axis=-1)
    points = surface.points(xy)
    normals = surface.normals(xy)
    weights = (
        radial
        * radial_weights[:, None]
        * (2 * np.pi / azimuthal_order)
        * surface.area_jacobian(xy)
    )

    return SurfaceGrid(points, normals, weights)
