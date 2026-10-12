"""Optional Matplotlib views in physical coordinates."""

from vecdiff_general.visualization.animation import animate_polarization
from vecdiff_general.visualization.fields import plot_intensity
from vecdiff_general.visualization.polarization import (
    plot_ellipses,
    plot_focal_polarization,
    plot_polarization,
    plot_polarization_scalar,
)

__all__ = [
    "plot_intensity",
    "plot_ellipses",
    "plot_polarization",
    "plot_focal_polarization",
    "plot_polarization_scalar",
    "animate_polarization",
]
