"""Optional Matplotlib views in physical coordinates."""

from .animation import animate_polarization
from .fields import plot_intensity
from .polarization import (
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
