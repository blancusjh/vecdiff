"""Optional Matplotlib views in physical coordinates."""

from .animation import animate_polarization
from .fields import plot_components
from .polarization import plot_ellipses

__all__ = ["plot_components", "plot_ellipses", "animate_polarization"]
