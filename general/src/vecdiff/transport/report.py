"""What a transport leg could not carry."""

from dataclasses import dataclass

from vecdiff._printing import guarded, number


@dataclass(frozen=True, repr=False)
class TransportReport:
    """What the eikonal leg could not carry, and how well its map was inverted.

    ``azimuthal_orders`` lists the orders the axisymmetric leg carried along
    its meridian; it is empty for the lattice leg. ``caustics`` is the caustic
    index of the carried rays: the signed number of transverse directions in
    which their tube collapsed on the leg. Rays with another index reach the
    output plane from the far side of a caustic, where the map folds; they
    carry no field, and ``folded_relative_amplitude`` is their share of the
    carried amplitude. ``undersampled_relative_amplitude`` is the share on
    rays that leave the surface steeper than the output lattice can sample,
    and ``band_removed_relative_amplitude`` the share in propagating plane
    waves that the lattice leg's re-expansion removes next to the light
    circle (beyond sin = 0.995 in the transmitted medium).
    """

    unsupported_relative_amplitude: float
    dark_samples: int
    newton_residual: float
    azimuthal_orders: tuple = ()
    caustics: int = 0
    folded_relative_amplitude: float = 0.0
    undersampled_relative_amplitude: float = 0.0
    band_removed_relative_amplitude: float = 0.0

    @guarded
    def __repr__(self):
        """Its shares and counts, compactly, in one line."""
        orders = f", orders={self.azimuthal_orders}" if self.azimuthal_orders else ""
        return (
            f"TransportReport(unsupported={number(self.unsupported_relative_amplitude, 3)}, "
            f"folded={number(self.folded_relative_amplitude, 3)}, "
            f"undersampled={number(self.undersampled_relative_amplitude, 3)}, "
            f"band_removed={number(self.band_removed_relative_amplitude, 3)}, "
            f"dark={self.dark_samples}, caustics={self.caustics}, "
            f"residual={number(self.newton_residual, 3)}{orders})"
        )

    def __str__(self):
        """What was not carried, in one line: relative amplitudes and counts."""
        orders = f", orders {self.azimuthal_orders}" if self.azimuthal_orders else ""
        return (
            f"unsupported {self.unsupported_relative_amplitude:.1e}, "
            f"folded {self.folded_relative_amplitude:.1e}, "
            f"undersampled {self.undersampled_relative_amplitude:.1e}, "
            f"band removed {self.band_removed_relative_amplitude:.1e}, "
            f"dark samples {self.dark_samples}, caustics {self.caustics}, "
            f"Newton residual {self.newton_residual:.1e}{orders}"
        )
