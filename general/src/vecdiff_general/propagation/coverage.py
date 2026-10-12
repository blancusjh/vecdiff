"""Spatial coverage of a target, independent of interface transmission.

Complete coverage extends the existing forward modal representation within
its material regions. It is not an evanescent or finite-edge solver.
"""

import importlib.util
from dataclasses import dataclass

import numpy as np

from vecdiff_general._arrays import frozen_array
from vecdiff_general.fields import ElectricField
from vecdiff_general.sampling import PointGrid


def validate_coverage(mode):
    if not isinstance(mode, str) or mode not in ("reduced", "complete"):
        raise ValueError("coverage must be 'reduced' or 'complete'")
    return mode


@dataclass(frozen=True)
class CoverageReport:
    mode: str
    computed: np.ndarray
    regions: np.ndarray
    extended: np.ndarray
    representation: str = "forward propagating modes on continued/open graph faces"

    def __post_init__(self):
        object.__setattr__(self, "computed", frozen_array(self.computed, dtype=bool))
        object.__setattr__(self, "regions", frozen_array(self.regions, dtype=int))
        object.__setattr__(self, "extended", frozen_array(self.extended, dtype=bool))


def region_indices(grid, interfaces, wavelength, side=None):
    """Assign points using actual graph heights, with explicit one-sided traces."""
    if side not in (None, "minus", "plus"):
        raise ValueError("side must be 'minus' or 'plus'")
    region = np.zeros(grid.shape, dtype=int)
    previous_plus = np.ones(grid.shape, bool)
    tolerance = 1e-10 * wavelength
    for interface in interfaces:
        surface = interface.surface
        local = (grid.points - surface.frame.origin) @ surface.frame.basis
        distance = local[..., 2] - surface.height(local[..., :2])
        if not np.isfinite(distance).all():
            raise ValueError("target lies outside the defined surface chart")
        boundary = abs(distance) <= tolerance
        if np.any(boundary) and side is None:
            raise ValueError(
                "target touches an interface; specify side='minus' or side='plus'"
            )
        plus = (distance > tolerance) | (boundary & (side == "plus"))
        if np.any(plus & ~previous_plus):
            raise ValueError(
                "interface graphs intersect or are out of order at the target"
            )
        region += plus
        previous_plus = plus
    return region


class CoverageEvaluator:
    """Cache graph envelopes, classify observations, then evaluate their modes."""

    def __init__(self):
        self._bounds = {}

    def clear_cache(self):
        self._bounds.clear()

    def classify(self, grid, interfaces, source, *, mode, side=None):
        validate_coverage(mode)
        wavelength = source.expansion.wavelength0
        regions = region_indices(grid, interfaces, wavelength, side)
        reduced = np.ones(grid.shape, dtype=bool)
        for i, interface in enumerate(interfaces):
            # An equal-index bookkeeping interface does not obstruct coverage.
            if interface.minus.material.n == interface.plus.material.n:
                continue
            surface = interface.surface
            key = (id(source.grid), id(surface))
            if key not in self._bounds:
                local = (
                    source.grid.points - surface.frame.origin
                ) @ surface.frame.basis
                height = surface.height(local[..., :2])
                self._bounds[key] = (
                    source.grid,
                    surface,
                    float(height.min()),
                    float(height.max()),
                )
            _, _, lo, hi = self._bounds[key]
            z = (grid.points - surface.frame.origin) @ surface.frame.normal
            tolerance = 1e-10 * wavelength
            # Exclude the sampled graph's whole height band in reduced mode.
            reduced &= np.where(regions <= i, z <= lo + tolerance, z >= hi - tolerance)
        computed = reduced if mode == "reduced" else np.ones(grid.shape, dtype=bool)
        return CoverageReport(mode, computed, regions, computed & ~reduced)

    @staticmethod
    def electric(spectra, grid, report, *, backend="auto", **kwargs):
        if backend == "auto":
            backend = "nufft" if importlib.util.find_spec("finufft") else "direct"
        values = np.full(grid.shape + (3,), np.nan + 1j * np.nan)
        for index in np.unique(report.regions[report.computed]):
            mask = report.computed & (report.regions == index)
            values[mask] = (
                spectra[index]
                .expansion.electric(
                    PointGrid(grid.points[mask]), backend=backend, **kwargs
                )
                .values
            )
        return ElectricField(
            grid, spectra[0].expansion.wavelength0, values, valid=report.computed
        )
