"""Spatial coverage of a target, independent of interface transmission.

Complete coverage extends the existing forward modal representation within
its material regions. It is not an evanescent or finite-edge solver.
"""

from dataclasses import dataclass

import numpy as np

from vecdiff._arrays import frozen_array
from vecdiff._backend import bd, to_numpy
from vecdiff.fields import ElectricField, ElectromagneticField, PlaneSpectrum
from vecdiff.sampling import CartesianGrid, PointGrid


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
    grid: object = None
    sampling: dict | None = None

    def __post_init__(self):
        object.__setattr__(self, "computed", frozen_array(self.computed, dtype=bool))
        object.__setattr__(self, "regions", frozen_array(self.regions, dtype=int))
        object.__setattr__(self, "extended", frozen_array(self.extended, dtype=bool))


def region_indices(grid, interfaces, wavelength, side=None):
    """Assign points using actual graph heights, with explicit one-sided traces."""
    if side not in (None, "minus", "plus"):
        raise ValueError("side must be 'minus' or 'plus'")
    region = bd.zeros(grid.shape, dtype=int)
    previous_plus = bd.ones(grid.shape, bool)
    tolerance = 1e-10 * wavelength
    for interface in interfaces:
        surface = interface.surface
        local = (grid.points - surface.frame.origin) @ surface.frame.basis
        distance = local[..., 2] - surface.height(local[..., :2])
        if not bd.isfinite(distance).all():
            raise ValueError("target lies outside the defined surface chart")
        boundary = abs(distance) <= tolerance
        if bd.any(boundary) and side is None:
            raise ValueError(
                "target touches an interface; specify side='minus' or side='plus'"
            )
        plus = (distance > tolerance) | (boundary & (side == "plus"))
        if bd.any(plus & ~previous_plus):
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
        wavelength = source.wavelength0
        regions = region_indices(grid, interfaces, wavelength, side)
        reduced = bd.ones(grid.shape, dtype=bool)
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
                xy = local[..., :2].reshape(-1, 2)
                # Chart points beyond the graph's domain do not meet the surface.
                inside = surface.domain(xy)
                height = surface.height(xy[inside] if bd.any(inside) else xy[:1] * 0)
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
            reduced &= bd.where(regions <= i, z <= lo + tolerance, z >= hi - tolerance)
        computed = reduced if mode == "reduced" else bd.ones(grid.shape, dtype=bool)
        return CoverageReport(mode, computed, regions, computed & ~reduced)

    @staticmethod
    def electric(spectra, grid, report, **summation):
        """E on ``grid`` from the spectrum of each point's region; NaN where not computed."""
        values = bd.full(grid.shape + (3,), bd.nan + 1j * bd.nan)
        for index, mask, field in region_fields(
            spectra, grid, report, False, summation
        ):
            values[mask] = field.values.reshape(-1, 3)
        return ElectricField(
            grid, spectra[0].wavelength0, values, valid=report.computed
        )

    @staticmethod
    def evaluate(spectra, grid, report, **summation):
        """E and H on ``grid`` from the spectrum of each point's region."""
        values = bd.full(grid.shape + (3,), bd.nan + 1j * bd.nan)
        magnetic = bd.full(grid.shape + (3,), bd.nan + 1j * bd.nan)
        for index, mask, field in region_fields(spectra, grid, report, True, summation):
            values[mask] = field.electric.values.reshape(-1, 3)
            magnetic[mask] = field.magnetic.reshape(-1, 3)
        electric = ElectricField(
            grid, spectra[0].wavelength0, values, valid=report.computed
        )
        return ElectromagneticField(electric, magnetic)


def lattice_offset(spectrum, grid):
    """The distance from a PlaneSpectrum's lattice to ``grid``, if ``grid`` is it moved.

    None unless ``grid`` is that lattice translated along its normal.
    """
    plane = spectrum.grid
    if not (
        isinstance(grid, CartesianGrid)
        and grid.shape == plane.shape
        and bd.array_equal(grid.u, plane.u)
        and bd.array_equal(grid.v, plane.v)
        and bd.allclose(grid.frame.basis, plane.frame.basis, rtol=0, atol=1e-12)
    ):
        return None
    delta = grid.frame.origin - plane.frame.origin
    if bd.linalg.norm(delta @ plane.frame.basis[:, :2]) > 1e-12 * spectrum.wavelength0:
        return None
    return float(delta @ plane.frame.normal)


def region_fields(spectra, grid, report, magnetic, summation):
    """(region, mask, field) for each region with computed points of ``grid``.

    A grid that is wholly in one region and is that region's FFT lattice
    moved along its normal is propagated by FFT; other points are summed.
    """
    for index in to_numpy(bd.unique(report.regions[report.computed])).tolist():
        spectrum = spectra[index]
        mask = report.computed & (report.regions == index)
        offset = (
            lattice_offset(spectrum, grid)
            if isinstance(spectrum, PlaneSpectrum) and bool(bd.all(mask))
            else None
        )
        if offset is not None:
            yield index, mask, spectrum.propagate(offset, magnetic=magnetic)
            continue
        points = PointGrid(grid.points[mask])
        method = spectrum.evaluate if magnetic else spectrum.electric
        yield index, mask, method(points, **summation)
