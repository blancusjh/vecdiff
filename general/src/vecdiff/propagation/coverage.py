"""Which segments of the path reach each point of a target, and where they are computed.

A point lies in the segment between two encounters when it is on the side
of the first that the light leaves into and on the side of the second that
it arrives from, the sides taken on the actual surfaces (their graphs, or
the whole closed surface). Several segments of one region reach the same
point (in front of a mirror the incident and the reflected light), and the
field there is their sum: the light of the path. A point no segment
reaches receives no light of this path.

Each segment's spectrum is referred to the plane of the encounter that
formed it. In the band of heights its surface spans there, the spectrum is
the continuation of the carried field beyond that plane: "complete"
coverage takes it as the field (the method's field, which the leg carries
to that order), "reduced" leaves the band out. It is not an evanescent or
finite-edge solver.
"""

from dataclasses import dataclass

import numpy as np

from vecdiff._arrays import frozen_array
from vecdiff._backend import bd, to_numpy
from vecdiff._printing import block, grid_summary, guarded, shape
from vecdiff.fields import ElectricField, ElectromagneticField, PlaneSpectrum
from vecdiff.sampling import CartesianGrid, PointGrid
from vecdiff.surfaces.surface import ImplicitSurface

MODES = ("complete", "reduced")


def validate_coverage(mode):
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError("coverage must be 'reduced' or 'complete'")
    return mode


@dataclass(frozen=True, repr=False)
class CoverageReport:
    """The target's points: computed, their segments, and those in a surface's band.

    ``segments`` (segments, *shape) marks the segments of the path that
    reach each point; ``regions`` is the first of them (-1 for none);
    ``extended`` marks the points computed in a band that reduced coverage
    leaves out.
    """

    mode: str
    computed: np.ndarray
    regions: np.ndarray
    extended: np.ndarray
    representation: str = "forward propagating modes on continued/open graph faces"
    grid: object = None
    sampling: dict | None = None
    segments: np.ndarray = None

    def __post_init__(self):
        object.__setattr__(self, "computed", frozen_array(self.computed, dtype=bool))
        object.__setattr__(self, "regions", frozen_array(self.regions, dtype=int))
        object.__setattr__(self, "extended", frozen_array(self.extended, dtype=bool))
        if self.segments is not None:
            object.__setattr__(
                self, "segments", frozen_array(self.segments, dtype=bool)
            )

    @guarded
    def __repr__(self):
        """Its mode, shape and target grid in a few words; no mask is read."""
        grid = "" if self.grid is None else f", on {grid_summary(self.grid)}"
        return f"CoverageReport(mode={self.mode!r}, {shape(self.computed.shape)}{grid})"

    @guarded
    def __str__(self):
        """How many points are computed, reached by each segment and extended."""
        computed = to_numpy(self.computed)
        lines = [
            ("grid", None if self.grid is None else repr(self.grid)),
            ("computed", f"{int(computed.sum())} of {computed.size} points"),
            ("extended", f"{int(to_numpy(self.extended).sum())} points in a band"),
        ]
        if self.segments is not None:
            reach = to_numpy(self.segments).reshape(len(self.segments), -1).sum(-1)
            lines.append(
                ("segments", ", ".join(f"[{i}] {int(n)}" for i, n in enumerate(reach)))
            )
        lines.append(("representation", self.representation))
        return block(f"CoverageReport ({self.mode} coverage)", lines)


def signed_side(surface, points):
    """Where points lie from a surface: > 0 on its plus side, < 0 on its minus side.

    A closed surface is sided by its implicit form everywhere; a graph by
    z - h(x, y) over its chart, which a target must not leave.
    """
    local = (points - surface.frame.origin) @ surface.frame.basis
    if isinstance(surface, ImplicitSurface):
        return surface.implicit(local)[0]
    distance = local[..., 2] - surface.height(local[..., :2])
    if not bd.isfinite(distance).all():
        raise ValueError("target lies outside the defined surface chart")
    return distance


def segment_regions(elements, source_region):
    """The region of each segment of the path."""
    if not elements:
        return (source_region,)
    return (elements[0].incident,) + tuple(e.outgoing for e in elements)


def segment_membership(grid, elements, wavelength, side=None):
    """(segments, *shape) booleans: which segments of the path reach each point.

    ``side`` ("minus" or "plus") says which side a point on a surface is
    taken on, minus being the side the light arrives from at that element.
    """
    if side not in (None, "minus", "plus"):
        raise ValueError("side must be 'minus' or 'plus'")
    tolerance = 1e-10 * wavelength
    shape = grid.shape
    if not elements:
        return bd.ones((1,) + shape, dtype=bool)
    plus = []
    for element in elements:
        distance = signed_side(element.surface, grid.points)
        boundary = abs(distance) <= tolerance
        if bd.any(boundary) and side is None:
            raise ValueError(
                "target touches an interface; specify side='minus' or side='plus'"
            )
        plus.append((distance > tolerance) | (boundary & (side == "plus")))
    leaves = [
        p if element.branch == "transmitted" else ~p
        for p, element in zip(plus, elements)
    ]
    members = [~plus[0]]
    members += [leaves[k] & ~plus[k + 1] for k in range(len(elements) - 1)]
    members.append(leaves[-1])
    return bd.stack(members)


def check_regions(members, regions):
    """Raise where a point lies in segments of different regions: the surfaces cross."""
    for a in range(len(regions)):
        for b in range(a + 1, len(regions)):
            if regions[a] != regions[b] and bd.any(members[a] & members[b]):
                raise ValueError(
                    "interface graphs intersect or are out of order at the target"
                )


class CoverageEvaluator:
    """Classify a target's points by segment and band, then sum their spectra."""

    def __init__(self):
        self._bounds = {}

    def clear_cache(self):
        self._bounds.clear()

    def classify(self, grid, elements, source, *, mode, side=None, bands=None):
        """The CoverageReport of ``grid`` for the path of ``elements``.

        ``bands`` holds, per element, the heights (lo, hi) along its normal
        that its surface spans where it carries light, or None where it
        excludes nothing; by default they are measured over the incident
        chart.
        """
        validate_coverage(mode)
        wavelength = source.wavelength0
        members = segment_membership(grid, elements, wavelength, side)
        check_regions(members, segment_regions(elements, source.region))
        reduced = bd.ones(grid.shape, dtype=bool)
        tolerance = 1e-10 * wavelength
        for i, element in enumerate(elements):
            band = self._band(element, source) if bands is None else bands[i]
            if band is None:
                continue
            lo, hi = band
            surface = element.surface
            z = (grid.points - surface.frame.origin) @ surface.frame.normal
            inside = (z > lo + tolerance) & (z < hi - tolerance)
            reduced &= ~(inside & (members[i] | members[i + 1]))
        computed = reduced if mode == "reduced" else bd.ones(grid.shape, dtype=bool)
        reached = bd.any(members, axis=0)
        first = bd.where(reached, bd.argmax(members, axis=0), -1)
        return CoverageReport(
            mode, computed, first, computed & ~reduced, segments=members
        )

    def _band(self, element, source):
        """The heights the surface spans over the incident chart; None at a
        surface between equal indices, which carries nothing away."""
        far = element.far
        if far is not None and element.incident.material.n == far.material.n:
            return None
        surface = element.surface
        key = (id(source.grid), id(surface))
        if key not in self._bounds:
            local = (source.grid.points - surface.frame.origin) @ surface.frame.basis
            xy = local[..., :2].reshape(-1, 2)
            # Chart points beyond the graph's domain do not meet the surface.
            inside = surface.domain(xy)
            height = surface.height(xy[inside] if bd.any(inside) else xy[:1] * 0)
            self._bounds[key] = (
                source.grid,
                surface,
                (float(height.min()), float(height.max())),
            )
        return self._bounds[key][2]

    @staticmethod
    def electric(spectra, grid, report, **summation):
        """E on ``grid``: the sum of its segments' spectra; NaN where not computed."""
        values = blank(grid, report)
        for mask, field in segment_fields(spectra, grid, report, False, summation):
            values[mask] += field.values.reshape(-1, 3)
        return ElectricField(
            grid, spectra[0].wavelength0, values, valid=report.computed
        )

    @staticmethod
    def evaluate(spectra, grid, report, **summation):
        """E and H on ``grid``: the sum of its segments' spectra."""
        values, magnetic = blank(grid, report), blank(grid, report)
        for mask, field in segment_fields(spectra, grid, report, True, summation):
            values[mask] += field.electric.values.reshape(-1, 3)
            magnetic[mask] += field.magnetic.reshape(-1, 3)
        electric = ElectricField(
            grid, spectra[0].wavelength0, values, valid=report.computed
        )
        return ElectromagneticField(electric, magnetic)


def blank(grid, report):
    """Zero where computed (no segment there: no light of the path), NaN elsewhere."""
    return bd.where(
        report.computed[..., None],
        bd.zeros(grid.shape + (3,), complex),
        bd.nan + 1j * bd.nan,
    )


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


def segment_fields(spectra, grid, report, magnetic, summation):
    """(mask, field) for each segment with computed points of ``grid``.

    A grid wholly in one segment that is its spectrum's FFT lattice moved
    along its normal is propagated by FFT; other points are summed.
    """
    members = report.segments
    alone = bool(bd.all(bd.sum(members, axis=0) <= 1))
    for index in range(len(spectra)):
        mask = report.computed & members[index]
        if not bool(bd.any(mask)):
            continue
        spectrum = spectra[index]
        offset = (
            lattice_offset(spectrum, grid)
            if isinstance(spectrum, PlaneSpectrum) and alone and bool(bd.all(mask))
            else None
        )
        if offset is not None:
            yield mask, spectrum.propagate(offset, magnetic=magnetic)
            continue
        points = PointGrid(grid.points[mask])
        method = spectrum.evaluate if magnetic else spectrum.electric
        yield mask, method(points, **summation)
