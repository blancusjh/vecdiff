"""Electric-field propagation through an optical system onto placed targets.

Spectral and local Fresnel share forward surface-equivalence reconstruction.
Returning branches and finite material edges require a separate model.
"""

import importlib.util
from dataclasses import dataclass, replace

import numpy as np

from vecdiff_general.fields import (
    ElectricField,
    TransverseElectricField,
)
from vecdiff_general.fields.angular_spectrum import angular_spectrum
from vecdiff_general.fields.spectrum import PlaneSpectrum
from vecdiff_general.propagation.coverage import CoverageEvaluator, validate_coverage
from vecdiff_general.propagation.homogeneous import PlaneTransport
from vecdiff_general.propagation.surface_interaction import (
    LocalFresnel,
    SpectralFresnel,
)
from vecdiff_general.propagation.surface_transfer import SurfaceTransport
from vecdiff_general.propagation.targets import target_grid
from vecdiff_general.sampling import CartesianGrid
from vecdiff_general.system import DielectricInterface, OpticalSystem


@dataclass(frozen=True)
class PropagationSolution:
    """Outgoing spectrum and the boundary responses used to construct it."""

    spectrum: PlaneSpectrum
    interactions: tuple
    observation_boundary: float | None = None
    stages: tuple = ()
    interfaces: tuple = ()
    treatment: str = "spectral"
    coverage: str = "reduced"

    def field(
        self,
        target=None,
        *,
        x=None,
        y=None,
        z=None,
        window=None,
        shape=None,
        sampling="auto",
        side=None,
        backend="auto",
        **kwargs,
    ):
        """Electric field on a placed/sampled target or a coordinate plane.

        Points are assigned to their physical region between the open graph
        faces. Each region uses its forward reconstructed spectrum: this does
        not add reflected paths or the evanescent near field. At a boundary,
        explicitly choose side='minus' or 'plus'; these are reconstructed
        one-sided fields, not the prescribed Fresnel traces.
        """
        source = self.stages[0] if self.stages else self.spectrum
        grid, _ = target_grid(
            source,
            self.interfaces,
            target,
            x=x,
            y=y,
            z=z,
            window=window,
            shape=shape,
            sampling=sampling,
        )
        evaluator = CoverageEvaluator()
        report = evaluator.classify(
            grid, self.interfaces, source, mode=self.coverage, side=side
        )
        return evaluator.electric(
            self.stages or (self.spectrum,), grid, report, backend=backend, **kwargs
        )

    def evaluate(self, grid, *, backend="auto", **kwargs):
        plane = self.spectrum.grid
        displacement = grid.points - plane.frame.origin
        if self.observation_boundary is not None and np.any(
            displacement @ plane.frame.normal
            < self.observation_boundary - 1e-12 * self.spectrum.expansion.wavelength0
        ):
            raise ValueError("observation must be beyond the final surface envelope")
        if (
            isinstance(grid, CartesianGrid)
            and grid.shape == plane.shape
            and np.array_equal(grid.u, plane.u)
            and np.array_equal(grid.v, plane.v)
            and np.allclose(grid.frame.basis, plane.frame.basis, rtol=0, atol=1e-12)
        ):
            delta = grid.frame.origin - plane.frame.origin
            if (
                np.linalg.norm(delta @ plane.frame.basis[:, :2])
                <= 1e-12 * self.spectrum.expansion.wavelength0
            ):
                return self.spectrum.propagate(float(delta @ plane.frame.normal))
        if backend == "auto":
            backend = "nufft" if importlib.util.find_spec("finufft") else "direct"
        return self.spectrum.evaluate(grid, backend=backend, **kwargs)


class PropagationPipeline:
    """Compose homogeneous transport and spectral or local Fresnel transmission.

    An empty path is valid. Surfaces must be nonintersecting graphs sharing the
    input FFT frame orientation and transverse origin. Finite material edges
    and stops require an explicit model and are not silently approximated here.
    Sampled electric inputs are completed internally using the system's incident
    medium. PlaneSpectrum inputs and solve/evaluate remain supported. Call the
    operator with a target to return an ElectricField; geometry stays in the
    OpticalSystem and observations belong to each operator invocation.
    """

    def __init__(
        self,
        system=(),
        *,
        transmission="spectral",
        coverage="reduced",
        incident_region=None,
        max_sine=0.85,
        transport_tolerance=1e-10,
    ):
        self.system = (
            system
            if isinstance(system, OpticalSystem)
            else OpticalSystem(tuple(system))
        )
        if any(x is not None for x in self.system.apertures + self.system.stops):
            raise ValueError(
                "finite material edges and stops are not supported by this forward pipeline"
            )
        self.interfaces = self.system.interfaces
        self.incident_region = incident_region or self.system.incident_region
        if self.interfaces:
            if (
                self.incident_region is not None
                and self.incident_region != self.interfaces[0].minus
            ):
                raise ValueError("incident region disagrees with the system")
            self.incident_region = self.interfaces[0].minus
        if not all(isinstance(x, DielectricInterface) for x in self.interfaces):
            raise TypeError("pipeline steps must be DielectricInterface objects")
        for left, right in zip(self.interfaces, self.interfaces[1:]):
            if left.plus != right.minus:
                raise ValueError(
                    "successive interactions disagree about the intervening medium"
                )
        if not 0 < max_sine < 1 or not 0 < transport_tolerance < 1:
            raise ValueError("transport bandwidth and tolerance must lie in (0, 1)")
        if transmission is None or (
            isinstance(transmission, str) and transmission == "spectral"
        ):
            transmission = SpectralFresnel()
        elif isinstance(transmission, str) and transmission == "local":
            transmission = LocalFresnel()
        if not hasattr(transmission, "transmit"):
            raise ValueError(
                "transmission must be 'spectral', 'local', or a Fresnel treatment"
            )
        self.coverage = validate_coverage(coverage)
        self._coverage_evaluator = CoverageEvaluator()
        self.transmission = transmission
        self.treatment = (
            "local"
            if isinstance(transmission, LocalFresnel)
            else "spectral"
            if isinstance(transmission, SpectralFresnel)
            else type(transmission).__name__
        )
        self.max_sine, self.transport_tolerance = max_sine, transport_tolerance
        self.clear_cache()
        self.last_solution = None
        self.last_sampling = None

    def clear_cache(self):
        """Discard prepared operators and the last incident field's solution."""
        self._key, self._operators = None, ()
        self._input, self._source = None, None
        self._solved_source = None
        self._stages, self._responses = [], []
        self.last_solution, self.last_sampling = None, None
        self.last_coverage = None
        self._coverage_evaluator.clear_cache()

    def _source_spectrum(self, source):
        if source is None:
            source = self.system.incident_field
        if source is None:
            raise ValueError("supply an incident field or set system.incident_field")
        if source is self._input or source is self._source:
            return self._source
        original = source
        if isinstance(source, (ElectricField, TransverseElectricField)):
            if self.incident_region is None:
                raise ValueError(
                    "a sampled incident field requires the system's incident_region"
                )
            source = angular_spectrum(
                source, self.incident_region, evanescent="discard"
            )
        if not isinstance(source, PlaneSpectrum):
            raise TypeError(
                "incident field must be an ElectricField, TransverseElectricField, or PlaneSpectrum"
            )
        if (
            self.incident_region is not None
            and source.expansion.region != self.incident_region
        ):
            raise ValueError("source medium differs from the system's incident medium")
        self._input, self._source = original, source
        return source

    def __call__(
        self,
        incident_field=None,
        target_surface=None,
        *,
        x=None,
        y=None,
        z=None,
        window=None,
        shape=None,
        sampling="auto",
        side=None,
        backend="auto",
        **kwargs,
    ):
        """Return an ElectricField; target coordinates are global, in metres.

        Repeated targets for the same immutable incident object reuse already
        solved interfaces. Only encounters up to the requested region are run.
        """
        from vecdiff_general.sampling.grids import Grid
        from vecdiff_general.surfaces.surface import GraphSurface

        if target_surface is None and isinstance(incident_field, (GraphSurface, Grid)):
            target_surface, incident_field = incident_field, None
        source = self._source_spectrum(incident_field)
        grid, report = target_grid(
            source,
            self.interfaces,
            target_surface,
            x=x,
            y=y,
            z=z,
            window=window,
            shape=shape,
            sampling=sampling,
        )
        coverage = self._coverage_evaluator.classify(
            grid, self.interfaces, source, mode=self.coverage, side=side
        )
        through = (
            int(coverage.regions[coverage.computed].max())
            if np.any(coverage.computed)
            else 0
        )
        solution = self.solve(source, through=through)
        self.last_solution, self.last_sampling = solution, report
        self.last_coverage = coverage
        return self._coverage_evaluator.electric(
            solution.stages, grid, coverage, backend=backend, **kwargs
        )

    def _prepare(self, source, through):
        grid, wavelength = source.grid, source.expansion.wavelength0
        if not isinstance(grid, CartesianGrid):
            raise TypeError("pipeline requires a CartesianGrid PlaneSpectrum")
        key = (
            wavelength,
            grid.u.tobytes(),
            grid.v.tobytes(),
            grid.frame.basis.tobytes(),
        )
        # Check placements on every call, including reuse with another source.
        for interface in self.interfaces:
            frame = interface.surface.frame
            if (
                not np.allclose(frame.basis, grid.frame.basis, rtol=0, atol=1e-12)
                or np.linalg.norm(
                    (frame.origin - grid.frame.origin) @ grid.frame.basis[:, :2]
                )
                > 1e-12 * wavelength
            ):
                raise ValueError(
                    "pipeline surfaces must share the source orientation and transverse origin"
                )
        operators = list(self._operators) if key == self._key else []
        if len(operators) >= through:
            return
        for interface in self.interfaces[len(operators) : through]:
            plane = CartesianGrid(grid.u, grid.v, interface.surface.frame)
            args = dict(max_sine=self.max_sine, tolerance=self.transport_tolerance)
            operators.append(
                (
                    SurfaceTransport(
                        plane, interface.surface, interface.minus, wavelength, **args
                    ),
                    SurfaceTransport(
                        plane, interface.surface, interface.plus, wavelength, **args
                    ),
                )
            )
        for left, right in zip(operators, operators[1:]):
            if np.any(
                (right[0].grid.points - left[0].grid.points) @ grid.frame.normal <= 0
            ):
                raise ValueError(
                    "successive surface charts intersect or are out of order"
                )
        self._key, self._operators = key, tuple(operators)

    def solve(self, source=None, *, through=None):
        """Prepare reusable forward fields; normal callers can use operator(...)."""
        source = self._source_spectrum(source)
        through = len(self.interfaces) if through is None else through
        if not isinstance(through, int) or not 0 <= through <= len(self.interfaces):
            raise ValueError("through must index a region in the optical system")
        if self.interfaces and source.expansion.region != self.interfaces[0].minus:
            raise ValueError(
                "source medium differs from the first interface incident medium"
            )
        self._prepare(source, through)
        if self._solved_source is not source:
            self._stages, self._responses = [source], []
            self._solved_source = source
        start = len(self._stages) - 1
        current = self._stages[-1]
        for interface, (qin, qout) in zip(
            self.interfaces[start:through], self._operators[start:through]
        ):
            distance = float(
                (qin.plane.frame.origin - current.grid.frame.origin)
                @ current.grid.frame.normal
            )
            current = PlaneTransport().spectrum(current, distance)
            if interface.minus.material.n == interface.plus.material.n:
                # Equal-index transport is an identity, without a spectral cut
                # or finite-chart reconstruction of an already known solution.
                current = PlaneSpectrum(
                    current.grid, replace(current.expansion, region=interface.plus)
                )
                self._stages.append(current)
                self._responses.append(None)
                continue
            response = self.transmission.transmit(current, interface, qin)
            current = qout.radiate(response.field)
            self._responses.append(response)
            self._stages.append(current)
        boundary = (
            None
            if through == 0
            else float(max(0, self._operators[through - 1][1].height.max()))
        )
        result = PropagationSolution(
            self._stages[through],
            tuple(r for r in self._responses[:through] if r is not None),
            boundary,
            tuple(self._stages[: through + 1]),
            self.interfaces[:through],
            self.treatment,
            self.coverage,
        )
        self.last_solution = result
        return result
