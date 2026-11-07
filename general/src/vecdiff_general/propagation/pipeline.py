"""Electric-field propagation through an optical system onto placed targets.

Two realizations of the surface legs. transport="wave" carries the field to
each surface exactly (SurfaceTransport) and back to a plane spectrum after a
spectral, local or carrier-directed Fresnel transmission. transport="eikonal"
replaces those legs by the plane-surface-plane map along a declared carrier
(EikonalTransport), on a lattice or, for a rotationally symmetric system, on
one meridian with Hankel spectra per azimuthal order. Returning branches and
finite material edges require a separate model.
"""

import importlib.util
from dataclasses import dataclass, replace

from vecdiff_general._backend import bd
from vecdiff_general.fields import (
    ElectricField,
    TransverseElectricField,
)
from vecdiff_general.fields.angular_spectrum import angular_spectrum
from vecdiff_general.fields.beams import _Beam
from vecdiff_general.fields.spectrum import PlaneSpectrum
from vecdiff_general.geometry.frames import Frame
from vecdiff_general.propagation._axisymmetric import (
    AxialTransport,
    BeamField,
    footprint,
)
from vecdiff_general.propagation.coverage import CoverageEvaluator, validate_coverage
from vecdiff_general.propagation.eikonal import (
    EikonalTransport,
    RefractedRays,
    check_window,
)
from vecdiff_general.propagation.homogeneous import PlaneTransport
from vecdiff_general.propagation.surface_interaction import (
    CarrierFresnel,
    LocalFresnel,
    SpectralFresnel,
)
from vecdiff_general.propagation.surface_transfer import SurfaceTransport
from vecdiff_general.propagation.targets import target_grid
from vecdiff_general.sampling import CartesianGrid, PolarGrid
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
        method="auto",
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
            self.stages or (self.spectrum,), grid, report, method=method, **kwargs
        )

    def evaluate(self, grid, *, method="auto", **kwargs):
        plane = self.spectrum.grid
        displacement = grid.points - plane.frame.origin
        if self.observation_boundary is not None and bd.any(
            displacement @ plane.frame.normal
            < self.observation_boundary - 1e-12 * self.spectrum.expansion.wavelength0
        ):
            raise ValueError("observation must be beyond the final surface envelope")
        if (
            isinstance(self.spectrum, PlaneSpectrum)
            and isinstance(grid, CartesianGrid)
            and grid.shape == plane.shape
            and bd.array_equal(grid.u, plane.u)
            and bd.array_equal(grid.v, plane.v)
            and bd.allclose(grid.frame.basis, plane.frame.basis, rtol=0, atol=1e-12)
        ):
            delta = grid.frame.origin - plane.frame.origin
            if (
                bd.linalg.norm(delta @ plane.frame.basis[:, :2])
                <= 1e-12 * self.spectrum.expansion.wavelength0
            ):
                return self.spectrum.propagate(float(delta @ plane.frame.normal))
        if method == "auto":
            method = "nufft" if importlib.util.find_spec("finufft") else "direct"
        return self.spectrum.evaluate(grid, method=method, **kwargs)


class PropagationPipeline:
    """Compose homogeneous transport with transmission at each surface.

    An empty path is valid. Surfaces must be nonintersecting graphs sharing the
    input FFT frame orientation and transverse origin. Finite material edges
    and stops require an explicit model and are not silently approximated here.
    Sampled electric inputs are completed internally using the system's incident
    medium. PlaneSpectrum inputs and solve/evaluate remain supported. Call the
    operator with a target to return an ElectricField; geometry stays in the
    OpticalSystem and observations belong to each operator invocation.

    transport="wave" (default) takes transmission "spectral" (default), "local",
    "carrier" or a Fresnel treatment. transport="eikonal" carries the field
    with the carrier dyadic along each surface's plane-surface-plane map; its
    reference planes sit at ``reference_planes`` offsets from each surface
    vertex along the axis (None: the vertex planes), and ``reference_fans``
    weight the carried rays at each surface: None carries every supported
    ray, (start, end) rolls them off between those incidence angles in
    degrees, a callable receives the RayMap. "carrier" transmission and
    eikonal transport need the incident field's carrier, passed as
    ``carrier=`` with the field. Eikonal transport crosses any sequence of
    surfaces: after each refraction the carrier is the family of refracted
    rays (Appendix B), labelled on the first plane, with its Appendix A
    derivatives; carrier transmission crosses one refracting surface.

    Eikonal transport also takes a beam from fields as the incident field
    (its own carrier unless carrier= is given). Surfaces of revolution about
    the beam's axis, with a carrier symmetric about it, are then carried on
    one meridian with Hankel spectra per azimuthal order, at a cost linear in
    the aperture instead of quadratic; a system without that symmetry needs
    the beam sampled on a CartesianGrid. ``axisymmetric`` makes the choice
    explicit: None (default) takes the meridian for beams and the lattice
    for sampled fields, True takes the meridian for either (a sampled field
    is read on rings and must be a few azimuthal orders about its grid's
    centre), False the lattice.
    """

    def __init__(
        self,
        system=(),
        *,
        transmission=None,
        coverage="reduced",
        incident_region=None,
        max_sine=0.85,
        transport_tolerance=1e-10,
        transport="wave",
        reference_planes=None,
        reference_fans=None,
        axisymmetric=None,
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
        if transport not in ("wave", "eikonal"):
            raise ValueError('transport must be "wave" or "eikonal"')
        self.transport = transport
        if transport == "eikonal":
            if transmission not in (None, "carrier"):
                raise ValueError(
                    "eikonal transport transmits with the carrier dyadic; "
                    "transmission must be None or 'carrier'"
                )
            transmission = "carrier"
            offsets = (
                (0.0,) * len(self.interfaces)
                if reference_planes is None
                else tuple(float(x) for x in reference_planes)
            )
            if len(offsets) != len(self.interfaces):
                raise ValueError("give one reference-plane offset per interface")
            self.reference_planes = offsets
            fans = (
                (None,) * len(self.interfaces)
                if reference_fans is None
                else tuple(check_window(fan) for fan in reference_fans)
            )
            if len(fans) != len(self.interfaces):
                raise ValueError("give one reference fan per interface")
            if axisymmetric not in (None, True, False):
                raise ValueError("axisymmetric must be None, True or False")
            self.reference_fans, self.axisymmetric = fans, axisymmetric
        elif reference_planes is not None:
            raise ValueError("reference_planes place the planes of eikonal transport")
        elif reference_fans is not None or axisymmetric is not None:
            raise ValueError(
                "reference_fans and axisymmetric choose how eikonal transport is carried"
            )
        else:
            self.reference_planes = None
            self.reference_fans, self.axisymmetric = None, None
        if transmission == "carrier":
            pass
        elif transmission is None or (
            isinstance(transmission, str) and transmission == "spectral"
        ):
            transmission = SpectralFresnel()
        elif isinstance(transmission, str) and transmission == "local":
            transmission = LocalFresnel()
        if transmission != "carrier" and not hasattr(transmission, "transmit"):
            raise ValueError(
                "transmission must be 'spectral', 'local', 'carrier', "
                "or a Fresnel treatment"
            )
        self.coverage = validate_coverage(coverage)
        self._coverage_evaluator = CoverageEvaluator()
        self.transmission = transmission
        self.treatment = (
            "eikonal"
            if transport == "eikonal"
            else "carrier"
            if transmission == "carrier"
            else "local"
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
        self._carrier = None
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
        if isinstance(source, _Beam):
            source = self._beam_source(source)
        elif isinstance(source, (ElectricField, TransverseElectricField)):
            if self.incident_region is None:
                raise ValueError(
                    "a sampled incident field requires the system's incident_region"
                )
            source = angular_spectrum(
                source, self.incident_region, evanescent="discard"
            )
        if not isinstance(source, (PlaneSpectrum, BeamField)):
            raise TypeError(
                "incident field must be an ElectricField, TransverseElectricField, "
                "PlaneSpectrum, or a beam for eikonal transport"
            )
        if (
            self.incident_region is not None
            and source.expansion.region != self.incident_region
        ):
            raise ValueError("source medium differs from the system's incident medium")
        self._input, self._source = original, source
        return source

    def _beam_source(self, beam):
        """The beam as a first stage, charted on the first reference plane."""
        if self.transport != "eikonal":
            raise TypeError(
                "a beam is an incident field of transport='eikonal'; "
                "sample it on a CartesianGrid for wave transport"
            )
        if self.axisymmetric is False:
            raise ValueError(
                "axisymmetric=False carries a sampled field: evaluate the beam "
                "on a CartesianGrid and pass that field"
            )
        axis = beam.frame
        origin = axis.origin
        if self.interfaces:
            frame = self.interfaces[0].surface.frame
            first = frame.origin + self.reference_planes[0] * frame.normal
            origin = origin + float((first - origin) @ axis.normal) * axis.normal
        plane = Frame(origin, axis.basis)
        chart = PolarGrid(
            bd.linspace(0.0, footprint(beam, plane), 9),
            2 * bd.pi * bd.arange(8) / 8,
            plane,
        )
        return BeamField(beam, chart)

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
        method="auto",
        carrier=None,
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
            if bd.any(coverage.computed)
            else 0
        )
        solution = self.solve(source, through=through, carrier=carrier)
        self.last_solution, self.last_sampling = solution, report
        self.last_coverage = coverage
        return self._coverage_evaluator.electric(
            solution.stages, grid, coverage, method=method, **kwargs
        )

    def _prepare(self, source, through, carrier, axial):
        grid, wavelength = source.grid, source.expansion.wavelength0
        if axial:
            self._prepare_axial(source, through, carrier)
            return
        if not isinstance(grid, CartesianGrid):
            raise TypeError("pipeline requires a CartesianGrid PlaneSpectrum")
        key = (
            wavelength,
            grid.u.tobytes(),
            grid.v.tobytes(),
            grid.frame.basis.tobytes(),
            id(carrier) if self.transport == "eikonal" else None,
        )
        # Check placements on every call, including reuse with another source.
        for interface in self.interfaces:
            frame = interface.surface.frame
            if (
                not bd.allclose(frame.basis, grid.frame.basis, rtol=0, atol=1e-12)
                or bd.linalg.norm(
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
        if self.transport == "eikonal":
            for index in range(len(operators), through):
                interface = self.interfaces[index]
                if interface.minus.material.n == interface.plus.material.n:
                    operators.append(None)
                    continue
                frame = interface.surface.frame
                origin = frame.origin + self.reference_planes[index] * frame.normal
                last = next((leg for leg in reversed(operators) if leg), None)
                # The next leg reads the previous leg's output lattice, and its
                # carrier is the family that leg refracted.
                lattice = grid if last is None else last.output_plane
                family = carrier
                if last is not None:
                    first = next(leg for leg in operators if leg)
                    family = RefractedRays(last.map, (first.plane.u, first.plane.v))
                plane = CartesianGrid(lattice.u, lattice.v, Frame(origin, frame.basis))
                operators.append(
                    EikonalTransport(
                        plane,
                        interface,
                        family,
                        wavelength,
                        window=self.reference_fans[index],
                    )
                )
            self._key, self._operators = key, tuple(operators)
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
            if bd.any(
                (right[0].grid.points - left[0].grid.points) @ grid.frame.normal <= 0
            ):
                raise ValueError(
                    "successive surface charts intersect or are out of order"
                )
        self._key, self._operators = key, tuple(operators)

    def _prepare_axial(self, source, through, carrier):
        """Meridian legs about the incident field's axis, one per refracting surface."""
        wavelength = source.expansion.wavelength0
        beam = isinstance(source, BeamField)
        if beam:
            axis, identity = source.axis, ("beam", id(source.beam))
        else:
            grid = source.grid
            axis, identity = grid.frame, (grid.u.tobytes(), grid.v.tobytes())
        key = ("axial", wavelength, identity, axis.origin.tobytes(), id(carrier))
        operators = list(self._operators) if key == self._key else []
        for index in range(len(operators), through):
            interface = self.interfaces[index]
            if interface.minus.material.n == interface.plus.material.n:
                operators.append(None)
                continue
            frame = interface.surface.frame
            plane = Frame(
                frame.origin + self.reference_planes[index] * frame.normal, axis.basis
            )
            last = next((leg for leg in reversed(operators) if leg), None)
            if last is not None:
                # Labels on the first plane: the fan can only narrow from there.
                reach, scale = last.edge, last.edge
                family = RefractedRays(last.map)
            elif beam:
                reach, scale = None, footprint(source.beam, plane)
            else:
                grid = source.grid
                reach = float(min(-grid.u[0], grid.u[-1], -grid.v[0], grid.v[-1]))
                scale = reach
            if last is None:
                family = carrier
            operators.append(
                AxialTransport(
                    interface,
                    family,
                    axis,
                    wavelength,
                    offset=self.reference_planes[index],
                    window=self.reference_fans[index],
                    reach=reach,
                    scale=scale,
                )
            )
        self._key, self._operators = key, tuple(operators)

    def solve(self, source=None, *, through=None, carrier=None):
        """Prepare reusable forward fields; normal callers can use operator(...)."""
        source = self._source_spectrum(source)
        through = len(self.interfaces) if through is None else through
        if not isinstance(through, int) or not 0 <= through <= len(self.interfaces):
            raise ValueError("through must index a region in the optical system")
        if self.interfaces and source.expansion.region != self.interfaces[0].minus:
            raise ValueError(
                "source medium differs from the first interface incident medium"
            )
        if carrier is None and isinstance(source, BeamField):
            carrier = source.carrier
        axial = self.transport == "eikonal" and (
            isinstance(source, BeamField) or self.axisymmetric is True
        )
        if self.treatment in ("eikonal", "carrier"):
            if carrier is None:
                raise ValueError(
                    f"{self.treatment} transmission needs the incident carrier: "
                    "pass carrier= (a fields.Carrier) with the field"
                )
            refracting = [
                x
                for x in self.interfaces[:through]
                if x.minus.material.n != x.plus.material.n
            ]
            if len(refracting) > 1 and self.treatment == "carrier":
                raise ValueError(
                    "the carrier is not yet continued through refraction; "
                    "carrier transmission crosses one refracting surface"
                )
        if carrier is not self._carrier:
            self._solved_source = None
            self._carrier = carrier
        self._prepare(source, through, carrier, axial)
        if self._solved_source is not source:
            self._stages, self._responses = [source], []
            self._solved_source = source
        start = len(self._stages) - 1
        current = self._stages[-1]
        for interface, operator in zip(
            self.interfaces[start:through], self._operators[start:through]
        ):
            if axial:
                if operator is None:
                    current = (
                        PlaneSpectrum(
                            current.grid,
                            replace(current.expansion, region=interface.plus),
                        )
                        if isinstance(current, PlaneSpectrum)
                        else current.relabeled(interface.plus)
                    )
                    response = None
                else:
                    if isinstance(current, PlaneSpectrum):
                        distance = float(
                            (operator.plane.origin - current.grid.frame.origin)
                            @ current.grid.frame.normal
                        )
                        current = PlaneTransport().spectrum(current, distance)
                    current, response = operator.transmit(current)
                self._responses.append(response)
                self._stages.append(current)
                continue
            if operator is None:
                # An index-matched surface between eikonal legs: the identity.
                current = PlaneSpectrum(
                    current.grid, replace(current.expansion, region=interface.plus)
                )
                self._stages.append(current)
                self._responses.append(None)
                continue
            qin, qout = (operator, None) if self.transport == "eikonal" else operator
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
            if self.transport == "eikonal":
                current, response = qin.transmit(current)
            else:
                transmission = (
                    CarrierFresnel(carrier)
                    if self.transmission == "carrier"
                    else self.transmission
                )
                response = transmission.transmit(current, interface, qin)
                current = qout.radiate(response.field)
            self._responses.append(response)
            self._stages.append(current)
        if through == 0:
            boundary = None
        elif axial:
            last = self._operators[through - 1]
            boundary = None if last is None else last.boundary
        elif self.transport == "eikonal":
            last = self._operators[through - 1]
            boundary = None if last is None else last.boundary
        else:
            boundary = float(max(0, self._operators[through - 1][1].height.max()))
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
