"""The method: light through an optical system, carried along its rays across each surface.

``Propagator(system)(light)`` carries the light across each refracting
surface along the rays of its carrier (``vecdiff.transport``), and re-expands
it in plane waves in the homogeneous region that follows. The result is a
``Solution``: one Spectrum per region, one TransportReport per refracting
surface, and the field (E and H together) on any target.
"""

from dataclasses import dataclass, field, replace

from vecdiff._backend import bd
from vecdiff.fields import (
    BeamSpectrum,
    ElectricField,
    HankelSpectrum,
    PlaneSpectrum,
    Spectrum,
    TransverseElectricField,
)
from vecdiff.fields.angular_spectrum import angular_spectrum
from vecdiff.fields.beam_spectrum import footprint
from vecdiff.fields.beams import _Beam
from vecdiff.geometry.frames import Frame
from vecdiff.propagation.coverage import CoverageEvaluator, validate_coverage
from vecdiff.propagation.homogeneous import PlaneTransport
from vecdiff.propagation.targets import target_grid
from vecdiff.sampling import CartesianGrid, PolarGrid
from vecdiff.system import DielectricInterface, OpticalSystem
from vecdiff.transport.dyadic import check_window
from vecdiff.transport.lattice import LatticeTransport
from vecdiff.transport.meridian import MeridianTransport
from vecdiff.transport.rays import RefractedRays

REALIZATIONS = ("auto", "meridian", "lattice")


@dataclass(frozen=True, eq=False)
class Solution:
    """Light through an optical system: the field of every region it crosses.

    ``spectra`` holds one Spectrum per region, the incident light first and
    the light leaving the last surface (``spectrum``) last. ``reports`` holds
    what was not carried at each refracting surface, and ``legs`` the
    prepared transport legs that carried it. ``field`` gives E and H on a
    target, each point from the spectrum of its own region.
    """

    spectra: tuple
    interfaces: tuple
    reports: tuple
    legs: tuple = ()
    _coverage: CoverageEvaluator = field(
        default_factory=CoverageEvaluator, repr=False, compare=False
    )

    @property
    def spectrum(self):
        """The Spectrum of the last region: the light leaving the system."""
        return self.spectra[-1]

    def coverage(
        self,
        target=None,
        *,
        x=None,
        y=None,
        z=None,
        window=None,
        shape=None,
        side=None,
        coverage="reduced",
    ):
        """The target's grid, the region of each point and which points are computed.

        The target is a sampled grid, a placed surface, or one of x=, y=, z=
        for a coordinate plane; a surface or plane is sampled over ``window``
        (u_min, u_max, v_min, v_max), by default the projected input chart and
        path, at ``shape`` (n_v, n_u), by default 0.4 wavelengths in the
        densest medium. Points are assigned to their region between the
        surfaces; on a surface, ``side`` ("minus" or "plus") says which.
        coverage "reduced" leaves out the band of heights a curved surface
        spans, where neither neighbouring spectrum is the field; "complete"
        takes each point's region spectrum there as well.
        """
        validate_coverage(coverage)
        source = self.spectra[0]
        grid, sampling = target_grid(
            source,
            self.interfaces,
            target,
            x=x,
            y=y,
            z=z,
            window=window,
            shape=shape,
        )
        report = self._coverage.classify(
            grid, self.interfaces, source, mode=coverage, side=side
        )
        return replace(report, grid=grid, sampling=sampling)

    def field(
        self,
        target=None,
        *,
        x=None,
        y=None,
        z=None,
        window=None,
        shape=None,
        side=None,
        coverage="reduced",
        method="auto",
        tolerance=None,
        chunk_size=None,
        threads=None,
    ):
        """E and H (an ElectromagneticField, SI units) on a target.

        The target and its sampling are those of ``coverage``; points left
        out are NaN and marked invalid. The summation options are those of
        every Spectrum.
        """
        report = self.coverage(
            target,
            x=x,
            y=y,
            z=z,
            window=window,
            shape=shape,
            side=side,
            coverage=coverage,
        )
        return self._coverage.evaluate(
            self.spectra,
            report.grid,
            report,
            method=method,
            tolerance=tolerance,
            chunk_size=chunk_size,
            threads=threads,
        )

    def electric(
        self,
        target=None,
        *,
        x=None,
        y=None,
        z=None,
        window=None,
        shape=None,
        side=None,
        coverage="reduced",
        method="auto",
        tolerance=None,
        chunk_size=None,
        threads=None,
    ):
        """E alone (an ElectricField) on a target, as ``field`` gives it."""
        report = self.coverage(
            target,
            x=x,
            y=y,
            z=z,
            window=window,
            shape=shape,
            side=side,
            coverage=coverage,
        )
        return self._coverage.electric(
            self.spectra,
            report.grid,
            report,
            method=method,
            tolerance=tolerance,
            chunk_size=chunk_size,
            threads=threads,
        )

    def __str__(self):
        lines = [
            f"Solution through {len(self.interfaces)} surface(s), "
            f"{len(self.reports)} carried; leaving as "
            f"{type(self.spectrum).__name__} in {self.spectrum.region.name}"
        ]
        refracting = [
            (index, interface)
            for index, interface in enumerate(self.interfaces, 1)
            if interface.minus.material.n != interface.plus.material.n
        ]
        for (index, interface), report in zip(refracting, self.reports):
            lines.append(
                f"  surface {index} ({type(interface.surface).__name__}, "
                f"{interface.minus.name} -> {interface.plus.name}): {report}"
            )
        return "\n".join(lines)


def optical_system(system):
    """The OpticalSystem of ``system`` (one, or a sequence of DielectricInterface).

    Finite material edges and stops are refused: no leg carries their
    scattering yet.
    """
    system = system if isinstance(system, OpticalSystem) else OpticalSystem(system)
    if any(x is not None for x in system.apertures + system.stops):
        raise ValueError(
            "finite material edges and stops are not supported by the propagation yet"
        )
    if not all(isinstance(x, DielectricInterface) for x in system.interfaces):
        raise TypeError("the system's encounters must be DielectricInterface objects")
    return system


def incident_region(system):
    """The region light enters the system from, or None if nothing says."""
    if system.interfaces:
        return system.interfaces[0].minus
    return system.incident_region


def sampled_spectrum(light, region):
    """A sampled incident field as the PlaneSpectrum of its propagating waves."""
    if region is None:
        raise ValueError(
            "a sampled incident field needs the region it is in: "
            "give the system an interface or incident_region"
        )
    return angular_spectrum(light, region, evanescent="discard")


def check_placement(interfaces, grid, wavelength):
    """Raise unless every surface shares the lattice's orientation and transverse origin."""
    for interface in interfaces:
        frame = interface.surface.frame
        offset = (frame.origin - grid.frame.origin) @ grid.frame.basis[:, :2]
        if (
            not bd.allclose(frame.basis, grid.frame.basis, rtol=0, atol=1e-12)
            or bd.linalg.norm(offset) > 1e-12 * wavelength
        ):
            raise ValueError(
                "the surfaces must share the lattice's orientation and "
                "transverse origin"
            )


def lattice_key(grid, wavelength):
    """What legs prepared on a lattice depend on, besides geometry and carrier."""
    return (wavelength, grid.u.tobytes(), grid.v.tobytes(), grid.frame.basis.tobytes())


def check_incident(spectrum, region):
    if not isinstance(spectrum, Spectrum):
        raise TypeError(
            "light must be a beam (fields.PlaneBeam, fields.FocusedBeam), "
            "a sampled field (fields.ElectricField, fields.TransverseElectricField) "
            "or a fields.Spectrum"
        )
    if region is not None and spectrum.region != region:
        raise ValueError("the light is not in the system's incident region")
    return spectrum


def axis_of(source):
    """The frame whose normal is the axis of the light: the beam's, else its plane's."""
    return source.beam.frame if isinstance(source, BeamSpectrum) else source.plane


class Propagator:
    """The eikonal-angular-spectrum method, prepared for one optical system.

    Calling it with light returns the Solution. Across each refracting
    surface the field is carried along the rays of its carrier, from a
    reference plane in front of the surface to the same plane behind it
    (``transport``); between surfaces it propagates as plane waves.

    ``planes`` places the reference planes: None puts them at each surface's
    vertex, otherwise one offset per surface along its axis. ``fans`` weight
    the carried rays at each surface: None carries every ray the surface
    supports, (start, end) rolls them off between those incidence angles in
    degrees, a callable receives the RayMap; one per surface.
    ``realization`` chooses how the legs are carried: "meridian" carries one
    meridian of a system of surfaces of revolution about the light's axis,
    with Hankel spectra per azimuthal order, at a cost linear in the
    aperture; "lattice" carries a lattice of any geometry; "auto" takes the
    meridian for a beam and the lattice for a sampled field or a lattice
    spectrum. ``lattice`` (a CartesianGrid) is the lattice a beam is carried
    on, read exactly at the preimages; a sampled field brings its own.

    Legs depend on the geometry, the carrier and the incident lattice or
    beam, not on the field: they are prepared once and reused for every
    call with the same ones.
    """

    def __init__(
        self, system=(), *, planes=None, fans=None, realization="auto", lattice=None
    ):
        self.system = optical_system(system)
        self.interfaces = self.system.interfaces
        self.region = incident_region(self.system)
        count = len(self.interfaces)
        self.planes = (
            (0.0,) * count if planes is None else tuple(float(x) for x in planes)
        )
        if len(self.planes) != count:
            raise ValueError("give one reference-plane offset per surface")
        self.fans = (
            (None,) * count if fans is None else tuple(check_window(f) for f in fans)
        )
        if len(self.fans) != count:
            raise ValueError("give one fan per surface")
        if realization not in REALIZATIONS:
            raise ValueError("realization must be 'auto', 'meridian' or 'lattice'")
        if lattice is not None and not isinstance(lattice, CartesianGrid):
            raise TypeError("lattice must be a CartesianGrid")
        if lattice is not None and realization == "meridian":
            raise ValueError("a lattice carries the beam off the meridian")
        self.realization, self.lattice = realization, lattice
        self._key, self._legs, self._carrier = None, (), None
        self._beam_input, self._beam_source = None, None

    def __call__(self, light=None, carrier=None):
        """The Solution for ``light``, carried along ``carrier``.

        ``light`` is a beam (its own carrier unless ``carrier`` is given), a
        sampled field or a Spectrum in the system's incident region, by
        default the system's incident_field. A sampled field or a plane
        spectrum needs its carrier (a fields.Carrier).
        """
        if light is None:
            light = self.system.incident_field
        if light is None:
            raise ValueError("pass the light, or set the system's incident_field")
        source = self._incident(light)
        family = carrier
        if family is None and isinstance(source, BeamSpectrum):
            family = source.carrier
        refracting = any(
            x.minus.material.n != x.plus.material.n for x in self.interfaces
        )
        if family is None and refracting:
            raise ValueError(
                "the method carries light along its carrier: pass carrier= "
                "(a fields.Carrier) with the field, or use "
                "alternatives.WavePropagator for a field without one"
            )
        legs = self._prepare(source, family)
        spectra, reports = [source], []
        for interface, leg in zip(self.interfaces, legs):
            current = spectra[-1]
            if leg is None:
                # An index-matched surface: the identity, without a cut.
                spectra.append(current.relabeled(interface.plus))
                continue
            if isinstance(current, PlaneSpectrum):
                # The other spectra are read at any depth.
                plane = leg.plane if isinstance(leg.plane, Frame) else leg.plane.frame
                distance = float(
                    (plane.origin - current.plane.origin) @ current.plane.normal
                )
                current = PlaneTransport().spectrum(current, distance)
            current, report = leg.transmit(current)
            spectra.append(current)
            reports.append(report)
        return Solution(
            tuple(spectra),
            self.interfaces,
            tuple(reports),
            tuple(leg for leg in legs if leg is not None),
        )

    def _incident(self, light):
        """The light as the Spectrum of the first region."""
        if isinstance(light, _Beam):
            # One BeamSpectrum, and so one carrier, per beam: its legs are reused.
            if light is not self._beam_input:
                source = check_incident(self._beam(light), self.region)
                self._beam_input, self._beam_source = light, source
            return self._beam_source
        if self.lattice is not None:
            raise ValueError(
                "lattice= carries a beam; a sampled field brings its own lattice"
            )
        elif isinstance(light, (ElectricField, TransverseElectricField)):
            source = sampled_spectrum(light, self.region)
        else:
            source = light
        return check_incident(source, self.region)

    def _beam(self, beam):
        """The beam as a Spectrum, charted where its first leg reads it."""
        if self.lattice is not None:
            return BeamSpectrum(beam, self.lattice)
        if self.realization == "lattice":
            raise ValueError(
                "realization='lattice' carries a beam on a lattice: pass lattice= "
                "(a CartesianGrid), or a sampled field"
            )
        axis = beam.frame
        origin = axis.origin
        if self.interfaces:
            frame = self.interfaces[0].surface.frame
            first = frame.origin + self.planes[0] * frame.normal
            origin = origin + float((first - origin) @ axis.normal) * axis.normal
        plane = Frame(origin, axis.basis)
        chart = PolarGrid(
            bd.linspace(0.0, footprint(beam, plane), 9),
            2 * bd.pi * bd.arange(8) / 8,
            plane,
        )
        return BeamSpectrum(beam, chart)

    def _prepare(self, source, carrier):
        """One leg per surface (None where the index does not change), reused."""
        meridian = isinstance(source, HankelSpectrum) or (
            isinstance(source.grid, PolarGrid) or self.realization == "meridian"
        )
        if meridian and self.realization == "lattice":
            raise ValueError(
                "this light is carried on the meridian (a HankelSpectrum, or a "
                "beam on a polar chart); realization='lattice' cannot read it"
            )
        key = self._meridian_key(source) if meridian else self._lattice_key(source)
        key += (id(carrier),)
        legs = list(self._legs) if key == self._key else []
        build = self._meridian_leg if meridian else self._lattice_leg
        for index in range(len(legs), len(self.interfaces)):
            interface = self.interfaces[index]
            if interface.minus.material.n == interface.plus.material.n:
                legs.append(None)
                continue
            legs.append(build(index, legs, source, carrier))
        # The carrier is held so that its id in the key stays its own.
        self._key, self._legs, self._carrier = key, tuple(legs), carrier
        return self._legs

    def _lattice_key(self, source):
        grid = source.grid
        if not isinstance(grid, CartesianGrid):
            raise TypeError("the lattice realization needs light on a CartesianGrid")
        # Placements are checked on every call, also when legs are reused.
        check_placement(self.interfaces, grid, source.wavelength0)
        return ("lattice",) + lattice_key(grid, source.wavelength0)

    def _lattice_leg(self, index, legs, source, carrier):
        interface = self.interfaces[index]
        frame = interface.surface.frame
        origin = frame.origin + self.planes[index] * frame.normal
        last = next((leg for leg in reversed(legs) if leg), None)
        # The next leg reads the previous leg's output lattice, and its
        # carrier is the family that leg refracted.
        lattice = source.grid if last is None else last.output_plane
        family = carrier
        if last is not None:
            first = next(leg for leg in legs if leg)
            family = RefractedRays(last.map, (first.plane.u, first.plane.v))
        plane = CartesianGrid(lattice.u, lattice.v, Frame(origin, frame.basis))
        return LatticeTransport(
            plane, interface, family, source.wavelength0, window=self.fans[index]
        )

    @staticmethod
    def _meridian_key(source):
        grid = source.grid
        if isinstance(source, BeamSpectrum):
            identity = ("beam", id(source.beam))
        elif isinstance(grid, PolarGrid):
            identity = (grid.radius.tobytes(),)
        else:
            identity = (grid.u.tobytes(), grid.v.tobytes())
        origin = axis_of(source).origin.tobytes()
        return ("meridian", source.wavelength0, identity, origin)

    def _meridian_leg(self, index, legs, source, carrier):
        interface = self.interfaces[index]
        axis = axis_of(source)
        last = next((leg for leg in reversed(legs) if leg), None)
        extent = None
        if last is not None:
            # Labels on the first plane: the fan can only narrow from there.
            reach, scale = last.edge, last.edge
            family = RefractedRays(last.map)
        else:
            family = carrier
            frame = interface.surface.frame
            plane = Frame(frame.origin + self.planes[index] * frame.normal, axis.basis)
            if isinstance(source, BeamSpectrum):
                reach, scale = None, footprint(source.beam, plane)
                # A surface that carries every ray leaves the fan to the beam.
                extent = 2 * scale
            elif isinstance(source, HankelSpectrum):
                reach = scale = source.extent
            else:
                grid = source.grid
                reach = float(min(-grid.u[0], grid.u[-1], -grid.v[0], grid.v[-1]))
                scale = reach
        return MeridianTransport(
            interface,
            family,
            axis,
            source.wavelength0,
            offset=self.planes[index],
            window=self.fans[index],
            reach=reach,
            scale=scale,
            extent=extent,
        )
