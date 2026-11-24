"""The method: light through an optical system, carried along its rays across each encounter.

``Propagator(system)(light)`` carries the light across each encounter of the
system's path (a refraction, a reflection, a stop) along the rays of its
carrier (``vecdiff.transport``), and re-expands it in plane waves in the
region that follows. Each leg meets its surface along the propagation: a
surface met against its normal is met as its reverse, and its axis turns
over with the light (``propagation.path``). The result is a ``Solution``:
one Spectrum per segment of the path, one TransportReport per carried
encounter, and the field (E and H together) on any target.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from vecdiff._backend import bd, to_numpy
from vecdiff._printing import block, fan, guarded, sequence, shorten
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
from vecdiff.media.absorption import TRANSPORT, require_transparent
from vecdiff.propagation import _summary
from vecdiff.propagation.coverage import CoverageEvaluator, validate_coverage
from vecdiff.propagation.homogeneous import PlaneTransport
from vecdiff.propagation.path import orient
from vecdiff.propagation.targets import target_grid
from vecdiff.sampling import CartesianGrid, PointGrid, PolarGrid
from vecdiff.surfaces.surface import GraphSurface, ReversedSurface
from vecdiff.system import ELEMENTS, OpticalSystem, Stop
from vecdiff.transport.dyadic import check_window
from vecdiff.transport.lattice import LatticeTransport
from vecdiff.transport.meridian import MeridianTransport
from vecdiff.transport.rays import RefractedRays

REALIZATIONS = ("auto", "meridian", "lattice")


@dataclass(frozen=True, eq=False, repr=False)
class Solution:
    """Light through an optical system: the field of every segment of its path.

    ``spectra`` holds one Spectrum per segment, the incident light first and
    the light leaving the last element (``spectrum``) last; ``elements`` the
    path's elements as the light meets them. ``reports`` holds what was not
    carried at each carried encounter, and ``legs`` the prepared transport
    legs that carried it. ``field`` gives E and H on a target: at each point
    the sum of the spectra of the segments that reach it (in front of a
    mirror, the incident and the reflected light).

    ``bands`` are the heights each element's surface spans where it carries
    light, and ``carriers`` the leg of each element (None where it carries
    nothing away, an identity), which give the automatic windows.
    """

    spectra: tuple
    elements: tuple
    reports: tuple
    legs: tuple = ()
    bands: tuple | None = None
    carriers: tuple | None = None
    _coverage: CoverageEvaluator = field(
        default_factory=CoverageEvaluator, repr=False, compare=False
    )

    @property
    def spectrum(self):
        """The Spectrum of the last segment: the light leaving the system."""
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
        coverage="complete",
    ):
        """The target's grid, the segments of each point and which points are computed.

        The target is a sampled grid, a placed surface, or one of x=, y=, z=
        for a coordinate plane; a surface or plane is sampled over ``window``
        (u_min, u_max, v_min, v_max), by default the footprint of the rays
        that reach it, padded by their diffraction (else the projected input
        chart and path), at ``shape`` (n_v, n_u), by default 0.4 wavelengths
        in the densest medium. Points are assigned to the segments that
        reach them; on a surface, ``side`` ("minus", the side the light
        arrives from there, or "plus") says which. coverage "complete" takes
        each segment's spectrum everywhere in its region; "reduced" leaves
        out the band of heights a curved surface spans, where the spectrum
        is the continuation of the carried field beyond its plane.
        """
        validate_coverage(coverage)
        source = self.spectra[0]
        grid, sampling = target_grid(
            source,
            self.elements,
            target,
            x=x,
            y=y,
            z=z,
            window=window,
            shape=shape,
            footprint=self._footprint if self.carriers is not None else None,
        )
        report = self._coverage.classify(
            grid,
            self.elements,
            source,
            mode=coverage,
            side=side,
            bands=self.bands,
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
        coverage="complete",
        method="auto",
        tolerance=None,
        chunk_size=None,
        threads=None,
    ):
        """E and H (an ElectromagneticField, SI units) on a target.

        The target and its sampling are those of ``coverage``; points left
        out are NaN and marked invalid, points no segment reaches receive no
        light (zero). The summation options are those of every Spectrum.
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
        coverage="complete",
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

    def _footprint(self, target):
        """The window of a target surface that the light's rays reach, or None.

        The rays of the segments that reach the target's origin, where their
        light is at least a hundredth of its brightest intensity, are carried
        straight to its plane; their extent is padded by four diffraction
        widths lambda / (n NA) of their cone, and at least 25 wavelengths.
        Returns that window and the largest index of those segments, which
        sets the sampling. The incident light has no rays of its own: None
        where only it, or no segment, reaches the target.
        """
        if not isinstance(target, GraphSurface):
            return None
        centre = PointGrid(bd.asarray(target.frame.origin)[None])
        try:
            members = self._coverage.classify(
                centre, self.elements, self.spectra[0], mode="complete", side="plus"
            ).segments
        except ValueError:
            return None
        # The incident light has no rays of its own; the segments that carried
        # light set the window.
        reached = np.flatnonzero(to_numpy(members)[1:, 0]) + 1
        if not len(reached):
            return None
        frame = target.frame
        normal, axes = to_numpy(frame.normal), to_numpy(frame.basis)[:, :2]
        origin = to_numpy(frame.origin)
        uv, pad = [], 0.0
        for index in reached:
            leg = next((g for g in self.carriers[:index][::-1] if g is not None), None)
            if leg is None:
                return None
            points, directions = (to_numpy(a) for a in leg.footprint())
            # The rays that carry light: a hundredth of the brightest intensity.
            light = to_numpy(
                bd.linalg.norm(
                    self.spectra[index].electric(PointGrid(bd.asarray(points))).values,
                    axis=-1,
                )
            )
            lit = light >= 0.1 * light.max()
            points, directions = points[lit], directions[lit]
            along = directions @ normal
            keep = np.abs(along) > 1e-3
            if not keep.any():
                return None
            points, directions = points[keep], directions[keep]
            travel = ((origin - points) @ normal) / along[keep]
            uv.append((points + travel[:, None] * directions - origin) @ axes)
            mean = directions.mean(axis=0)
            mean /= np.linalg.norm(mean)
            spread = np.linalg.norm(
                directions - np.outer(directions @ mean, mean), axis=-1
            )
            n = abs(self.spectra[index].region.material.n)
            step = self.spectra[0].wavelength0 / n
            pad = max(pad, 25 * step, 4 * step / max(float(spread.max()), 1e-6))
        uv = np.concatenate(uv)
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        width = float(np.max(hi - lo))
        pad = min(pad, max(0.25 * width, 25 * step))
        index = max(abs(self.spectra[k].region.material.n) for k in reached)
        return (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad), index

    @guarded
    def __repr__(self):
        """One line: how many elements, how many carried, what leaves and where."""
        return _summary.summary(self)

    @guarded
    def __str__(self):
        """The table of what each element's leg carried, and what leaves the path."""
        return _summary.text(self)

    @guarded
    def _repr_html_(self):
        return _summary.html(self)

    @guarded
    def _repr_markdown_(self):
        return _summary.markdown(self)


def surface_name(surface):
    """The name of a surface's kind, the same seen from either side."""
    while isinstance(surface, ReversedSurface):
        surface = surface.base
    return type(surface).__name__


def describe(index, element):
    """One line naming an element of the path and its regions."""
    if isinstance(element, Stop):
        return f"stop {index} (radius {element.radius:g}, in {element.incident.name})"
    name = surface_name(element.surface)
    if element.branch == "transmitted":
        return (
            f"surface {index} ({name}, "
            f"{element.incident.name} -> {element.outgoing.name})"
        )
    if element.branch == "mirror":
        return f"surface {index} ({name}, mirror in {element.incident.name})"
    return (
        f"surface {index} ({name}, reflected in {element.incident.name}, "
        f"{element.far.name} behind)"
    )


def diffracted(last, element, wavelength0):
    """How far beyond a stop's edge the next leg carries labels: the width
    the edge's diffraction spreads to on its plane, 6 wavelengths in the
    medium and three Fresnel widths sqrt(lambda d / n) over the distance d
    between the planes."""
    if not isinstance(last.encounter, Stop):
        return 0.0
    frame = element.surface.frame
    n = last.outgoing.material.n
    distance = abs(float((frame.origin - last.plane.origin) @ last.plane.normal))
    return 6 * wavelength0 / n + 3 * float(np.sqrt(wavelength0 * distance / n))


def optical_system(system):
    """The OpticalSystem of ``system``: one, one element, or a sequence of them."""
    if isinstance(system, OpticalSystem):
        return system
    if isinstance(system, ELEMENTS):
        return OpticalSystem(system)
    return OpticalSystem(*system)


def identity(element):
    """Whether an element carries nothing away: a surface between equal indices,
    without an edge."""
    return (
        element.branch == "transmitted"
        and not isinstance(element, Stop)
        and element.aperture is None
        and element.incident.material.n == element.far.material.n
    )


def sampled_spectrum(light, region):
    """A sampled incident field as the PlaneSpectrum of its propagating waves,
    or, in an absorbing region where every wave decays, of all of them."""
    if region is None:
        raise ValueError(
            "a sampled incident field needs the medium it is in: give the "
            "system an element that names it, or medium="
        )
    evanescent = "retain" if region.material.absorbing else "discard"
    return angular_spectrum(light, region, evanescent=evanescent)


def check_absorption(elements):
    """Raise AbsorbingMediumError where a leg would carry light along rays
    through an absorbing medium: every encounter but an identity and a
    reflection from an absorbing medium behind."""
    for index, element in enumerate(elements, 1):
        if element.incident is None or identity(element):
            continue
        what = f"carrying light across element {index} ({type(element).__name__})"
        require_transparent(element.incident, what, TRANSPORT)
        if element.branch == "transmitted":
            require_transparent(element.far, what, TRANSPORT)


def lattice_key(grid, wavelength):
    """What legs prepared on a lattice depend on, besides geometry and carrier."""
    return (wavelength, grid.u.tobytes(), grid.v.tobytes(), grid.frame.basis.tobytes())


def check_placement(elements, grid, wavelength):
    """Raise unless every surface shares the lattice's orientation and transverse origin."""
    for element in elements:
        frame = element.surface.frame
        offset = (frame.origin - grid.frame.origin) @ grid.frame.basis[:, :2]
        if (
            not bd.allclose(frame.basis, grid.frame.basis, rtol=0, atol=1e-12)
            or bd.linalg.norm(offset) > 1e-12 * wavelength
        ):
            raise ValueError(
                "the surfaces must share the lattice's orientation and "
                "transverse origin"
            )


def check_incident(spectrum, region):
    """The light as a Spectrum of ``region``.

    Light in another region of the same material is the light arriving
    there: the system says where it is.
    """
    if not isinstance(spectrum, Spectrum):
        raise TypeError(
            "light must be a beam (fields.PlaneBeam, fields.FocusedBeam), "
            "a sampled field (fields.ElectricField, fields.TransverseElectricField) "
            "or a fields.Spectrum"
        )
    if region is not None and spectrum.region != region:
        if spectrum.region.material != region.material:
            raise ValueError(
                f"the light is in {spectrum.region.material.name} (n = "
                f"{spectrum.region.material.n:g}), not in the system's incident "
                f"region ({region.material.name}, n = {region.material.n:g})"
            )
        spectrum = spectrum.relabeled(region)
    return spectrum


def axis_of(source):
    """The frame whose normal is the axis of the light: the beam's, else its plane's."""
    return source.beam.frame if isinstance(source, BeamSpectrum) else source.plane


def axis_direction(source, carrier):
    """The way the light's axis ray travels from its frame's origin."""
    frame = axis_of(source)
    if isinstance(source, BeamSpectrum):
        return frame.normal
    if carrier is not None and hasattr(carrier, "directions"):
        direction = carrier.directions(frame.origin[None])[0]
        if bool(bd.all(bd.isfinite(direction))):
            return direction
    return source.direction * frame.normal


class Propagator:
    """The eikonal-angular-spectrum method, prepared for one optical system.

    Calling it with light returns the Solution. Across each encounter of the
    path the field is carried along the rays of its carrier, from a
    reference plane at the surface to the same plane on the side the light
    leaves into (``transport``); between encounters it propagates as plane
    waves. A stop is carried as an encounter with its screen: what passes
    the hole goes on, cut there, and diffracts from the cut. Each leg's
    surface and axis follow the light: after a reflection the surfaces are
    met from their other side (``propagation.path``).

    ``planes`` places the reference planes: None puts them at each element's
    vertex (a stop's on its screen), otherwise one offset per element along
    the propagation. ``fans`` weight the carried rays at each element: None
    carries every ray it supports (within a stop's hole or a face's edge),
    (start, end) rolls them off between those incidence angles in degrees, a
    callable receives the RayMap; one per element. ``realization`` chooses
    how the legs are carried: "meridian" carries one meridian of a system of
    surfaces of revolution about the light's axis, with Hankel spectra per
    azimuthal order, at a cost linear in the aperture (mirrors on the axis
    turn the axis over); "lattice" carries a lattice of any geometry (folds,
    tilts); "auto" takes the meridian for a beam and the lattice for a
    sampled field, a lattice spectrum or a beam given ``lattice``.
    ``lattice`` (a CartesianGrid) is the lattice a beam is carried on, read
    exactly at the preimages; a sampled field brings its own.

    Legs depend on the geometry, the carrier and the incident lattice or
    beam, not on the field: they are prepared once and reused for every
    call with the same ones.
    """

    def __init__(
        self, system=(), *, planes=None, fans=None, realization="auto", lattice=None
    ):
        self.system = optical_system(system)
        check_absorption(self.system.elements)
        count = len(self.system.elements)
        self.planes = (
            (0.0,) * count if planes is None else tuple(float(x) for x in planes)
        )
        if len(self.planes) != count:
            raise ValueError("give one reference-plane offset per element")
        self.fans = (
            (None,) * count if fans is None else tuple(check_window(f) for f in fans)
        )
        if len(self.fans) != count:
            raise ValueError("give one fan per element")
        if realization not in REALIZATIONS:
            raise ValueError("realization must be 'auto', 'meridian' or 'lattice'")
        if lattice is not None and not isinstance(lattice, CartesianGrid):
            raise TypeError("lattice must be a CartesianGrid")
        if lattice is not None and realization == "meridian":
            raise ValueError("a lattice carries the beam off the meridian")
        self.realization, self.lattice = realization, lattice
        self._key, self._legs, self._carrier = None, (), None
        self._path_key, self._path = None, None
        self._beam_input, self._beam_source = None, None

    @property
    def elements(self):
        return self.system.elements

    @guarded
    def __repr__(self):
        """Its system in one line, the realization and any lattice."""
        lattice = "" if self.lattice is None else f", lattice={self.lattice!r}"
        return shorten(
            f"Propagator({self.system!r}, realization={self.realization!r}{lattice})"
        )

    @guarded
    def __str__(self):
        """Its system's table, then its reference planes, fans and realization."""
        fans = [fan(f) for f in self.fans]
        lines = [
            ("realization", self.realization),
            ("lattice", None if self.lattice is None else repr(self.lattice)),
            ("planes", f"offsets {sequence(self.planes)} from each vertex"),
            ("fans", "; ".join(f"{i}: {f}" for i, f in enumerate(fans, 1)) or "none"),
            ("legs prepared", str(len(self._legs))),
        ]
        return f"{self.system}\n" + block("Propagator", lines)

    def __call__(self, light, carrier=None):
        """The Solution for ``light``, carried along ``carrier``.

        ``light`` is a beam (its own carrier unless ``carrier`` is given), a
        sampled field or a Spectrum in the medium the system's path starts
        in. A sampled field or a plane spectrum needs its carrier (a
        fields.Carrier).
        """
        source = self._incident(light)
        system = self.system
        if system.region is None:
            # No element names the medium: the light's own.
            system = OpticalSystem(*system.elements, medium=source.region)
            source = check_incident(source, system.region)
            check_absorption(system.elements)
        family = carrier
        if family is None and isinstance(source, BeamSpectrum):
            family = source.carrier
        if family is None and not all(identity(e) for e in system.elements):
            raise ValueError(
                "the method carries light along its carrier: pass carrier= "
                "(a fields.Carrier) with the field, or use "
                "alternatives.WavePropagator for a field without one"
            )
        path = self._orient(system, source, family)
        legs = self._prepare(source, family, path)
        spectra, reports = [source], []
        for element, leg in zip(path.elements, legs):
            current = spectra[-1]
            if leg is None:
                # An index-matched surface: the identity, without a cut.
                spectra.append(current.relabeled(element.outgoing))
                continue
            current, report = leg.transmit(self._moved(current, leg))
            spectra.append(current)
            reports.append(report)
        return Solution(
            tuple(spectra),
            path.elements,
            tuple(reports),
            tuple(leg for leg in legs if leg is not None),
            bands=tuple(None if leg is None else leg.band for leg in legs),
            carriers=legs,
        )

    @staticmethod
    def _moved(current, leg):
        """A PlaneSpectrum moved by FFT onto a parallel plane of the next leg.

        The other spectra, and a PlaneSpectrum for a plane that is not
        parallel (after a reflection, a fold), are read at any point.
        """
        if not isinstance(current, PlaneSpectrum):
            return current
        plane = leg.plane if isinstance(leg.plane, Frame) else leg.plane.frame
        if not bd.allclose(plane.basis, current.plane.basis, rtol=0, atol=1e-12):
            return current
        distance = float((plane.origin - current.plane.origin) @ current.plane.normal)
        return PlaneTransport().spectrum(current, distance)

    def _incident(self, light):
        """The light as the Spectrum of the first region."""
        region = self.system.region
        if isinstance(light, _Beam):
            # One BeamSpectrum, and so one carrier, per beam: its legs are reused.
            if light is not self._beam_input:
                source = check_incident(self._beam(light), region)
                self._beam_input, self._beam_source = light, source
            return self._beam_source
        if self.lattice is not None:
            raise ValueError(
                "lattice= carries a beam; a sampled field brings its own lattice"
            )
        elif isinstance(light, (ElectricField, TransverseElectricField)):
            source = sampled_spectrum(light, region)
        else:
            source = light
        return check_incident(source, region)

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
        if self.elements:
            frame = self.elements[0].surface.frame
            first = frame.origin + self.planes[0] * frame.normal
            origin = origin + float((first - origin) @ axis.normal) * axis.normal
        plane = Frame(origin, axis.basis)
        chart = PolarGrid(
            bd.linspace(0.0, footprint(beam, plane), 9),
            2 * bd.pi * bd.arange(8) / 8,
            plane,
        )
        return BeamSpectrum(beam, chart)

    def _orient(self, system, source, carrier):
        """The path as this light meets it, traced once per axis."""
        axis = axis_of(source)
        direction = bd.asarray(axis_direction(source, carrier), dtype=float)
        key = (id(system), axis.origin.tobytes(), axis.basis.tobytes())
        key += (to_numpy(direction).tobytes(),)
        if key != self._path_key:
            self._path = orient(system.elements, axis, direction, source.wavelength0)
            self._path_key, self._system = key, system
        return self._path

    def _prepare(self, source, carrier, path):
        """One leg per element (None where it carries nothing away), reused."""
        meridian = isinstance(source, HankelSpectrum) or (
            isinstance(source.grid, PolarGrid) or self.realization == "meridian"
        )
        if meridian and self.realization == "lattice":
            raise ValueError(
                "this light is carried on the meridian (a HankelSpectrum, or a "
                "beam on a polar chart); realization='lattice' cannot read it"
            )
        key = self._meridian_key(source) if meridian else self._lattice_key(source)
        key += (id(carrier), self._path_key)
        legs = list(self._legs) if key == self._key else []
        build = self._meridian_leg if meridian else self._lattice_leg
        for index in range(len(legs), len(path.elements)):
            if identity(path.elements[index]):
                legs.append(None)
                continue
            legs.append(build(index, legs, source, carrier, path))
        # The carrier is held so that its id in the key stays its own.
        self._key, self._legs, self._carrier = key, tuple(legs), carrier
        return self._legs

    def _lattice_key(self, source):
        grid = source.grid
        if not isinstance(grid, CartesianGrid):
            raise TypeError("the lattice realization needs light on a CartesianGrid")
        return ("lattice",) + lattice_key(grid, source.wavelength0)

    def _lattice_leg(self, index, legs, source, carrier, path):
        element = path.elements[index]
        frame = element.surface.frame
        origin = frame.origin + self.planes[index] * frame.normal
        last = next((leg for leg in reversed(legs) if leg), None)
        # The next leg reads the previous leg's output lattice, and its
        # carrier is the family that leg carried on.
        lattice = source.grid if last is None else last.output_plane
        family = carrier
        if last is not None:
            first = next(leg for leg in legs if leg)
            family = RefractedRays(last.map, (first.plane.u, first.plane.v))
        elif not isinstance(source, BeamSpectrum):
            # A sampled field is read on its own lattice: the first plane is it.
            check_placement((element,), source.grid, source.wavelength0)
        plane = CartesianGrid(lattice.u, lattice.v, Frame(origin, frame.basis))
        return LatticeTransport(
            plane, element, family, source.wavelength0, window=self.fans[index]
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

    def _meridian_leg(self, index, legs, source, carrier, path):
        element = path.elements[index]
        axis = path.axes[index]
        last = next((leg for leg in reversed(legs) if leg), None)
        extent = None
        if last is not None:
            # Labels on the first plane: the fan can only narrow from there,
            # but beyond a stop's edge it also carries the light the edge
            # diffracts.
            reach = scale = last.edge + diffracted(last, element, source.wavelength0)
            family = RefractedRays(last.map)
        else:
            family = carrier
            frame = element.surface.frame
            plane = Frame(frame.origin + self.planes[index] * frame.normal, axis.basis)
            if isinstance(source, BeamSpectrum):
                reach, scale = None, footprint(source.beam, plane)
                # A surface that carries every ray leaves the fan to the beam;
                # otherwise it ends where the surface stops carrying rays (a
                # Stop bounds it where that is too wide).
                extent = 2 * scale
            elif isinstance(source, HankelSpectrum):
                reach = scale = source.extent
            else:
                grid = source.grid
                reach = float(min(-grid.u[0], grid.u[-1], -grid.v[0], grid.v[-1]))
                scale = reach
        return MeridianTransport(
            element,
            family,
            axis,
            source.wavelength0,
            offset=self.planes[index],
            window=self.fans[index],
            reach=reach,
            scale=scale,
            extent=extent,
        )
