"""Wave transport through an optical system: another method, kept for checks."""

from vecdiff._backend import bd
from vecdiff.alternatives.surface_interaction import LocalFresnel, SpectralFresnel
from vecdiff.alternatives.surface_transfer import SurfaceTransport
from vecdiff.fields import ElectricField, PlaneSpectrum, TransverseElectricField
from vecdiff.fields.beams import _Beam
from vecdiff.propagation.homogeneous import PlaneTransport
from vecdiff.propagation.propagator import (
    Solution,
    check_incident,
    check_placement,
    lattice_key,
    optical_system,
    sampled_spectrum,
)
from vecdiff.sampling import CartesianGrid
from vecdiff.system import Refraction


def refractions(system):
    """The system, if its path is refractions of unbounded faces met along their normals."""
    system = optical_system(system)
    if not all(isinstance(e, Refraction) and e.edge is None for e in system.elements):
        raise ValueError(
            "wave transport carries refractions only: finite material edges, "
            "stops and reflections are not supported by it"
        )
    return system


TREATMENTS = {"spectral": SpectralFresnel, "local": LocalFresnel}


def treatment(transmission):
    """The Fresnel treatment named by ``transmission``, or the one given."""
    if isinstance(transmission, str) and transmission in TREATMENTS:
        return TREATMENTS[transmission]()
    if isinstance(transmission, str) or not hasattr(transmission, "transmit"):
        raise ValueError(
            "transmission must be 'spectral', 'local' or a Fresnel treatment"
        )
    return transmission


class WavePropagator:
    """Light through an optical system by wave transport, without a carrier.

    At each surface the field is transported exactly from its plane spectrum
    to the surface (``SurfaceTransport``), transmitted there by a Fresnel
    treatment, and radiated back into a plane spectrum of the next region.
    ``transmission`` is "spectral" (``SpectralFresnel``, Fresnel resolved by
    incident direction), "local" (``LocalFresnel``, one local direction from
    the field's phase gradient) or a treatment with a ``transmit`` method.
    ``max_sine`` is the band of plane waves the surface transport keeps and
    ``transport_tolerance`` the tolerance of its kernel and inverse.

    It is the check on ``propagation.Propagator``, not the method: it takes a
    sampled field or a PlaneSpectrum, whose lattice it transports, and returns
    the same Solution, whose ``reports`` are the treatment's and whose
    ``legs`` the (incident, transmitted) surface transports of each surface.
    """

    def __init__(
        self,
        system=(),
        *,
        transmission="spectral",
        max_sine=0.85,
        transport_tolerance=1e-10,
    ):
        self.system = refractions(system)
        self.elements = self.system.elements
        self.region = self.system.region
        self.transmission = treatment(transmission)
        if not 0 < max_sine < 1 or not 0 < transport_tolerance < 1:
            raise ValueError("max_sine and transport_tolerance must lie in (0, 1)")
        self.max_sine, self.transport_tolerance = max_sine, transport_tolerance
        self._key, self._legs = None, ()

    def __call__(self, light):
        """The Solution for ``light``, a sampled field or a PlaneSpectrum."""
        source = check_incident(self._incident(light), self.region)
        legs = self._prepare(source)
        spectra, reports = [source], []
        for element, (incident, transmitted) in zip(self.elements, legs):
            current = spectra[-1]
            distance = float(
                (incident.plane.frame.origin - current.plane.origin)
                @ current.plane.normal
            )
            current = PlaneTransport().spectrum(current, distance)
            if element.incident.material.n == element.outgoing.material.n:
                # Equal-index transport is an identity, without a spectral cut
                # or finite-chart reconstruction of an already known solution.
                spectra.append(current.relabeled(element.outgoing))
                continue
            response = self.transmission.transmit(current, element, incident)
            spectra.append(transmitted.radiate(response.field))
            reports.append(response.report)
        return Solution(tuple(spectra), self.elements, tuple(reports), legs)

    def _incident(self, light):
        if isinstance(light, _Beam):
            raise TypeError(
                "a beam is light for propagation.Propagator, which carries it "
                "along its carrier; sample it on a CartesianGrid for wave transport"
            )
        if isinstance(light, (ElectricField, TransverseElectricField)):
            return sampled_spectrum(light, self.region)
        if not isinstance(light, PlaneSpectrum):
            raise TypeError("wave transport takes a sampled field or a PlaneSpectrum")
        return light

    def _prepare(self, source):
        """The incident and transmitted surface transports of every surface, reused."""
        grid, wavelength = source.grid, source.wavelength0
        if not isinstance(grid, CartesianGrid):
            raise TypeError("wave transport requires a CartesianGrid PlaneSpectrum")
        # Placements are checked on every call, also when transports are reused.
        check_placement(self.elements, grid, wavelength)
        key = lattice_key(grid, wavelength)
        if key == self._key:
            return self._legs
        options = dict(max_sine=self.max_sine, tolerance=self.transport_tolerance)
        legs = []
        for element in self.elements:
            plane = CartesianGrid(grid.u, grid.v, element.surface.frame)
            legs.append(
                tuple(
                    SurfaceTransport(
                        plane, element.surface, region, wavelength, **options
                    )
                    for region in (element.incident, element.outgoing)
                )
            )
        for left, right in zip(legs, legs[1:]):
            depth = (right[0].grid.points - left[0].grid.points) @ grid.frame.normal
            if bd.any(depth <= 0):
                raise ValueError(
                    "successive surface charts intersect or are out of order"
                )
        self._key, self._legs = key, tuple(legs)
        return self._legs
