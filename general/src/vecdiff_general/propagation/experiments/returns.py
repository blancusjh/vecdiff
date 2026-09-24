"""Internal returns of a two-surface element: the operational Debye series.

A field entering an element through its front surface is transmitted (T) and
reflected (R) at every later encounter. What is transmitted leaves; what is
reflected stays inside and travels back to the other surface, where the same
happens again. With the interior spectra as the state, pass p + 1 is

    a_{p+1} = R_j P a_p,       exit_p = T_j P a_p,

where P is the exact homogeneous transport (ASM) between the two vertex
planes followed by Q onto the next surface, and j alternates between the back
(odd p) and front (even p) surfaces. The entering field is a_1 = T_front s
and the external reflection is exit_0 = R_front s. Summed over passes,

    a = sum_p (R_f P R_b P)^k (a_1 + R_b P a_1) = (I - M)^{-1} (...),

with the round trip M = R_f P R_b P, a Neumann series that is the Debye series
of the element. For a sphere every pass has an exact counterpart
(``references.debye.SphereDebyeSeries``): pass p is the Debye term p.

Backward travel reuses the forward machinery on reversed charts: the same
physical surface is described in a frame whose normal points back along the
axis, so Q, Q^-1, ASM and the interface interaction are unchanged. The two
lattices are mirror images, so traces move between them by reversing rows.

Model limits, all reported rather than hidden: light incident outside the
entrance edge is not carried (its norm is returned); beyond a surface edge the
chart's flat extension is crossed with flat Fresnel (the sidewall model of
``propagate_high_frequency``); the lattice is periodic.
"""

from dataclasses import dataclass

import numpy as np

from ...fields import ElectricField, ElectromagneticField
from ...fields.spectrum import PlaneWaveExpansion
from ...geometry.frames import Frame
from ...interfaces import DielectricInterface
from ...sampling import CartesianGrid
from ...surfaces import Plane, SphericalCap
from ..high_frequency import _chart_surface, _masked
from ..operators import PlanarASM, SurfaceTransport

REVERSED = np.diag([1.0, -1.0, -1.0])


def reversed_surface(surface):
    """The same surface in a frame looking back along the axis."""
    frame = surface.frame
    if not np.allclose(frame.basis, np.eye(3), atol=1e-12, rtol=0):
        raise ValueError("element surfaces need the identity frame orientation")
    back = Frame(frame.origin, REVERSED)
    if isinstance(surface, SphericalCap):
        return SphericalCap(-surface.radius, back)
    if isinstance(surface, Plane):
        return Plane(back)
    raise NotImplementedError("returns are implemented for planes and spheres")


def _mirror(trace, grid):
    """A chart trace on the mirror lattice of the same physical surface."""
    return ElectromagneticField(
        ElectricField(grid, trace.electric.wavelength0, trace.electric.values[::-1]),
        trace.magnetic[::-1],
    )


@dataclass(frozen=True)
class ReturnsSolution:
    """Spectra of one run.

    internal[p - 1] is pass p inside the element: forward (on the front
    lattice) for odd p, backward (on the reversed back lattice) for even p.
    exits[p] is what leaves on encounter p: 0 and even p backward through the
    front surface (reversed front lattice), odd p forward through the back
    surface. reports holds every Q^-1 report in order of use. outside is the
    relative L2 norm of the incident trace beyond the entrance edge, which is
    not carried.
    """

    internal: tuple
    exits: dict
    reports: tuple
    outside: float


class ElementReturns:
    """Two-surface element with internal returns (see module docstring).

    front, back: Plane or SphericalCap in identity-oriented frames on the
    axis, front first. axis: symmetric, odd-length 1-D lattice coordinates
    shared by both vertex planes. apertures: surface edges (radius) of front and
    back. interaction: an interface interaction such as OrderInteraction.
    """

    def __init__(
        self,
        front,
        back,
        inside,
        outside,
        axis,
        wavelength0,
        apertures,
        interaction,
        *,
        max_sine=0.85,
        transport_tolerance=1e-10,
        inverse_tolerance=1e-9,
        max_iterations=400,
        mode_rtol=1e-10,
        support_rtol=0.0,
    ):
        axis = np.asarray(axis, float)
        if len(axis) % 2 == 0 or not np.array_equal(axis, -axis[::-1]):
            raise ValueError("axis must be odd-length and symmetric about zero")
        separation = back.frame.origin[2] - front.frame.origin[2]
        if (
            separation <= 0
            or np.any(back.frame.origin[:2])
            or np.any(front.frame.origin[:2])
        ):
            raise ValueError("vertices must lie on the z axis, front first")
        self.separation = separation
        self.interaction = interaction
        # Modes below mode_rtol of the largest are not sent to the interaction,
        # and chart points where the incident trace is below support_rtol of
        # its maximum get none (both bound the neglected field by that size).
        self.mode_rtol, self.support_rtol = mode_rtol, support_rtol
        self.wavelength0 = wavelength0
        self.inverse = dict(tolerance=inverse_tolerance, max_iterations=max_iterations)
        options = dict(max_sine=max_sine, tolerance=transport_tolerance)

        def transports(surface, aperture, regions):
            plane = CartesianGrid(axis, axis, surface.frame)
            chart = _chart_surface(surface, aperture)
            return [
                SurfaceTransport(plane, chart, region, wavelength0, **options)
                for region in regions
            ]

        a_front, a_back = apertures
        front_b, back_b = reversed_surface(front), reversed_surface(back)
        # (incident-side, transmitted-side) transports of each oriented encounter.
        self.entrance = transports(front, a_front, (outside, inside))
        self.exit = transports(back, a_back, (inside, outside))
        self.back_reversed = transports(back_b, a_back, (inside,))[0]
        self.front_reversed = transports(front_b, a_front, (inside, outside))
        self.interfaces = {
            "entrance": DielectricInterface(front, outside, inside),
            "back": DielectricInterface(back, inside, outside),
            "front": DielectricInterface(front_b, inside, outside),
        }
        radius = np.linalg.norm(self.entrance[0].xy, axis=-1)
        self.clear_front = radius <= a_front
        self.clear_back = np.linalg.norm(self.exit[0].xy, axis=-1) <= a_back
        self.everywhere = np.ones_like(self.clear_front)

    def _encounter(self, name, transport, spectrum, clear, mask):
        incident = transport.forward(spectrum)
        size = np.linalg.norm(incident.electric.values, axis=-1)
        mask = mask & (size >= self.support_rtol * size.max())
        incident = _masked(incident, mask)
        modes = spectrum.expansion
        size = np.linalg.norm(modes.amplitudes, axis=-1)
        amplitudes = np.where(
            (size >= self.mode_rtol * size.max())[:, None], modes.amplitudes, 0
        )
        modes = PlaneWaveExpansion(
            modes.wavevectors, amplitudes, modes.origin, modes.wavelength0, modes.region
        )
        return self.interaction(
            self.interfaces[name], modes, transport, incident, mask, clear
        )

    def run(self, source, passes):
        """Source: PlaneSpectrum in the outside medium on the front lattice."""
        if passes < 1:
            raise ValueError("at least one internal pass is required")
        reports, exits, internal = [], {}, []
        q_out, q_in = self.entrance
        trace = q_out.forward(source)
        outside = float(
            np.linalg.norm(trace.electric.values[~self.clear_front])
            / np.linalg.norm(trace.electric.values)
        )
        encounter = self._encounter(
            "entrance", q_out, source, self.clear_front, self.clear_front
        )
        reflected, report = self.front_reversed[1].inverse(
            _mirror(encounter.reflected, self.front_reversed[1].grid), **self.inverse
        )
        exits[0] = reflected
        reports.append(report)
        state, report = q_in.inverse(encounter.transmitted, **self.inverse)
        reports.append(report)
        for p in range(1, passes + 1):
            internal.append(state)
            arriving = PlanarASM().spectrum(state, self.separation)
            if p % 2:
                q_arrive, q_leave = self.exit
                q_return = self.back_reversed
                encounter = self._encounter(
                    "back", q_arrive, arriving, self.clear_back, self.everywhere
                )
            else:
                q_arrive, q_leave = self.front_reversed
                q_return = q_in
                encounter = self._encounter(
                    "front", q_arrive, arriving, self.clear_front, self.everywhere
                )
            exits[p], report = q_leave.inverse(encounter.transmitted, **self.inverse)
            reports.append(report)
            if p < passes:
                state, report = q_return.inverse(
                    _mirror(encounter.reflected, q_return.grid), **self.inverse
                )
                reports.append(report)
        return ReturnsSolution(tuple(internal), exits, tuple(reports), outside)
