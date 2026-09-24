"""Experimental R Q^-1 T^[N] Q R for direction-resolved analytic incidence.

The local coefficient expansion is followed by the production homogeneous Q
inverse. The inverse does not solve dielectric matching or fit any reference.
The default Fresnel propagation API is unaffected by this opt-in experiment.
"""

from time import perf_counter

import numpy as np

from ...fields import ElectricField, ElectromagneticField
from ...fields.spectrum import PlaneWaveExpansion
from ..high_frequency import _chart_surface
from ..operators import SurfaceTransport
from .taylor_interface import interface_expansion


class TaylorInterfaceOperator:
    """Assemble orders 0…N on one graph, then apply the same Q^-1 to each.

    source must be an analytic PlaneWaveExpansion. Every populated direction
    is treated separately, never replaced by a phase-gradient effective ray.
    height and length_scale follow interface_expansion's normalized-graph API.
    An aperture is an absorbing pupil, exactly as in the primary forward path.
    This is not a complete finite-body or multiple-encounter calculation.
    """

    def __init__(
        self,
        source,
        interface,
        grid,
        *,
        height,
        length_scale,
        order=3,
        aperture=None,
        max_sine=0.92,
        batch_size=64,
    ):
        if not isinstance(source, PlaneWaveExpansion):
            raise TypeError(
                "curvature expansion requires direction-resolved plane waves"
            )
        if source.region != interface.minus:
            raise ValueError("incident spectrum and interface medium disagree")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not isinstance(order, int) or not 0 <= order <= 3:
            raise ValueError("order must be zero through three")
        start = perf_counter()
        q = SurfaceTransport(
            grid,
            _chart_surface(interface.surface, aperture),
            interface.plus,
            source.wavelength0,
            max_sine=max_sine,
            tolerance=2e-12,
        )
        self.transport, self.order = q, order
        mask = (
            np.ones(grid.shape, bool)
            if aperture is None
            else np.linalg.norm(q.xy, axis=-1) <= aperture
        )
        positions = np.flatnonzero(mask.ravel())
        if not len(positions):
            raise ValueError("aperture contains no samples")
        self.traces = {}
        arrays = {
            branch: [np.zeros((order + 1, q.size, 3), complex) for _ in range(2)]
            for branch in ("incident", "reflected", "transmitted")
        }
        basis, origin = interface.surface.frame.basis, interface.surface.frame.origin
        local_xy = q.xy.reshape(-1, 2)
        ni, nt = interface.minus.material.n, interface.plus.material.n
        k = 2 * np.pi * ni / source.wavelength0
        for wavevector, amplitude in zip(source.wavevectors, source.amplitudes):
            if not np.any(amplitude):
                continue
            if np.any(abs(wavevector.imag) > 1e-12 * k):
                raise ValueError(
                    "curvature expansion currently requires propagating incidence"
                )
            direction = wavevector.real @ basis / k
            local_amplitude = (amplitude @ basis) * np.exp(
                1j * np.dot(wavevector, origin - source.origin)
            )
            for first in range(0, len(positions), batch_size):
                ids = positions[first : first + batch_size]
                expansion = interface_expansion(
                    height,
                    local_xy[ids],
                    n_incident=ni,
                    n_transmitted=nt,
                    direction=direction,
                    amplitude=local_amplitude,
                    order=order,
                    length_scale=length_scale,
                )
                expected = (q.grid.points.reshape(-1, 3)[ids] - origin) @ basis
                expected_normals = q.grid.normals.reshape(-1, 3)[ids] @ basis
                if not np.allclose(
                    expansion.points, expected, rtol=1e-11, atol=1e-12 * length_scale
                ) or not np.allclose(
                    expansion.normals, expected_normals, atol=1e-11, rtol=0
                ):
                    raise ValueError(
                        "Taylor height/gradient does not match the interface geometry"
                    )
                for branch, (electric, magnetic) in arrays.items():
                    for n in range(order + 1):
                        trace = expansion.trace(branch, source.wavelength0, order=n)
                        electric[n, ids] += trace.electric.values @ basis.T
                        magnetic[n, ids] += trace.magnetic @ basis.T
        for branch, (electric, magnetic) in arrays.items():
            self.traces[branch] = tuple(
                ElectromagneticField(
                    ElectricField(
                        q.grid,
                        source.wavelength0,
                        electric[n].reshape(grid.shape + (3,)),
                    ),
                    magnetic[n].reshape(grid.shape + (3,)),
                )
                for n in range(order + 1)
            )
        self.assembly_seconds = perf_counter() - start

    def spectrum(self, order, *, tolerance=1e-10, max_iterations=500):
        """Q^-1 T^[N] Q for the selected order; returns spectrum and Q^-1 report."""
        if not isinstance(order, int) or not 0 <= order <= self.order:
            raise ValueError("requested order is outside the assembled expansion")
        return self.transport.inverse(
            self.traces["transmitted"][order],
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
