"""Transmission and reflection at dielectric boundaries.

Planar scattering is exact mode by mode. Curved-surface transmission applies
tangent-plane Fresnel to individual incident directions."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import BarycentricInterpolator

from vecdiff_general.fields import ElectricField, ElectromagneticField
from vecdiff_general.fields.spectrum import PlaneSpectrum, PlaneWaveExpansion
from vecdiff_general.fourier.nufft import phase_factor
from vecdiff_general.media import Z0
from vecdiff_general.propagation._directions import (
    direction_from_wavevector,
    reflected_direction,
    transmitted_direction,
)
from vecdiff_general.propagation._fresnel import (
    _fresnel_basis,
    apply_fresnel,
    apply_transmission,
    fresnel_reflection_matrix,
    fresnel_transmission_matrix,
)
from vecdiff_general.propagation.surface_transfer import BandPlanCache
from vecdiff_general.sampling.grids import Grid
from vecdiff_general.system import PlanarInterface


@dataclass(frozen=True)
class InterfaceSolution:
    interface: PlanarInterface
    incident: PlaneWaveExpansion
    reflected: PlaneWaveExpansion
    transmitted: PlaneWaveExpansion
    incident_side: str

    def evaluate(self, grid: Grid, *, side):
        """Total E/H in one half-space. The side is required for boundary traces."""

        if side not in ("minus", "plus"):
            raise ValueError("side must be 'minus' or 'plus'")

        frame = self.interface.frame
        signed_distance = (grid.points - frame.origin) @ frame.normal
        coordinate_scale = max(
            1.0, float(np.max(np.abs(grid.points))), float(np.max(np.abs(frame.origin)))
        )
        tolerance = 32 * np.finfo(float).eps * coordinate_scale

        if (side == "minus" and np.any(signed_distance > tolerance)) or (
            side == "plus" and np.any(signed_distance < -tolerance)
        ):
            raise ValueError("observation points lie outside the requested half-space")

        if side != self.incident_side:
            return self.transmitted.evaluate(grid)

        incident = self.incident.evaluate(grid)
        reflected = self.reflected.evaluate(grid)

        return ElectromagneticField(
            ElectricField(
                grid,
                self.incident.wavelength0,
                incident.electric.values + reflected.electric.values,
            ),
            incident.magnetic + reflected.magnetic,
        )


def scatter_plane(incident, interface):
    """Scatter every mode, preserving coherent phase and both branches.

    Incident evanescent modes are outside this first implementation's domain.
    Transmitted evanescent waves under TIR are retained. Exact grazing and
    critical incidence require a separate limiting treatment and are rejected.
    """

    if isinstance(incident, PlaneSpectrum):
        incident = incident.expansion

    if not isinstance(incident, PlaneWaveExpansion) or not isinstance(
        interface, PlanarInterface
    ):
        raise TypeError("expected a plane spectrum and PlanarInterface")

    if incident.region == interface.minus:
        normal, destination, side = interface.frame.normal, interface.plus, "minus"
    elif incident.region == interface.plus:
        normal, destination, side = -interface.frame.normal, interface.minus, "plus"
    else:
        raise ValueError("incident region is not adjacent to the interface")

    occupied = np.any(incident.amplitudes != 0, axis=-1)
    k = incident.wavevectors[occupied]
    amplitudes = incident.amplitudes[occupied]

    phase = phase_factor(k, (interface.frame.origin - incident.origin)[None, :])[0]
    amplitudes = amplitudes * phase[:, None]
    vacuum_wavenumber = 2 * np.pi / incident.wavelength0
    incident_index = incident.region.material.n
    transmitted_index = destination.material.n

    direction_incident = direction_from_wavevector(
        k, vacuum_wavenumber * incident_index
    )
    direction_transmitted = transmitted_direction(
        direction_incident, normal, incident_index, transmitted_index
    )
    direction_reflected = reflected_direction(direction_incident, normal)

    transmission = fresnel_transmission_matrix(
        direction_incident,
        direction_transmitted,
        normal,
        incident_index,
        transmitted_index,
    )
    reflection = fresnel_reflection_matrix(
        direction_incident,
        direction_transmitted,
        normal,
        incident_index,
        transmitted_index,
    )

    electric_transmitted = apply_fresnel(amplitudes, transmission)
    electric_reflected = apply_fresnel(amplitudes, reflection)

    wavevector_transmitted = (
        vacuum_wavenumber * transmitted_index * direction_transmitted
    )
    wavevector_reflected = vacuum_wavenumber * incident_index * direction_reflected

    def make(vectors, values, region):
        return PlaneWaveExpansion(
            vectors, values, interface.frame.origin, incident.wavelength0, region
        )

    return InterfaceSolution(
        interface,
        make(k, amplitudes, incident.region),
        make(wavevector_reflected, electric_reflected, incident.region),
        make(wavevector_transmitted, electric_transmitted, destination),
        side,
    )


@dataclass(frozen=True)
class TransmissionReport:
    retained_modes: int
    omitted_modal_l2: float
    transport_ranks: tuple[int, ...]


@dataclass(frozen=True)
class Transmission:
    """Transmitted E/H on the interface, before outgoing reconstruction."""

    field: ElectromagneticField
    report: TransmissionReport | LocalTransmissionReport


class SpectralFresnel:
    """Apply local tangent-plane Fresnel separately to incident directions.

    Inputs are forward propagating PlaneSpectra on the incident transport grid.
    `angular_limit` bounds each local transverse direction coordinate; `cutoff`
    is relative modal amplitude. Removed modal L2 is always reported. The exit
    interpolation uses a fine central patch and coarser outer patches. Plans
    are cached for fixed geometry/material/wavelength and modal support.

    ``coefficients="unity"`` sets t_s=t_p=1 for a vector unity-transmission control.
    Refraction, polarization rotation, E/H completion and transport stay the
    same; this does not replace vector transport with a scalar Helmholtz solver.
    """

    def __init__(
        self,
        *,
        nodes=18,
        outer_nodes=6,
        central_width=0.3,
        angular_limit=0.65,
        cutoff=1e-7,
        workers=2,
        threads=None,
        cache_bytes=1024**3,
        coefficients="fresnel",
    ):
        if coefficients not in ("fresnel", "unity"):
            raise ValueError("coefficients must be 'fresnel' or 'unity'")
        if (
            not isinstance(nodes, int)
            or not isinstance(outer_nodes, int)
            or min(nodes, outer_nodes) < 2
        ):
            raise ValueError("angular node counts must be integers >= 2")
        if not 0 < central_width < angular_limit < 1 / np.sqrt(2):
            raise ValueError("require 0 < central_width < angular_limit < 1/sqrt(2)")
        if not 0 <= cutoff < 1:
            raise ValueError("modal cutoff must lie in [0, 1)")
        self.nodes, self.outer_nodes = nodes, outer_nodes
        self.coefficients = coefficients
        self.central_width, self.angular_limit, self.cutoff = (
            central_width,
            angular_limit,
            cutoff,
        )
        self.threads = threads or os.cpu_count() or 1
        self._plans = BandPlanCache(workers=workers, cache_bytes=cache_bytes)

    def transmit(self, spectrum, interface, transport):
        """Return coherent transmitted E/H; no single resultant ray is inferred."""
        if not isinstance(spectrum, PlaneSpectrum):
            raise TypeError("direction-resolved transmission requires a PlaneSpectrum")
        modes = spectrum.expansion
        if modes.region != interface.minus or transport.region != interface.minus:
            raise ValueError(
                "spectrum and incident transport must belong to interface.minus"
            )
        if modes.wavelength0 != transport.wavelength0:
            raise ValueError("spectrum and transport wavelengths differ")
        if transport.surface is not interface.surface:
            raise ValueError("transport must evaluate this interface surface")
        if spectrum.grid.shape != transport.plane.shape or not np.allclose(
            spectrum.grid.points,
            transport.plane.points,
            rtol=0,
            atol=1e-12 * modes.wavelength0,
        ):
            raise ValueError("spectrum must use the incident transport plane")
        n1, n2 = interface.minus.material.n, interface.plus.material.n
        basis = transport.plane.frame.basis
        k = modes.wavevectors @ basis
        occupied = np.any(modes.amplitudes != 0, axis=-1)
        if np.any(abs(k[occupied].imag) > 1e-12 * transport.k) or np.any(
            k[occupied, 2].real <= 0
        ):
            raise ValueError(
                "incident evanescent/backward waves require a different transport model"
            )
        if n1 == n2:
            return Transmission(
                transport.forward(spectrum),
                TransmissionReport(int(occupied.sum()), 0.0, (transport.rank,)),
            )
        magnitude = np.linalg.norm(modes.amplitudes, axis=-1)
        direction_xy = k[:, :2].real / transport.k
        keep = (magnitude > self.cutoff * magnitude.max()) & (
            np.max(abs(direction_xy), axis=-1) <= self.angular_limit
        )
        omitted = float(
            np.linalg.norm(magnitude[~keep]) / max(np.linalg.norm(magnitude), 1e-300)
        )
        out = np.zeros((transport.size, 6), complex)
        ranks = []
        if np.any(keep):
            if np.any(keep & ~transport.active.ravel()):
                raise ValueError(
                    "retained modes exceed the incident transport bandwidth"
                )
            wavevectors = k[keep].real
            d = wavevectors / np.linalg.norm(wavevectors, axis=-1, keepdims=True)
            a = modes.amplitudes[keep] @ basis
            bounds = np.max(abs(d[:, :2]), axis=0) * (1 + 1e-12)
            normals = transport.grid.normals.reshape(-1, 3) @ basis
            edges = [np.array([-v, v]) for v in bounds]
            if n1 > n2:
                c = self.central_width
                edges = [
                    np.array([-v, -c, c, v]) if v > c else np.array([-v, v])
                    for v in bounds
                ]
            for tx in range(len(edges[0]) - 1):
                for ty in range(len(edges[1]) - 1):
                    lo = np.array([edges[0][tx], edges[1][ty]])
                    hi = np.array([edges[0][tx + 1], edges[1][ty + 1]])
                    select = np.all(
                        (d[:, :2] >= lo) & ((d[:, :2] < hi) | (lo == hi)), axis=-1
                    )
                    if not np.any(select):
                        continue
                    central = np.all(lo >= -self.central_width - 1e-7) and np.all(
                        hi <= self.central_width + 1e-7
                    )
                    nt = self.nodes if central or n1 <= n2 else self.outer_nodes
                    cheb = np.cos(np.pi * (np.arange(nt) + 0.5) / nt)
                    axes, weights = [], []
                    for j in range(2):
                        if hi[j] == lo[j]:
                            axes.append(np.array([lo[j]]))
                            weights.append(np.ones((int(select.sum()), 1)))
                        else:
                            axis = cheb * (hi[j] - lo[j]) / 2 + (hi[j] + lo[j]) / 2
                            axes.append(axis)
                            weights.append(
                                BarycentricInterpolator(axis, np.eye(nt), axis=0)(
                                    d[select, j]
                                )
                            )
                    ids = np.flatnonzero(keep)[select]
                    plan = self._plans.plan(transport, ids)
                    ranks.append(plan.rank)
                    amplitudes = a[select, :2] * transport.size

                    def node(index, axes=axes, weights=weights, ids=ids, plan=plan):
                        ix, iy = index
                        x, y = axes[0][ix], axes[1][iy]
                        direction = np.array([x, y, np.sqrt(1 - x * x - y * y)])
                        lattice = np.zeros((transport.size, 2), complex)
                        lattice[ids] = (
                            amplitudes
                            * (weights[0][:, ix] * weights[1][:, iy])[:, None]
                        )
                        wave = plan(
                            lattice.reshape(transport.plane.shape + (2,))
                        ).reshape(-1, 2)
                        return apply_transmission(
                            direction,
                            normals,
                            n1,
                            n2,
                            wave,
                            coefficients=self.coefficients,
                        )

                    # Nodes are independent; summing in node order keeps the
                    # result identical to a serial loop.
                    # Batches bound the number of pending node fields in memory.
                    indices = list(np.ndindex(len(axes[0]), len(axes[1])))
                    batch = 2 * self.threads
                    with ThreadPoolExecutor(self.threads) as pool:
                        for start in range(0, len(indices), batch):
                            for contribution in pool.map(
                                node, indices[start : start + batch]
                            ):
                                out += contribution
        shape = transport.grid.shape + (3,)
        field = ElectromagneticField(
            ElectricField(
                transport.grid, modes.wavelength0, (out[:, :3] @ basis.T).reshape(shape)
            ),
            (out[:, 3:] @ basis.T).reshape(shape) / Z0,
        )
        return Transmission(
            field, TransmissionReport(int(keep.sum()), omitted, tuple(ranks))
        )


# Compatibility name; both names select exactly the same implementation.
DirectionResolvedFresnel = SpectralFresnel


@dataclass(frozen=True)
class LocalTransmissionReport:
    """Single-direction diagnostics, not estimates of physical field error."""

    retained_modes: int
    omitted_modal_l2: float
    transport_ranks: tuple[int, ...]
    eikonal_relative_residual: float
    longitudinal_relative_amplitude: float
    dark_samples: int
    discarded_surface_relative_amplitude: float


class LocalFresnel:
    r"""Pointwise Fresnel using an internally estimated vector phase gradient.

    k_eff = Im(sum_c conj(E_c) grad(E_c)) / sum_c |E_c|².
    Spatial derivatives are computed from the incident modal representation,
    then restricted to the surface. This equals the phase gradient for a
    common scalar phase and real vector envelope; general polarization and
    interfering branches can violate that assumption. Eikonal magnitude and
    longitudinal-field residuals are checked before using the direction.

    This is a single-branch approximation, not an automatic branch separator.
    No Poynting direction or user-supplied eikonal is used. Dark samples are
    zeroed below the relative amplitude cutoff. The dyadic projects onto the
    inferred incident polarization plane; that loss is explicitly reported.
    """

    def __init__(
        self,
        *,
        validity_tolerance=0.1,
        cutoff=1e-6,
        coefficients="fresnel",
        modal_cutoff=1e-7,
        angular_limit=0.65,
    ):
        if not 0 < validity_tolerance < 1 or not 0 <= cutoff < 1:
            raise ValueError(
                "local validity tolerance and cutoff must lie in (0, 1) and [0, 1)"
            )
        if coefficients not in ("fresnel", "unity"):
            raise ValueError("coefficients must be 'fresnel' or 'unity'")
        self.validity_tolerance = validity_tolerance
        self.cutoff, self.coefficients = cutoff, coefficients
        if not 0 <= modal_cutoff < 1 or not 0 < angular_limit < 1:
            raise ValueError("invalid local incident angular band or modal cutoff")
        self.modal_cutoff, self.angular_limit = modal_cutoff, angular_limit

    def transmit(self, spectrum, interface, transport):
        if (
            transport.surface is not interface.surface
            or spectrum.expansion.region != interface.minus
        ):
            raise ValueError(
                "local transmission requires the incident interface transport"
            )
        modes = spectrum.expansion
        magnitude = np.linalg.norm(modes.amplitudes, axis=-1)
        k = modes.wavevectors @ transport.plane.frame.basis
        occupied = magnitude > 0
        if np.any(abs(k[occupied].imag) > 1e-12 * transport.k) or np.any(
            k[occupied, 2].real <= 0
        ):
            raise ValueError(
                "local Fresnel requires forward propagating incident modes"
            )
        keep = (
            (magnitude > self.modal_cutoff * magnitude.max())
            & (np.max(abs(k[:, :2].real / transport.k), axis=-1) <= self.angular_limit)
            & transport.active.ravel()
        )
        omitted = float(
            np.linalg.norm(magnitude[~keep]) / max(np.linalg.norm(magnitude), 1e-300)
        )
        retained = PlaneSpectrum(
            spectrum.grid,
            replace(modes, amplitudes=np.where(keep[:, None], modes.amplitudes, 0)),
        )
        electric, derivative = transport.electric_jet(retained)
        intensity = np.sum(abs(electric) ** 2, axis=-1)
        lit = intensity > self.cutoff**2 * intensity.max()
        wavevector = np.divide(
            np.imag(np.sum(electric.conj()[..., :, None] * derivative, axis=-2)),
            intensity[..., None],
            out=np.zeros_like(electric.real),
            where=lit[..., None],
        )
        length = np.linalg.norm(wavevector, axis=-1)
        if np.any(lit & (length <= 1e-12 * transport.k)):
            raise ValueError(
                "local phase gradient is undefined in an illuminated region; use spectral transmission"
            )
        direction = np.divide(
            wavevector,
            length[..., None],
            out=np.zeros_like(wavevector),
            where=lit[..., None],
        )
        direction[~lit] = transport.plane.frame.normal
        weights = intensity * transport.grid.weights * lit
        total = max(float(weights.sum()), 1e-300)
        eikonal = float(
            np.sqrt(np.sum(weights * (length / transport.k - 1) ** 2) / total)
        )
        longitudinal = np.sum(direction * electric, axis=-1)
        projection = float(
            np.sqrt(
                np.sum(abs(longitudinal) ** 2 * transport.grid.weights * lit) / total
            )
        )
        if max(eikonal, projection) > self.validity_tolerance:
            raise ValueError(
                f"local single-direction approximation failed: eikonal residual={eikonal:.3g}, "
                f"longitudinal amplitude={projection:.3g}; use spectral transmission"
            )
        cosine = np.sum(direction * transport.grid.normals, axis=-1)
        if np.any(lit & (cosine <= 1e-10)):
            raise ValueError(
                "local transmission requires forward, non-grazing incidence on the surface; "
                f"unsupported relative amplitude={np.sqrt(weights[cosine <= 1e-10].sum() / total):.3g}"
            )
        n1, n2 = interface.minus.material.n, interface.plus.material.n
        s, pi, pt, dt, ts, tp = _fresnel_basis(
            direction, transport.grid.normals, n1, n2
        )
        if self.coefficients == "unity":
            ts, tp = np.ones_like(ts), np.ones_like(tp)
        transmitted = (ts * np.sum(s * electric, axis=-1))[..., None] * s + (
            tp * np.sum(pi * electric, axis=-1)
        )[..., None] * pt
        transmitted[~lit] = 0
        field = ElectromagneticField(
            ElectricField(transport.grid, spectrum.expansion.wavelength0, transmitted),
            n2 / Z0 * np.cross(dt, transmitted),
        )
        report = LocalTransmissionReport(
            int(keep.sum()),
            omitted,
            (transport.rank,),
            eikonal,
            projection,
            int(np.count_nonzero(~lit)),
            float(
                np.sqrt(
                    np.sum(intensity[~lit] * transport.grid.weights[~lit])
                    / max(np.sum(intensity * transport.grid.weights), 1e-300)
                )
            ),
        )
        return Transmission(field, report)
