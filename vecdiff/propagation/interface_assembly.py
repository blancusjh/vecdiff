"""Coherent repeated per-k Fresnel encounters in an ordered interface assembly.

This implementation uses a fixed, explicit Cartesian transverse spectral lattice
and global Cartesian vector amplitudes. It supports separated z-graph surfaces;
all retained modes must propagate in every region. Evanescent gaps belong to
propagate_layers. Curved encounters retain the local physical-optics model.
Convergence of feedback is not convergence of the dielectric boundary problem.
"""
from dataclasses import dataclass
import numpy as np
from ..fields.electric_spectrum import ElectricSpectrum
from ..interfaces.dielectric_interface import DielectricInterface
from ..surfaces.surface import Plane
from .interface_transform import interface_transform
from .multiple_scattering import coherent_feedback


@dataclass(frozen=True)
class AssemblyElectricField:
    assembly: object
    forward: tuple
    backward: tuple
    feedback: object
    surface_envelopes: tuple

    def evaluate(self, points, *, region, backend="direct"):
        """Total E,H in a specified region, outside the adjacent source envelopes.

The surface spectra have a half-space validity condition. A label cannot make
their continuation through a curved source envelope a valid interior field.
"""
        if not isinstance(region, int) or not 0 <= region < len(self.forward):
            raise ValueError("invalid region")
        q = np.asarray(points, float)
        if q.shape[-1:] != (3,) or not np.isfinite(q).all():
            raise ValueError("points must be finite (...,3)")
        lower = self.surface_envelopes[region-1][1] if region else -np.inf
        upper = self.surface_envelopes[region][0] if region < len(self.surface_envelopes) else np.inf
        if np.any(q[..., 2] < lower) or np.any(q[..., 2] > upper):
            raise ValueError("observations must lie in the region's source-free spectral slab")
        ef, hf = self.forward[region].evaluate(q, backend=backend)
        eb, hb = self.backward[region].evaluate(q, backend=backend)
        return ef+eb, hf+hb


def propagate_interfaces(incident, assembly, grid, *, samplings=None,
                         backend="direct", rtol=1e-10, max_iterations=1000,
                         method="gmres"):
    """Construct and solve the coherent interaction map of every interface.

No user-supplied round-trip closure is needed. The unknowns are the Ex/Ey
spectral amplitudes of both propagation branches in each internal region;
Ez is reconstructed by k.E=0 at every application. Phases use global positions,
so propagation between interfaces is already included in exp(i k.Q).

The lattice is an explicit bandwidth/window approximation for curved surfaces.
It must contain only non-grazing propagating modes in ALL media. Refine its
bandwidth, period, surface quadrature, and feedback tolerance independently.
Curved reverse encounters may reject back-facing modes rather than inventing
shadow coupling. Closed spheres and overlapping surface envelopes are excluded.
    """
    interfaces = assembly.interfaces
    count = len(interfaces)
    if incident.medium != assembly.media[0]:
        raise ValueError("incident medium does not match assembly")
    samplings = (None,)*count if samplings is None else tuple(samplings)
    if len(samplings) != count:
        raise ValueError("one quadrature (or None for an infinite plane) per interface")
    if backend not in ("direct", "nufft"):
        raise ValueError("backend must be direct or nufft")
    envelopes = []
    for interface, sampling in zip(interfaces, samplings):
        if getattr(interface.surface, 'is_closed', False):
            raise ValueError("closed bodies require a different spectral domain decomposition")
        if sampling is None:
            if not isinstance(interface.surface, Plane):
                raise ValueError("curved surfaces need explicit quadrature")
            normal = interface.normal_sign*interface.surface.frame.rotation[:, 2]
            if not np.allclose(normal, [0, 0, 1], atol=1e-12, rtol=0):
                raise ValueError("infinite planes must be oriented along +z")
            z = float(interface.surface.frame.origin[2]); envelopes.append((z, z))
        else:
            if sampling.surface is not interface.surface:
                raise ValueError("quadrature belongs to a different surface")
            if np.any(interface.normal_sign*sampling.normals[:, 2] <= 0):
                raise ValueError("assembly requires forward-oriented z-graph surfaces")
            envelopes.append((float(min(sampling.points[:, 2])), float(max(sampling.points[:, 2]))))
    if any(a[1] >= b[0] for a, b in zip(envelopes, envelopes[1:])):
        raise ValueError("surface z envelopes must be separated and ordered")
    kx, ky = (x.ravel() for x in grid.kxy)
    kxy = np.stack((kx, ky), axis=-1)
    wavevectors = []
    for medium in assembly.media:
        k = medium.wavenumber(incident.wavelength)
        kz2 = k*k-kx*kx-ky*ky
        if np.any(kz2 <= 1e-12*k*k):
            raise ValueError("lattice includes grazing/evanescent modes; reduce bandwidth or use propagate_layers")
        wavevectors.append(np.column_stack((kx, ky, np.sqrt(kz2))))
    modes = len(kx)

    def spectrum(region, direction, xy):
        k = wavevectors[region].copy(); k[:, 2] *= direction
        a = np.column_stack((xy, -np.sum(k[:, :2]*xy, axis=-1)/k[:, 2]))
        return ElectricSpectrum(k, a, incident.wavelength, assembly.media[region])

    def project(field, region, direction):
        if not isinstance(field, ElectricSpectrum):
            field = field.spectrum(grid, direction=direction, backend=backend)
        if np.any(field.wavevectors.imag):
            raise ValueError("evanescent incidence/output is not supported in this assembly")
        # Preserve sparse input and exact plane transforms without interpolation.
        dkx = 2*np.pi/(len(grid.x)*grid.dx)
        dky = 2*np.pi/(len(grid.y)*grid.dy)
        ix = np.rint(field.wavevectors[:, 0].real/dkx).astype(int) % len(grid.x)
        iy = np.rint(field.wavevectors[:, 1].real/dky).astype(int) % len(grid.y)
        index = iy*len(grid.x)+ix
        delta = np.linalg.norm(field.wavevectors[:, :2].real-kxy[index], axis=-1)
        if np.any(delta > 1e-9*assembly.media[region].wavenumber(incident.wavelength)):
            raise ValueError("plane-wave transverse wavevectors must lie on the supplied lattice")
        if np.any(direction*field.wavevectors[:, 2].real <= 0):
            raise ValueError("spectrum has the wrong propagation branch")
        a = np.zeros((modes, 2), complex)
        np.add.at(a, index, field.amplitudes[:, :2])
        return a

    incident_xy = project(incident, 0, 1)
    incident = spectrum(0, 1, incident_xy)
    reverse = tuple(DielectricInterface(i.surface, i.transmitted_medium, i.incident_medium, -i.normal_sign)
                    for i in interfaces)

    def encounter(j, direction, xy):
        region = j if direction == 1 else j+1
        if not np.any(xy):
            zero = np.zeros_like(xy)
            return zero, zero.copy()
        field = spectrum(region, direction, xy)
        result = interface_transform(field, interfaces[j] if direction == 1 else reverse[j], samplings[j])
        return (project(result.reflected, region, -direction),
                project(result.transmitted, region+direction, direction))

    def scatter(state, illuminate):
        forward = [np.zeros((modes, 2), complex) for _ in range(count+1)]
        backward = [x.copy() for x in forward]
        for j in range(count):
            f = incident_xy if j == 0 and illuminate else (state[j-1, 0] if j else forward[0])
            b = state[j, 1] if j < count-1 else backward[-1]
            rf, tf = encounter(j, 1, f)
            rb, tb = encounter(j, -1, b)
            backward[j] = rf+tb
            forward[j+1] = tf+rb
        return forward, backward

    def internal(forward, backward):
        return np.stack([np.stack((forward[j], backward[j])) for j in range(1, count)])

    state = np.zeros((count-1, 2, modes, 2), complex)
    if count > 1:
        injection = internal(*scatter(state, True))
        feedback = coherent_feedback(injection, lambda x: internal(*scatter(x, False)),
                                     rtol=rtol, max_iterations=max_iterations, method=method)
        state = feedback.state
    else:
        feedback = None
    forward, backward = scatter(state, True)
    forward[0] = incident_xy
    return AssemblyElectricField(assembly,
        tuple(spectrum(j, 1, a) for j, a in enumerate(forward)),
        tuple(spectrum(j, -1, a) for j, a in enumerate(backward)), feedback, tuple(envelopes))
