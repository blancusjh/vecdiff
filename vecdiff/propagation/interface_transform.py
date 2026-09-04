"""Linear spectral transformation at an interface, followed by propagation.

The infinite-plane path is an exact Fresnel transformation. Curved surfaces
use local tangent-plane Fresnel data for EVERY incident k. Their subsequent
Maxwell radiation is exact for those supplied currents, but the boundary data
are a physical-optics approximation, not a globally solved boundary trace.
"""
from dataclasses import dataclass
import numpy as np
from ..interfaces.fresnel import solve
from ..fields.electric_spectrum import ElectricSpectrum
from ..surfaces.surface import Plane
from .surface_radiation import SurfaceRadiation


@dataclass(frozen=True)
class InterfaceFields:
    reflected: object
    transmitted: object
    boundary: object = None


@dataclass(frozen=True)
class BoundaryData:
    incident_E: np.ndarray
    incident_H: np.ndarray
    reflected_E: np.ndarray
    reflected_H: np.ndarray
    transmitted_E: np.ndarray
    transmitted_H: np.ndarray


def transform_plane(spectrum, interface):
    if not isinstance(interface.surface, Plane): raise TypeError("exact plane transform requires Plane")
    if spectrum.medium != interface.incident_medium: raise ValueError("incident medium does not match spectrum")
    frame = interface.surface.frame
    result = solve(spectrum.wavevectors, spectrum.amplitudes, interface.normal_sign*frame.rotation[:, 2],
                   interface.incident_medium, interface.transmitted_medium, spectrum.wavelength)
    def branch(k, e, medium):
        phase = np.exp(1j*np.sum((spectrum.wavevectors-k)*frame.origin, axis=-1))
        return ElectricSpectrum(k, e*phase[:, None], spectrum.wavelength, medium)
    return InterfaceFields(branch(result.reflected_k, result.reflected_E, interface.incident_medium),
                           branch(result.transmitted_k, result.transmitted_E, interface.transmitted_medium))


def boundary_data(spectrum, interface, sampling, *, illuminated_only=False):
    """Accumulate per-mode boundary phasors before a single return transform.

illuminated_only=True explicitly sets transmitted/reflected traces to zero
on each mode's shadow side. It is a single-encounter approximation; it does
not trace exit faces, internal reflections, or diffraction into shadow.
    """
    if sampling.surface is not interface.surface: raise ValueError("sampling belongs to a different surface")
    if spectrum.medium != interface.incident_medium: raise ValueError("incident medium does not match spectrum")
    if np.any(spectrum.wavevectors.imag): raise ValueError("curved-interface incident spectrum must be propagating")
    n = interface.normal_sign*sampling.normals
    values = [np.zeros_like(sampling.points, complex) for _ in range(6)]
    ei, hi, er, hr, et, ht = values
    k0 = 2*np.pi/spectrum.wavelength
    for k, amplitude, h in zip(spectrum.wavevectors, spectrum.amplitudes, spectrum.magnetic_amplitudes):
        if not np.any(amplitude): continue
        phase = np.exp(1j*sampling.points @ k)[:, None]
        ei += phase*amplitude; hi += phase*h
        lit = n @ k.real > 1e-12*np.linalg.norm(k)
        if not illuminated_only and not np.all(lit):
            raise ValueError("surface includes back-facing/grazing samples; set illuminated_only explicitly for single-encounter approximation")
        if not np.any(lit): continue
        f = solve(k, amplitude, n[lit], interface.incident_medium, interface.transmitted_medium, spectrum.wavelength)
        er[lit] += phase[lit]*f.reflected_E; et[lit] += phase[lit]*f.transmitted_E
        hr[lit] += phase[lit]*np.cross(f.reflected_k, f.reflected_E)/k0
        ht[lit] += phase[lit]*np.cross(f.transmitted_k, f.transmitted_E)/k0
    return BoundaryData(*values)


def interface_transform(spectrum, interface, sampling=None, *, illuminated_only=False):
    if isinstance(interface.surface, Plane) and sampling is None:
        return transform_plane(spectrum, interface)
    if sampling is None: raise ValueError("a curved or finite interface needs explicit surface quadrature")
    b = boundary_data(spectrum, interface, sampling, illuminated_only=illuminated_only)
    reflected = SurfaceRadiation.from_boundary(sampling, b.reflected_E, b.reflected_H,
                    spectrum.wavelength, interface.incident_medium, normal_sign=-interface.normal_sign)
    transmitted = SurfaceRadiation.from_boundary(sampling, b.transmitted_E, b.transmitted_H,
                    spectrum.wavelength, interface.transmitted_medium, normal_sign=interface.normal_sign)
    return InterfaceFields(reflected, transmitted, b)
