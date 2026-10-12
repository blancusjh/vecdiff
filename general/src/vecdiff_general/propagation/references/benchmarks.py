"""Shared geometries and illumination for independent reference solvers."""

import numpy as np

from vecdiff_general.fields.spectrum import PlaneWaveExpansion
from vecdiff_general.geometry import Frame
from vecdiff_general.media import Dielectric, HomogeneousRegion
from vecdiff_general.surfaces import Plane, SphericalCap
from vecdiff_general.system import DielectricInterface, OpticalSystem
from vecdiff_general.volumes import VolumeBetweenSurfaces


def illumination(wavelength0=532e-9, index=1.5):
    exterior = HomogeneousRegion("exterior", Dielectric(1.0))
    interior = HomogeneousRegion("interior", Dielectric(index))
    source = PlaneWaveExpansion(
        [[0, 0, 2 * np.pi / wavelength0]], [[1, 0, 0]], [0, 0, 0], wavelength0, exterior
    )
    return source, interior


def lens_case(
    diameter_wavelengths,
    *,
    shape="plano-convex",
    wavelength0=532e-9,
    index=1.5,
    weak=False,
):
    """Lens geometry; weak=True holds sag near 2 lambda for large lenses."""
    if diameter_wavelengths <= 0 or shape not in ("plano-convex", "biconvex"):
        raise ValueError("positive diameter and a supported lens shape required")
    source, interior = illumination(wavelength0, index)
    radius = diameter_wavelengths * wavelength0 / 2
    curvature = max(4 * radius, radius**2 / (4 * wavelength0)) if weak else 4 * radius
    thickness = 0.6 * radius
    front = Plane() if shape == "plano-convex" else SphericalCap(curvature)
    back = SphericalCap(-curvature, Frame([0, 0, thickness]))
    volume = VolumeBetweenSurfaces(front, back, radius)
    system = OpticalSystem(
        (
            DielectricInterface(front, source.region, interior),
            DielectricInterface(back, interior, source.region),
        ),
        (radius, radius),
    )
    return source, interior, volume, system
