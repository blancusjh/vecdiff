"""Pure transfer audit: exact Mie traces, no Fresnel T or imposed aperture.

The graph implementation has finite numerical coverage. A complete
closed-surface reference and a Weyl propagating/evanescent split independently
separate that limitation from the mathematical field continuation formula.
Run with --output validation/reference_data/pure_surface_transport.json.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from vecdiff_general.comparisons.surface_radiation import SurfaceRadiation
from vecdiff_general.fields import ElectricField, ElectromagneticField
from vecdiff_general.fields.spectrum import PlaneWaveExpansion
from vecdiff_general.geometry import Frame
from vecdiff_general.media.medium import Z0
from vecdiff_general.propagation.references.boundary import (
    boundary_meridian,
    closed_box_grid,
)
from vecdiff_general.propagation.references.mie import MieSphere
from vecdiff_general.propagation.references.surface_transport import (
    curved_surface,
    problem,
    relative_fields,
    scattered,
)
from vecdiff_general.propagation.surface_transfer import SurfaceTransport
from vecdiff_general.sampling import CartesianGrid, PointGrid
from vecdiff_general.volumes import SphereVolume


def setup():
    source, sphere, glass = problem()
    mie = MieSphere(source, sphere, glass)
    # Include off-grid points to expose interpolation/atom representation error.
    axis = np.linspace(-1, 1, 9)
    x, y = np.meshgrid(axis + 0.071, axis - 0.033)
    xy = np.stack((x, y), axis=-1).reshape(-1, 2)
    probes = PointGrid(np.column_stack((xy, np.full(len(xy), 4.3))))
    return mie, curved_surface(2, 0.4, 24), probes


def errors(actual, exact, incident):
    """Same error numerator with both total-field and scattered-field scales."""
    residual = ElectromagneticField(
        ElectricField(exact.grid, 1, actual.electric.values - incident.electric.values),
        actual.magnetic - incident.magnetic,
    )
    target = ElectromagneticField(
        ElectricField(exact.grid, 1, exact.electric.values - incident.electric.values),
        exact.magnetic - incident.magnetic,
    )
    return {
        "total": relative_fields(actual, exact),
        "scattered_scale": relative_fields(residual, target),
    }


def global_case(*, length=24, count=96, max_sine=0.99):
    mie, surface, probes = setup()
    axis = (np.arange(count) - count // 2) * length / count
    plane = CartesianGrid(axis, axis, Frame([0, 0, 2]))
    q = SurfaceTransport(plane, surface, mie.exterior, 1, max_sine=max_sine)
    # Exact TOTAL Mie E and H, unchanged. No use of T, taper, or fitted reference.
    trace = mie.evaluate(q.grid)
    inverse, report = q.inverse(trace, tolerance=1e-10, max_iterations=400)
    radiated = q.radiate(trace)
    exact = mie.evaluate(probes)
    incident = mie.source.evaluate(probes)
    return {
        "implementation": "SurfaceTransport",
        "length": length,
        "count": count,
        "max_sine": max_sine,
        "input": "exact total Mie E/H samples; no taper",
        "inverse": errors(inverse.evaluate(probes), exact, incident),
        "radiate": errors(radiated.evaluate(probes), exact, incident),
        "trace_report": asdict(report),
    }


def evanescent_spectrum(radiation, *, order=48, azimuths=96, cutoff=6):
    """Independent Weyl reference for kz=i*k*t, 0<t<cutoff.

    With k_perp=k*sqrt(1+t^2), the Weyl measure is i*k^2/(8*pi^2) dt dphi
    multiplying [Z (I-khat khat^T) J~ - khat x M~]. No conjugation enters
    the transverse projector. This is reference quadrature, not the finite-chart Q API.
    The origin at max(source z) prevents growing source-side exponentials.
    """
    k = 2 * np.pi * radiation.region.material.n / radiation.wavelength0
    impedance = Z0 / radiation.region.material.n
    nodes, weights = leggauss(order)
    t, weights = (nodes + 1) * cutoff / 2, weights * cutoff / 2
    t, phi = np.meshgrid(t, np.arange(azimuths) * 2 * np.pi / azimuths, indexing="ij")
    directions = np.stack(
        (np.sqrt(1 + t * t) * np.cos(phi), np.sqrt(1 + t * t) * np.sin(phi), 1j * t), -1
    ).reshape(-1, 3)
    waves = k * directions
    source = radiation.grid.points.reshape(-1, 3)
    origin = np.array([0, 0, source[:, 2].max()])
    currents = np.concatenate(
        (impedance * radiation.electric_current, radiation.magnetic_current), -1
    ).reshape(-1, 6)
    currents *= radiation.grid.weights.reshape(-1, 1)
    moments = np.concatenate(
        [
            np.exp(-1j * waves[start : start + 128] @ (source - origin).T) @ currents
            for start in range(0, len(waves), 128)
        ]
    )
    j, m = moments[:, :3], moments[:, 3:]
    bracket = (
        j - directions * np.sum(directions * j, -1)[:, None] - np.cross(directions, m)
    )
    measure = np.broadcast_to(weights[:, None] * 2 * np.pi / azimuths, t.shape).ravel()
    amplitudes = 1j * k * k / (8 * np.pi**2) * measure[:, None] * bracket
    amplitudes[:, 2] = (
        -(waves[:, 0] * amplitudes[:, 0] + waves[:, 1] * amplitudes[:, 1]) / waves[:, 2]
    )
    return PlaneWaveExpansion(
        waves, amplitudes, origin, radiation.wavelength0, radiation.region
    )


def complete_reference_case(*, surface_order=24, spectral_order=48, cutoff=6):
    mie, _, probes = setup()
    boundary = boundary_meridian(SphereVolume([0, 0, 0], 0.6), surface_order).surface(
        2 * surface_order
    )
    evaluate = scattered(mie)
    # Exact full-sphere trace of one Maxwell solution, without any window.
    radiation = SurfaceRadiation.from_field(evaluate(boundary), mie.exterior)
    exact = evaluate(probes)
    full = radiation.evaluate(probes)
    propagating = radiation.angular_spectrum(
        polar_order=spectral_order, azimuthal_order=2 * spectral_order
    ).evaluate(probes)
    evanescent = evanescent_spectrum(
        radiation, order=spectral_order, azimuths=2 * spectral_order, cutoff=cutoff
    ).evaluate(probes)
    combined = ElectromagneticField(
        ElectricField(
            probes, 1, propagating.electric.values + evanescent.electric.values
        ),
        propagating.magnetic + evanescent.magnetic,
    )
    return {
        "surface_order": surface_order,
        "spectral_order": spectral_order,
        "evanescent_cutoff": cutoff,
        "source_radius": 0.6,
        "full_surface_integral": relative_fields(full, exact),
        "propagating_only": relative_fields(propagating, exact),
        "propagating_plus_evanescent": relative_fields(combined, exact),
    }


def total_field_box_case(order=80):
    """Exact TOTAL Mie field transferred inward into a source-free exterior box."""
    mie, _, probes = setup()
    boundary = closed_box_grid([0, 0, 3.3], [2, 2, 1.3], order=order)
    radiation = SurfaceRadiation.from_field(
        mie.evaluate(boundary), mie.exterior, normal_sign=-1
    )
    return {
        "order": order,
        "error": relative_fields(radiation.evaluate(probes), mie.evaluate(probes)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "definition": "exact Mie traces -> transfer only -> independent exact Mie observations",
        "geometry": "sphere radius 0.3, n=1.5, wavelength=1; h1=2+0.4 cos(2pi x/24)cos(2pi y/24); target z=4.3",
        "observations": "9x9 points: x in [-1,1]+0.071, y in [-1,1]-0.033",
        "global": [],
        "complete_reference": [],
        "total_field_box": [],
    }
    for band in (0.85, 0.99, 0.999):
        row = global_case(max_sine=band)
        result["global"].append(row)
        print("global", band, row["radiate"], flush=True)
    for order, cutoff in ((16, 6), (32, 6), (48, 6), (64, 8)):
        row = complete_reference_case(spectral_order=order, cutoff=cutoff)
        result["complete_reference"].append(row)
        print("complete", row, flush=True)
    for order in (32, 64, 96):
        row = total_field_box_case(order)
        result["total_field_box"].append(row)
        print("total field box", row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
