"""Record small reproductions supporting the general architecture review.

Run with the project interpreter. These observations document current gaps;
they are not regression tests requiring those gaps to remain unfixed.
"""

import json
from pathlib import Path

import numpy as np
import vecdiff_general as optics
from vecdiff_general.comparisons import SurfaceRadiation

ROOT = Path(__file__).resolve().parents[1]


def shared_material_boundary():
    faces = [optics.Plane(optics.Frame(origin=[0, 0, z])) for z in (0, 1, 2)]
    regions = (
        optics.MaterialRegion(
            "first",
            optics.VolumeBetweenSurfaces(faces[0], faces[1], 1),
            optics.Dielectric(1.5),
        ),
        optics.MaterialRegion(
            "second",
            optics.VolumeBetweenSurfaces(faces[1], faces[2], 1),
            optics.Dielectric(1.7),
        ),
    )
    matter = optics.MaterialDistribution(optics.Dielectric(1), regions)

    try:
        value = matter.refractive_index([0, 0, 1])
    except (TypeError, ValueError) as error:
        return {"shared_face_error": str(error)}

    return {"shared_face_index": float(value)}


def aperture_and_surface_chart():
    air = optics.HomogeneousRegion("air", optics.Dielectric(1))
    glass = optics.HomogeneousRegion("glass", optics.Dielectric(1.5))
    face = optics.Plane()
    interface = optics.DielectricInterface(face, air, glass)
    quadrature = optics.sample_surface(face, 5e-6, radial_order=8, azimuthal_order=16)
    source = optics.PlaneWaveExpansion(
        np.array([[0, 0, 2 * np.pi / 532e-9]]),
        np.array([[1, 0, 0]]),
        np.zeros(3),
        532e-9,
        air,
    )
    radius = np.linalg.norm(quadrature.points[..., :2], axis=-1)
    report = {
        "quadrature_nodes_outside_smaller_aperture": int(np.sum(radius > 1e-6)),
        "quadrature_retains_chart": all(
            hasattr(quadrature, name)
            for name in ("coordinates", "coordinate_system", "array_order")
        ),
    }

    try:
        optics.propagate_interfaces(
            source,
            optics.OpticalSystem((interface,), (1e-6,)),
            quadrature,
        )
    except (TypeError, ValueError) as error:
        report["oversized_quadrature_rejected"] = str(error)
    else:
        report["oversized_quadrature_rejected"] = False

    return report


def nonsmooth_axis():
    cone = optics.AxisymmetricSurface(lambda radius: radius, np.ones_like)

    return {
        "normal_returned_at_nondifferentiable_apex": cone.normals(
            np.array([[0.0, 0.0]])
        ).tolist(),
    }


def typed_grid_surface_currents():
    grid = optics.PolarGrid(
        np.array([0.5e-6, 1e-6]),
        np.arange(8) * 2 * np.pi / 8,
    )
    currents = np.zeros(grid.shape + (3,), dtype=complex)
    region = optics.HomogeneousRegion("air", optics.Dielectric(1))

    try:
        SurfaceRadiation(grid, currents, currents, 532e-9, region)
    except TypeError as error:
        return {"polar_grid_error": str(error)}

    return {"polar_grid_accepted": True}


def fft_lattice_contract():
    grid = optics.CartesianGrid(np.arange(2) * 1e-6, np.arange(2) * 1e-6)
    region = optics.HomogeneousRegion("air", optics.Dielectric(1))
    expansion = optics.PlaneWaveExpansion(
        np.tile([0, 0, 2 * np.pi / 532e-9], (4, 1)),
        np.tile([1, 0, 0], (4, 1)),
        grid.points[1, 1],
        532e-9,
        region,
    )

    # Four identical physical modes are valid, but they are not an FFT lattice.
    try:
        spectrum = optics.PlaneSpectrum(grid, expansion)
    except (TypeError, ValueError) as error:
        return {"incompatible_fft_lattice_rejected": str(error)}

    direct = spectrum.evaluate(grid).electric.values
    propagated = spectrum.propagate(0).electric.values

    return {
        "zero_distance_E_disagreement_V_per_m": float(
            np.max(np.abs(direct - propagated))
        ),
    }


def main():
    report = {
        "shared_boundary": shared_material_boundary(),
        "aperture_and_chart": aperture_and_surface_chart(),
        "axis_regularity": nonsmooth_axis(),
        "surface_grid_contract": typed_grid_surface_currents(),
        "fft_lattice_contract": fft_lattice_contract(),
    }
    output = ROOT / "validation/results/general_architecture_review.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
