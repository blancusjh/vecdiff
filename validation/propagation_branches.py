"""Independent saved-reference check of the local/spectral public operator.

Run outside notebooks. This is the R/lambda=100, n=1.5, radius-35-lambda
illumination benchmark, not general macroscopic validation. Both methods use
exactly the same source, optical system, grid, and detector. No phase/gain fit.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import vecdiff_general as o
from scipy.special import erfc


def run():
    root = Path(__file__).resolve().parents[1]
    directory = root / "validation/reference_data"
    with np.load(directory / "focal_pipeline.npz") as data:
        reference = {k: data[k].copy() for k in data.files}
    with np.load(directory / "focal_spectral_N280_a18_c1e-07_t3.npz") as data:
        previous = data["order0_plane_E"].copy()
    air = o.HomogeneousRegion("air", o.Dielectric(1))
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    axis = (np.arange(280) - 140) * 130 / 280
    grid = o.CartesianGrid(axis, axis, o.Frame([0, 0, -105]))
    rho = np.linalg.norm(grid.points[..., :2], axis=-1)
    values = np.stack((0.5 * erfc((rho - 35) / 1.5), np.zeros_like(rho)), axis=-1)
    incident = o.TransverseElectricField(grid, 1, values)
    system = o.OpticalSystem(
        [
            o.DielectricInterface(
                o.SphericalCap(100, o.Frame([0, 0, -100])), air, glass
            ),
            o.DielectricInterface(
                o.SphericalCap(-100, o.Frame([0, 0, 100])), glass, air
            ),
        ]
    )
    target = o.PointGrid(reference["plane"])
    rows = []
    arrays = {"points": target.points}

    def relative(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    for method in ("local", "spectral"):
        propagator = o.PropagationPipeline(system, transmission=method)
        tick = perf_counter()
        field = propagator(incident, target, tolerance=1e-12)
        seconds = perf_counter() - tick
        complete = propagator.last_solution.evaluate(target, tolerance=1e-12)
        assert relative(field.values, complete.electric.values) < 1e-10
        row = dict(
            method=method,
            operator_E_s=seconds,
            complex_E_Debye_relative=relative(
                field.values, reference["exact_plane_p1"]
            ),
            complex_H_Debye_relative=relative(
                complete.magnetic, reference["exact_plane_p1_H"]
            ),
            complex_E_Mie_relative=relative(
                field.values, reference["exact_plane_total"]
            ),
            interfaces=[vars(r.report) for r in propagator.last_solution.interactions],
        )
        if method == "spectral":
            row["complex_E_saved_spectral_relative"] = relative(field.values, previous)
            assert row["complex_E_saved_spectral_relative"] < 1e-10
        rows.append(row)
        arrays[method + "_E"] = field.values
        arrays[method + "_H"] = complete.magnetic
        report = dict(
            radius_over_wavelength=100,
            sphere_index=1.5,
            beam_radius_over_wavelength=35,
            detector="121x121 points, x/y in [-12,12] lambda, z=146.9541667 lambda",
            notes=[
                "Local estimates phase gradients internally; it is not the historical Poynting adapter.",
                "Timings include source completion, operator preparation, solving, and electric detector evaluation.",
                "Full-Mie comparison includes physical branches omitted by both candidates.",
            ],
            results=rows,
        )
        (directory / "propagation_branches.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        np.savez_compressed(directory / "propagation_branches.npz", **arrays)
        print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    run()
