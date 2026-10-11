"""Public pipeline versus the saved direction-resolved Fresnel focal benchmark.

This uses only the installed core, NumPy and SciPy. No notebooks, historical Git
code or reference solvers enter the candidate calculation. Comparisons occur
only after the candidate is complete.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import vecdiff_general as o
from scipy.special import erfc


def run(count=280):
    root = Path(__file__).resolve().parents[1]
    axis = (np.arange(count) - count // 2) * 130 / count
    air = o.HomogeneousRegion("air", o.Dielectric(1))
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    grid = o.CartesianGrid(axis, axis, o.Frame([0, 0, -105]))
    x, y = np.meshgrid(axis, axis)
    values = np.zeros(grid.shape + (2,), complex)
    values[..., 0] = 0.5 * erfc((np.hypot(x, y) - 35) / 1.5)
    started = perf_counter()
    source = o.angular_spectrum(
        o.TransverseElectricField(grid, 1, values), air, evanescent="discard"
    )
    path = o.PropagationPipeline(
        [
            o.DielectricInterface(
                o.SphericalCap(100, o.Frame([0, 0, -100])), air, glass
            ),
            o.DielectricInterface(
                o.SphericalCap(-100, o.Frame([0, 0, 100])), glass, air
            ),
        ]
    )
    solution = path.solve(source)
    solve_seconds = perf_counter() - started
    data = root / "validation/reference_data"
    with (
        np.load(data / "focal_pipeline.npz") as reference,
        np.load(data / "focal_spectral_N280_a18_c1e-07_t3.npz") as previous,
    ):
        started = perf_counter()
        field = solution.evaluate(
            o.PointGrid(reference["plane"]), backend="nufft", tolerance=1e-12
        )
        evaluation_seconds = perf_counter() - started

        def relative(a, b):
            return float(np.linalg.norm(a - b) / np.linalg.norm(b))

        result = dict(
            count=count,
            solve_seconds=solve_seconds,
            evaluation_seconds=evaluation_seconds,
            relative_E_saved=relative(
                field.electric.values, previous["order0_plane_E"]
            ),
            relative_H_saved=relative(field.magnetic, previous["order0_plane_H"]),
            relative_E_p1=relative(field.electric.values, reference["exact_plane_p1"]),
            relative_H_p1=relative(field.magnetic, reference["exact_plane_p1_H"]),
            interfaces=[
                dict(
                    retained_modes=x.report.retained_modes,
                    omitted_modal_l2=x.report.omitted_modal_l2,
                    transport_ranks=x.report.transport_ranks,
                )
                for x in solution.interactions
            ],
        )
    assert result["relative_E_saved"] < 1e-10, result
    assert result["relative_H_saved"] < 1e-10, result
    assert result["relative_E_p1"] < 6.3e-5, result
    (data / "pipeline_regression.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    run()
