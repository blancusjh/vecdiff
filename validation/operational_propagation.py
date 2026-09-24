"""Reproducible macro-lens timing, sampling and angular-band experiment.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python validation/operational_propagation.py
The JSON is measured on this machine, without fitting gain or phase.
"""

import json
from time import perf_counter

import numpy as np
import vecdiff_general as optics


def experiment(count, max_sine=0.85):
    wavelength = 532e-9
    air = optics.HomogeneousRegion("air", optics.Dielectric(1))
    glass = optics.HomogeneousRegion("glass", optics.Dielectric(1.5))
    output = optics.HomogeneousRegion("output", optics.Dielectric(1))
    radius, aperture, thickness = 2e-3, 200e-6, 100e-6
    front = optics.Plane()
    back = optics.SphericalCap(-radius, optics.Frame([0, 0, thickness]))
    system = optics.OpticalSystem(
        (
            optics.DielectricInterface(front, air, glass),
            optics.DielectricInterface(back, glass, output),
        ),
        (aperture, aperture),
    )

    def envelope(points):
        r = np.linalg.norm(points[..., :2], axis=-1)
        taper = np.cos(np.pi / 2 * np.clip((r - 160e-6) / 40e-6, 0, 1)) ** 4
        return (np.exp(-((r / 150e-6) ** 2)) * taper)[..., None] * [1, 0, 0]

    source = optics.EikonalField(
        lambda p: p[..., 2], lambda p: np.array([0, 0, 1]), envelope, wavelength, air
    )
    axis = (np.arange(count) - count // 2) * 600e-6 / count
    grid = optics.CartesianGrid(axis, axis)
    start = perf_counter()
    solution = optics.propagate_interfaces(source, system, grid, max_sine=max_sine)
    elapsed = perf_counter() - start
    Q = solution.transports[-1][1]
    reconstructed = Q.forward(solution.transmitted[0])
    local = solution.encounters[0][-1].transmitted
    focus = thickness + radius / 0.5
    probes = optics.PointGrid([[0, 0, focus], [2e-6, 0, focus], [0, 2e-6, focus]])
    values = solution.evaluate(probes, backend="direct").electric.values
    report = solution.inverse_reports[0][-1]
    return dict(
        count=count,
        max_sine=max_sine,
        seconds=elapsed,
        rank=Q.rank,
        iterations=report.iterations,
        inverse_residual=report.relative_residual,
        magnetic_error=float(
            np.linalg.norm(reconstructed.magnetic - local.magnetic)
            / np.linalg.norm(local.magnetic)
        ),
        flux_error=float(abs(reconstructed.power() / local.power() - 1)),
    ), values


if __name__ == "__main__":
    results, fields = zip(
        *(experiment(n, band) for n, band in ((384, 0.85), (512, 0.85), (512, 0.2)))
    )
    output = dict(
        geometry_m=dict(
            radius=0.002, aperture=0.0002, thickness=0.0001, cell_width=0.0006
        ),
        wavelength_m=532e-9,
        runs=results,
        grid_change=float(
            np.linalg.norm(fields[0] - fields[1]) / np.linalg.norm(fields[1])
        ),
        angular_band_change=float(
            np.linalg.norm(fields[2] - fields[1]) / np.linalg.norm(fields[1])
        ),
    )
    print(json.dumps(output, indent=2))
    assert output["grid_change"] < 0.002
    assert output["angular_band_change"] < 0.002
    assert max(r["flux_error"] for r in results) < 0.001
