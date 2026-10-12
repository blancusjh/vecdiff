"""Accuracy and wall-time audit of complete transfer on non-spherical surfaces.

Exact displaced electric dipoles provide a solution independent of any sphere
or transport kernel. Both direct dyadic radiation and the full Weyl spectrum
(propagating + evanescent) receive identical E/H traces. These are reference
implementations, not the restricted SurfaceTransport or phase-space APIs.
"""

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss

from vecdiff_general.comparisons.surface_radiation import SurfaceRadiation
from vecdiff_general.fields import ElectricField, ElectromagneticField
from vecdiff_general.media import Dielectric, HomogeneousRegion
from vecdiff_general.media.medium import Z0
from vecdiff_general.propagation.references.boundary import closed_box_grid
from vecdiff_general.propagation.references.pure_surface_transport import (
    evanescent_spectrum,
)
from vecdiff_general.propagation.references.surface_transport import relative_fields
from vecdiff_general.sampling import PointGrid, SurfaceGrid

SHAPES = ("ellipsoid", "box", "cylinder", "rippled")
AIR = HomogeneousRegion("air", Dielectric(1))


def boundary(shape, order):
    """Outward surface quadrature; lengths in vacuum wavelengths (lambda0=1)."""
    u, w = leggauss(order)
    phi = np.arange(2 * order) * np.pi / order
    u, phi = np.meshgrid(u, phi, indexing="ij")
    measure = np.broadcast_to(w[:, None] * np.pi / order, u.shape)
    s = np.sqrt(1 - u * u)
    direction = np.stack((s * np.cos(phi), s * np.sin(phi), u), -1)
    if shape == "ellipsoid":
        c, s = np.cos(0.41), np.sin(0.41)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        axes = np.array([0.65, 0.45, 0.55])
        points = (direction * axes) @ rotation.T + [0.03, -0.02, 0.04]
        area_vector = (np.prod(axes) * direction / axes) @ rotation.T
    elif shape == "rippled":
        x, y, z = np.moveaxis(direction, -1, 0)
        radius = 0.6 * (1 + 0.16 * (x**3 - 3 * x * y * y) + 0.1 * x * y * z)
        gradient = 0.6 * np.stack(
            (
                0.48 * (x * x - y * y) + 0.1 * y * z,
                -0.96 * x * y + 0.1 * x * z,
                0.1 * x * y,
            ),
            -1,
        )
        tangential = gradient - direction * np.sum(direction * gradient, -1)[..., None]
        points = radius[..., None] * direction
        area_vector = (
            radius[..., None] ** 2 * direction - radius[..., None] * tangential
        )
    elif shape == "box":
        return closed_box_grid([0, 0, 0], [0.6, 0.5, 0.55], order=order)
    elif shape == "cylinder":
        radius, half_height = 0.55, 0.6
        cp, sp = np.cos(phi), np.sin(phi)
        side = np.stack((radius * cp, radius * sp, half_height * u), -1)
        normal = np.stack((cp, sp, np.zeros_like(u)), -1)
        r = radius * (u + 1) / 2
        points, normals, weights = [side], [normal], [radius * half_height * measure]
        for sign in (-1, 1):
            points.append(
                np.stack((r * cp, r * sp, np.full_like(r, sign * half_height)), -1)
            )
            normals.append(np.broadcast_to([0, 0, sign], side.shape))
            weights.append(r * radius / 2 * measure)
        return SurfaceGrid(
            np.concatenate(points), np.concatenate(normals), np.concatenate(weights)
        )
    else:
        raise ValueError(f"unknown shape: {shape}")
    jacobian = np.linalg.norm(area_vector, axis=-1)
    return SurfaceGrid(points, area_vector / jacobian[..., None], measure * jacobian)


def exact_field(grid):
    """Coherent fields of two displaced, differently polarized Hertz dipoles."""
    electric = np.zeros_like(grid.points, complex)
    magnetic = np.zeros_like(electric)
    k = 2 * np.pi
    sources = [
        ([0.08, -0.06, 0.03], [1, 0.3j, 0.2]),
        ([-0.09, 0.04, -0.07], [-0.2j, 0.4, 0.3]),
    ]
    for location, moment in sources:
        displacement = grid.points - location
        radius = np.linalg.norm(displacement, axis=-1)[..., None]
        direction = displacement / radius
        moment = np.asarray(moment) / k**2
        radial = direction * (direction @ moment)[..., None]
        phase = np.exp(1j * k * radius)
        electric += phase * (
            k**2 / radius * (moment - radial)
            + (1 / radius**3 - 1j * k / radius**2) * (3 * radial - moment)
        )
        magnetic += (
            phase
            * k**2
            / (Z0 * radius)
            * (1 + 1j / (k * radius))
            * np.cross(direction, moment)
        )
    return ElectromagneticField(ElectricField(grid, 1, electric), magnetic)


def receiver(count=9):
    """An asymmetric curved destination entirely above every source surface."""
    x, y = np.meshgrid(
        np.linspace(-0.4, 0.4, count) + 0.027, np.linspace(-0.4, 0.4, count) - 0.019
    )
    return PointGrid(np.stack((x, y, 1.2 + 0.2 * x * x - 0.12 * x * y + 0.1 * y), -1))


def timed(function, repeats):
    """Wall seconds: one untimed warmup, then independent repetitions."""
    function()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        value = function()
        samples.append(perf_counter() - start)
    return value, {
        "median": float(np.median(samples)),
        "min": min(samples),
        "max": max(samples),
        "samples": samples,
    }


def add_fields(first, second):
    return ElectromagneticField(
        ElectricField(first.grid, 1, first.electric.values + second.electric.values),
        first.magnetic + second.magnetic,
    )


def case(
    shape,
    *,
    order=32,
    spectral_order=48,
    cutoff=8,
    count=9,
    repeats=3,
    evaluate=exact_field,
    reference="two analytic dipoles",
):
    grid, geometry_time = timed(lambda: boundary(shape, order), repeats)
    probes = receiver(count)
    trace, trace_time = timed(lambda: evaluate(grid), repeats)
    truth, truth_time = timed(lambda: evaluate(probes), repeats)
    radiation, current_time = timed(
        lambda: SurfaceRadiation.from_field(trace, AIR), repeats
    )
    direct, direct_time = timed(lambda: radiation.evaluate(probes), repeats)
    propagating, prop_time = timed(
        lambda: radiation.angular_spectrum(
            polar_order=spectral_order, azimuthal_order=2 * spectral_order
        ),
        repeats,
    )
    evanescent, ev_time = timed(
        lambda: evanescent_spectrum(
            radiation, order=spectral_order, azimuths=2 * spectral_order, cutoff=cutoff
        ),
        repeats,
    )
    spectrum_field, eval_time = timed(
        lambda: add_fields(propagating.evaluate(probes), evanescent.evaluate(probes)),
        repeats,
    )

    # Time the complete spectral path as well; a sum of separate medians is not
    # itself an end-to-end timing measurement.
    def spectral_transfer():
        p = radiation.angular_spectrum(
            polar_order=spectral_order, azimuthal_order=2 * spectral_order
        )
        e = evanescent_spectrum(
            radiation, order=spectral_order, azimuths=2 * spectral_order, cutoff=cutoff
        )
        return add_fields(p.evaluate(probes), e.evaluate(probes))

    _, spectral_time = timed(spectral_transfer, repeats)
    return {
        "shape": shape,
        "reference": reference,
        "surface_order": order,
        "spectral_order": spectral_order,
        "evanescent_cutoff": cutoff,
        "source_nodes": int(np.prod(grid.shape)),
        "target_nodes": int(np.prod(probes.shape)),
        "spectral_modes": 4 * spectral_order**2,
        "direct_error": relative_fields(direct, truth),
        "weyl_error": relative_fields(spectrum_field, truth),
        "weyl_vs_direct": relative_fields(spectrum_field, direct),
        "seconds": {
            "geometry": geometry_time,
            "source_field": trace_time,
            "target_reference": truth_time,
            "currents": current_time,
            "direct_transfer": direct_time,
            "propagating_setup": prop_time,
            "evanescent_setup": ev_time,
            "spectral_evaluation": eval_time,
            "spectral_transfer": spectral_time,
        },
    }


def bem_case(repeats=3):
    """Transfer a non-spherical BEM solution; no second scattering solve in transport."""
    from vecdiff_general.propagation.references.bem import DielectricBEM
    from vecdiff_general.propagation.references.mesh import SurfaceMesh
    from vecdiff_general.propagation.references.surface_transport import (
        problem,
        scattered,
    )

    source, sphere, glass = problem()
    mesh = SurfaceMesh.sphere(sphere, 2)
    mesh = SurfaceMesh(mesh.vertices * [1, 0.22 / 0.3, 0.18 / 0.3], mesh.triangles)
    start = perf_counter()
    solution = DielectricBEM(source, mesh, glass, field_far_quadrature=8)
    setup_seconds = perf_counter() - start
    row = case(
        "box",
        order=24,
        repeats=repeats,
        evaluate=scattered(solution),
        reference="PMCHWT dielectric ellipsoid; comparison to the same BEM solution",
    )
    row["bem"] = {
        "semiaxes": [0.3, 0.22, 0.18],
        "index": 1.5,
        "unknowns": solution.unknowns,
        "triangles": len(mesh.triangles),
        "setup_seconds_single_run": setup_seconds,
        "cached_solve": solution.loaded_from is not None,
        "algebraic_residual": solution.algebraic_residual,
        "scope": "transport consistency; not an analytic error bound for the BEM scattering solution",
    }
    return row


def additional_checks(repeats=3):
    """Vary spectral order separately from surface order; then receiver count."""
    result = {"spectral_convergence": [], "receiver_scaling": []}
    for spectral_order in (24, 32, 64):
        row = case("rippled", order=40, spectral_order=spectral_order, repeats=repeats)
        result["spectral_convergence"].append(row)
        print("spectral order", spectral_order, row["weyl_error"], flush=True)
    for count in (21, 41):
        row = case("rippled", order=24, count=count, repeats=repeats)
        result["receiver_scaling"].append(row)
        print(
            "receiver nodes",
            count**2,
            "direct/spectral seconds",
            row["seconds"]["direct_transfer"]["median"],
            row["seconds"]["spectral_transfer"]["median"],
            flush=True,
        )
    result["bem"] = bem_case(repeats)
    return result


def plot_results(data, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for shape in SHAPES:
        rows = [r for r in data["cases"] if r["shape"] == shape]
        left.semilogy(
            [r["surface_order"] for r in rows],
            [max(r["weyl_error"].values()) for r in rows],
            "o-",
            label=shape,
        )
    left.set(
        xlabel="Surface quadrature order",
        ylabel="Relative complex E/H error",
        title="Full spectral transport: convergence",
    )
    left.grid(alpha=0.25)
    left.legend()
    rows = [r for r in data["cases"] if r["surface_order"] == 40]
    x = np.arange(len(rows))
    right.bar(
        x - 0.18,
        [r["seconds"]["direct_transfer"]["median"] for r in rows],
        width=0.36,
        label="Direct surface integral",
    )
    right.bar(
        x + 0.18,
        [r["seconds"]["spectral_transfer"]["median"] for r in rows],
        width=0.36,
        label="Full spectrum: setup + evaluation",
    )
    right.set(
        xticks=x,
        xticklabels=[r["shape"] for r in rows],
        ylabel="Wall time (s)",
        title="81 receiver points; surface order 40",
        ylim=(0, 1.4 * max(r["seconds"]["spectral_transfer"]["median"] for r in rows)),
    )
    right.legend(fontsize=8)
    fig.suptitle(
        f"Complete field transfer — {data['hardware']['cpu']}; median of {data['repeats']} runs"
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    data = {
        "hardware": {
            "cpu": cpu,
            "logical_cpus": os.cpu_count(),
            "platform": platform.platform(),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "threads": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "timing": "perf_counter wall seconds; one warmup then repeated samples; arrays in memory; no file I/O in timed regions",
        "wavelength0": 1,
        "repeats": args.repeats,
        "cases": [],
    }
    for shape in SHAPES:
        for order in (12, 24, 40):
            row = case(shape, order=order, repeats=args.repeats)
            data["cases"].append(row)
            print(
                shape,
                order,
                "errors",
                row["direct_error"],
                row["weyl_error"],
                "seconds",
                row["seconds"]["direct_transfer"]["median"],
                row["seconds"]["spectral_transfer"]["median"],
                flush=True,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(data, indent=2) + "\n")
    data.update(additional_checks(args.repeats))
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    plot_results(data, args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
