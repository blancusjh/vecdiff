"""Windowed Q/Q-inverse stress tests, without a Fresnel interface model.

Run from the repository root::

    python -m vecdiff_general.propagation.references.surface_transport \
        --output validation/reference_data/surface_transport.json

Mie and BEM comparisons use scattered fields: a large incident plane wave must
not hide a poor reconstruction. All lengths below are in vacuum wavelengths
(wavelength0=1 metre). The finite chart is tapered only outside the probe patch;
the reference on the destination is the original, untapered Maxwell solution.
Consequently these errors include deliberate alteration of the source trace;
use ``pure_surface_transport`` for the corrected exact-trace transfer audit.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from vecdiff_general.fields import ElectricField, ElectromagneticField
from vecdiff_general.fields.spectrum import PlaneSpectrum, PlaneWaveExpansion
from vecdiff_general.fourier.cartesian import FFT2, frequency_grid
from vecdiff_general.geometry import Frame
from vecdiff_general.media import Dielectric, HomogeneousRegion
from vecdiff_general.media.medium import Z0
from vecdiff_general.propagation.references.bem import DielectricBEM
from vecdiff_general.propagation.references.mesh import SurfaceMesh
from vecdiff_general.propagation.references.mie import MieSphere
from vecdiff_general.propagation.surface_transfer import SurfaceTransport
from vecdiff_general.sampling import CartesianGrid, PointGrid
from vecdiff_general.surfaces import FreeformSurface, SphericalCap
from vecdiff_general.volumes import SphereVolume


def problem():
    air = HomogeneousRegion("air", Dielectric(1))
    glass = HomogeneousRegion("glass", Dielectric(1.5))
    source = PlaneWaveExpansion([[0, 0, 2 * np.pi]], [[1, 0, 0]], [0, 0, 0], 1, air)
    return source, SphereVolume([0, 0, 0], 0.3), glass


def scattered(solution):
    """Subtract the analytic incident field, including its magnetic field."""

    def evaluate(grid):
        total = solution.evaluate(grid, side="exterior")
        incident = solution.source.evaluate(grid)
        return ElectromagneticField(
            ElectricField(grid, 1, total.electric.values - incident.electric.values),
            total.magnetic - incident.magnetic,
        )

    return evaluate


def dipole(grid):
    """Exact electric Hertz dipole at the origin, exp(-i omega t).

    The moment is scaled so E = exp(ikr)[k^2(p-rhat(rhat.p))/r +
    (3 rhat(rhat.p)-p)(1/r^3-ik/r^2)]. H has SI impedance normalization.
    This direct formula uses no angular-spectrum or surface integral code.
    """
    r = np.linalg.norm(grid.points, axis=-1)[..., None]
    direction = grid.points / r
    p = np.array([1, 0.3j, 0])
    radial = direction * (direction @ p)[..., None]
    k = 2 * np.pi
    phase = np.exp(1j * k * r)
    electric = phase * (
        k**2 / r * (p - radial) + (1 / r**3 - 1j * k / r**2) * (3 * radial - p)
    )
    magnetic = phase * k**2 / (Z0 * r) * (1 + 1j / (k * r)) * np.cross(direction, p)
    return ElectromagneticField(ElectricField(grid, 1, electric), magnetic)


def curved_surface(height, amplitude, period):
    f = 2 * np.pi / period
    return FreeformSurface(
        lambda xy: height + amplitude * np.cos(f * xy[..., 0]) * np.cos(f * xy[..., 1]),
        lambda xy: (
            -amplitude
            * f
            * np.stack(
                (
                    np.sin(f * xy[..., 0]) * np.cos(f * xy[..., 1]),
                    np.cos(f * xy[..., 0]) * np.sin(f * xy[..., 1]),
                ),
                axis=-1,
            )
        ),
    )


def relative_fields(actual, expected, weights=None):
    if weights is None:
        weights = np.ones(expected.electric.grid.shape)
    weight = np.sqrt(weights)[..., None]
    return {
        name: float(np.linalg.norm((a - b) * weight) / np.linalg.norm(b * weight))
        for name, a, b in (
            ("E", actual.electric.values, expected.electric.values),
            ("H", actual.magnetic, expected.magnetic),
        )
    }


def tapered(field, window):
    return ElectromagneticField(
        ElectricField(
            field.electric.grid, 1, field.electric.values * window[..., None]
        ),
        field.magnetic * window[..., None],
    )


def sampled_spectrum(field, region, max_sine):
    """Explicit band selection before Maxwell completion; no grazing division.

    The sampled transverse E is prescribed data; truncation/tapering means the
    original complete E/H need not belong to this finite modal space.
    """
    grid = field.electric.grid
    fx, fy = frequency_grid(grid.coordinates)
    ku, kv, k = 2 * np.pi * fx, 2 * np.pi * fy, 2 * np.pi * region.material.n
    active = ku**2 + kv**2 <= (max_sine * k) ** 2
    kz = np.sqrt((k**2 - ku**2 - kv**2).astype(complex))
    xy = FFT2((field.electric.values @ grid.frame.basis)[..., :2]) / np.prod(grid.shape)
    xy[~active] = 0
    longitudinal = -(ku * xy[..., 0] + kv * xy[..., 1]) / np.where(active, kz, 1)
    amplitudes = (
        np.concatenate((xy, longitudinal[..., None]), axis=-1) @ grid.frame.basis.T
    )
    waves = (
        ku[..., None] * grid.frame.u
        + kv[..., None] * grid.frame.v
        + kz[..., None] * grid.frame.normal
    )
    return PlaneSpectrum(
        grid,
        PlaneWaveExpansion(
            waves.reshape(-1, 3),
            amplitudes.reshape(-1, 3),
            grid.points[grid.shape[0] // 2, grid.shape[1] // 2],
            1,
            region,
        ),
    )


def transfer_case(evaluate, *, length=24, count=96, max_sine=0.85, height=2):
    """Plane -> curved trace, and curved trace -> independent second surface.

    The same 9x9 probe coordinates [-1,1]^2 are used at every refinement.
    Surface shape stays fixed as the computational window grows (period 24).
    """
    source, _, _ = problem()
    axis = (np.arange(count) - count // 2) * length / count
    plane = CartesianGrid(axis, axis, Frame([0, 0, height]))
    first = SurfaceTransport(
        plane, curved_surface(height, 0.4, 24), source.region, 1, max_sine=max_sine
    )
    second = SurfaceTransport(
        plane, curved_surface(height + 2, 0.3, 24), source.region, 1, max_sine=max_sine
    )
    # A cosine-squared edge is one inside |x|,|y| <= 0.3 L, zero at 0.5 L.
    t = np.clip((abs(plane.points[..., :2]) - 0.3 * length) / (0.2 * length), 0, 1)
    window = np.prod(np.cos(np.pi * t / 2) ** 2, axis=-1)
    trace = tapered(evaluate(first.grid), window)
    recovered, report = first.inverse(trace, tolerance=1e-10, max_iterations=400)
    radiated = first.radiate(trace)
    initial = sampled_spectrum(
        tapered(evaluate(plane), window), source.region, max_sine
    )
    indices = count // 2 + np.rint(np.linspace(-1, 1, 9) * count / length).astype(int)
    np.testing.assert_allclose(axis[indices], np.linspace(-1, 1, 9), atol=1e-14)
    subset = np.ix_(indices, indices)

    def patch(field):
        grid = PointGrid(field.electric.grid.points[subset])
        return ElectromagneticField(
            ElectricField(grid, 1, field.electric.values[subset]),
            field.magnetic[subset],
        )

    def errors(field, transport):
        actual = patch(field)
        return relative_fields(
            actual, evaluate(actual.electric.grid), transport.grid.weights[subset]
        )

    return {
        "length": length,
        "count": count,
        "max_sine": max_sine,
        "height": height,
        "forward": errors(first.forward(initial), first),
        "inverse_transfer": errors(second.forward(recovered), second),
        "radiate_transfer": errors(second.forward(radiated), second),
        "trace_report": asdict(report),
        "rank": first.rank,
    }


def modal_case(*, scale=1, sag=0.7, rotated=True):
    """Exact, independently evaluated, three-plane-wave Maxwell solution.

    Periodic charts remove edge truncation. Translation, rotation, complex
    polarization and SI rescaling exercise phase and vector conventions.
    """
    angle = 0.41 if rotated else 0
    c, s = np.cos(angle), np.sin(angle)
    basis = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    frame = Frame(scale * np.array([0.2, -0.3, 0.5]), basis)
    axis = (np.arange(32) - 16) * (12 * scale / 32)
    plane = CartesianGrid(axis, axis, frame)
    region = HomogeneousRegion("background", Dielectric(1.2))
    k = 2 * np.pi * 1.2 / scale
    transverse = 2 * np.pi / (12 * scale) * np.array([[1, 2], [4, -2], [-3, 1]])
    kz = np.sqrt(k**2 - np.sum(transverse**2, axis=-1))
    wavevectors = np.column_stack((transverse, kz)) @ basis.T
    electric = np.array([[1, 0.3j], [0.2j, 0.5], [-0.4, 0.2j]])
    longitudinal = -np.sum(transverse * electric, axis=-1) / kz
    source = PlaneWaveExpansion(
        wavevectors,
        np.column_stack((electric, longitudinal)) @ basis.T,
        frame.origin + scale * np.array([0.17, -0.23, 0.31]),
        scale,
        region,
    )
    from vecdiff_general.fields.angular_spectrum import angular_spectrum

    initial = angular_spectrum(source.evaluate(plane).electric, region)
    f = 2 * np.pi / (12 * scale)
    surface = FreeformSurface(
        lambda xy: sag * scale * np.cos(f * xy[..., 0]) * np.cos(f * xy[..., 1]),
        lambda xy: (
            -sag
            * scale
            * f
            * np.stack(
                (
                    np.sin(f * xy[..., 0]) * np.cos(f * xy[..., 1]),
                    np.cos(f * xy[..., 0]) * np.sin(f * xy[..., 1]),
                ),
                axis=-1,
            )
        ),
        frame,
    )
    q = SurfaceTransport(plane, surface, region, scale)
    independent_trace = source.evaluate(q.grid)
    recovered, report = q.inverse(
        independent_trace, tolerance=1e-11, max_iterations=400
    )
    radiated = q.radiate(independent_trace)
    target = PointGrid(plane.points[::4, ::4] + 3 * scale * frame.normal)
    expected = source.evaluate(target)
    return {
        "scale": scale,
        "sag": sag,
        "rotated": rotated,
        "forward": relative_fields(
            q.forward(initial), independent_trace, q.grid.weights
        ),
        "inverse_transfer": relative_fields(recovered.evaluate(target), expected),
        "radiate_transfer": relative_fields(radiated.evaluate(target), expected),
        "trace_report": asdict(report),
    }


def single_interface_case():
    """Exact first-Debye-term sphere transmission, before any internal return.

    An optical-size sphere is a positive control for predominantly forward
    fields. This is a single-interface solution, not closed-body Mie scattering.
    """
    from scipy.special import erfc

    from vecdiff_general.propagation.references.debye import sphere_single_interface

    radius = 300 / (2 * np.pi)
    half = 0.45 * radius
    count = 2 * int(np.ceil(half / 0.5))
    count += (-count) % 16
    axis = (np.arange(count) - count // 2) * (2 * half / count)
    plane = CartesianGrid(axis, axis)
    glass = HomogeneousRegion("glass", Dielectric(1.5))
    q = SurfaceTransport(plane, SphericalCap(radius), glass, 1)

    def evaluate(grid):
        e, h = sphere_single_interface(grid.points, 1, radius, 1, 1.5)["transmitted"]
        return ElectromagneticField(ElectricField(grid, 1, e), h)

    points = q.grid.points
    edge = 0.5 * erfc(
        (np.hypot(points[..., 0], points[..., 1]) - 0.36 * radius) / (0.025 * radius)
    )
    trace = tapered(evaluate(q.grid), edge)
    coordinate = np.linspace(-0.15, 0.15, 9) * radius
    x, y = np.meshgrid(coordinate, coordinate)
    height = radius - np.sqrt(radius**2 - 2 * (0.15 * radius) ** 2) + 2
    probes = PointGrid(np.stack((x, y, np.full_like(x, height)), axis=-1))
    expected = evaluate(probes)
    recovered, report = q.inverse(trace, tolerance=1e-10, max_iterations=400)
    return {
        "reference": "single-interface Debye sphere",
        "ka_incident": 300,
        "radius": radius,
        "count": count,
        "max_sine": 0.85,
        "probe_height": height,
        "probe_half_width": 0.15 * radius,
        "inverse_transfer": relative_fields(recovered.evaluate(probes), expected),
        "radiate_transfer": relative_fields(
            q.radiate(trace).evaluate(probes), expected
        ),
        "trace_report": asdict(report),
    }


def run_benchmarks(include_bem=True):
    source, sphere, glass = problem()
    mie = MieSphere(source, sphere, glass)
    rows = []
    # Independent controls: spacing, width, angular band, and source distance.
    settings = [
        (24, 96, 0.85, 2),
        (24, 192, 0.85, 2),
        (16, 64, 0.85, 2),
        (32, 128, 0.85, 2),
        (24, 96, 0.95, 2),
        (24, 96, 0.99, 2),
        (24, 96, 0.999, 2),
        (48, 192, 0.999, 2),
        (24, 96, 0.99, 0.8),
    ]
    for label, evaluate in (
        ("Mie scattered", scattered(mie)),
        ("Hertz dipole", dipole),
    ):
        for length, count, band, height in settings:
            row = {
                "reference": label,
                **transfer_case(
                    evaluate, length=length, count=count, max_sine=band, height=height
                ),
            }
            rows.append(row)
            print(
                label, length, count, band, height, row["inverse_transfer"], flush=True
            )

    checks = [
        {"reference": "analytic plane-wave superposition", **modal_case(scale=scale)}
        for scale in (1, 532e-9)
    ]
    checks.append(single_interface_case())
    from vecdiff_general.propagation.references.diagnostics import maxwell_residuals

    checks.append(
        {
            "reference": "Hertz dipole Maxwell residuals",
            "step": 0.0025,
            "residuals": maxwell_residuals(
                dipole,
                [[0.4, -0.3, 1.2], [-0.2, 0.1, 0.6]],
                wavelength0=1,
                index=1,
                step=0.0025,
            ),
        }
    )
    coordinate = np.linspace(-1, 1, 9)
    x, y = np.meshgrid(coordinate, coordinate)
    xy = np.stack((x, y), axis=-1)
    probes = PointGrid(curved_surface(4, 0.3, 24).points(xy))
    exact = scattered(mie)(probes)
    finer = MieSphere(source, sphere, glass, order=mie.order + 6)
    checks.append(
        {
            "reference": "Mie order refinement",
            "order": mie.order,
            "change": relative_fields(scattered(finer)(probes), exact),
        }
    )
    if include_bem:
        for shape in ("sphere", "spheroid"):
            previous = None
            for level in (1, 2, 3):
                mesh = SurfaceMesh.sphere(sphere, level)
                if shape == "spheroid":
                    mesh = SurfaceMesh(mesh.vertices * [1, 1, 1.5], mesh.triangles)
                solution = DielectricBEM(
                    source, mesh, glass, cache=".cache/transport_bem"
                )
                evaluate = scattered(solution)
                field = evaluate(probes)
                check = {
                    "reference": "BEM " + shape,
                    "level": level,
                    "unknowns": solution.unknowns,
                    "algebraic_residual": solution.algebraic_residual,
                }
                if shape == "sphere":
                    check["vs_mie"] = relative_fields(field, exact)
                if previous is not None:
                    check["mesh_change"] = relative_fields(field, previous)
                previous = field
                checks.append(check)
                # Transport errors are against THIS BEM field; its discretization
                # error is separately measured above, never assigned to Q.
                row = {
                    "reference": "BEM " + shape,
                    "level": level,
                    **transfer_case(evaluate, max_sine=0.99),
                }
                rows.append(row)
                print(
                    row["reference"], level, row["inverse_transfer"], check, flush=True
                )
    return {
        "wavelength0": 1,
        "sphere_radius": 0.3,
        "index": 1.5,
        "probe_patch": "9x9 nodes, [-1,1]^2; physical surface area weights",
        "surface_heights": "h1=z+0.4 cos(2pi x/24)cos(2pi y/24); h2=z+2+0.3 cos(2pi x/24)cos(2pi y/24)",
        "taper": "product cosine-squared edges, unity to 0.3L, zero at 0.5L",
        "error": "relative complex vector L2, no phase or amplitude alignment",
        "reference_checks": checks,
        "cases": rows,
    }


def plot_results(result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, reference in zip(axes, ("Mie scattered", "Hertz dipole")):
        rows = [
            r
            for r in result["cases"]
            if r["reference"] == reference
            and r["length"] == 24
            and r["count"] == 96
            and r["height"] == 2
        ]
        rows.sort(key=lambda r: r["max_sine"])
        for method, label in (
            ("forward", "Q: plane to surface"),
            ("inverse_transfer", "Q₂ Q₁⁺: least squares"),
            ("radiate_transfer", "Q₂ radiation: E and H"),
        ):
            ax.plot(
                [r["max_sine"] for r in rows],
                [100 * max(r[method].values()) for r in rows],
                "o-",
                label=label,
            )
        ax.set(
            title=reference,
            xlabel="Angular cutoff sin(θmax)",
            ylabel="max(E, H) relative error (%)",
        )
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-bem", action="store_true")
    args = parser.parse_args()
    result = run_benchmarks(not args.skip_bem)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    plot_results(result, args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
