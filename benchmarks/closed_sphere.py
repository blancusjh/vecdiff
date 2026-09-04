"""Independent closed-sphere convergence benchmark; no Mie data enter the solve."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from vecdiff import Medium, Sphere, plane_wave, sample_surface, solve_closed_interface
from vecdiff.observables.electromagnetism import boundary_residuals, poynting
from references.mie import fields as mie_fields


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))


def case(radius=.5, order=12, *, wavelength=1., sphere_index=1.5, offset=.5):
    """Raw errors at independent surface/bulk points, without fitted normalization."""
    started = time.perf_counter()
    sphere = Sphere(radius)
    sources = sample_surface(sphere, (-1, 1), (0, 2*np.pi), order, 2*order)
    boundary = sample_surface(sphere, (-1, 1), (0, 2*np.pi), 2*order, 3*order)
    incident = plane_wave(wavelength=wavelength)
    field = solve_closed_interface(incident, Medium(sphere_index), boundary, sources,
                                    inward_offset=offset*radius, outward_offset=offset*radius)
    # Same held-out nodes for every order: not a new error functional each time.
    check = sample_surface(sphere, (-1, 1), (.027, 2*np.pi+.027), 31, 47)
    eo, ho = field.evaluate(check.points, region="exterior")
    ei, hi = field.evaluate(check.points, region="interior")
    bc = boundary_residuals(eo, ho, ei, hi, check.normals, Medium(), Medium(sphere_index),
                            weights=check.weights, electric_scale=1., magnetic_scale=1.)
    errors, bulk = {}, []
    for region, scale in [("exterior", 1.3), ("interior", .7)]:
        points = check.points[::13]*scale
        e, h = field.evaluate(points, region=region)
        em, hm = mie_fields(points, radius, wavelength=wavelength, sphere_index=sphere_index)
        errors[region+"_E"] = relative(e, em)
        errors[region+"_H"] = relative(h, hm)
        bulk.extend((e, h))
    # Total outgoing flux through a closed sphere vanishes for a lossless object.
    eflux, hflux = field.evaluate(check.points*1.3, region="exterior")
    flux = np.sum(np.sum(poynting(eflux, hflux)*check.normals, axis=-1)*check.weights*1.3**2)
    incident_flux = poynting([[1, 0, 0]], [[0, 1, 0]])[0, 2]*np.pi*radius**2
    ecenter, _ = field.evaluate([[0, 0, 0]], region="interior")
    row = dict(radius=radius, wavelength=wavelength, sphere_index=sphere_index, order=order,
               source_count=len(sources.points), boundary_count=len(boundary.points), offset_fraction=offset,
               fit_residual=field.fit_residual, held_out_boundary=bc, errors_vs_mie=errors,
               rank=field.rank, unknowns=len(field.coefficients), condition_number=field.condition_number,
               total_closed_flux_relative=float(abs(flux)/incident_flux),
               center_electric_norm2=float(np.sum(abs(ecenter)**2)), seconds=time.perf_counter()-started)
    return row, np.concatenate(bulk), field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radii", nargs="+", type=float, default=[.25, .5, 1.])
    parser.add_argument("--orders", nargs="+", type=int, default=[8, 12, 16])
    parser.add_argument("--offset", type=float, default=.5)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/closed_sphere.json"))
    args = parser.parse_args()
    from importlib.metadata import version
    report = dict(versions={n: version(n) for n in ("numpy", "scipy", "miepython")},
                   conventions="exp(-i omega t), H=Z0 H_SI, vacuum wavelength=1", cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for radius in args.radii:
        previous = None
        for order in args.orders:
            row, bulk, _ = case(radius, order, offset=args.offset)
            if previous is not None:
                row["bulk_change_from_previous_order"] = relative(bulk, previous)
            previous = bulk
            report["cases"].append(row)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
