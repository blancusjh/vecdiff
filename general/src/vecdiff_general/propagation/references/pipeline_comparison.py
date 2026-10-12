"""Fixed closed-body references versus the complete continued-face pipeline.

The numerical models and observation domain are explicit. This is a comparison
of forward open-face propagation and full-wave closed-body scattering, not a
claim that their physical boundary-value problems coincide. Plotting and
notebook display dependencies are loaded only by ``show_comparison``.
"""

import os
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import vecdiff_general as optics
from vecdiff_general.propagation.references import (
    DielectricBEM,
    MieSphere,
    SurfaceMesh,
    surface_mesh,
)


def relative(actual, reference):
    return float(
        np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-300)
    )


def timed_field(solution, points, **kwargs):
    start = perf_counter()
    field = solution.evaluate(optics.PointGrid(points), **kwargs)
    return field, perf_counter() - start


def lens_cases():
    F = optics.Frame
    return {
        "Plano-convex": optics.VolumeBetweenSurfaces(
            optics.Plane(), optics.SphericalCap(-3, frame=F([0, 0, 0.8])), 1.5
        ),
        "Biconvex": optics.VolumeBetweenSurfaces(
            optics.SphericalCap(3), optics.SphericalCap(-3, frame=F([0, 0, 1.2])), 1.5
        ),
        "Meniscus": optics.VolumeBetweenSurfaces(
            optics.SphericalCap(2.2), optics.SphericalCap(6, frame=F([0, 0, 0.8])), 1.5
        ),
        "Even asphere": optics.VolumeBetweenSurfaces(
            optics.Plane(),
            optics.EvenAsphere(
                -1 / 3, conic=-1, coefficients=(-0.015,), frame=F([0, 0, 0.8])
            ),
            1.5,
        ),
        "Cartesian oval": optics.VolumeBetweenSurfaces(
            optics.Plane(),
            optics.CartesianOval.from_conjugates(
                1.5, -np.inf, 1, 6, frame=F([0, 0, 0.8])
            ),
            1.5,
        ),
        "Astigmatic freeform": optics.VolumeBetweenSurfaces(
            optics.Plane(),
            optics.FreeformSurface(
                lambda p: (
                    -(p[..., 0] ** 2) / (2 * 2.6)
                    - p[..., 1] ** 2 / (2 * 4.2)
                    + 0.025 * p[..., 0] * p[..., 1]
                ),
                lambda p: np.stack(
                    (
                        -p[..., 0] / 2.6 + 0.025 * p[..., 1],
                        -p[..., 1] / 4.2 + 0.025 * p[..., 0],
                    ),
                    axis=-1,
                ),
                frame=F([0, 0, 0.85]),
            ),
            1.5,
        ),
        "Two freeform faces": optics.VolumeBetweenSurfaces(
            optics.FreeformSurface(
                lambda p: (
                    0.04 * p[..., 0] ** 2
                    + 0.025 * p[..., 1] ** 2
                    + 0.015 * p[..., 0] * p[..., 1]
                ),
                lambda p: np.stack(
                    (
                        0.08 * p[..., 0] + 0.015 * p[..., 1],
                        0.05 * p[..., 1] + 0.015 * p[..., 0],
                    ),
                    axis=-1,
                ),
            ),
            optics.FreeformSurface(
                lambda p: (
                    -0.12 * (p[..., 0] - 0.25) ** 2
                    - 0.065 * (p[..., 1] + 0.15) ** 2
                    + 0.012 * p[..., 0] ** 3
                    - 0.01 * p[..., 0] * p[..., 1]
                ),
                lambda p: np.stack(
                    (
                        -0.24 * (p[..., 0] - 0.25)
                        + 0.036 * p[..., 0] ** 2
                        - 0.01 * p[..., 1],
                        -0.13 * (p[..., 1] + 0.15) - 0.01 * p[..., 0],
                    ),
                    axis=-1,
                ),
                frame=F([0, 0, 1.1]),
            ),
            1.5,
        ),
    }


def _lens_mesh(body, h):
    # Warp a closed flat-cylinder mesh to the two exact graph faces.
    spacing = body.back.frame.origin[2] - body.front.frame.origin[2]
    base = surface_mesh(
        optics.VolumeBetweenSurfaces(
            optics.Plane(),
            optics.Plane(frame=optics.Frame([0, 0, spacing])),
            body.rim_radius,
        ),
        h,
    )
    p = base.vertices.copy()
    xy = p[:, :2]
    t = p[:, 2] / spacing
    lower = body.front.height(xy)
    upper = spacing + body.back.height(xy)
    assert np.all(upper > lower)
    p[:, 2] = (1 - t) * lower + t * upper
    return SurfaceMesh(p, base.triangles)


def scaled_body(body, radius):
    """Scale every length, preserving shape; radius means sphere or rim radius."""
    original = body.radius if isinstance(body, optics.SphereVolume) else body.rim_radius
    scale = radius / original
    if isinstance(body, optics.SphereVolume):
        return optics.SphereVolume(body.center * scale, radius)

    def face(surface):
        return optics.FreeformSurface(
            lambda xy: scale * surface.height(xy / scale),
            lambda xy: surface.gradient(xy / scale),
            frame=optics.Frame(surface.frame.origin * scale, surface.frame.basis),
        )

    return optics.VolumeBetweenSurfaces(face(body.front), face(body.back), radius)


def compare_closed_body(
    name,
    body,
    source,
    interior,
    *,
    count=128,
    width=12,
    nodes=18,
    threads=4,
    workers=1,
    mesh_edge=0.20,
    sphere_mesh_level=3,
    boundary_band=0.08,
    scale_domain=False,
    adaptive_bem=False,
    bem_max_unknowns=5000,
    focal_halfwidth=2,
    meridian_shape=(151, 89),
    coverage="reduced",
):
    """Return complex fields, trace differences and complete wall times.

    Width, mesh-edge scale and boundary band are in vacuum wavelengths. The
    source must be the analytic plane-wave expansion required by Mie/BEM.
    Mie is used only for a sphere. Each body receives one complete public
    pipeline solve, with shared observation coordinates. BEM is cold and runs
    only when the fixed-resolution mesh fits the declared solver capacity.
    """
    wavelength = source.wavelength0
    pipeline_count, pipeline_width = count, width * wavelength
    pipeline_nodes, pipeline_threads, pipeline_workers = nodes, threads, workers
    mesh_edge *= wavelength
    boundary_band *= wavelength
    radius = body.radius if isinstance(body, optics.SphereVolume) else body.rim_radius
    domain_scale = radius / wavelength if scale_domain else 1.0

    def continued_face(face, core, tail):
        """Exact face to core; C1 radial continuation with radius bounded by core+tail.

        Beyond core, s(r)=core+tail*(1-exp(-(r-core)/tail)). Its derivative is
        one at the join and tends to zero. The continuation is a physical modeling
        choice, not a material rim or an aperture mask.
        """

        def mapping(xy):
            xy = np.asarray(xy, float)
            radius = np.linalg.norm(xy, axis=-1)
            outside = radius > core
            excess = np.maximum(radius - core, 0)
            mapped_radius = np.where(
                outside, core - tail * np.expm1(-excess / tail), radius
            )
            derivative = np.where(outside, np.exp(-excess / tail), 1.0)
            scale = np.divide(
                mapped_radius, radius, out=np.ones_like(radius), where=radius != 0
            )
            direction = np.divide(
                xy,
                radius[..., None],
                out=np.zeros_like(xy),
                where=radius[..., None] != 0,
            )
            return xy * scale[..., None], scale, derivative, direction

        def height(xy):
            return face.height(mapping(xy)[0])

        def gradient(xy):
            mapped, scale, derivative, direction = mapping(xy)
            base = face.gradient(mapped)
            radial = np.sum(base * direction, axis=-1)
            return (
                scale[..., None] * base
                + ((derivative - scale) * radial)[..., None] * direction
            )

        return optics.FreeformSurface(height, gradient, frame=face.frame)

    def pipeline_geometry(body):
        if isinstance(body, optics.SphereVolume):
            radius = body.radius
            front = optics.SphericalCap(
                radius, optics.Frame(body.center - [0, 0, radius])
            )
            back = optics.SphericalCap(
                -radius, optics.Frame(body.center + [0, 0, radius])
            )
            core, tail = 0.95 * radius, 0.035 * radius
        else:
            front, back = body.front, body.back
            radius = body.rim_radius
            core, tail = radius, 0.05 * radius
        return [continued_face(face, core, tail) for face in (front, back)], core, tail

    def run_pipeline(body, interior, *, compare_traces):
        start = perf_counter()
        faces, core, tail = pipeline_geometry(body)
        axis = (
            (np.arange(pipeline_count) - pipeline_count // 2)
            * pipeline_width
            / pipeline_count
        )
        entrance_z = faces[0].frame.origin[2] - 2 * wavelength
        grid = optics.CartesianGrid(axis, axis, optics.Frame([0, 0, entrance_z]))
        electric = source.evaluate(grid).electric.values[..., :2]
        incident = optics.TransverseElectricField(grid, wavelength, electric)
        interfaces = [
            optics.DielectricInterface(faces[0], source.region, interior),
            optics.DielectricInterface(faces[1], interior, source.region),
        ]
        path = optics.PropagationPipeline(
            optics.OpticalSystem(interfaces),
            coverage=coverage,
            max_sine=0.95,
            transmission=optics.SpectralFresnel(
                nodes=pipeline_nodes,
                angular_limit=0.65,
                cutoff=1e-7,
                workers=pipeline_workers,
                threads=pipeline_threads,
                cache_bytes=128 * 1024**2,
            ),
        )
        preparation = perf_counter() - start
        start = perf_counter()
        solution = path.solve(incident)
        solve = perf_counter() - start
        # Reconstruct the intermediate spectrum from the already computed first
        # interface response, without repeating transmission or inferring a ray.
        start = perf_counter()
        transports = [
            optics.SurfaceTransport(
                optics.CartesianGrid(axis, axis, face.frame),
                face,
                region,
                wavelength,
                max_sine=0.95,
            )
            for face, region in zip(
                faces if compare_traces else (), (interior, source.region)
            )
        ]
        intermediate = solution.stages[1]
        reconstructed = (
            [
                transports[0].forward(intermediate),
                transports[1].forward(solution.spectrum),
            ]
            if compare_traces
            else []
        )
        trace_setup = perf_counter() - start
        front_points, back_points = [
            response.field.grid.points for response in solution.interactions
        ]
        return dict(
            solution=solution,
            path=path,
            incident=incident,
            intermediate=intermediate,
            faces=faces,
            core=core,
            tail=tail,
            axis=axis,
            reconstructed=reconstructed,
            envelope=(
                front_points[..., 2].min(),
                front_points[..., 2].max(),
                back_points[..., 2].min(),
                back_points[..., 2].max(),
            ),
            timing={
                "preparation s": preparation,
                "solve s": solve,
                "interface reconstruction s": trace_setup,
            },
        )

    def pipeline_meridian(candidate, body, valid):
        start = perf_counter()
        if candidate["path"].coverage == "complete":
            from vecdiff_general.fourier.nufft import NUFFT

            path = candidate["path"]
            electric = path(
                candidate["incident"],
                optics.PointGrid(meridian),
                side="plus",
                backend="nufft",
            )
            report = path.last_coverage
            # E was evaluated through the public operator. H is requested only
            # for this explicit E/H benchmark, using exactly the same regions.
            magnetic = np.empty_like(electric.values)
            for index in np.unique(report.regions):
                mask = report.regions == index
                modes = path.last_solution.stages[index].expansion
                magnetic[mask] = NUFFT(
                    modes.magnetic_amplitudes,
                    modes.wavevectors,
                    meridian[mask] - modes.origin,
                    backend="nufft",
                )
            candidate["extended_points"] = int(report.extended.sum())
            field = optics.ElectromagneticField(
                optics.ElectricField(
                    optics.PointGrid(meridian[report.computed]),
                    wavelength,
                    electric.values[report.computed],
                ),
                magnetic[report.computed],
            )
            return field, report.computed, electric.values, perf_counter() - start
        front_min, front_max, back_min, back_max = candidate["envelope"]
        inside = body.contains(meridian)
        masks = [
            valid & (Z < front_min),
            valid & inside & (Z >= front_max) & (Z <= back_min),
            valid & (Z >= back_max),
        ]
        electric = np.full(meridian.shape, np.nan + 1j * np.nan)
        magnetic = np.full_like(electric, np.nan + 1j * np.nan)
        for mask, model in zip(
            masks, (source, candidate["intermediate"], candidate["solution"])
        ):
            if np.any(mask):
                field, _ = timed_field(
                    model,
                    meridian[mask],
                    **({"backend": "nufft"} if model is not source else {}),
                )
                electric[mask], magnetic[mask] = field.electric.values, field.magnetic
        coverage = np.logical_or.reduce(masks)
        field = optics.ElectromagneticField(
            optics.ElectricField(
                optics.PointGrid(meridian[coverage]), wavelength, electric[coverage]
            ),
            magnetic[coverage],
        )
        return field, coverage, electric, perf_counter() - start

    def interface_probes(bem, body, which):
        # Facet centroids give genuine one-sided BEM traces, avoiding mesh edges.
        mesh = bem.mesh
        corners = mesh.corners
        if isinstance(body, optics.SphereVolume):
            select = np.all(
                (corners[..., 2] - body.center[2]) * (-1 if which == 0 else 1)
                >= -1e-10,
                axis=1,
            )
        else:
            face = (body.front, body.back)[which]
            heights = face.frame.origin[2] + face.height(corners[..., :2])
            select = np.all(abs(corners[..., 2] - heights) < 1e-8 * wavelength, axis=1)
        points = corners[select].mean(axis=1)
        radius = (
            body.radius if isinstance(body, optics.SphereVolume) else body.rim_radius
        )
        points = points[np.linalg.norm(points[:, :2], axis=-1) <= 0.8 * radius]
        return points[np.linspace(0, len(points) - 1, min(49, len(points)), dtype=int)]

    def compare_interfaces(name, candidate, body, references):
        if not references:
            return [], 0.0, {}
        start = perf_counter()
        rows = []
        reference_seconds = {name: 0.0 for name in references}
        for which, (face, response, reconstructed) in enumerate(
            zip(
                candidate["faces"],
                candidate["solution"].interactions,
                candidate["reconstructed"],
            )
        ):
            if "BEM" in references:
                mesh_points = interface_probes(references["BEM"], body, which)
                sample_ids = None
            else:
                angles = np.arange(49) * np.pi * (3 - np.sqrt(5))
                radii = 0.8 * radius * np.sqrt((np.arange(49) + 0.5) / 49)
                xy = np.stack((radii * np.cos(angles), radii * np.sin(angles)), axis=-1)
                axis = candidate["axis"]
                column_ids = np.abs(axis[:, None] - xy[:, 0]).argmin(axis=0)
                row_ids = np.abs(axis[:, None] - xy[:, 1]).argmin(axis=0)
                sample_ids = row_ids, column_ids
                xy = np.stack((axis[column_ids], axis[row_ids]), axis=-1)
                mesh_points = face.points(xy)
            xy = mesh_points[:, :2]
            points = face.points(xy)
            normals = face.normals(xy)
            samples = []
            for field in (response.field, reconstructed):
                values = [
                    (
                        RegularGridInterpolator(
                            (candidate["axis"], candidate["axis"]), data
                        )(xy[:, ::-1])
                        if sample_ids is None
                        else data[sample_ids]
                    )
                    for data in (field.electric.values, field.magnetic)
                ]
                samples.append(values)
            for reference_name, model in references.items():
                reference_start = perf_counter()
                reference_points = points if reference_name == "Mie" else mesh_points
                field, elapsed = timed_field(
                    model,
                    reference_points,
                    side="interior" if which == 0 else "exterior",
                )
                reference_seconds[reference_name] += perf_counter() - reference_start

                def tangent(values):
                    return (
                        values - np.sum(values * normals, axis=-1)[..., None] * normals
                    )

                row = {
                    "geometry": name,
                    "face": "entrance" if which == 0 else "exit",
                    "reference": reference_name,
                    "samples": len(xy),
                    "pipeline trace sampling": (
                        "bilinear at BEM facets"
                        if sample_ids is None
                        else "exact Q grid points"
                    ),
                    "max facet offset / λ": float(
                        np.max(np.linalg.norm(reference_points - points, axis=-1))
                        / wavelength
                    ),
                }
                for label, values in zip(("T trace", "reconstructed trace"), samples):
                    row[label + " E %"] = 100 * relative(
                        values[0], field.electric.values
                    )
                    row[label + " H %"] = 100 * relative(values[1], field.magnetic)
                    row[label + " tangent E %"] = 100 * relative(
                        tangent(values[0]), tangent(field.electric.values)
                    )
                    row[label + " tangent H %"] = 100 * relative(
                        tangent(values[1]), tangent(field.magnetic)
                    )
                rows.append(row)
        elapsed = perf_counter() - start - sum(reference_seconds.values())
        return rows, elapsed, reference_seconds

    extent = min(2.2 * domain_scale, 0.45 * width) if scale_domain else 2.2
    x = np.linspace(-extent, extent, meridian_shape[1]) * wavelength
    z = np.linspace(-1.45, 8.05, meridian_shape[0]) * wavelength * domain_scale
    X, Z = np.meshgrid(x, z)
    meridian = np.stack((X, np.zeros_like(X), Z), axis=-1)
    focal_axis = np.linspace(-focal_halfwidth, focal_halfwidth, 81) * wavelength
    FX, FY = np.meshgrid(focal_axis, focal_axis)
    results, accuracy_rows, timing_rows, interface_rows = {}, [], [], []

    def body_band(body, points):
        if isinstance(body, optics.SphereVolume):
            return (
                abs(np.linalg.norm(points - body.center, axis=-1) - body.radius)
                < boundary_band
            )
        xy = points[..., :2]
        rho = np.linalg.norm(xy, axis=-1)
        clipped = (
            xy * np.minimum(1, body.rim_radius / np.maximum(rho, 1e-15))[..., None]
        )
        lo = body.front.frame.origin[2] + body.front.height(clipped)
        hi = body.back.frame.origin[2] + body.back.height(clipped)
        zz = points[..., 2]
        faces = (rho < body.rim_radius + boundary_band) & (
            (abs(zz - lo) < boundary_band) | (abs(zz - hi) < boundary_band)
        )
        rim = (
            (abs(rho - body.rim_radius) < boundary_band)
            & (zz > lo - boundary_band)
            & (zz < hi + boundary_band)
        )
        return faces | rim

    def run_case(name, body, interior, *, sphere=False):
        start = perf_counter()
        if sphere:
            area = 4 * np.pi * radius**2
        else:
            spacing = body.back.frame.origin[2] - body.front.frame.origin[2]
            area = 2 * np.pi * radius * (radius + spacing)
        expected_unknowns = int(
            np.ceil(1.2 * 3 * area / (np.sqrt(3) / 4 * mesh_edge**2))
        )
        mesh = None
        if not adaptive_bem or radius <= 3 * wavelength:
            mesh = (
                (
                    surface_mesh(body, mesh_edge)
                    if adaptive_bem
                    else SurfaceMesh.sphere(body, sphere_mesh_level)
                )
                if sphere
                else _lens_mesh(body, mesh_edge)
            )
            expected_unknowns = 2 * len(mesh.edges)
        bem = None
        references, setup_times = {}, {}
        if mesh is not None and expected_unknowns <= bem_max_unknowns:
            with patch.dict(os.environ, {"VECDIFF_BEM_CACHE": ""}):
                bem = DielectricBEM(
                    source, mesh, interior, max_unknowns=bem_max_unknowns
                )
            references["BEM"] = bem
            setup_times["BEM"] = perf_counter() - start
        if sphere:
            start = perf_counter()
            references = {"Mie": MieSphere(source, body, interior), **references}
            setup_times["Mie"] = perf_counter() - start
        candidate = run_pipeline(body, interior, compare_traces=bool(references))
        end = body.center[2] + body.radius if sphere else body.back.frame.origin[2]
        scan_start = max(end, candidate["envelope"][-1]) + 0.15 * wavelength
        scan_count = (
            max(301, int(np.ceil((z[-1] - scan_start) / (0.25 * wavelength))) + 1)
            if scale_domain
            else 301
        )
        scan_z = np.linspace(scan_start, z[-1], scan_count)
        scan_points = np.stack(
            (np.zeros_like(scan_z), np.zeros_like(scan_z), scan_z), axis=-1
        )
        focus_reference = (
            "Mie" if sphere else ("BEM" if bem is not None else "Pipeline")
        )
        axial, axial_times = ({}, {})
        axial[focus_reference], axial_times[focus_reference] = timed_field(
            references.get(focus_reference, candidate["solution"]),
            scan_points,
            **({"backend": "nufft"} if focus_reference == "Pipeline" else {}),
        )
        peak = int(np.argmax(axial[focus_reference].electric.norm2()))
        focus_z = scan_z[peak]
        focal_points = np.stack((FX, FY, np.full_like(FX, focus_z)), axis=-1)
        focal, focal_times, bulk, bulk_times = ({}, {}, {}, {})
        for method, model in references.items():
            if method not in axial:
                axial[method], axial_times[method] = timed_field(model, scan_points)
            focal[method], focal_times[method] = timed_field(model, focal_points)
        if "Pipeline" not in axial:
            axial["Pipeline"], axial_times["Pipeline"] = timed_field(
                candidate["solution"], scan_points, backend="nufft"
            )
        focal["Pipeline"], focal_times["Pipeline"] = timed_field(
            candidate["solution"], focal_points, backend="nufft"
        )
        valid = ~body_band(body, meridian)
        bulk["Pipeline"], coverage, pipeline_map, bulk_times["Pipeline"] = (
            pipeline_meridian(candidate, body, valid)
        )
        for method, model in references.items():
            bulk[method], bulk_times[method] = timed_field(model, meridian[coverage])
        interface, pipeline_trace_s, reference_trace_s = compare_interfaces(
            name, candidate, body, references
        )
        interface_rows.extend(interface)
        downstream = meridian[coverage, 2] >= candidate["envelope"][-1]
        pairs = [("Pipeline", method) for method in references]
        if sphere and bem is not None:
            pairs.append(("BEM", "Mie"))
        for method, reference in pairs:
            row = {
                "geometry": name,
                "candidate": method,
                "reference": reference,
                "plane z / λ": float(focus_z / wavelength),
                "focal E %": 100
                * relative(
                    focal[method].electric.values, focal[reference].electric.values
                ),
                "focal H %": 100
                * relative(focal[method].magnetic, focal[reference].magnetic),
                "meridian E %": 100
                * relative(
                    bulk[method].electric.values, bulk[reference].electric.values
                ),
                "meridian H %": 100
                * relative(bulk[method].magnetic, bulk[reference].magnetic),
                "downstream E %": 100
                * relative(
                    bulk[method].electric.values[downstream],
                    bulk[reference].electric.values[downstream],
                ),
                "downstream H %": 100
                * relative(
                    bulk[method].magnetic[downstream],
                    bulk[reference].magnetic[downstream],
                ),
                "meridian coverage %": float(100 * coverage.sum() / valid.sum()),
            }
            accuracy_rows.append(row)
        for method in (*references, "Pipeline"):
            preparation = (
                candidate["timing"]["preparation s"]
                if method == "Pipeline"
                else setup_times[method]
            )
            solve = candidate["timing"]["solve s"] if method == "Pipeline" else 0.0
            trace_time = (
                candidate["timing"]["interface reconstruction s"] + pipeline_trace_s
                if method == "Pipeline"
                else reference_trace_s[method]
            )
            row = {
                "geometry": name,
                "method": method,
                "setup + solve s": preparation + solve,
                "focal field s": focal_times[method],
                "full focal calculation s": preparation + solve + focal_times[method],
                "axial field s": axial_times[method],
                "meridian field s": bulk_times[method],
                "interface fields s": trace_time,
                "all requested fields s": preparation
                + solve
                + focal_times[method]
                + axial_times[method]
                + bulk_times[method]
                + trace_time,
                "meridian points": int(coverage.sum()),
                "focal points": int(FX.size),
            }
            timing_rows.append(row)
        results[name] = dict(
            body=body,
            focal=focal,
            axial=axial,
            bulk=bulk,
            coverage=coverage,
            scan_z=scan_z,
            focus_z=focus_z,
            envelope=candidate["envelope"],
            core=candidate["core"],
            tail=candidate["tail"],
            triangles=len(mesh.triangles) if bem is not None else None,
            unknowns=expected_unknowns,
            bem_residual=float(bem.algebraic_residual) if bem is not None else None,
            bem_computed=bem is not None,
            bem_unknowns_estimated=mesh is None,
            bem_matrix_gib=16 * expected_unknowns**2 / 1024**3,
            bem_max_unknowns=bem_max_unknowns,
            coverage_percent=float(
                100
                * coverage.sum()
                / (
                    coverage.size
                    if candidate["path"].coverage == "complete"
                    else valid.sum()
                )
            ),
            coverage_mode=candidate["path"].coverage,
            extended_points=candidate.get("extended_points", 0),
            radius_over_wavelength=radius / wavelength,
            count=count,
            spacing_over_wavelength=width / count,
            width_over_wavelength=width,
            focus_reference=focus_reference,
            mie_order=references["Mie"].order if sphere else None,
            focus_at_scan_boundary=peak in (0, len(scan_z) - 1),
        )

    run_case(name, body, interior, sphere=isinstance(body, optics.SphereVolume))
    result = results[name]
    result.update(
        name=name,
        wavelength=wavelength,
        x=x,
        z=z,
        focal_axis=focal_axis,
        accuracy=accuracy_rows,
        timing=timing_rows,
        interfaces=interface_rows,
    )
    return result


def show_comparison(result):
    """Show each case's fields first, followed by compact differences and times."""
    from html import escape

    import matplotlib.pyplot as plt
    from IPython.display import HTML, display

    def table_html(rows):
        if not rows:
            return "<p>No independent reference computed at this size.</p>"
        keys = list(rows[0])

        def cell(value):
            text = (
                f"{value:.3g}"
                if isinstance(value, (float, np.floating))
                else str(value)
            )
            return escape(text)

        return (
            "<div style='overflow-x:auto'><table><tr>"
            + "".join(f"<th>{escape(key)}</th>" for key in keys)
            + "</tr>"
            + "".join(
                "<tr>" + "".join(f"<td>{cell(row[key])}</td>" for key in keys) + "</tr>"
                for row in rows
            )
            + "</table></div>"
        )

    def outline(ax):
        body = result["body"]
        if isinstance(body, optics.SphereVolume):
            theta = np.linspace(0, 2 * np.pi, 401)
            ax.plot(
                (body.center[2] + body.radius * np.cos(theta)) / wavelength,
                body.radius * np.sin(theta) / wavelength,
                color="0.5",
                lw=0.7,
            )
        else:
            axis = np.linspace(-body.rim_radius, body.rim_radius, 301)
            xy = np.stack((axis, np.zeros_like(axis)), axis=-1)
            front = body.front.frame.origin[2] + body.front.height(xy)
            back = body.back.frame.origin[2] + body.back.height(xy)
            ax.plot(front / wavelength, axis / wavelength, color="0.5", lw=0.7)
            ax.plot(back / wavelength, axis / wavelength, color="0.5", lw=0.7)
            for j in (0, -1):
                ax.plot(
                    [front[j] / wavelength, back[j] / wavelength],
                    [axis[j] / wavelength] * 2,
                    color="0.5",
                    lw=0.7,
                )

    wavelength = result["wavelength"]
    methods = [
        method for method in ("Pipeline", "Mie", "BEM") if method in result["focal"]
    ]
    coverage = result["coverage"]
    x, z, focal_axis = result["x"], result["z"], result["focal_axis"]
    extent = [
        z[0] / wavelength,
        z[-1] / wavelength,
        x[0] / wavelength,
        x[-1] / wavelength,
    ]
    focal_extent = [focal_axis[0] / wavelength, focal_axis[-1] / wavelength] * 2
    fig, axes = plt.subplots(
        3,
        len(methods),
        figsize=(3.7 * len(methods), 7.2),
        sharey="row",
        constrained_layout=True,
    )
    for row, key in enumerate(("bulk", "focal")):
        maximum = max(
            np.max(result[key][method].electric.norm2()) for method in methods
        )
        for col, method in enumerate(methods):
            values = result[key][method].electric.norm2()
            if row == 0:
                image = np.full((len(z), len(x)), np.nan)
                image[coverage] = values
                values = image.T
            image = axes[row, col].imshow(
                values,
                origin="lower",
                extent=extent if row == 0 else focal_extent,
                cmap="hot",
                aspect="auto" if row == 0 else "equal",
                vmin=0,
                vmax=maximum,
            )
            axes[row, col].set(
                title=f"{method} · {'meridian' if row == 0 else 'focal plane'}",
                xlabel="z / λ" if row == 0 else "x / λ",
            )
            if row == 0:
                outline(axes[row, col])
        axes[row, 0].set_ylabel("x / λ" if row == 0 else "y / λ")
        fig.colorbar(image, ax=list(axes[row]), shrink=0.8, label="|E|² (V²/m²)")
    for col, method in enumerate(methods):
        ex = result["focal"][method].electric.values[len(focal_axis) // 2, :, 0]
        axes[2, col].plot(focal_axis / wavelength, ex.real, label="Re Ex")
        axes[2, col].plot(focal_axis / wavelength, ex.imag, "--", label="Im Ex")
        axes[2, col].set(xlabel="x / λ; y = 0")
        axes[2, col].legend(fontsize=8)
    axes[2, 0].set_ylabel("Ex (V/m)")
    fig.suptitle(
        f"{result['name']} · observation plane z/λ={result['focus_z'] / wavelength:.3f}"
        + (" · scan-boundary maximum" if result.get("focus_at_scan_boundary") else "")
    )
    plt.show()
    plt.close(fig)

    differences = [
        {
            "Comparison": f"{row['candidate']} / {row['reference']}",
            "Focal E (%)": row["focal E %"],
            "Focal H (%)": row["focal H %"],
            "Meridian E (%)": row["meridian E %"],
            "Meridian H (%)": row["meridian H %"],
        }
        for row in result["accuracy"]
    ]
    times = [
        {
            "Method": row["method"],
            "Setup + solve (s)": row["setup + solve s"],
            "Complete focal (s)": row["full focal calculation s"],
            "All requested fields (s)": row["all requested fields s"],
        }
        for row in result["timing"]
    ]
    display(
        HTML(
            "<b>Complex-field differences</b>"
            + table_html(differences)
            + "<b>Execution time</b>"
            + table_html(times)
        )
    )

    interface = [
        {
            "Face": row["face"],
            "Reference": row["reference"],
            "Facet offset / λ": row["max facet offset / λ"],
            "T E (%)": row["T trace E %"],
            "T H (%)": row["T trace H %"],
            "Reconstructed E (%)": row["reconstructed trace E %"],
            "Reconstructed H (%)": row["reconstructed trace H %"],
        }
        for row in result["interfaces"]
    ]
    tangent = [
        {
            "Face": row["face"],
            "Reference": row["reference"],
            "T E (%)": row["T trace tangent E %"],
            "T H (%)": row["T trace tangent H %"],
            "Reconstructed E (%)": row["reconstructed trace tangent E %"],
            "Reconstructed H (%)": row["reconstructed trace tangent H %"],
        }
        for row in result["interfaces"]
    ]
    coverage_percent = result["coverage_percent"]
    bem_description = (
        f"BEM: {result['triangles']} triangles, {result['unknowns']} unknowns; "
        f"algebraic residual {result['bem_residual']:.2g}."
        if result["bem_computed"]
        else f"BEM not computed: {'estimated ' if result['bem_unknowns_estimated'] else ''}"
        f"{result['unknowns']} unknowns exceed the {result['bem_max_unknowns']} dense-solver limit."
    )
    display(
        HTML(
            "<details><summary>Interface traces and solver details</summary>"
            + "<p>Three-component complex E/H, entrance glass side and exit air side; "
            "analytic pipeline faces versus BEM facets. T is the prescribed transmitted trace.</p>"
            + table_html(interface)
            + "<p>Tangential components, projected with the analytic normal:</p>"
            + table_html(tangent)
            + f"<p>{bem_description} Meridian coverage: {coverage_percent:.1f}%.</p>"
            + "<p>Chart-interpolated pipeline traces include sampling error. BEM facet offsets "
            "and fixed-mesh uncertainty also contribute; these discrepancies are not error bounds.</p>"
            + "</details>"
        )
    )


def size_study(name, body, source, interior, ratios=(3, 10, 30, 100)):
    """Scale the body at fixed wavelength, increasing the grid with its size.

    The sphere has an independent Mie reference at every size. BEM keeps
    h/lambda=0.20 and is computed only within the native dense-solver limit.
    Radius is the sphere radius or the nonspherical body's rim radius.
    """
    from scipy.fft import next_fast_len

    # Resolve the configured max_sine=0.95 band in every material, with a
    # ten-percent Nyquist margin. Wavelength and spacing are in vacuum units.
    max_index = max(source.region.material.n, interior.material.n)
    spacing = min(0.4, 0.9 / (2 * max_index * 0.95))
    results = []
    for ratio in ratios:
        width = max(12, 4 * ratio)
        count = next_fast_len(max(128, int(np.ceil(width / spacing))), real=True)
        results.append(
            compare_closed_body(
                name,
                scaled_body(body, ratio * source.wavelength0),
                source,
                interior,
                count=count,
                width=width,
                scale_domain=True,
                adaptive_bem=True,
            )
        )
    return results


def show_size_study(series):
    """Display size-dependent fields before the accuracy and timing tables."""
    from html import escape

    import matplotlib.pyplot as plt
    from IPython.display import HTML, display

    sphere = "Mie" in series[0]["focal"]
    reference = "Mie" if sphere else "BEM"
    columns = [
        ("Pipeline", "bulk"),
        (reference, "bulk"),
        ("Pipeline", "focal"),
        (reference, "focal"),
    ]
    if sphere:
        columns.append(("BEM", "focal"))
    fig, axes = plt.subplots(
        len(series),
        len(columns),
        figsize=(2.75 * len(columns), 2.35 * len(series)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, result in enumerate(series):
        ratio = result["radius_over_wavelength"]
        wavelength = result["wavelength"]
        colorbar_columns = {
            domain: max(
                col
                for col, (method, kind) in enumerate(columns)
                if kind == domain and method in result[domain]
            )
            for domain in ("bulk", "focal")
        }
        for col, (method, domain) in enumerate(columns):
            ax = axes[row, col]
            if method not in result[domain]:
                ax.set_facecolor("0.94")
                ax.text(
                    0.5,
                    0.5,
                    "BEM not computed\ndense limit exceeded",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                ax.set(xticks=[], yticks=[])
                continue
            values = result[domain][method].electric.norm2()
            maximum = max(
                np.max(field.electric.norm2()) for field in result[domain].values()
            )
            if domain == "bulk":
                image = np.full((len(result["z"]), len(result["x"])), np.nan)
                image[result["coverage"]] = values
                values = image.T
                extent = [
                    result["z"][0] / (ratio * wavelength),
                    result["z"][-1] / (ratio * wavelength),
                    result["x"][0] / (ratio * wavelength),
                    result["x"][-1] / (ratio * wavelength),
                ]
            else:
                extent = [
                    result["focal_axis"][0] / wavelength,
                    result["focal_axis"][-1] / wavelength,
                ] * 2
            image = ax.imshow(
                values,
                origin="lower",
                extent=extent,
                cmap="hot",
                vmin=0,
                vmax=maximum,
                aspect="auto" if domain == "bulk" else "equal",
            )
            if row == 0:
                ax.set_title(
                    f"{method}\n{'meridian' if domain == 'bulk' else 'focal plane'}",
                    fontsize=9,
                )
            if col == 0:
                ax.set_ylabel(f"R/λ = {ratio:g}\nx / R")
            if row == len(series) - 1:
                ax.set_xlabel("z / R" if domain == "bulk" else "x / λ")
            if col == colorbar_columns[domain]:
                fig.colorbar(image, ax=ax, shrink=0.65, label="|E|²")
    fig.suptitle(
        f"{series[0]['name']}: increasing R/λ; incident Ex = 1 V/m", fontsize=11
    )
    plt.show()
    plt.close(fig)

    def table(rows):
        keys = list(rows[0])

        def fmt(value):
            return (
                f"{value:.4g}"
                if isinstance(value, (float, np.floating))
                else escape(str(value))
            )

        return (
            "<div style='overflow-x:auto'><table><tr>"
            + "".join(f"<th>{escape(key)}</th>" for key in keys)
            + "</tr>"
            + "".join(
                "<tr>" + "".join(f"<td>{fmt(row[key])}</td>" for key in keys) + "</tr>"
                for row in rows
            )
            + "</table></div>"
        )

    rows, resources, traces = [], [], []
    for result in series:
        times = {r["method"]: r for r in result["timing"]}
        difference = next(
            (
                r
                for r in result["accuracy"]
                if r["candidate"] == "Pipeline" and r["reference"] == reference
            ),
            {},
        )
        rows.append(
            {
                "R/λ": result["radius_over_wavelength"],
                "Grid": result["count"],
                "Δ/λ": result["spacing_over_wavelength"],
                "Pipeline focal (s)": times["Pipeline"]["full focal calculation s"],
                "Pipeline all fields (s)": times["Pipeline"]["all requested fields s"],
                f"{reference} focal (s)": times.get(reference, {}).get(
                    "full focal calculation s", "—"
                ),
                "BEM focal (s)": times.get("BEM", {}).get(
                    "full focal calculation s", "—"
                ),
                "Focal E (%)": difference.get("focal E %", "—"),
                "Focal H (%)": difference.get("focal H %", "—"),
            }
        )
        resources.append(
            {
                "R/λ": result["radius_over_wavelength"],
                "BEM unknowns": ("≈ " if result["bem_unknowns_estimated"] else "")
                + str(result["unknowns"]),
                "One dense matrix (GiB)": result["bem_matrix_gib"],
                "BEM computed": result["bem_computed"],
                "Meridian coverage (%)": result["coverage_percent"],
                "Meridian E (%)": difference.get("meridian E %", "—"),
                "Meridian H (%)": difference.get("meridian H %", "—"),
            }
        )
        for record in result["interfaces"]:
            if record["reference"] != reference:
                continue
            traces.append(
                {
                    "R/λ": result["radius_over_wavelength"],
                    "Face": record["face"],
                    "Sampling": record["pipeline trace sampling"],
                    "T E (%)": record["T trace E %"],
                    "T H (%)": record["T trace H %"],
                    "T tangent E (%)": record["T trace tangent E %"],
                    "T tangent H (%)": record["T trace tangent H %"],
                    "Reconstructed E (%)": record["reconstructed trace E %"],
                    "Reconstructed H (%)": record["reconstructed trace H %"],
                    "Reconstructed tangent E (%)": record[
                        "reconstructed trace tangent E %"
                    ],
                    "Reconstructed tangent H (%)": record[
                        "reconstructed trace tangent H %"
                    ],
                }
            )
    display(
        HTML(
            "<p>Colors in each row/domain share one maximum: black is zero and white is that joint maximum. "
            "The complex fields used for differences are not normalized or fitted.</p>"
            "<b>Complete execution times and complex focal differences</b>"
            + table(rows)
        )
    )
    display(
        HTML(
            "<details><summary>Meridional/interfaces and BEM capacity</summary>"
            + table(resources)
            + (table(traces) if traces else "")
            + "<p>Unknown counts above R/λ=3 are area-based estimates; their matrix storage is also an estimate, "
            "excluding factorization/workspace. Dashes mean no independent reference was computed. "
            "BEM is unavailable for large nonspherical fields; their plane is chosen from the pipeline's axial maximum.</p>"
            + "</details>"
        )
    )
