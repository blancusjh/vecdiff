"""Reproducible cases for notebook 07; no notebook-only physics implementations."""

from time import perf_counter

import numpy as np

from ...fields import ElectricField, ElectromagneticField
from ...fields.spectrum import PlaneWaveExpansion
from ...geometry import Frame
from ...interfaces.dielectric_interface import DielectricInterface
from ...interfaces.optical_system import OpticalSystem
from ...media import Dielectric, HomogeneousRegion
from ...observables.diagnostics import interface_residuals
from ...sampling import CartesianGrid, PointGrid, SurfaceGrid
from ...surfaces import Plane, SphericalCap
from ...volumes import SphereVolume, VolumeBetweenSurfaces
from ..high_frequency import propagate_high_frequency
from .bem import DielectricBEM
from .boundary import boundary_meridian, closed_box_grid
from .diagnostics import boundary_residuals, closed_flux_residual, maxwell_residuals
from .fem import DielectricFEM
from .mesh import SurfaceMesh
from .meshing import surface_mesh, tetrahedral_mesh
from .mie import MieSphere


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
    """Notebook-05 geometry; weak=True holds sag near 2 lambda for large lenses."""
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


def fast_boundary_checks(solution, system):
    """Local T versus reconstructed transmitted fields, per interface.

    The reflected side remains the local Fresnel trace. These are transmission
    reconstruction checks, NOT a global two-sided scattering solution.
    Samples outside the clear aperture are excluded from jump diagnostics.
    """
    rows = []
    for index, (interface, aperture, encounter, spectrum, transports) in enumerate(
        zip(
            system.interfaces,
            system.apertures,
            solution.encounters[0],
            solution.interface_spectra[0],
            solution.transports,
        )
    ):
        Q = transports[1]
        inside = np.linalg.norm(Q.xy, axis=-1) < 0.95 * aperture
        grid = SurfaceGrid(
            Q.grid.points[inside], Q.grid.normals[inside], Q.grid.weights[inside]
        )

        def subset(field):
            return ElectromagneticField(
                ElectricField(
                    grid, field.electric.wavelength0, field.electric.values[inside]
                ),
                field.magnetic[inside],
            )

        incoming = ElectromagneticField(
            ElectricField(
                Q.grid,
                Q.wavelength0,
                encounter.incident.electric.values
                + encounter.reflected.electric.values,
            ),
            encounter.incident.magnetic + encounter.reflected.magnetic,
        )
        for kind, transmitted in (
            ("local_T", encounter.transmitted),
            ("reconstructed_transmission", Q.forward(spectrum)),
        ):
            residual = interface_residuals(
                subset(incoming),
                subset(transmitted),
                grid.normals,
                interface.minus.material,
                interface.plus.material,
                reference_electric=1,
                reference_index=1,
            )
            flux_error = abs(
                encounter.incident.power()
                + encounter.reflected.power()
                - transmitted.power()
            ) / max(abs(encounter.incident.power()), 1e-300)
            rows.append(
                dict(
                    interface=index,
                    check=kind,
                    flux_error=float(flux_error),
                    **residual,
                )
            )
    return rows


def comparison_geometry(geometry, diameter, wavelength0=532e-9, *, weak=False):
    if geometry != "sphere":
        return lens_case(diameter, shape=geometry, wavelength0=wavelength0, weak=weak)
    source, interior = illumination(wavelength0)
    radius = diameter * wavelength0 / 2
    volume = SphereVolume([0, 0, 0], radius)
    front = SphericalCap(radius, Frame([0, 0, -radius]))
    back = SphericalCap(-radius, Frame([0, 0, radius]))
    # The current forward eikonal path cannot describe returning/TIR branches.
    # Use a declared central chart; full-wave solvers still see the whole sphere.
    # 0.45R is a computational support, not the sphere's edge: light beyond it
    # would meet the sphere, so it is declared as a stop rather than passing by.
    aperture = 0.45 * radius
    system = OpticalSystem(
        (
            DielectricInterface(front, source.region, interior),
            DielectricInterface(back, interior, source.region),
        ),
        (aperture, aperture),
        (aperture, aperture),
    )
    return source, interior, volume, system


class ForwardComparison:
    """Historical bypass experiment; not the method reported by compare_case.

    The main API includes absorbing pupils. E_inc + E_lens - E_equal-index-pupil
    restores an approximate bypass field for this comparison only. This is not
    a replacement scattering solver, and retains neither rim scattering nor
    repeated internal reflections. ``raw`` is the unmodified operational result.
    """

    def __init__(self, source, system, grid):
        self.source, self.system = source, system
        self.raw = propagate_high_frequency(source, system, grid)
        control_region = HomogeneousRegion(
            "equal-index control", source.region.material
        )
        control = OpticalSystem(
            tuple(
                DielectricInterface(
                    i.surface,
                    source.region if j == 0 else control_region,
                    source.region
                    if j == len(system.interfaces) - 1
                    else control_region,
                )
                for j, i in enumerate(system.interfaces)
            ),
            system.apertures,
        )
        self.control = propagate_high_frequency(source, control, grid)

    def evaluate(self, grid, *, backend="auto"):
        field = self.raw.evaluate(grid, backend=backend)
        baseline = self.control.evaluate(grid, backend=backend)
        incident = self.source.evaluate(grid)
        return ElectromagneticField(
            ElectricField(
                grid,
                self.source.wavelength0,
                incident.electric.values
                + field.electric.values
                - baseline.electric.values,
            ),
            incident.magnetic + field.magnetic - baseline.magnetic,
        )


def compare_case(
    geometry,
    diameter,
    *,
    wavelength0=532e-9,
    grids=(32, 48),
    window_apertures=3.0,
    bem_subdivisions=(1, 2),
    bem_divisors=(4, 6),
    fem_sizes=(0.26, 0.18),
    run_bem=True,
    run_fem=True,
    weak=False,
    max_fem_unknowns=40000,
    interaction=None,
):
    """One geometry/parameter group, main method first at every resolution.

    Returned rows distinguish physical field differences, mesh changes,
    algebraic residuals and local forward-path checks. No automatic 'ground
    truth' designation is based on a solved linear system. Native references
    are bounded first-order implementations, not macroscopic accelerated solvers.
    window_apertures is the full FFT-cell width divided by the clear radius.
    Vary it independently of grids to distinguish periodic replicas from spacing.
    interaction is passed to propagate_high_frequency (for example
    experiments.orders.OrderInteraction(3)); None keeps the eikonal Fresnel T.
    """
    if not np.isfinite(window_apertures) or window_apertures <= 2:
        raise ValueError("window_apertures must be finite and greater than two")
    source, interior, volume, system = comparison_geometry(
        geometry, diameter, wavelength0, weak=weak
    )
    radius = diameter * wavelength0 / 2
    exit_z = system.interfaces[-1].surface.frame.origin[2]
    z = exit_z + min(0.4 * wavelength0, 0.5 * radius)
    probes = PointGrid(
        np.array(
            [
                [0, 0, z],
                [0.2 * radius, 0, z],
                [-0.2 * radius, 0.1 * radius, z],
                [0, 0.25 * radius, z],
            ]
        )
    )
    output = dict(
        geometry=geometry,
        diameter_lambda=diameter,
        volume=volume,
        system=system,
        source=source,
        interior=interior,
        probes=probes,
        solutions={},
        rows=[],
        boundary=[],
        window_apertures=window_apertures,
    )
    rows = output["rows"]
    field_history = {}
    exact = MieSphere(source, volume, interior) if geometry == "sphere" else None
    exact_field = exact.evaluate(probes).electric.values if exact is not None else None

    def record(name, resolution, solution, seconds, mesh_seconds=0):
        start = perf_counter()
        field = solution.evaluate(probes).electric.values
        evaluation = perf_counter() - start
        previous = field_history.get(name)
        change = (
            None
            if previous is None
            else float(np.linalg.norm(field - previous) / np.linalg.norm(field))
        )
        field_history[name] = field
        output["solutions"][name] = solution
        row = dict(
            geometry=geometry,
            diameter_lambda=diameter,
            method=name,
            resolution=resolution,
            setup_seconds=seconds,
            mesh_seconds=mesh_seconds,
            evaluation_seconds=evaluation,
            mesh_change=change,
            field_gap=None,
            status="refinement required",
        )
        if exact_field is not None:
            row["field_gap"] = float(
                np.linalg.norm(field - exact_field) / np.linalg.norm(exact_field)
            )
        if name == "Ours":
            row["window_apertures"] = window_apertures
            row["transform_backend"] = "Cartesian FFT ASM; variable-height Q"
            row["field_definition"] = (
                "unmodified R Q^-1 T Q R; edges keep outside light, stops cut"
                if interaction is None
                else f"R Q^-1 T Q R with {type(interaction).__name__}"
                f"(order {getattr(interaction, 'order', '?')}); edges keep outside light, stops cut"
            )
            checks = fast_boundary_checks(solution, system)
            reconstructed = [
                r for r in checks if r["check"] == "reconstructed_transmission"
            ]
            for key in (
                "tangential_E",
                "tangential_H",
                "normal_D",
                "normal_B",
                "flux_error",
            ):
                row[key] = max(r[key] for r in reconstructed)
            row["boundary_scope"] = "forward trace + local reflection"
            row["flux_scope"] = "transmission reconstruction"
            row["inverse_residual"] = max(
                r.relative_residual for r in solution.inverse_reports[0]
            )
            row["status"] = (
                "central chart behind a 0.45R stop; the reference is the whole sphere"
                if geometry == "sphere"
                else "single pass: no rim scattering or internal returns"
            )
            for r in checks:
                output["boundary"].append(dict(method=name, resolution=resolution, **r))
        else:
            boundary = (
                boundary_meridian(volume, 23).surface(8, phase=0.17)
                if name == "Mie"
                else solution.mesh.boundary_grid()
            )
            row.update(boundary_residuals(solution, boundary))
            row["boundary_scope"] = (
                "full boundary; analytical sphere"
                if name == "Mie"
                else "full boundary; mesh face centres"
            )
            row["algebraic_residual"] = getattr(solution, "algebraic_residual", None)
            row["unknowns"] = getattr(solution, "unknowns", None)
            row["assembly_seconds"] = getattr(solution, "assembly_seconds", None)
            row["solve_seconds"] = getattr(solution, "solve_seconds", None)
            center = np.array([0, 0, 0 if geometry == "sphere" else exit_z / 2])
            if geometry == "sphere":
                enclosing = boundary_meridian(
                    SphereVolume(center, radius + 0.2 * wavelength0), 24
                ).surface(12, phase=0.13)
            else:
                enclosing = closed_box_grid(
                    center, np.array([radius, radius, exit_z / 2]) + 0.2 * wavelength0
                )
            row["flux_error"] = abs(closed_flux_residual(solution, enclosing))
            row["flux_scope"] = "net closed-surface total power"
            if name == "Mie":
                finer = MieSphere(source, volume, interior, order=solution.order + 6)
                row["mesh_change"] = float(
                    np.linalg.norm(finer.evaluate(probes).electric.values - field)
                    / np.linalg.norm(field)
                )
                row["status"] = "analytic multipole reference"
        if name == "FEM":
            row["maxwell_scope"] = (
                "weak discrete equations; strong curl-H not applicable"
            )
            row["maxwell_max"] = None
        else:
            # Avoid an unbounded macro Mie flux integration / derivative study.
            diagnostics = maxwell_residuals(
                solution.evaluate, probes.points[:2], wavelength0=wavelength0, index=1
            )
            row.update(diagnostics)
            row["maxwell_max"] = max(diagnostics.values())
            row["maxwell_scope"] = "homogeneous exterior finite differences"
        rows.append(row)

    for count in grids:
        aperture = system.apertures[0]
        axis = (np.arange(count) - count // 2) * window_apertures * aperture / count
        grid = CartesianGrid(axis, axis, system.interfaces[0].surface.frame)
        start = perf_counter()
        try:
            solution = propagate_high_frequency(
                source, system, grid, interaction=interaction
            )
        except ValueError as exc:
            if "TIR" not in str(exc) and "incident" not in str(exc):
                raise
            rows.append(
                dict(
                    geometry=geometry,
                    diameter_lambda=diameter,
                    method="Ours",
                    resolution=count,
                    status=f"unsupported forward branch: {exc}",
                )
            )
            continue
        record("Ours", count, solution, perf_counter() - start)
    if run_bem:
        resolutions = bem_subdivisions if geometry == "sphere" else bem_divisors
        for resolution in resolutions:
            start = perf_counter()
            mesh = (
                SurfaceMesh.sphere(volume, resolution)
                if geometry == "sphere"
                else surface_mesh(volume, 2 * radius / resolution)
            )
            mesh_seconds = perf_counter() - start
            start = perf_counter()
            solution = DielectricBEM(source, mesh, interior)
            record(
                "BEM",
                len(mesh.triangles),
                solution,
                perf_counter() - start,
                mesh_seconds,
            )
    else:
        rows.append(
            dict(
                geometry=geometry,
                diameter_lambda=diameter,
                method="BEM",
                status="not run: dense mesh budget",
            )
        )
    if run_fem:
        for h in fem_sizes:
            start = perf_counter()
            mesh = tetrahedral_mesh(
                volume,
                edge_size=h * wavelength0,
                physical_padding=0.65 * wavelength0,
                pml_width=0.4 * wavelength0,
            )
            mesh_seconds = perf_counter() - start
            start = perf_counter()
            try:
                solution = DielectricFEM(
                    source, mesh, interior, max_unknowns=max_fem_unknowns
                )
            except ValueError as exc:
                if "unknowns" not in str(exc):
                    raise
                rows.append(
                    dict(
                        geometry=geometry,
                        diameter_lambda=diameter,
                        method="FEM",
                        resolution=h,
                        status=str(exc),
                        mesh_seconds=mesh_seconds,
                    )
                )
                continue
            record("FEM", h, solution, perf_counter() - start, mesh_seconds)
    else:
        rows.append(
            dict(
                geometry=geometry,
                diameter_lambda=diameter,
                method="FEM",
                status="not run: volume mesh budget",
            )
        )
    if exact is not None:
        if diameter <= 20:
            start = perf_counter()
            exact = MieSphere(source, volume, interior)
            record("Mie", exact.order, exact, perf_counter() - start)
        else:
            # Mie multipoles remain affordable; dense surface flux quadrature does not.
            start = perf_counter()
            exact = MieSphere(source, volume, interior)
            elapsed = perf_counter() - start
            start = perf_counter()
            exact.evaluate(probes)
            evaluation = perf_counter() - start
            q = exact.efficiencies()
            output["solutions"]["Mie"] = exact
            rows.append(
                dict(
                    geometry=geometry,
                    diameter_lambda=diameter,
                    method="Mie",
                    resolution=exact.order,
                    setup_seconds=elapsed,
                    evaluation_seconds=evaluation,
                    field_gap=0.0,
                    flux_error=abs(q["extinction"] - q["scattering"]) / q["extinction"],
                    flux_scope="optical theorem",
                    status="analytic; boundary sampling not run at this size",
                )
            )
    # Lens differences use BEM as a comparator, never a certified exact answer.
    if exact_field is None and "BEM" in field_history:
        reference = field_history["BEM"]
        for name, solution in output["solutions"].items():
            finest = next(
                r
                for r in reversed(rows)
                if r["method"] == name and "setup_seconds" in r
            )
            finest["field_gap"] = float(
                np.linalg.norm(field_history[name] - reference)
                / np.linalg.norm(reference)
            )
    return output
