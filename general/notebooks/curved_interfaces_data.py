"""Data for notebook 11 (curved interfaces): lenses against BEM, spheres against Mie.

Lenses (6 wavelengths): each is solved by

* our forward chain with the per-plane-wave interaction at order 0 and 3
  (``OrderInteraction``); no stops, so light passing outside the lens is kept;
* the native dielectric BEM at two element sizes (lambda/3 and lambda/4);
* Stratton-Chu radiation of our own exit-face field, as an independent check of
  the surface-to-plane step (Q^-1).

A sphere under a Gaussian beam wider than itself: Lorenz-Mie on the whole
y = 0 plane (``mie_picture``) and our field on the same grid (``sphere_ours``:
the sphere met on its closed surface and the field outside carried through
intermediate planes, ``experiments.intermediate``) at order 0 and 3.

Results are cached in output/notebook11 under a fingerprint of every numerical
input and of the solver sources, so a changed solver or parameter recomputes.

usage: curved_interfaces_study.py [lens name ... | sphere]
"""

import hashlib
import inspect
import json
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import vecdiff_general as o
from vecdiff_general.comparisons.surface_radiation import SurfaceRadiation
from vecdiff_general.fields import ElectricField, ElectromagneticField
from vecdiff_general.observables.maxwell import interface_l2_residuals
from vecdiff_general.propagation import high_frequency, operators
from vecdiff_general.propagation.experiments import orders, polar_lens, taylor_interface
from vecdiff_general.propagation.experiments.orders import OrderInteraction
from vecdiff_general.propagation.high_frequency import propagate_high_frequency
from vecdiff_general.propagation.references import DielectricBEM, bem, meshing
from vecdiff_general.propagation.references.meshing import surface_mesh
from vecdiff_general.sampling import SurfaceGrid
from vecdiff_general.volumes import VolumeBetweenSurfaces

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "output" / "notebook11"
WAVELENGTH = 532e-9
INDEX = 1.5
DIAMETER = 6.0  # wavelengths
BEM_SIZES = (1 / 3, 1 / 4)  # element edge / wavelength; the finer is the reference
GRID, WINDOW = 256, 24  # FFT samples; window width / clear radius
# name: (front radius, back radius, centre thickness), wavelengths; None = plane.
LENSES = {
    # Thick lenses, a^2/(lambda T) = 4-5: outside the single-pass validity range.
    "plano-convex": (None, -8.0, 1.8),
    "convex-plano": (8.0, None, 1.8),
    "biconvex": (10.0, -10.0, 2.2),
    "asymmetric biconvex": (20.0, -7.5, 2.2),
    # Thin lenses, a^2/(lambda T) = 12.9: the order comparisons of notebook 11.
    "thin plano-convex": (None, -10.0, 0.7),
    "thin convex-plano": (10.0, None, 0.7),
    "thin biconvex": (20.0, -20.0, 0.7),
    "thin asymmetric biconvex": (40.0, -13.0, 0.7),
    # Thickness sweep at fixed curvature.
    "biconvex R20, T 0.6": (20.0, -20.0, 0.6),
    "biconvex R20, T 0.9": (20.0, -20.0, 0.9),
    "biconvex R20, T 1.2": (20.0, -20.0, 1.2),
    "biconvex R20, T 1.8": (20.0, -20.0, 1.8),
    "biconvex R20, T 2.4": (20.0, -20.0, 2.4),
}
MAIN = (
    "thin plano-convex",
    "thin convex-plano",
    "thin biconvex",
    "thin asymmetric biconvex",
)
THICK = ("plano-convex", "convex-plano", "biconvex", "asymmetric biconvex")
SWEEP = tuple(name for name in LENSES if name.startswith("biconvex R20"))


def element_fresnel(name):
    """a^2 / (lambda T): large when the rim zone sqrt(lambda T) is small."""
    return (DIAMETER / 2) ** 2 / LENSES[name][2]


def surfaces(name):
    front_r, back_r, thickness = LENSES[name]
    lam = WAVELENGTH
    front = o.Plane() if front_r is None else o.SphericalCap(front_r * lam)
    frame = o.Frame([0, 0, thickness * lam])
    back = o.Plane(frame) if back_r is None else o.SphericalCap(back_r * lam, frame)
    return front, back


def lens(name):
    """(source, system, volume) for a unit x-polarized plane wave along +z."""
    air = o.HomogeneousRegion("air", o.Dielectric(1.0))
    glass = o.HomogeneousRegion("glass", o.Dielectric(INDEX))
    front, back = surfaces(name)
    a = DIAMETER / 2 * WAVELENGTH
    system = o.OpticalSystem(
        (
            o.DielectricInterface(front, air, glass),
            o.DielectricInterface(back, glass, air),
        ),
        apertures=(a, a),
    )
    source = o.PlaneWaveExpansion(
        [[0, 0, 2 * np.pi / WAVELENGTH]], [[1.0, 0, 0]], [0, 0, 0], WAVELENGTH, air
    )
    return source, system, VolumeBetweenSurfaces(front, back, a)


def paraxial_focus(name):
    """Back focal distance from the exit vertex (thick-lens matrices), metres."""
    front, back = surfaces(name)
    n = INDEX
    d = back.frame.origin[2]
    p1 = 0.0 if isinstance(front, o.Plane) else (n - 1) / front.radius
    p2 = 0.0 if isinstance(back, o.Plane) else (1 - n) / back.radius
    power = p1 + p2 - d / n * p1 * p2
    if power == 0:
        return np.inf
    return (1 - d / n * p1) / power


def probes(name):
    """Axis, meridional plane y = 0 and nothing else; the focal plane follows BEM."""
    lam, a = WAVELENGTH, DIAMETER / 2 * WAVELENGTH
    exit_z = LENSES[name][2] * lam
    end = exit_z + max(2.0 * paraxial_focus(name), 12 * lam)
    z_axis = np.linspace(exit_z + 0.5 * lam, end, 801)
    x_map = np.linspace(-1.4 * a, 1.4 * a, 81)
    z_map = np.linspace(exit_z + 0.5 * lam, end, 121)
    return z_axis, x_map, z_map


def focal_window(name, focus):
    """Focal-plane axis wide enough for three secondary maxima on each side.

    For an Airy focus the third side lobe peaks at 1.85 λ/NA and the fourth
    zero lies at 2.12 λ/NA (NA = a / sqrt(a² + d²), d the detector distance
    behind the exit vertex); 2.3 λ/NA includes it. These small lenses have a low
    Fresnel number: the detector (the axial intensity maximum) lies far short
    of the paraxial focus, the focal spot is one broad peak, and beyond about
    |x| = a the plane is dominated by light passing outside the lens, whose edge
    fringes are the secondary maxima. The window therefore also reaches 2a + 6λ.
    Sampled at λ/6 or finer. The notebook counts the maxima it shows.
    """
    lam, a = WAVELENGTH, DIAMETER / 2 * WAVELENGTH
    d = focus - LENSES[name][2] * lam
    na = a / np.hypot(a, d)
    half = max(3 * lam, 2.3 * lam / na, 2 * a + 6 * lam)
    step = min(lam / (8 * na), lam / 6)
    count = 2 * int(np.ceil(half / step)) + 1
    return np.linspace(-half, half, count)


def _digest(*parts):
    return hashlib.sha256("".join(parts).encode()).hexdigest()[:16]


def _parameters(name):
    return json.dumps(
        dict(
            name=name,
            lens=LENSES[name],
            D=DIAMETER,
            n=INDEX,
            lam=WAVELENGTH,
            bem=BEM_SIZES,
        ),
        sort_keys=True,
    )


def bem_fingerprint(name):
    """BEM fields depend only on the lens, the probes and the BEM sources."""
    return _digest(
        _parameters(name),
        inspect.getsource(bem),
        inspect.getsource(meshing),
        inspect.getsource(probes),
        inspect.getsource(bem_fields),
        inspect.getsource(focal_window),
    )


def ours_fingerprint(name):
    modules = (high_frequency, operators, orders, polar_lens, taylor_interface)
    return _digest(
        bem_fingerprint(name),
        json.dumps(dict(grid=GRID, window=WINDOW)),
        "".join(inspect.getsource(m) for m in modules),
        Path(__file__).read_text(),
    )


def _points(*arrays):
    return o.PointGrid(np.concatenate([a.reshape(-1, 3) for a in arrays]))


def _relative(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _disk_power(field_values, magnetic, axis):
    """Integrated S_z over the focal disk of radius 1.5 wavelengths, watts."""
    s = 0.5 * np.real(np.cross(field_values, magnetic.conj()))[..., 2]
    xx, yy = np.meshgrid(axis, axis)
    inside = np.hypot(xx, yy) <= 1.5 * WAVELENGTH
    return float(np.sum(s * inside) * (axis[1] - axis[0]) ** 2)


def exit_boundary(solution, system):
    """Local interface jumps on the exit face (inside the edge), relative."""
    encounter = solution.encounters[0][-1]
    grid = encounter.grid
    inside = (
        np.linalg.norm(solution.transports[-1][1].xy, axis=-1) <= system.apertures[-1]
    )
    ids = np.flatnonzero(inside.ravel())
    cut_grid = SurfaceGrid(
        grid.points.reshape(-1, 3)[ids],
        grid.normals.reshape(-1, 3)[ids],
        grid.weights.reshape(-1)[ids],
    )

    def cut(field):
        return ElectromagneticField(
            ElectricField(
                cut_grid, WAVELENGTH, field.electric.values.reshape(-1, 3)[ids]
            ),
            field.magnetic.reshape(-1, 3)[ids],
        )

    incident, reflected = cut(encounter.incident), cut(encounter.reflected)
    minus = ElectromagneticField(
        ElectricField(
            cut_grid, WAVELENGTH, incident.electric.values + reflected.electric.values
        ),
        incident.magnetic + reflected.magnetic,
    )
    interface = system.interfaces[-1]
    return interface_l2_residuals(
        minus,
        cut(encounter.transmitted),
        incident,
        cut_grid,
        n_minus=interface.minus.material.n,
        n_plus=interface.plus.material.n,
    )


def convergent_share(solution, system, interaction):
    """Share of the exit's incident spectral power within the convergent band."""
    back = system.interfaces[-1].surface
    if isinstance(back, o.Plane):
        return None
    # Tables are keyed by (radius, wavelength, n_in, n_out, curvature).
    limit = next(v for key, v in interaction.limits.items() if key[2] == INDEX)
    internal = solution.interface_spectra[0][0].expansion
    q = solution.transports[-1][0]
    normals = q.grid.normals.reshape(-1, 3)
    keep = np.linalg.norm(q.xy.reshape(-1, 2), axis=-1) <= system.apertures[-1]
    weight = np.sum(abs(internal.amplitudes) ** 2, -1)
    occupied = weight > 0
    k = internal.wavevectors[occupied].real
    d = k / np.linalg.norm(k, axis=-1, keepdims=True)
    w = weight[occupied]
    cosine = np.clip(normals[keep] @ d.T, -1, 1)
    active = np.arccos(cosine) <= limit
    return float(np.sum(active * w[None]) / (active.shape[0] * np.sum(w)))


def _grids(name):
    z_axis, x_map, z_map = probes(name)
    axis_pts = np.stack((0 * z_axis, 0 * z_axis, z_axis), -1)
    zz, xx = np.meshgrid(z_map, x_map)
    return z_axis, x_map, z_map, axis_pts, np.stack((xx, 0 * xx, zz), -1)


def bem_fields(name, verbose=True):
    """BEM at both element sizes on the axis, the meridian and the focal plane.

    The detector plane is the on-axis maximum of the finer BEM more than one
    wavelength behind the exit vertex.
    """
    lam = WAVELENGTH
    source, system, volume = lens(name)
    z_axis, x_map, z_map, axis_pts, map_pts = _grids(name)
    arrays, info, solvers = {}, {}, {}
    for size in BEM_SIZES:
        label = f"BEM λ/{round(1 / size)}"
        start = perf_counter()
        solvers[label] = DielectricBEM(
            source,
            surface_mesh(volume, size * lam),
            system.interfaces[0].plus,
            max_unknowns=40000,
        )
        values = solvers[label].evaluate(_points(axis_pts, map_pts))
        n_axis = len(z_axis)
        arrays[f"{label}|axis"] = values.electric.values[:n_axis]
        arrays[f"{label}|axis_h"] = values.magnetic[:n_axis]
        arrays[f"{label}|map"] = values.electric.values[n_axis:].reshape(map_pts.shape)
        info[f"{label} unknowns"] = solvers[label].unknowns
        info[f"{label} seconds"] = perf_counter() - start
        if verbose:
            print(
                f"  {name}: {label}, {solvers[label].unknowns} unknowns, "
                f"{info[f'{label} seconds']:.0f} s",
                flush=True,
            )
    reference = f"BEM λ/{round(1 / BEM_SIZES[-1])}"
    intensity = np.sum(abs(arrays[f"{reference}|axis"]) ** 2, -1)
    beyond = z_axis > (LENSES[name][2] + 1.0) * lam
    focus = float(z_axis[np.argmax(np.where(beyond, intensity, 0))])
    focal_axis = focal_window(name, focus)
    fx, fy = np.meshgrid(focal_axis, focal_axis)
    focal_pts = np.stack((fx, fy, np.full_like(fx, focus)), -1)
    for label, solver in solvers.items():
        values = solver.evaluate(o.PointGrid(focal_pts))
        arrays[f"{label}|focal"] = values.electric.values
        arrays[f"{label}|focal_h"] = values.magnetic
    arrays.update(
        z_axis=z_axis,
        x_map=x_map,
        z_map=z_map,
        focal_axis=focal_axis,
        detector_z=np.array(focus),
    )
    return arrays, info


def _slug(name):
    """File-safe name: letters, digits and underscores only (no dots)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def _cached(stem, build):
    CACHE.mkdir(parents=True, exist_ok=True)
    # Append extensions explicitly: with_suffix would treat a dot in a lens
    # name as an extension and make different lenses share one file.
    npz, meta = Path(f"{stem}.npz"), Path(f"{stem}.json")
    if npz.exists() and meta.exists():
        with np.load(npz) as data:
            return {key: data[key] for key in data.files}, json.loads(meta.read_text())
    arrays, info = build()
    np.savez_compressed(npz, **arrays)
    meta.write_text(json.dumps(info, indent=2) + "\n")
    return arrays, info


def ours_fields(name, reference, verbose=True):
    """Our order-0 and order-3 runs and their Stratton-Chu checks."""
    lam = WAVELENGTH
    source, system, _ = lens(name)
    a = system.apertures[0]
    z_axis, _, _, axis_pts, map_pts = _grids(name)
    focal_axis, focus = reference["focal_axis"], float(reference["detector_z"])
    fx, fy = np.meshgrid(focal_axis, focal_axis)
    focal_pts = np.stack((fx, fy, np.full_like(fx, focus)), -1)
    grid_axis = (np.arange(GRID) - GRID // 2) * WINDOW * a / GRID
    grid = o.CartesianGrid(grid_axis, grid_axis)
    arrays, info = {}, {}
    for order in (0, 3):
        label = f"Ours N={order}"
        start = perf_counter()
        # complete_curl only changes H at order 0 (for the boundary audit).
        interaction = OrderInteraction(order, complete_curl=True)
        solution = propagate_high_frequency(
            source, system, grid, interaction=interaction
        )
        values = solution.evaluate(
            _points(axis_pts, map_pts, focal_pts), backend="nufft"
        )
        n_axis, n_map = len(z_axis), map_pts[..., 0].size
        e, h = values.electric.values, values.magnetic
        arrays[f"{label}|axis"] = e[:n_axis]
        arrays[f"{label}|axis_h"] = h[:n_axis]
        arrays[f"{label}|map"] = e[n_axis : n_axis + n_map].reshape(map_pts.shape)
        arrays[f"{label}|focal"] = e[n_axis + n_map :].reshape(focal_pts.shape)
        arrays[f"{label}|focal_h"] = h[n_axis + n_map :].reshape(focal_pts.shape)
        info[f"{label} seconds"] = perf_counter() - start
        info[f"{label} exit jumps"] = exit_boundary(solution, system)
        if order:
            info["convergent incidence (deg)"] = {
                str(key): np.degrees(v) for key, v in interaction.limits.items()
            }
            info["share of exit spectral power with order 3 active"] = convergent_share(
                solution, system, interaction
            )
        # Stratton-Chu radiation of our exit-face field (within a + 8 lambda),
        # plus the light that passed outside the lens, which is exact ASM.
        start = perf_counter()
        exit_trace = solution.encounters[0][-1].transmitted
        q = solution.transports[-1][1]
        keep = np.flatnonzero((np.linalg.norm(q.xy, axis=-1) <= a + 8 * lam).ravel())
        surface = SurfaceGrid(
            q.grid.points.reshape(-1, 3)[keep],
            q.grid.normals.reshape(-1, 3)[keep],
            q.grid.weights.reshape(-1)[keep],
        )
        cut = ElectromagneticField(
            ElectricField(
                surface, lam, exit_trace.electric.values.reshape(-1, 3)[keep]
            ),
            exit_trace.magnetic.reshape(-1, 3)[keep],
        )
        axis_grid = o.PointGrid(axis_pts)
        radiated = SurfaceRadiation.from_field(cut, q.region).evaluate(axis_grid)
        passing = solution.exterior_spectra[0][0].evaluate(axis_grid, backend="nufft")
        arrays[f"Stratton–Chu N={order}|axis"] = (
            radiated.electric.values + passing.electric.values
        )
        info[f"Stratton–Chu N={order} seconds"] = perf_counter() - start
        if verbose:
            print(f"  {name}: {label} {info[f'{label} seconds']:.0f} s", flush=True)
    return arrays, info


def case(name, verbose=True):
    """(summary, arrays) for one lens; BEM and our runs are cached separately."""
    slug = _slug(name)
    bem_arrays, bem_info = _cached(
        CACHE / f"bem_{slug}_{bem_fingerprint(name)}", lambda: bem_fields(name, verbose)
    )
    ours_arrays, ours_info = _cached(
        CACHE / f"ours_{slug}_{ours_fingerprint(name)}",
        lambda: ours_fields(name, bem_arrays, verbose),
    )
    arrays = {**bem_arrays, **ours_arrays}
    lam = WAVELENGTH
    z_axis = arrays["z_axis"]
    window = z_axis > (LENSES[name][2] + 1.0) * lam
    reference = f"BEM λ/{round(1 / BEM_SIZES[-1])}"
    ref_axis = arrays[f"{reference}|axis"]
    labels = sorted({key.split("|")[0] for key in arrays if "|" in key})
    metrics = {}
    for label in labels:
        axis = arrays[f"{label}|axis"]
        metrics[f"{label} axial error vs {reference}"] = _relative(
            axis[window], ref_axis[window]
        )
        metrics[f"{label} peak |E|^2"] = float(
            np.max(np.sum(abs(axis[window]) ** 2, -1))
        )
        if f"{label}|focal_h" in arrays:
            metrics[f"{label} focal disk power"] = _disk_power(
                arrays[f"{label}|focal"],
                arrays[f"{label}|focal_h"],
                arrays["focal_axis"],
            )
    metrics["axial change N=0 to N=3"] = _relative(
        arrays["Ours N=3|axis"][window], arrays["Ours N=0|axis"][window]
    )
    metrics["BEM refinement change"] = _relative(
        arrays[f"BEM λ/{round(1 / BEM_SIZES[0])}|axis"][window], ref_axis[window]
    )
    for order in (0, 3):
        metrics[f"Stratton–Chu vs Q^-1 N={order}"] = _relative(
            arrays[f"Stratton–Chu N={order}|axis"][window],
            arrays[f"Ours N={order}|axis"][window],
        )
    for key in (
        "Ours N=0 exit jumps",
        "Ours N=3 exit jumps",
        "share of exit spectral power with order 3 active",
    ):
        metrics[key] = ours_info.get(key)
    result = dict(
        name=name,
        lens=LENSES[name],
        diameter_lambda=DIAMETER,
        paraxial_focus_lambda=paraxial_focus(name) / lam,
        detector_z_lambda=float(arrays["detector_z"]) / lam,
        exit_vertex_lambda=LENSES[name][2],
        convergent_incidence_deg=ours_info.get("convergent incidence (deg)", {}),
        metrics=metrics,
        timings={
            k: v for k, v in {**bem_info, **ours_info}.items() if k.endswith("seconds")
        },
    )
    return result, arrays


def sphere_control(verbose=True):
    """BEM (lambda/3) versus Mie for a glass sphere of diameter 6 lambda, on axis."""
    from vecdiff_general.propagation.references import MieSphere
    from vecdiff_general.volumes import SphereVolume

    def build():
        lam = WAVELENGTH
        radius = DIAMETER / 2 * lam
        air = o.HomogeneousRegion("air", o.Dielectric(1.0))
        glass = o.HomogeneousRegion("glass", o.Dielectric(INDEX))
        source = o.PlaneWaveExpansion(
            [[0, 0, 2 * np.pi / lam]], [[1.0, 0, 0]], [0, 0, 0], lam, air
        )
        z = np.linspace(radius + lam, radius + 20 * lam, 400)
        points = o.PointGrid(np.stack((0 * z, 0 * z, z), -1))
        volume = SphereVolume([0, 0, 0], radius)
        start = perf_counter()
        solver = DielectricBEM(
            source, surface_mesh(volume, lam / 3), glass, max_unknowns=40000
        )
        arrays = dict(
            z=z,
            mie=MieSphere(source, volume, glass).evaluate(points).electric.values,
            bem=solver.evaluate(points).electric.values,
        )
        if verbose:
            print(
                f"  sphere control: {solver.unknowns} unknowns, "
                f"{perf_counter() - start:.0f} s",
                flush=True,
            )
        return arrays, dict(unknowns=solver.unknowns)

    key = _digest(
        _parameters("plano-convex"),
        inspect.getsource(bem),
        inspect.getsource(meshing),
        inspect.getsource(sphere_control),
    )
    return _cached(CACHE / f"sphere_control_{key}", build)


# Exact picture of a sphere under a beam wider than itself. radius, waist,
# grid step (vacuum wavelengths); the waist is at the sphere centre.
MIE_PICTURE = (5.0, 6.0, 1 / 12)


def _axial_total(series, points, incident=None):
    """Incident, scattered and total E of an AxialBeamSphere at points.

    Inside the sphere the scattered part is left at zero and the total is the
    interior field. ``incident`` replaces the continuous beam outside (for
    example by the lattice beam of the iteration).
    """
    r = np.linalg.norm(points - series.center, axis=-1)
    inside = r <= series.radius
    shape = (len(points), 3)
    parts = dict(
        incident=np.zeros(shape, complex),
        scattered=np.zeros(shape, complex),
        total=np.zeros(shape, complex),
    )
    parts["incident"] = series.fields(points, "exterior", ["incident"])[0][0]
    if incident is not None:
        parts["incident"][~inside] = incident[~inside]
    parts["scattered"][~inside] = series.fields(points[~inside], "exterior", ["mie"])[0][0]
    parts["total"][~inside] = parts["incident"][~inside] + parts["scattered"][~inside]
    parts["total"][inside] = series.fields(points[inside], "interior", ["mie"])[0][0]
    return parts


def mie_picture(verbose=True):
    """|E| of the incident, scattered and total fields on the y = 0 plane."""
    from vecdiff_general.propagation.references import axial_beam
    from vecdiff_general.propagation.references.axial_beam import AxialBeamSphere

    radius, waist, step = MIE_PICTURE

    def build():
        start = perf_counter()
        series = AxialBeamSphere(waist, radius, radius, radius, INDEX)
        x = np.arange(-4 * radius, 4 * radius + 1e-9, step)
        z = np.arange(-3 * radius, 7 * radius + 1e-9, step)
        zz, xx = np.meshgrid(z, x, indexing="ij")
        points = np.stack((xx, 0 * xx, zz), -1).reshape(-1, 3)
        parts = _axial_total(series, points)
        arrays = {k: np.linalg.norm(v, axis=-1).reshape(zz.shape).astype(np.float32)
                  for k, v in parts.items()}
        arrays.update(x=x, z=z)
        info = dict(radius=radius, waist=waist, step=step, order=series.order,
                    seconds=perf_counter() - start)
        if verbose:
            print(f"  Mie picture: {points.shape[0]} points, {info['seconds']:.0f} s", flush=True)
        return arrays, info

    fingerprint = _digest(
        json.dumps(dict(picture=MIE_PICTURE, index=INDEX)),
        inspect.getsource(axial_beam),
        inspect.getsource(_axial_total),
    )
    return _cached(CACHE / f"mie_picture_{fingerprint}", build)


# Our field for the same picture: internal passes computed (pass 0, the
# external reflection and shadow, is always included) and the half width of
# the lattice of the intermediate planes, in vacuum wavelengths.
SPHERE_PASSES = 1
SPHERE_HALF = 48.0


def sphere_beam(radius):
    """Air, glass, the sphere centre and the incident beam as plane waves.

    E_x = exp(-rho^2 / w^2) on the plane through the centre, as the FFT
    spectrum of a 48-wavelength lattice (images below 1e-9 in the picture);
    numerically empty modes (below 1e-12 of the largest) are dropped.
    """
    from vecdiff_general.propagation.operators import SurfaceTransport

    air = o.HomogeneousRegion("air", o.Dielectric(1.0))
    glass = o.HomogeneousRegion("glass", o.Dielectric(INDEX, "glass"))
    center = np.array([0.0, 0.0, radius])
    axis = np.arange(-96, 97) / 4
    frame = o.Frame(center)
    plane = o.CartesianGrid(axis, axis, frame)
    flat = SurfaceTransport(plane, o.Plane(frame), air, 1.0, max_sine=0.95)
    x, y = plane.points[..., 0], plane.points[..., 1]
    electric = np.zeros(plane.shape + (3,), complex)
    electric[..., 0] = np.exp(-(x * x + y * y) / MIE_PICTURE[1] ** 2)
    trace = o.ElectromagneticField(
        o.ElectricField(flat.grid, 1.0, electric), np.zeros_like(electric)
    )
    spectrum, _ = flat.inverse(trace)
    modes = spectrum.expansion
    size = np.linalg.norm(modes.amplitudes, axis=-1)
    keep = size >= 1e-12 * size.max()
    beam = o.PlaneWaveExpansion(
        modes.wavevectors[keep], modes.amplitudes[keep], modes.origin, 1.0, air
    )
    return air, glass, center, beam


def sphere_ours(order, verbose=True):
    """Our |E| on the grid of ``mie_picture``: incident, scattered and total.

    The sphere is met by the beam and iterated for SPHERE_PASSES internal
    passes on the closed surface; the field outside is carried through the
    intermediate planes (``experiments.intermediate``). Only our operators
    are used: the per-wave interaction of the given order, Q^-1 and ASM.
    """
    from vecdiff_general.propagation.experiments import intermediate
    from vecdiff_general.propagation.experiments.intermediate import (
        IntermediateSphere,
        PlaneMarch,
    )

    radius, _, step = MIE_PICTURE

    def build():
        start = perf_counter()
        air, glass, center, beam = sphere_beam(radius)
        sphere = IntermediateSphere(beam, center, radius, glass, air, OrderInteraction(order))
        x = np.arange(-4 * radius, 4 * radius + 1e-9, step)
        z = np.arange(-3 * radius, 7 * radius + 1e-9, step)
        march = PlaneMarch(z, SPHERE_HALF, step, center, radius, 1.0)
        solution = sphere.run(
            SPHERE_PASSES, march, terms=SPHERE_PASSES, verbose=verbose
        )
        zz, xx = np.meshgrid(z, x, indexing="ij")
        points = np.stack((xx, 0 * xx, zz), -1).reshape(-1, 3)
        inside = np.linalg.norm(points - center, axis=-1) <= radius
        columns = np.flatnonzero(np.isclose(march.x[:, None], x[None], atol=1e-9).any(1))
        scattered = np.zeros((len(points), 3), complex)
        for up, down in solution.emissions.values():
            rows = march.march(up, 1) + march.march(down, -1)
            scattered += rows[:, columns].reshape(-1, 3)
        scattered[inside] = 0
        incident = beam.evaluate(o.PointGrid(points)).electric.values
        total = incident + scattered
        total[inside] = sum(
            state.evaluate(o.PointGrid(points[inside])).electric.values
            for state in solution.internal
        )
        arrays = {
            name: np.linalg.norm(value, axis=-1).reshape(zz.shape).astype(np.float32)
            for name, value in (("incident", incident), ("scattered", scattered), ("total", total))
        }
        arrays.update(x=x, z=z)
        info = dict(
            radius=radius,
            order=order,
            passes=SPHERE_PASSES,
            directions=len(sphere.transport.wavevectors),
            surface_samples=len(sphere.points),
            band=list(march.shell),
            fit_residuals=[r.relative_residual for r in solution.reports],
            seconds=perf_counter() - start,
        )
        if verbose:
            print(f"  ours, order {order}: {info['seconds']:.0f} s", flush=True)
        return arrays, info

    fingerprint = _digest(
        json.dumps(dict(picture=MIE_PICTURE, index=INDEX, passes=SPHERE_PASSES,
                        half=SPHERE_HALF, order=order)),
        "".join(
            inspect.getsource(m)
            for m in (intermediate, orders, polar_lens, taylor_interface, operators)
        ),
        inspect.getsource(sphere_beam),
        inspect.getsource(sphere_ours),
    )
    return _cached(CACHE / f"sphere_ours_order{order}_{fingerprint}", build)


if __name__ == "__main__":
    names = sys.argv[1:]
    if not names:
        sphere_control()
    for name in names or [*LENSES, "sphere"]:
        start = perf_counter()
        if name == "sphere":
            mie_picture()
            for order in (0, 3):
                sphere_ours(order)
        else:
            case(name)
        print(f"{name}: ready in {perf_counter() - start:.0f} s", flush=True)
