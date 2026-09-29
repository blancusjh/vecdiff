"""Data for notebook 07, section "6λ₀ lenses": BEM references and our fast path.

The BEM fields (two element sizes, lambda/3 and lambda/4) and the 6λ₀ sphere
control against Mie are cached in .cache/lenses under a fingerprint of the
lens, the probes and the BEM sources; functions entering that fingerprint
(probes, bem_fields, focal_window, sphere_control) must stay unchanged or the
references are recomputed (minutes each). Our field uses the production
R Q^-1 T Q R and takes about a second per lens.
"""

import hashlib
import inspect
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import vecdiff_general as o
from vecdiff_general.propagation.references import DielectricBEM, bem, meshing
from vecdiff_general.propagation.references.meshing import surface_mesh
from vecdiff_general.volumes import VolumeBetweenSurfaces

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "lenses"
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
    # Thin lenses, a^2/(lambda T) = 12.9: the order comparisons of notebook 06.
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


def _points(*arrays):
    return o.PointGrid(np.concatenate([a.reshape(-1, 3) for a in arrays]))


def _relative(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


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


def ours_axis(name, *, rim_coupling="auto"):
    """Our field on the BEM axis probes (complex E, V/m)."""
    source, system, _ = lens(name)
    a = system.apertures[0]
    z_axis, *_ = probes(name)
    grid_axis = (np.arange(GRID) - GRID // 2) * WINDOW * a / GRID
    solution = o.propagate_interfaces(
        source,
        system,
        o.CartesianGrid(grid_axis, grid_axis),
        rim_coupling=rim_coupling,
    )
    points = np.stack((0 * z_axis, 0 * z_axis, z_axis), -1)
    return solution.evaluate(o.PointGrid(points)).electric.values


def validity(name):
    """Axial errors of ours and BEM refinement, beyond one wavelength."""
    arrays, _ = _cached(
        CACHE / f"bem_{_slug(name)}_{bem_fingerprint(name)}",
        lambda: bem_fields(name),
    )
    z_axis = arrays["z_axis"]
    window = z_axis > (LENSES[name][2] + 1.0) * WAVELENGTH
    reference = arrays[f"BEM λ/{round(1 / BEM_SIZES[-1])}|axis"]
    coarse = arrays[f"BEM λ/{round(1 / BEM_SIZES[0])}|axis"]
    ours = ours_axis(name)
    uncoupled = ours_axis(name, rim_coupling="off")
    return dict(
        z=z_axis,
        bem=reference,
        ours=ours,
        uncoupled=uncoupled,
        error=_relative(ours[window], reference[window]),
        error_uncoupled=_relative(uncoupled[window], reference[window]),
        bem_change=_relative(coarse[window], reference[window]),
    )
