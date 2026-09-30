"""Data for notebook 07: macroscopic Sigma <-> plane operators against exact fields.

A plane wave in air meets a glass sphere (the exact single-interface field of
``references.debye.sphere_single_interface``, stable to ka ~ 1e5). Our pipeline:
analytic incident field on Sigma -> local Fresnel T -> explicit Q^-1
(SurfaceTransport.radiate) -> plane-wave evaluation. The Huygens trace is apodized by a smooth erf
edge at 0.36 R; the comparison window (|x|, |y| <= 0.15 R, just above the cap)
is far from that edge. Results are cached in the repository's .cache/.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import vecdiff_general as o
from scipy.special import erfc
from vecdiff_general.media.medium import Z0
from vecdiff_general.propagation.references.debye import sphere_single_interface

CACHE = Path(__file__).resolve().parents[2] / ".cache" / "interface_operators"
WAVELENGTH = 1.0
N_AIR, N_GLASS = 1.0, 1.5


def _key(**parameters):
    return hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[
        :16
    ]


def sphere_case(ka, *, spacing=0.5, probes=31, cache=True):
    """Error and cost of T + explicit Q^-1 for an air-to-glass sphere of size ka."""
    parameters = dict(ka=ka, spacing=spacing, probes=probes)
    path = CACHE / f"sphere_{_key(**parameters)}.json"
    if cache and path.exists():
        return json.loads(path.read_text())

    radius = ka * WAVELENGTH / (2 * np.pi * N_AIR)
    half = 0.45 * radius
    count = 2 * int(np.ceil(half / spacing))
    axis = (np.arange(count) - count // 2) * (2 * half / count)
    grid = o.CartesianGrid(axis, axis)
    air = o.HomogeneousRegion("air", o.Dielectric(N_AIR))
    glass = o.HomogeneousRegion("glass", o.Dielectric(N_GLASS))
    surface = o.SphericalCap(radius)

    clock = time.perf_counter()
    transport = o.SurfaceTransport(grid, surface, glass, WAVELENGTH)
    build = time.perf_counter() - clock

    points = transport.grid.points
    interface = o.DielectricInterface(surface, air, glass)
    edge = 0.5 * erfc(
        (np.hypot(points[..., 0], points[..., 1]) - 0.36 * radius) / (0.025 * radius)
    )
    electric = np.empty(points.shape, complex)
    magnetic = np.empty(points.shape, complex)
    clock = time.perf_counter()
    rows = max(1, 2**20 // count)  # local T in row blocks bounds its 3x3-matrix memory
    for start in range(0, count, rows):
        block = slice(start, start + rows)
        grid_block = o.SurfaceGrid(
            points[block], transport.grid.normals[block], transport.grid.weights[block]
        )
        phase = np.exp(1j * 2 * np.pi * N_AIR / WAVELENGTH * points[block][..., 2])[
            ..., None
        ]
        incident = o.ElectromagneticField(
            o.ElectricField(grid_block, WAVELENGTH, phase * np.array([1.0, 0, 0])),
            phase * np.array([0, N_AIR / Z0, 0]),
        )
        encounter = o.fresnel_transmission(
            incident, interface, np.broadcast_to([0, 0, 1.0], points[block].shape)
        )
        electric[block] = encounter.transmitted.electric.values * edge[block][..., None]
        magnetic[block] = encounter.transmitted.magnetic * edge[block][..., None]
    fresnel = time.perf_counter() - clock
    trace = o.ElectromagneticField(
        o.ElectricField(transport.grid, WAVELENGTH, electric), magnetic
    )
    del electric, magnetic
    clock = time.perf_counter()
    spectrum = transport.radiate(trace)
    inverse = time.perf_counter() - clock

    window = np.linspace(-0.15, 0.15, probes) * radius
    X, Y = np.meshgrid(window, window)
    height = radius - np.sqrt(radius**2 - 2 * (0.15 * radius) ** 2) + 2.0
    targets = np.stack((X, Y, np.full_like(X, height)), -1)
    clock = time.perf_counter()
    expansion = spectrum.expansion
    live = np.linalg.norm(expansion.amplitudes, axis=-1) > 0
    active = o.PlaneWaveExpansion(
        expansion.wavevectors[live],
        expansion.amplitudes[live],
        expansion.origin,
        WAVELENGTH,
        glass,
    )
    ours = active.evaluate(
        o.PointGrid(targets), backend="nufft", tolerance=1e-12
    ).electric.values
    evaluation = time.perf_counter() - clock
    exact = sphere_single_interface(targets, WAVELENGTH, radius, N_AIR, N_GLASS)[
        "transmitted"
    ][0]
    result = dict(
        parameters,
        radius_wavelengths=radius,
        lattice=count,
        seconds=dict(
            build=build, fresnel=fresnel, inverse=inverse, evaluation=evaluation
        ),
        relative_error=float(np.linalg.norm(ours - exact) / np.linalg.norm(exact)),
        rank=transport.rank,
    )
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1, default=float))
    return result


def _exit_face(ka, spacing):
    """Glass-to-air sphere exit (a converging face): lattice, surface, media."""
    radius = ka * WAVELENGTH / (2 * np.pi * N_GLASS)
    half = 0.45 * radius
    count = 2 * int(np.ceil(half / spacing))
    axis = (np.arange(count) - count // 2) * (2 * half / count)
    glass = o.HomogeneousRegion("glass", o.Dielectric(N_GLASS))
    air = o.HomogeneousRegion("air", o.Dielectric(N_AIR))
    return radius, o.CartesianGrid(axis, axis), o.SphericalCap(-radius), glass, air


def _edge(xy, radius):
    return 0.5 * erfc(
        (np.hypot(xy[..., 0], xy[..., 1]) - 0.36 * radius) / (0.025 * radius)
    )


def _transmit(surface, glass, air, xy, radius):
    """Local Fresnel T of the axial plane wave in glass at chart points xy."""
    points, normals = surface.points(xy), surface.normals(xy)
    grid = o.SurfaceGrid(points, normals, np.ones(points.shape[:-1]))
    phase = np.exp(1j * 2 * np.pi * N_GLASS / WAVELENGTH * points[..., 2])[..., None]
    incident = o.ElectromagneticField(
        o.ElectricField(grid, WAVELENGTH, phase * np.array([1.0, 0, 0])),
        phase * np.array([0, N_GLASS / Z0, 0]),
    )
    encounter = o.fresnel_transmission(
        incident,
        o.DielectricInterface(surface, glass, air),
        np.broadcast_to([0, 0, 1.0], points.shape),
    )
    edge = _edge(xy, radius)[..., None]
    return (
        points,
        normals,
        encounter.transmitted.electric.values * edge,
        encounter.transmitted.magnetic * edge,
        encounter.transmitted_directions,
    )


def _ray_landing(surface, glass, air, xy, radius, z):
    points, normals, E, _, d = _transmit(surface, glass, air, xy, radius)
    s = (z - points[..., 2]) / d[..., 2]
    return points, normals, E, d, s, points[..., :2] + s[..., None] * d[..., :2]


def ray_field(ka, z, targets_xy, *, spacing=0.5):
    """Geometrical-optics transport of the transmitted trace to the plane z.

    Each target is traced back to its surface point (the ray map is inverted by
    interpolation, then refined by Newton steps), where T is evaluated exactly.
    Phase: optical path; amplitude: ray-tube Jacobian |E|^2 (d.n) dA_Sigma =
    |E'|^2 d_z dA_plane. Undefined where the ray map folds (caustics).
    """
    from scipy.interpolate import griddata

    radius, grid, surface, glass, air = _exit_face(ka, spacing)
    # Coarse ray map on a sub-lattice for the initial inversion.
    coarse = np.linspace(-0.45 * radius, 0.45 * radius, 161)
    U, V = np.meshgrid(coarse, coarse)
    xy = np.stack((U, V), -1)
    *_, landing = _ray_landing(surface, glass, air, xy, radius, z)
    guess = np.stack(
        [
            griddata(
                landing.reshape(-1, 2),
                xy.reshape(-1, 2)[:, i],
                targets_xy,
                method="linear",
            )
            for i in (0, 1)
        ],
        -1,
    )
    field = np.full(targets_xy.shape[:-1] + (3,), np.nan + 0j)
    det = np.full(targets_xy.shape[:-1], np.nan)
    reached = np.isfinite(guess).all(-1)
    if not reached.any():
        return field, det
    guess, goal = guess[reached], targets_xy[reached]
    step = 1e-4 * radius

    def landing_and_jacobian(xy):
        base = _ray_landing(surface, glass, air, xy, radius, z)
        du = _ray_landing(surface, glass, air, xy + [step, 0], radius, z)[-1]
        dv = _ray_landing(surface, glass, air, xy + [0, step], radius, z)[-1]
        jac = np.stack(((du - base[-1]) / step, (dv - base[-1]) / step), -1)
        return base, jac

    for _ in range(8):  # Newton on landing(xy) = target, kept inside the chart
        base, jac = landing_and_jacobian(guess)
        with np.errstate(all="ignore"):
            update = np.linalg.solve(jac, (base[-1] - goal)[..., None])[..., 0]
        guess = guess - np.where(np.isfinite(update), update, 0)
        norm = np.hypot(guess[..., 0], guess[..., 1])
        guess *= np.minimum(1, 0.5 * radius / np.maximum(norm, 1e-300))[..., None]
    (points, normals, E, d, s, landed), jac = landing_and_jacobian(guess)
    # Geometrical optics is defined only where the ray actually lands there.
    converged = np.linalg.norm(landed - goal, axis=-1) < 1e-6 * radius
    jacobian = np.linalg.det(jac)
    area = surface.area_jacobian(guess)
    with np.errstate(all="ignore"):
        tube = np.sqrt(area * np.sum(d * normals, -1) / (np.abs(jacobian) * d[..., 2]))
    values = E * (tube * np.exp(1j * 2 * np.pi * N_AIR / WAVELENGTH * s))[..., None]
    values[~converged] = np.nan
    field[reached] = values
    det[reached] = np.where(converged, jacobian, np.nan)
    return field, det


def wave_spectrum(ka, *, spacing=0.5):
    """T + explicit Q^-1 on the exit face: the exact-to-tolerance wave reference."""
    radius, grid, surface, glass, air = _exit_face(ka, spacing)
    clock = time.perf_counter()
    transport = o.SurfaceTransport(grid, surface, air, WAVELENGTH)
    build = time.perf_counter() - clock
    clock = time.perf_counter()
    _, _, E, H, _ = _transmit(surface, glass, air, transport.xy, radius)
    fresnel = time.perf_counter() - clock
    trace = o.ElectromagneticField(o.ElectricField(transport.grid, WAVELENGTH, E), H)
    clock = time.perf_counter()
    expansion = transport.radiate(trace).expansion
    inverse = time.perf_counter() - clock
    live = np.linalg.norm(expansion.amplitudes, axis=-1) > 0
    active = o.PlaneWaveExpansion(
        expansion.wavevectors[live],
        expansion.amplitudes[live],
        expansion.origin,
        WAVELENGTH,
        air,
    )
    return active, dict(build=build, fresnel=fresnel, inverse=inverse)


def ray_inverse(ka, *, spacing=0.5):
    """Q^-1 by ray transport: Sigma -> its tangent plane (virtually in air).

    Every lattice node of the plane z = 0 receives the transmitted field of the
    ray through it (phase: optical path over the sag; amplitude: ray tube). The
    result is a plane field for exact ASM; rays are never used beyond it.
    """
    radius, grid, surface, glass, air = _exit_face(ka, spacing)
    clock = time.perf_counter()
    X, Y = np.meshgrid(grid.u, grid.v)
    field, _ = ray_field(ka, 0.0, np.stack((X, Y), -1), spacing=spacing)
    field = np.nan_to_num(field)  # no ray: outside the apodized aperture
    transverse = o.TransverseElectricField(grid, WAVELENGTH, field[..., :2])
    spectrum = o.angular_spectrum(transverse, air, evanescent="discard")
    seconds = time.perf_counter() - clock
    expansion = spectrum.expansion
    live = np.linalg.norm(expansion.amplitudes, axis=-1) > 0
    active = o.PlaneWaveExpansion(
        expansion.wavevectors[live],
        expansion.amplitudes[live],
        expansion.origin,
        WAVELENGTH,
        air,
    )
    return active, seconds


def inverse_comparison(
    ka=1000,
    *,
    planes=(2.0, 0.5, 1.0, 1.5, 1.8, 2.0),
    samples=241,
    spacing=0.5,
    cache=True,
):
    """Ray Q^-1 vs explicit wave Q^-1, both followed by exact ASM.

    Transverse cuts behind a converging glass-to-air sphere face. planes: the
    first entry in wavelengths behind the vertex, the rest in units of R (the
    paraxial focus is at 2R).
    """
    parameters = dict(ka=ka, planes=list(planes), samples=samples, spacing=spacing)
    path = CACHE / f"inverse_{_key(**parameters)}.json"
    if cache and path.exists():
        return json.loads(path.read_text())
    radius = ka * WAVELENGTH / (2 * np.pi * N_GLASS)
    wave, wave_seconds = wave_spectrum(ka, spacing=spacing)
    ray, ray_seconds = ray_inverse(ka, spacing=spacing)
    rows = []
    for index, plane in enumerate(planes):
        z = plane if index == 0 else plane * radius
        half = max(0.36 * radius * abs(1 - z / (2 * radius)) * 1.3, 25 * WAVELENGTH)
        x = np.linspace(-half, half, samples)
        targets = o.PointGrid(
            np.column_stack((x, np.zeros_like(x), np.full_like(x, z)))
        )
        a = wave.evaluate(targets, backend="nufft", tolerance=1e-12).electric.values
        b = ray.evaluate(targets, backend="nufft", tolerance=1e-12).electric.values
        rows.append(
            dict(
                z=z,
                x=x.tolist(),
                wave_intensity=(np.linalg.norm(a, axis=-1) ** 2).tolist(),
                ray_intensity=(np.linalg.norm(b, axis=-1) ** 2).tolist(),
                relative_error=float(np.linalg.norm(b - a) / np.linalg.norm(a)),
            )
        )
    result = dict(
        parameters,
        radius=radius,
        wave_seconds=wave_seconds,
        ray_seconds=ray_seconds,
        planes_result=rows,
    )
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, default=float))
    return result
