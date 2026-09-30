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


GROUND_SIZES = (1e3, 3e3, 1e4, 3e4, 1e5)
FACES = {
    # face: (incident index, transmitted index, interior flag, sign of cap radius)
    "entry": (N_AIR, N_GLASS, False, 1.0),  # air -> glass, convex toward the source
    "exit": (N_GLASS, N_AIR, True, -1.0),  # glass -> air, a converging lens exit
}


def ground_truth(ka, face="entry", *, window=21, surface=31, cache=True):
    """Exact single-interface sphere fields for benchmarking Q, T and Q^-1.

    ka is the size parameter in the incident medium; R = ka lambda / (2 pi n_i).
    The vertex is at the origin and the wave travels along +z, x polarized.
    Stored (complex SI phasors, unit incident |E|):
    * surface: a (surface x surface) chart grid over |x|, |y| <= 0.45 R on the
      cap; incident, reflected and transmitted E and H there, with normals;
    * window: a (window x window) grid over |x|, |y| <= 0.15 R on the plane two
      wavelengths past the cap, where the transmitted E and H live.
    The reference is exact Maxwell (stable Debye-series first term); its
    interface conditions are checked when stored.
    """
    n_i, n_t, interior, sign = FACES[face]
    parameters = dict(ka=float(ka), face=face, window=window, surface=surface)
    path = CACHE / "ground_truth" / f"{face}_ka{ka:.0e}_{_key(**parameters)}.npz"
    if cache and path.exists():
        return dict(np.load(path))
    radius = ka * WAVELENGTH / (2 * np.pi * n_i)
    cap = o.SphericalCap(sign * radius)
    chart = np.linspace(-0.45, 0.45, surface) * radius
    xy = np.stack(np.meshgrid(chart, chart), -1)
    points, normals = cap.points(xy), cap.normals(xy)
    probe = np.linspace(-0.15, 0.15, window) * radius
    X, Y = np.meshgrid(probe, probe)
    corner = cap.height(np.array([[0.15 * radius, 0.15 * radius]]))[0]
    height = (max(corner, 0.0) if sign > 0 else 0.0) + 2 * WAVELENGTH
    targets = np.stack((X, Y, np.full_like(X, height)), -1)
    clock = time.perf_counter()
    on_surface = sphere_single_interface(
        points, WAVELENGTH, radius, n_i, n_t, interior=interior
    )
    beyond = sphere_single_interface(
        targets, WAVELENGTH, radius, n_i, n_t, interior=interior
    )
    seconds = time.perf_counter() - clock
    jumps = [
        np.linalg.norm(
            np.cross(normals, on_surface["incident"][i] + on_surface["reflected"][i])
            - np.cross(normals, on_surface["transmitted"][i])
        )
        / np.linalg.norm(np.cross(normals, on_surface["transmitted"][i]))
        for i in (0, 1)
    ]
    data = dict(
        ka=ka,
        radius=radius,
        n_incident=n_i,
        n_transmitted=n_t,
        surface_points=points,
        surface_normals=normals,
        window_points=targets,
        seconds=seconds,
        tangential_E_jump=jumps[0],
        tangential_H_jump=jumps[1],
    )
    for key in ("incident", "reflected", "transmitted"):
        data[f"surface_{key}_E"], data[f"surface_{key}_H"] = on_surface[key]
    data["window_transmitted_E"], data["window_transmitted_H"] = beyond["transmitted"]
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)
    return data


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


# ----------------------------------------------------------------------------
# Ball lens: two interfaces, a focus behind the lens, exact Debye references.
# ----------------------------------------------------------------------------

BALL = dict(
    radius=100.0, n=1.5, waist=20.0, lattice=280, half=65.0, aperture=0.0, edge=1.5
)


def _profile(rho, parameters):
    """Incident E_x on the waist plane: Gaussian, or an erf-edged flat top."""
    if parameters["aperture"] > 0:
        from scipy.special import erfc

        return 0.5 * erfc((rho - parameters["aperture"]) / parameters["edge"])
    return np.exp(-(rho**2) / parameters["waist"] ** 2)


def _ball_sphere(parameters):
    """AxialBeamSphere for the beam _profile (its coefficients are linear in the
    beam: p and q follow from the profile's Hankel transform, as the class
    computes them for a Gaussian)."""
    from scipy.special import j0
    from vecdiff_general.propagation.references.axial_beam import (
        AxialBeamSphere,
        _pi_tau,
    )

    R = parameters["radius"]
    lens = AxialBeamSphere(
        waist=parameters["waist"],
        waist_z=-R - 5.0,
        center_z=0.0,
        radius=R,
        n_sphere=parameters["n"],
    )
    if parameters["aperture"] <= 0:
        return lens
    k = lens.k0 * lens.n_exterior
    top = min(
        np.pi / 2, np.arcsin(min(1.0, 2 * np.sqrt(37) / (k * parameters["edge"])))
    )
    t, w = np.polynomial.legendre.leggauss(max(400, 2 * lens.order))
    alpha = top / 2 * (t + 1)
    weight = top / 2 * w * np.sin(alpha)
    c = np.cos(alpha)
    reach = parameters["aperture"] + 8 * parameters["edge"]
    r, wr = np.polynomial.legendre.leggauss(int(4 * k * reach / np.pi) + 200)
    rho = reach / 2 * (r + 1)
    wrho = reach / 2 * wr
    # 2D Fourier transform of the radial profile / (4 pi^2): Hankel transform.
    transform = (
        j0(np.outer(k * np.sin(alpha), rho)) @ (_profile(rho, parameters) * rho * wrho)
    ) / (2 * np.pi)
    spectrum = transform * k**2 * np.exp(1j * k * c * (0.0 - (-R - 5.0)))
    pi, tau = _pi_tau(lens.order, c)
    n = lens.orders[:, None]
    scale = 2 * np.pi / (n * (n + 1))
    lens.p = scale[:, 0] * ((pi + c * tau) * spectrum * weight).sum(-1)
    lens.q = scale[:, 0] * ((tau + c * pi) * spectrum * weight).sum(-1)
    return lens


def _ball_worker(job):
    import warnings

    warnings.simplefilter("ignore")
    points, parameters = job
    E, H = _ball_sphere(parameters).fields(points, "exterior", [1, "mie", "incident"])
    return E, H


def ball_reference(points, *, cache_name, workers=10, **overrides):
    """Exact exterior fields behind the ball lens: Debye p = 1 and full Mie.

    p = 1 is the wave that crossed both faces once (the physics of our
    one-way pipeline); "total" is incident + Mie scattering, with every
    internal return. Computed in parallel over point chunks and cached.
    """
    from concurrent.futures import ProcessPoolExecutor

    parameters = dict(BALL, **overrides)
    points = np.asarray(points, float)
    key = _key(
        name=cache_name,
        shape=list(points.shape),
        **{k: float(v) for k, v in parameters.items()},
    )
    path = CACHE / "ball" / f"{cache_name}_{key}.npz"
    if path.exists():
        return dict(np.load(path))
    flat = points.reshape(-1, 3)
    chunks = np.array_split(flat, workers * 4)
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_ball_worker, [(c, parameters) for c in chunks]))
    E = np.concatenate([p[0] for p in parts], axis=1)
    H = np.concatenate([p[1] for p in parts], axis=1)
    shape = points.shape
    data = dict(
        points=points,
        p1_E=E[0].reshape(shape),
        p1_H=H[0].reshape(shape),
        total_E=(E[1] + E[2]).reshape(shape),
        total_H=(H[1] + H[2]).reshape(shape),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return data


def _beam_spectrum(grid, region, parameters, waist_z, plane_z):
    """The reference's beam: E_x Gaussian at z = waist_z, propagating modes only,
    as a PlaneSpectrum on ``grid`` (a lattice at plane_z)."""
    X, Y = np.meshgrid(grid.u, grid.v)
    values = np.zeros(grid.shape + (2,), complex)
    values[..., 0] = _profile(np.hypot(X, Y), parameters)
    at_waist = o.CartesianGrid(grid.u, grid.v, o.Frame([0, 0, waist_z]))
    spectrum = o.angular_spectrum(
        o.TransverseElectricField(at_waist, WAVELENGTH, values),
        region,
        evanescent="discard",
    )
    return _translate(spectrum, grid, plane_z - waist_z)


def _translate(spectrum, grid, distance):
    """The same plane-wave expansion referred to the parallel plane ``grid``."""
    expansion = spectrum.expansion
    phase = np.exp(1j * expansion.wavevectors[:, 2] * distance)
    phase[np.abs(expansion.wavevectors[:, 2].imag) > 0] = 0
    ny, nx = grid.shape
    moved = o.PlaneWaveExpansion(
        expansion.wavevectors,
        expansion.amplitudes * phase[:, None],
        grid.points[ny // 2, nx // 2],
        WAVELENGTH,
        expansion.region,
    )
    return o.fields.spectrum.PlaneSpectrum(grid, moved)


def _poynting_directions(E, H, normals):
    """Local energy-flow direction; in the dark (flux below 1e-8 of its peak,
    where it is roundoff) the surface normal, which carries no field there."""
    S = np.real(np.cross(E, np.conj(H)))
    norm = np.linalg.norm(S, axis=-1)
    lit = norm > 1e-8 * norm.max()
    out = normals.copy()
    out[lit] = S[lit] / norm[lit][:, None]
    return out


def ball_ours(**overrides):
    """Our pipeline through the ball lens with the explicit Q^-1 on lattices.

    Gaussian beam (the reference's) -> Q onto the entry cap (exact plane-wave
    evaluation) -> T -> explicit Q^-1 -> ASM to the exit tangent plane -> Q onto
    the exit cap -> T -> explicit Q^-1. Returns the outgoing expansion in air
    (propagating modes) and the stage times.
    """
    p = dict(BALL, **overrides)
    R, n = p["radius"], p["n"]
    air = o.HomogeneousRegion("air", o.Dielectric(1.0))
    glass = o.HomogeneousRegion("glass", o.Dielectric(n))
    count = p["lattice"]
    axis = (np.arange(count) - count // 2) * (2 * p["half"] / count)
    seconds = {}
    clock = time.perf_counter()
    entry_plane = o.CartesianGrid(axis, axis, o.Frame([0, 0, -R]))
    entry_cap = o.SphericalCap(R, o.Frame([0, 0, -R]))
    entry = o.SurfaceTransport(entry_plane, entry_cap, glass, WAVELENGTH)
    exit_plane = o.CartesianGrid(axis, axis, o.Frame([0, 0, R]))
    exit_cap = o.SphericalCap(-R, o.Frame([0, 0, R]))
    leave = o.SurfaceTransport(exit_plane, exit_cap, glass, WAVELENGTH)
    seconds["build"] = time.perf_counter() - clock

    clock = time.perf_counter()
    beam = _beam_spectrum(entry_plane, air, p, -R - 5.0, -R)
    values = beam.expansion.evaluate(
        o.PointGrid(entry.grid.points), backend="nufft", tolerance=1e-12
    )
    incident = o.ElectromagneticField(
        o.ElectricField(entry.grid, WAVELENGTH, values.electric.values), values.magnetic
    )
    trace = o.fresnel_transmission(
        incident,
        o.DielectricInterface(entry_cap, air, glass),
        _poynting_directions(
            incident.electric.values, incident.magnetic, entry.grid.normals
        ),
    ).transmitted
    inside = entry.radiate(
        o.ElectromagneticField(
            o.ElectricField(entry.grid, WAVELENGTH, trace.electric.values),
            trace.magnetic,
        )
    )
    seconds["entry face (Q, T, Q^-1)"] = time.perf_counter() - clock

    clock = time.perf_counter()
    at_exit = _translate(inside, exit_plane, 2 * R)
    on_exit = leave.forward(at_exit)
    trace = o.fresnel_transmission(
        on_exit,
        o.DielectricInterface(exit_cap, glass, air),
        _poynting_directions(
            on_exit.electric.values, on_exit.magnetic, leave.grid.normals
        ),
    ).transmitted
    out_transport = o.SurfaceTransport(exit_plane, exit_cap, air, WAVELENGTH)
    outgoing = out_transport.radiate(
        o.ElectromagneticField(
            o.ElectricField(out_transport.grid, WAVELENGTH, trace.electric.values),
            trace.magnetic,
        )
    ).expansion
    seconds["R + exit face (Q, T, Q^-1)"] = time.perf_counter() - clock
    live = np.linalg.norm(outgoing.amplitudes, axis=-1) > 0
    active = o.PlaneWaveExpansion(
        outgoing.wavevectors[live],
        outgoing.amplitudes[live],
        outgoing.origin,
        WAVELENGTH,
        air,
    )
    return active, seconds


def ball_comparison(
    *, aperture=35.0, meridional=(161, 241), focal=121, focal_half=12.0
):
    """Ours and exact (p = 1, full Mie) on the meridional section and the focal
    plane of the ball lens; everything cached (the full run takes minutes)."""
    parameters = dict(BALL, aperture=aperture)
    key = _key(
        kind="ball_comparison",
        meridional=list(meridional),
        focal=focal,
        focal_half=focal_half,
        **{k: float(v) for k, v in parameters.items()},
    )
    path = CACHE / "ball" / f"comparison_{key}.npz"
    if path.exists():
        return dict(np.load(path))
    R = parameters["radius"]
    x = np.linspace(-45, 45, meridional[0])
    z = np.linspace(R + 3, 2.4 * R, meridional[1])
    X, Z = np.meshgrid(x, z)
    section = np.stack((X, np.zeros_like(X), Z), -1)
    clock = time.perf_counter()
    ours, seconds = ball_ours(aperture=aperture)
    total_seconds = time.perf_counter() - clock
    reference = ball_reference(
        section,
        cache_name="meridional_flat" if aperture else "meridional",
        aperture=aperture,
    )
    focus = z[
        np.argmax(np.linalg.norm(reference["p1_E"][:, meridional[0] // 2], axis=-1))
    ]
    f = np.linspace(-focal_half, focal_half, focal)
    FX, FY = np.meshgrid(f, f)
    plane = np.stack((FX, FY, np.full_like(FX, focus)), -1)
    focal_reference = ball_reference(
        plane, cache_name="focal_flat" if aperture else "focal", aperture=aperture
    )

    def evaluate(pts):
        return ours.evaluate(
            o.PointGrid(pts.reshape(-1, 3)), backend="nufft", tolerance=1e-12
        )

    a, b = evaluate(section), evaluate(plane)
    data = dict(
        focus=focus,
        section=section,
        plane=plane,
        ours_section_E=a.electric.values.reshape(section.shape),
        ours_plane_E=b.electric.values.reshape(plane.shape),
        ours_plane_H=b.magnetic.reshape(plane.shape),
        seconds_total=total_seconds,
        **{f"exact_section_{k}": reference[f"{k}_E"] for k in ("p1", "total")},
        **{f"exact_plane_{k}": focal_reference[f"{k}_E"] for k in ("p1", "total")},
        **{f"exact_plane_{k}_H": focal_reference[f"{k}_H"] for k in ("p1", "total")},
        **{f"seconds {k}": v for k, v in seconds.items()},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return data


def phase_space_sphere(ka, *, tile, dense=101, cache=True):
    """Phase-space pipeline (Q, T, Q^-1, R on atoms) on the air-to-glass sphere.

    Plane wave in air, apodized by an erf edge at 0.36 R, as plane atoms at the
    vertex plane -> Q -> T -> Q^-1 -> R to the ground-truth window. Grid steps
    follow each stage's band: the window's, plus the non-quadratic surface
    residual on the chart and the refracted wavefront curvature on the output.
    Returns errors against the stored exact fields, stage times, and the
    field on a dense window (for maps).
    """
    from vecdiff_general.propagation import phase_space as ps

    path = (
        CACHE
        / "phase_space"
        / f"sphere_{_key(ka=float(ka), tile=float(tile), dense=dense)}.npz"
    )
    if cache and path.exists():
        return dict(np.load(path))
    truth = ground_truth(ka, "entry")
    R = float(truth["radius"])
    k, n2 = 2 * np.pi / WAVELENGTH, N_GLASS
    cap = o.SphericalCap(R)

    def field(xy):
        values = np.zeros(xy.shape[:-1] + (3,), complex)
        values[..., 0] = _edge(xy, R)
        return values

    sharpness = 4.0
    sigma = tile / sharpness
    window_band = 2 * np.sqrt(np.log(1e9)) / sigma
    support = tile + 2 * np.sqrt(np.log(1e9)) * sigma

    def plan(band):
        step = np.pi / (1.25 * band)
        return step, int(np.ceil(support / step / 2)) * 2 + 4

    p_step, p_size = plan(window_band)
    c_step, c_size = plan(window_band + k * (support / 2) ** 3 / (2 * R**3))
    o_step, o_size = plan(window_band + n2 * k * (support / 2) / (3 * R))
    air = o.HomogeneousRegion("air", o.Dielectric(N_AIR))
    glass = o.HomogeneousRegion("glass", o.Dielectric(n2))
    times = {}
    clock = time.perf_counter()
    atoms = ps.AtomField.from_function(
        field,
        lambda xy: np.zeros(xy.shape),
        k,
        extent=0.42 * R,
        tile=tile,
        step=p_step,
        size=p_size,
        sharpness=sharpness,
        tolerance=1e-9,
    )
    times["atoms"] = time.perf_counter() - clock
    clock = time.perf_counter()
    chart = ps.surface_transport(
        atoms, cap, N_AIR, size=c_size, step=c_step, tolerance=1e-8
    )
    times["Q"] = time.perf_counter() - clock
    clock = time.perf_counter()
    transmitted = ps.transmit_chart(
        chart, cap, o.DielectricInterface(cap, air, glass), WAVELENGTH
    )
    times["T"] = time.perf_counter() - clock
    clock = time.perf_counter()
    out = ps.radiate(
        transmitted, cap, n2, WAVELENGTH, size=o_size, step=o_step, tolerance=1e-8
    )
    times["Q^-1"] = time.perf_counter() - clock
    window = truth["window_points"]
    clock = time.perf_counter()
    moved = out.propagate_to(window[0, 0, 2])
    ours = moved.evaluate(window[..., :2].reshape(-1, 2)).reshape(window.shape)
    times["R + evaluation"] = time.perf_counter() - clock
    surface = truth["surface_points"]
    near = np.hypot(surface[..., 0], surface[..., 1]) < 0.25 * R
    E_inc, _ = chart.evaluate(surface[near][:, :2])
    E_tr, _ = transmitted.evaluate(surface[near][:, :2])

    def gap(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    side = np.linspace(-0.15, 0.15, dense) * R
    X, Y = np.meshgrid(side, side)
    dense_field = moved.evaluate(np.column_stack((X.ravel(), Y.ravel()))).reshape(
        X.shape + (3,)
    )
    exact_dense = sphere_single_interface(
        np.stack((X, Y, np.full_like(X, window[0, 0, 2])), -1), WAVELENGTH, R, N_AIR, n2
    )["transmitted"][0]
    data = dict(
        ka=ka,
        radius=R,
        tile=tile,
        atoms=len(atoms.centres),
        grids=np.array([p_size, c_size, o_size]),
        steps=np.array([p_step, c_step, o_step]),
        error_Q=gap(E_inc, truth["surface_incident_E"][near]),
        error_T=gap(E_tr, truth["surface_transmitted_E"][near]),
        error_window=gap(ours, truth["window_transmitted_E"]),
        dense_x=X,
        dense_y=Y,
        dense_ours=dense_field,
        dense_exact=exact_dense,
        **{f"seconds {k}": v for k, v in times.items()},
    )
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)
    return data
