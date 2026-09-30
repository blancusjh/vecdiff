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
