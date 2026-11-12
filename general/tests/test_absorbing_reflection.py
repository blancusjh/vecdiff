"""Reflection from an absorbing medium behind a surface: a metal mirror, a lossy glass.

The reflected rays stay in the transparent incident medium, so the map, its
Jacobian, the flux factor and the caustic count are those of the dielectric
reflection; only r_s and r_p change, through the complex index behind. They
are checked here against each plane wave's own boundary-value problem,
solved from the continuity of tangential E and H with an inhomogeneous
transmitted wave whose normal wavenumber is chosen on the decaying branch
(Im k_z > 0), without any Fresnel formula. The fields vary as exp(-i omega t).
"""

import numpy as np
import pytest
import vecdiff as o
from numpy.testing import assert_allclose
from vecdiff.media.medium import Z0
from vecdiff.transport.dyadic import carried, fresnel

bd, host = o.backend.bd, o.backend.to_numpy
F, S, P, Y = o.fields, o.surfaces, o.propagation, o.system
Frame = o.geometry.Frame
AIR = o.media.HomogeneousRegion("air", o.media.air)
INDICES = (1.5 + 0.02j, 0.05 + 3.3j, 2 + 4j, 1.0 + 1e-6j)


def boundary_value(k, e, n_i, n_t):
    """Reflected and transmitted plane waves of e exp(i k . r) on the plane z = 0.

    The transmitted wavevector has the incident tangential part and
    k_z = +-sqrt((n_t k0)^2 - k_x^2 - k_y^2), the root that decays into
    z > 0 (Im k_z >= 0). Each unknown wave is two amplitudes on a basis
    transverse to its own wavevector bilinearly (q . v = 0, complex q);
    tangential E and Z0 H = q x E / k0 are continuous. Returns
    (E_r, k_r, E_t, k_t).
    """
    k = np.asarray(k, complex)
    k0 = np.sqrt(k @ k).real / n_i
    k_r = k * np.array([1, 1, -1])
    k_z = np.sqrt(complex((n_t * k0) ** 2 - k[0] ** 2 - k[1] ** 2))
    if k_z.imag < 0:
        k_z = -k_z
    k_t = np.array([k[0], k[1], k_z])

    def basis(q):
        a = np.cross([0, 0, 1.0], q)
        if np.linalg.norm(a) < 1e-12 * np.linalg.norm(q):
            a = np.array([1.0, 0, 0], complex)
        return a, np.cross(q, a)

    def h(q, v):
        return np.cross(q, v) / k0

    waves = [(k_r, v) for v in basis(k_r)] + [(k_t, v) for v in basis(k_t)]
    matrix = np.array(
        [
            np.r_[sign * v[:2], sign * h(q, v)[:2]]
            for (q, v), sign in zip(waves, [1, 1, -1, -1])
        ]
    ).T
    x = np.linalg.solve(matrix, -np.r_[e[:2], h(k, e)[:2]])
    return (
        x[0] * waves[0][1] + x[1] * waves[1][1],
        k_r,
        x[2] * waves[2][1] + x[3] * waves[3][1],
        k_t,
    )


def normal_flux(e, k, k0=2 * np.pi):
    """<S_z> Z0 of the plane wave e exp(i k . r) at z = 0 (k complex)."""
    return 0.5 * np.real(np.cross(e, np.conj(np.cross(k, e) / k0)))[2]


def incident_wave(degrees, azimuth=0.3, n_i=1.0):
    """Direction, TE axis s and TM axis p = s x d of a plane wave."""
    theta = np.radians(degrees)
    d = np.array(
        [
            np.sin(theta) * np.cos(azimuth),
            np.sin(theta) * np.sin(azimuth),
            np.cos(theta),
        ]
    )
    s = np.cross([0, 0, 1.0], d)
    s = s / np.linalg.norm(s) if np.linalg.norm(s) > 1e-12 else np.array([0, 1.0, 0])
    return d, s, np.cross(s, d)


def carried_reflection(d, s, e, n_i, n_t):
    """The reflected E the leg's dyadic carries for one ray of direction d."""
    mapping = o.transport.InterfaceMap(
        S.Plane(), F.Carrier.collimated(d), n_i, n_t, -1.0, branch="reflected"
    )
    rays = mapping.forward(np.zeros((1, 2)))
    assert host(rays.valid).all()
    return host(
        carried(bd.asarray(e[None]), rays, bd.asarray(s[None]), n_i, n_t, "reflected")
    )[0]


POLARIZATIONS = {"s": (1, 0), "p": (0, 1), "elliptical": (0.6, 0.8j)}


@pytest.mark.parametrize("n_t", INDICES)
@pytest.mark.parametrize("name", list(POLARIZATIONS))
def test_the_carried_reflection_of_an_absorbing_medium_solves_the_boundary_problem(
    n_t, name
):
    """The dyadic's r_s, r_p, branch of c_t and p signs, 0 to 85 degrees:
    measured 7e-15 at worst. The reflected and absorbed powers add up."""
    a, b = POLARIZATIONS[name]
    for degrees in range(0, 86, 5):
        d, s, p = incident_wave(degrees)
        e = (a * s + b * p).astype(complex)
        got = carried_reflection(d, s, e, 1.0, n_t)
        e_r, k_r, e_t, k_t = boundary_value(2 * np.pi * d, e, 1.0, n_t)
        assert np.abs(got - e_r).max() < 1e-13
        incident = normal_flux(e, 2 * np.pi * d)
        R = -normal_flux(e_r, k_r) / incident
        absorbed = normal_flux(e_t, k_t) / incident
        assert 0 < R < 1 and absorbed > 0
        assert abs(R + absorbed - 1) < 1e-12


@pytest.mark.parametrize("n_t", INDICES)
def test_the_normal_reflectance_of_an_absorbing_medium_is_the_textbook_value(n_t):
    """R = |(n - 1) / (n + 1)|^2, an independent number, for both polarizations."""
    d, s, p = incident_wave(0.0)
    textbook = abs((n_t - 1) / (n_t + 1)) ** 2
    for e in (s, p, (s + 1j * p) / np.sqrt(2)):
        got = carried_reflection(d, s, e.astype(complex), 1.0, n_t)
        assert abs(np.vdot(got, got).real - textbook) < 1e-14


@pytest.mark.parametrize(
    "n_i, n_t, degrees",
    [(1.0, 1.5, 0), (1.0, 1.5, 40), (1.0, 1.5, 80), (1.5, 1.0, 20), (1.5, 1.0, 60)],
)
def test_vanishing_absorption_reproduces_the_dielectric_reflection(n_i, n_t, degrees):
    """|r(n + i kappa) - r(n)| falls with kappa, also beyond the critical angle,
    where the dielectric wave decays on the same branch, Im c_t > 0."""

    class Rays:
        cos_incident = bd.asarray([np.cos(np.radians(degrees))])

    lossless = np.array(fresnel(Rays, n_i, n_t, "reflected"))[:, 0]
    gaps = []
    for kappa in (1e-3, 1e-5, 1e-7, 1e-9):
        lossy = np.array(fresnel(Rays, n_i, complex(n_t, kappa), "reflected"))[:, 0]
        gaps.append(np.abs(lossy - lossless).max())
    assert all(b < 0.02 * a for a, b in zip(gaps, gaps[1:]))
    assert gaps[-1] < 1e-7


def plane_wave_leg(n_t, mode, polarization, step=0.3):
    """One lattice plane wave of air reflected by a plane over an absorbing
    medium, through the lattice leg and its re-expansion, and the exact wave."""
    behind = o.media.HomogeneousRegion("metal", o.media.Medium(n_t, "metal"))
    n = 64
    axis = (np.arange(n) - n // 2) * step
    frequency = 2 * np.pi * np.fft.fftfreq(n, step)
    kx, ky = frequency[mode[0]], frequency[mode[1]]
    kz = np.sqrt((2 * np.pi) ** 2 - kx**2 - ky**2)
    kappa = np.hypot(kx, ky)
    te = np.array([-ky, kx]) / kappa if kappa else np.array([0, 1.0])
    tm = np.array([kx, ky]) / kappa if kappa else np.array([1.0, 0])
    amplitude = polarization[0] * te + polarization[1] * tm
    grid = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, 0.0]))
    x, y = np.meshgrid(axis, axis)
    values = amplitude * np.exp(1j * (kx * x + ky * y))[..., None]
    incident = F.angular_spectrum(
        F.TransverseElectricField(grid, 1.0, values), AIR, evanescent="discard"
    )
    leg = o.transport.LatticeTransport(
        grid,
        Y.Reflection(S.Plane(), AIR, behind=behind),
        F.Carrier.collimated([kx, ky, kz]),
        1.0,
    )
    e = np.r_[amplitude, -(kx * amplitude[0] + ky * amplitude[1]) / kz]
    e_r, k_r, _, _ = boundary_value([kx, ky, kz], e, 1.0, n_t)
    back, report = leg.reflect(incident, method="direct")
    field = back.propagate(0.0)
    E, H = host(field.electric.values), host(field.magnetic)
    out = back.grid
    u, v = np.meshgrid(host(out.u), host(out.v))
    wave = np.exp(1j * (kx * u + ky * v))[..., None]
    h_r = np.cross(k_r, e_r) / (2 * np.pi * Z0)
    flux = normal_flux(e, np.array([kx, ky, kz]))
    return dict(
        # Relative to the incident wave: a nearly matched medium reflects 5e-7.
        E=abs(E - e_r * wave).max() / abs(e).max(),
        H=abs(H - h_r * wave).max() * Z0 / abs(e).max(),
        R=-0.5 * np.real(np.cross(E, np.conj(H)))[..., 2].mean() * Z0 / flux,
        exact_R=-normal_flux(e_r, k_r) / flux,
        report=report,
    )


@pytest.mark.parametrize("n_t", INDICES)
@pytest.mark.parametrize(
    "mode, polarization", [((0, 0), (1, 0)), ((5, 3), (0, 1)), ((18, 6), (0.6, -0.8j))]
)
def test_a_plane_wave_returns_from_an_absorbing_medium_as_its_exact_reflection(
    n_t, mode, polarization
):
    """The lattice leg and its re-expansion: E and H of the exact reflected
    wave to roundoff of the incident one, and its reflectance below one."""
    result = plane_wave_leg(n_t, mode, polarization)
    assert result["E"] < 1e-12 and result["H"] < 1e-12
    assert abs(result["R"] - result["exact_R"]) < 1e-12 and result["R"] < 1
    assert result["report"].caustics == 0


def line_carrier(x0, z0):
    """Rays that converge on the line x = x0, z = z0, parallel to y."""

    def toward(r):
        r = bd.asarray(r, float)
        d = bd.zeros(r.shape)
        d[..., 0], d[..., 2] = x0 - r[..., 0], z0 - r[..., 2]
        return d

    def gradient(r):
        d = toward(r)
        return d / bd.linalg.norm(d, axis=-1, keepdims=True)

    def hessian(r):
        d = toward(r)
        length = bd.linalg.norm(d, axis=-1)
        s = d / length[..., None]
        projector = bd.eye(3) - s[..., :, None] * s[..., None, :]
        projector[..., 1, 1] -= 1.0
        return -projector / length[..., None, None]

    return F.Carrier(lambda r: -bd.linalg.norm(toward(r), axis=-1), gradient, hessian)


def cylindrical_beam(size, n_t, focus=2.0, te=True, sine=0.3, tilt=0.0):
    """Relative error of a cylindrical beam reflected by a plane metal mirror
    against its plane-wave sum, each wave reflected by its boundary problem.

    The boundary is z = 0 and the reference plane z = -40 size; the beam
    converges on the line (x_f, focus * 40 size) behind the boundary.
    """
    plane = 40.0 * size
    z_f = focus * plane
    x_f = np.tan(tilt) * (z_f + plane)
    step = 0.4
    half = 1.6 * np.tan(np.arcsin(sine)) * (z_f + plane) + 20 + 2 * np.tan(tilt) * plane
    count = int(2 * half / step) // 2 * 2
    x = (np.arange(count) - count // 2) * step
    y = (np.arange(8) - 4) * step
    k = 2 * np.pi * np.fft.fftfreq(count, step)
    offset = k / (2 * np.pi) - np.sin(tilt)
    amplitude = np.exp(-((offset / (0.5 * sine)) ** 2)) * (abs(offset) < sine)
    kz = np.sqrt(np.maximum((2 * np.pi) ** 2 - k**2, 0.0))
    amplitude = amplitude * np.exp(-1j * (k * x_f + kz * (plane + z_f)))
    component = 1 if te else 0
    # Each plane wave of E_y (TE) or of the transverse TM wave with E_x.
    reflected = np.zeros(count, complex)
    for j in np.flatnonzero(amplitude):
        if te:
            e = np.array([0, 1.0, 0], complex)
        else:
            e = np.array([kz[j], 0, -k[j]], complex) / (2 * np.pi)
        e_r, _, _, _ = boundary_value([k[j], 0, kz[j]], e, 1.0, n_t)
        reflected[j] = e_r[component] / e[component]
    reflected *= amplitude * np.exp(2j * kz * plane)
    values = np.zeros((len(y), count, 2), complex)
    values[..., component] = amplitude @ np.exp(1j * np.outer(k, x))
    behind = o.media.HomogeneousRegion("metal", o.media.Medium(n_t, "metal"))
    grid = o.sampling.CartesianGrid(x, y, Frame([0, 0, -plane]))
    incident = F.angular_spectrum(
        F.TransverseElectricField(grid, 1.0, values), AIR, evanescent="discard"
    )
    leg = o.transport.LatticeTransport(
        grid,
        Y.Reflection(S.Plane(), AIR, behind=behind),
        line_carrier(x_f, z_f),
        1.0,
        seed_stride=2,
    )
    out, report = leg.carry(incident, method="direct")
    row = host(out.values)[len(y) // 2, :, component]
    exact = reflected @ np.exp(1j * np.outer(k, host(out.grid.u)))
    bright = host(leg.lit).reshape(out.grid.shape)[len(y) // 2] & (
        abs(exact) > 0.2 * abs(exact).max()
    )
    error = np.linalg.norm(row[bright] - exact[bright]) / np.linalg.norm(exact[bright])
    return error, report


@pytest.mark.parametrize(
    "n_t, te, tilt, sine",
    [
        (0.05 + 3.3j, True, 0.0, 0.3),
        (0.05 + 3.3j, False, 0.5, 0.3),
        (2 + 4j, False, 0.96, 0.12),
        (1.5 + 0.02j, False, 0.7, 0.12),
    ],
)
def test_a_beam_reflected_by_a_metal_mirror_tends_to_its_plane_wave_sum(
    n_t, te, tilt, sine
):
    """The leading order's error, halving as the system doubles (sizes 1, 2, 4).

    Measured: silver (0.05 + 3.3i) TE at normal incidence 9.4, 4.8, 2.4 %;
    silver TM about 29 degrees 6.7, 3.4, 1.7 %; 2 + 4i TM about 55 degrees
    13, 6.3, 3.2 %; lossy glass (1.5 + 0.02i) TM about 40 degrees, where its
    small r_p turns fast with the angle, 29, 15, 7.4 %.
    """
    errors, reports = zip(
        *(
            cylindrical_beam(size, n_t, te=te, tilt=tilt, sine=sine)
            for size in (1, 2, 4)
        )
    )
    assert all(r.caustics == 0 and r.folded_relative_amplitude == 0 for r in reports)
    assert errors[1] < 0.6 * errors[0] and errors[2] < 0.6 * errors[1]
    assert errors[2] < 0.1


def test_the_pseudo_brewster_minimum_of_a_lossy_glass_is_carried_on_a_curved_mirror():
    """On a convex sphere a collimated beam meets every incidence: the carried
    |r_p|^2 is least at the angle where the boundary problem's is, and that
    least value is not zero (absorption leaves a residual reflection)."""
    from scipy.optimize import minimize_scalar

    n_t = 1.5 + 0.02j
    radius, tilt = 100.0, np.radians(50.0)
    mapping = o.transport.InterfaceMap(
        S.SphericalCap(radius),
        F.Carrier.collimated([np.sin(tilt), 0, np.cos(tilt)]),
        1.0,
        n_t,
        -10.0,
        branch="reflected",
    )
    # Incidence about 50 to 65 degrees, varying across the meridian y = 0; the rays
    # that return to the plane.
    heights = np.linspace(-0.2, 0.2, 4000) * radius
    rays = mapping.forward(np.c_[heights, np.zeros_like(heights)])
    rays = rays.subset(host(rays.valid))
    assert (host(rays.caustics) == 0).all()
    # On the meridian y = 0 the TE axis is y, and p_i = s x s_i.
    s = bd.asarray(np.tile([0, 1.0, 0], (len(rays.valid), 1)))
    p = np.cross(host(s), host(rays.incident))
    reflected = host(
        carried(bd.asarray(p.astype(complex)), rays, s, 1.0, n_t, "reflected")
    )
    R = np.sum(np.abs(reflected) ** 2, axis=-1)
    theta = np.degrees(np.arccos(host(rays.cos_incident)))
    assert theta.min() < 52 and theta.max() > 60

    def exact(degrees):
        d, s_, p_ = incident_wave(degrees, azimuth=0.0)
        e_r, _, _, _ = boundary_value(2 * np.pi * d, p_.astype(complex), 1.0, n_t)
        return np.vdot(e_r, e_r).real

    best = minimize_scalar(
        exact, bounds=(40, 70), method="bounded", options={"xatol": 1e-9}
    )
    assert abs(theta[np.argmin(R)] - best.x) < 0.02
    assert abs(R.min() - best.fun) < 1e-8 and 1e-6 < best.fun < 1e-3
    assert_allclose(R[::400], [exact(t) for t in theta[::400]], rtol=0, atol=1e-14)


def concave_mirror(behind, realization):
    """A collimated beam focused by a concave sphere: the field near its focus."""
    radius = 160.0
    beam = F.PlaneBeam(
        F.profiles.Gaussian(12.0),
        F.polarization.x,
        wavelength0=1.0,
        region=o.media.air,
        frame=Frame([0, 0, -20.0]),
    )
    mirror = S.SphericalCap(-radius)
    options = {"fans": ((15.0, 25.0),)}
    if realization == "lattice":
        options["lattice"] = lattice(40.0)
    solution = P.Propagator(
        Y.Reflection(mirror, o.media.air, behind=behind), **options
    )(beam)
    points = rings(4.0, -radius / 2, 12, 4)
    return host(solution.spectrum.electric(points).values), solution


def lattice(half, step=0.4):
    count = int(2 * half / step) // 2 * 2
    axis = (np.arange(count) - count // 2) * step
    return o.sampling.CartesianGrid(axis, axis)


def rings(radius, z, count=30, azimuths=6):
    r, phi = np.meshgrid(
        np.linspace(0, radius, count), 2 * np.pi * np.arange(azimuths) / azimuths
    )
    return o.sampling.PointGrid(
        np.c_[(r * np.cos(phi)).ravel(), (r * np.sin(phi)).ravel(), np.full(r.size, z)]
    )


@pytest.mark.parametrize("realization", ["meridian", "lattice"])
def test_a_curved_metal_mirror_tends_to_the_ideal_mirror_as_its_index_grows(
    realization,
):
    """r_s -> -1 and r_p -> 1 as |n| -> infinity, as 2 c_i / n and 2 / (n c_i):
    the field at the focus moves from the ideal mirror's by about 2 / |n|."""
    ideal, _ = concave_mirror(None, realization)
    gaps = []
    for size in (10.0, 100.0, 1000.0):
        metal = o.media.Medium(size * (0.3 + 1j), "metal")
        field, solution = concave_mirror(metal, realization)
        gaps.append(np.linalg.norm(field - ideal) / np.linalg.norm(ideal))
        assert "metal" in str(solution) and solution.reports[0].caustics == 0
    assert all(8 < a / b < 12 for a, b in zip(gaps, gaps[1:]))
    assert gaps[0] < 3 / abs(10 * (0.3 + 1j))
