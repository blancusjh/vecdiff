"""The reflected branch of the eikonal leg: map, Jacobian, dyadic and lattice leg.

The reflected ray s_r = s - 2 c_i n returns to the same geometric plane in
the incident medium. Its map is checked against finite differences and
against mirrors that image a point stigmatically; the reflected dyadic
against each plane wave's own boundary-value problem, solved here from
the continuity of tangential E and H (or vanishing tangential E on a
perfect conductor) without any Fresnel formula.
"""

import numpy as np
import pytest
import vecdiff as o
from numpy.testing import assert_allclose

bd, host = o.backend.bd, o.backend.to_numpy
F, S = o.fields, o.surfaces
Frame = o.geometry.Frame
AIR = o.media.HomogeneousRegion("air", o.media.Dielectric(1.0))
GLASS = o.media.HomogeneousRegion("glass", o.media.Dielectric(1.5))
rng = np.random.default_rng(11)


def reflected_map(surface, carrier, offset, n=1.0):
    return o.transport.InterfaceMap(
        surface, carrier, n, 1.5, offset, branch="reflected"
    )


def finite_difference_jacobian(mapping, xy, step):
    columns = []
    for e in np.eye(2):
        near = host(mapping.forward(xy + step * e).output) - host(
            mapping.forward(xy - step * e).output
        )
        far = host(mapping.forward(xy + 2 * step * e).output) - host(
            mapping.forward(xy - 2 * step * e).output
        )
        columns.append((8 * near - far) / (12 * step))
    return np.stack(columns, axis=-1)


TILTED = S.FreeformSurface(
    lambda xy: 0.2 * xy[..., 0] - 0.1 * xy[..., 1],
    lambda xy: bd.broadcast_to(bd.asarray([0.2, -0.1]), bd.asarray(xy).shape),
    hessian_function=lambda xy: bd.zeros(bd.asarray(xy).shape + (2,)),
)


@pytest.mark.parametrize(
    "surface, carrier, offset, half",
    [
        (TILTED, F.Carrier.converging([5.0, -3.0, 400.0]), -20.0, 30.0),
        # convex toward the light, plane in front and behind the vertex
        (S.SphericalCap(80.0), F.Carrier.collimated([0.05, 0.0, 1.0]), -10.0, 30.0),
        (S.SphericalCap(80.0), F.Carrier.diverging([0, 0, -200.0]), 4.0, 25.0),
        # concave: a focus between the surface and the plane
        (S.SphericalCap(-60.0), F.Carrier.collimated([0, 0.03, 1.0]), -45.0, 12.0),
        (
            S.EvenAsphere(-1 / 70, conic=-0.6, coefficients=(2e-7,)),
            F.Carrier.converging([3, -2, 150]),
            -12.0,
            20.0,
        ),
    ],
)
def test_reflected_map_jacobian_is_the_derivative_of_the_map(
    surface, carrier, offset, half
):
    """Appendix A with ds_r = ds - 2 (dc_i n + c_i dn), and g from the tube's power."""
    mapping = reflected_map(surface, carrier, offset)
    xy = rng.uniform(-half, half, (120, 2))
    rays = mapping.forward(xy)
    assert host(rays.valid).all()
    assert (host(rays.transmitted)[:, 2] < 0).all()
    numeric = finite_difference_jacobian(mapping, xy, 1e-4)
    analytic = host(rays.jacobian)
    assert_allclose(analytic, numeric, atol=2e-9 * abs(numeric).max())
    det = np.linalg.det(analytic)
    caustics = host(rays.caustics)
    # Reflection reverses no orientation of the plane's own axes: the ratio
    # changes sign exactly once per collapsed direction.
    assert np.array_equal(det < 0, caustics % 2 == 1)
    sz, rz = host(rays.incident)[:, 2], host(rays.transmitted)[:, 2]
    expected = np.sqrt(np.abs(sz / (rz * det))) * (-1j) ** caustics
    assert_allclose(host(rays.flux), expected, rtol=1e-14)
    assert_allclose(host(rays.cos_transmitted), host(rays.cos_incident), rtol=0)


def test_concave_mirror_counts_the_focus_on_the_way_back():
    """A focus between the mirror and the plane: kappa = 2 and g < 0."""
    mapping = reflected_map(S.SphericalCap(-60.0), F.Carrier.collimated([0, 0, 1]), -45)
    rays = mapping.forward(rng.uniform(-8, 8, (50, 2)))
    assert (host(rays.caustics) == 2).all()
    assert (host(rays.flux).real < 0).all()
    # Before the focus: nothing collapses.
    near = reflected_map(S.SphericalCap(-60.0), F.Carrier.collimated([0, 0, 1]), -10)
    rays = near.forward(rng.uniform(-8, 8, (50, 2)))
    assert (host(rays.caustics) == 0).all() and (host(rays.flux).real > 0).all()


def test_plane_mirror_at_normal_incidence_is_the_identity_map():
    for offset in (-7.0, 0.0, 3.0):
        mapping = reflected_map(S.Plane(), F.Carrier.collimated([0, 0, 1]), offset, 1.3)
        xy = rng.uniform(-5, 5, (20, 2))
        rays = mapping.forward(xy)
        assert host(rays.valid).all()
        assert_allclose(host(rays.output), xy, atol=1e-15)
        assert_allclose(host(rays.jacobian), np.broadcast_to(np.eye(2), (20, 2, 2)))
        assert (host(rays.caustics) == 0).all()
        assert_allclose(host(rays.flux), 1.0, rtol=1e-15)
        assert_allclose(host(rays.path), -2 * 1.3 * offset, atol=1e-14)


def test_paraboloidal_mirror_images_the_collimated_beam_on_its_focus():
    curvature = -0.01
    focus = np.array([0.0, 0.0, 1 / (2 * curvature)])
    mapping = reflected_map(
        S.EvenAsphere(curvature, conic=-1.0), F.Carrier.collimated([0, 0, 1]), -10.0
    )
    xy = rng.uniform(-30, 30, (300, 2))
    rays = mapping.forward(xy)
    assert host(rays.valid).all()
    end = np.c_[host(rays.output), np.full(len(xy), -10.0)]
    toward = (focus - end) / np.linalg.norm(focus - end, axis=-1, keepdims=True)
    assert_allclose(host(rays.transmitted), toward, atol=1e-14)
    path = host(rays.path) + np.linalg.norm(focus - end, axis=-1)
    assert np.ptp(path) < 1e-13 * 100


def test_ellipsoidal_mirror_images_one_focus_on_the_other():
    radius, eccentricity = 100.0, 0.5
    axis = radius / (1 - eccentricity**2)
    near = np.array([0, 0, -axis * (1 - eccentricity)])
    far = np.array([0, 0, -axis * (1 + eccentricity)])
    surface = S.EvenAsphere(-1 / radius, conic=-(eccentricity**2))
    mapping = reflected_map(surface, F.Carrier.diverging(near), -20.0, 1.4)
    xy = rng.uniform(-25, 25, (300, 2))
    rays = mapping.forward(xy)
    assert host(rays.valid).all()
    start = np.c_[xy, np.full(len(xy), -20.0)]
    end = np.c_[host(rays.output), np.full(len(xy), -20.0)]
    toward = (far - end) / np.linalg.norm(far - end, axis=-1, keepdims=True)
    assert_allclose(host(rays.transmitted), toward, atol=1e-14)
    path = (
        1.4 * np.linalg.norm(start - near, axis=-1)
        + host(rays.path)
        + 1.4 * np.linalg.norm(far - end, axis=-1)
    )
    assert_allclose(path, 1.4 * 2 * axis, rtol=1e-14)


def test_reflected_branch_needs_rays_that_head_back_to_the_plane():
    with pytest.raises(ValueError, match="branch"):
        o.transport.InterfaceMap(
            S.Plane(), F.Carrier.collimated([0, 0, 1]), 1, 1.5, branch="back"
        )
    # Grazing a tilted mirror, the reflected ray does not return.
    rays = reflected_map(TILTED, F.Carrier.collimated([0.98, 0, 0.2]), -2).forward(
        np.zeros((1, 2))
    )
    assert not host(rays.valid).any() and host(rays.flux)[0] == 0


def boundary_value(k, e, n_i, n_t, mirror=False):
    """Reflected and transmitted plane waves of e exp(i k . r) on the plane z = 0.

    Each unknown wave is two amplitudes on a basis transverse to its own
    wavevector (bilinearly, for the decaying wave beyond the critical angle,
    Im k_z > 0); tangential E and Z0 H = k x E / k0 are continuous, or
    tangential E vanishes on a perfect conductor. Returns (E_r, k_r, E_t, k_t).
    """
    k = np.asarray(k, complex)
    k0 = np.sqrt(k @ k).real / n_i
    k_r = k * np.array([1, 1, -1])
    k_t = np.array([k[0], k[1], np.sqrt((n_t * k0) ** 2 - k[0] ** 2 - k[1] ** 2)])

    def basis(q):
        a = np.cross([0, 0, 1.0], q)
        if np.linalg.norm(a) < 1e-12 * np.linalg.norm(q):
            a = np.array([1.0, 0, 0])
        return a, np.cross(q, a)

    def h(q, v):
        return np.cross(q, v) / k0

    waves = [(k_r, v) for v in basis(k_r)] + [(k_t, v) for v in basis(k_t)]
    if mirror:
        a, b = (v for _, v in waves[:2])
        x = np.linalg.solve(np.stack([a[:2], b[:2]], -1), -e[:2])
        return x[0] * a + x[1] * b, k_r, None, None
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


def plane_wave_on_a_plane(n_i, n_t, mode, polarization, offset, branch, step=0.3):
    """The carried reflection and transmission of one lattice plane wave, and the exact ones.

    ``polarization`` is (TE, TM) on the plane's transverse axes of the wave.
    """
    from vecdiff.media.medium import Z0

    first = o.media.HomogeneousRegion("first", o.media.Dielectric(n_i))
    second = o.media.HomogeneousRegion("second", o.media.Dielectric(n_t))
    n = 64
    axis = (np.arange(n) - n // 2) * step
    frequency = 2 * np.pi * np.fft.fftfreq(n, step)
    kx, ky = frequency[mode[0]], frequency[mode[1]]
    k = 2 * np.pi * n_i
    kz = np.sqrt(k**2 - kx**2 - ky**2)
    kappa = np.hypot(kx, ky)
    te = np.array([-ky, kx]) / kappa if kappa else np.array([0, 1.0])
    tm = np.array([kx, ky]) / kappa if kappa else np.array([1.0, 0])
    amplitude = polarization[0] * te + polarization[1] * tm
    grid = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, offset]))
    x, y = np.meshgrid(axis, axis)
    values = amplitude * np.exp(1j * (kx * x + ky * y))[..., None]
    incident = F.angular_spectrum(
        F.TransverseElectricField(grid, 1.0, values), first, evanescent="discard"
    )
    interface = o.system.DielectricInterface(S.Plane(), first, second)
    carrier = F.Carrier.collimated([kx, ky, kz])
    e = np.r_[amplitude, -(kx * amplitude[0] + ky * amplitude[1]) / kz]
    e_r, k_r, e_t, k_t = boundary_value([kx, ky, kz], e, n_i, n_t, branch == "mirror")
    leg = o.transport.LatticeTransport(grid, interface, carrier, 1.0, branch=branch)
    carried, report = leg.carry(incident, method="direct")
    out = carried.grid
    u, v = np.meshgrid(host(out.u), host(out.v))
    # The incident wave meets z = 0 a distance -offset beyond the plane and
    # its reflection comes back as far.
    wave = np.exp(1j * (kx * u + ky * v - 2 * kz * offset))[..., None]
    lit = host(leg.lit).reshape(out.shape)
    result = {
        "lit": lit.mean(),
        "carried": abs(host(carried.values) - e_r[:2] * wave)[lit].max(),
        "report": report,
    }
    if offset != 0:
        return result
    # Every output point is lit: the re-expansion is the reflected plane wave.
    back, _ = leg.reflect(incident, method="direct")
    field = back.propagate(0.0)
    E, H = host(field.electric.values), host(field.magnetic)
    h_r = np.cross(k_r, e_r) / (2 * np.pi * Z0)
    occupied = np.any(host(back.expansion.amplitudes) != 0, axis=-1)
    flux = (
        0.5
        * np.real(np.cross(e, np.conj(np.cross([kx, ky, kz], e))))[2]
        / (2 * np.pi * Z0)
    )
    result.update(
        E=abs(E - e_r * wave).max() / abs(e_r).max(),
        H=abs(H - h_r * wave).max() / abs(h_r).max(),
        backward=(host(back.expansion.wavevectors.real)[occupied, 2] < 0).all(),
        R=-0.5 * np.real(np.cross(E, np.conj(H)))[..., 2].mean() / flux,
    )
    if e_t is not None and abs(k_t[2].imag) == 0:
        forward = o.transport.LatticeTransport(grid, interface, carrier, 1.0)
        ahead, _ = forward.transmit(incident, method="direct")
        field = ahead.propagate(0.0)
        E, H = host(field.electric.values), host(field.magnetic)
        result.update(
            E_t=abs(E - e_t * wave).max() / abs(e_t).max(),
            T=0.5 * np.real(np.cross(E, np.conj(H)))[..., 2].mean() / flux,
        )
    return result


PLANE_WAVES = {
    "air to glass, normal": (1.0, 1.5, (0, 0), (1, 0)),
    "air to glass, s": (1.0, 1.5, (5, 3), (1, 0)),
    "air to glass, p": (1.0, 1.5, (5, 3), (0, 1)),
    "air to glass, elliptical, 41 deg": (1.0, 1.5, (12, -4), (1, 0.5j)),
    "air to glass, elliptical, 76 deg": (1.0, 1.5, (18, 6), (0.6, -0.8j)),
    "glass to air, below critical": (1.5, 1.0, (8, 3), (0.3, 1j)),
    "glass to air, total reflection": (1.5, 1.0, (22, 5), (0.3, 1j)),
    "glass to air, total reflection, p": (1.5, 1.0, (25, -2), (0, 1)),
}


@pytest.mark.parametrize("name", list(PLANE_WAVES))
def test_reflected_plane_wave_is_the_exact_reflection_of_the_boundary(name):
    """E and H of the reflected wave satisfy the boundary conditions with the
    transmitted one, and the carried powers add up: R + T = 1."""
    n_i, n_t, mode, polarization = PLANE_WAVES[name]
    result = plane_wave_on_a_plane(n_i, n_t, mode, polarization, 0.0, "reflected")
    assert result["lit"] == 1 and result["backward"]
    assert result["carried"] < 3e-13
    # Phases k . r reach 80 rad on the lattice: roundoff of 1e-13 after the FFTs.
    assert result["E"] < 1e-12 and result["H"] < 1e-12
    if "total" in name:
        assert abs(result["R"] - 1) < 1e-12 and "T" not in result
    else:
        assert result["E_t"] < 1e-12
        assert abs(result["R"] + result["T"] - 1) < 1e-12
        assert 0.03 < result["R"] < 0.6


@pytest.mark.parametrize("offset", [-3.0, 2.5])
def test_reflected_plane_wave_is_exact_from_planes_before_and_behind_the_boundary(
    offset,
):
    for n_i, n_t, mode in ((1.0, 1.5, (12, -4)), (1.5, 1.0, (22, 5))):
        result = plane_wave_on_a_plane(n_i, n_t, mode, (1, 0.5j), offset, "reflected")
        assert result["lit"] > 0.4 and result["carried"] < 3e-13


def test_ideal_mirror_cancels_the_tangential_field_and_reflects_all_power():
    result = plane_wave_on_a_plane(1.0, 1.5, (7, 2), (1, 0.4 + 0.2j), 0.0, "mirror")
    assert result["carried"] < 3e-13 and result["E"] < 1e-12 and result["H"] < 1e-12
    assert abs(result["R"] - 1) < 1e-13
    behind = plane_wave_on_a_plane(1.0, 1.5, (0, 0), (1, 0.2j), -2.0, "mirror")
    assert behind["lit"] == 1 and behind["carried"] < 1e-14


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


def cylindrical_beam(
    size, focus, branch, n_i=1.0, n_t=1.5, te=True, sine=0.3, tilt=0, step=None
):
    """Relative error of a reflected cylindrical beam against its plane-wave sum.

    The boundary is z = 0 and the reference plane z = -40 size; the incident
    beam converges on the line (x_f, focus * 40 size), behind the boundary,
    so its reflection converges on its mirror image: on the way back to the
    plane when focus < 1. Each plane wave of the beam is reflected by the
    coefficients that the boundary-value test above confirmed.
    """
    plane = 40.0 * size
    z_f = focus * plane
    x_f = np.tan(tilt) * (z_f + plane)
    step = 0.4 / n_i if step is None else step
    half = 1.6 * np.tan(np.arcsin(sine)) * (z_f + plane) + 20 + 2 * np.tan(tilt) * plane
    count = int(2 * half / step) // 2 * 2
    x = (np.arange(count) - count // 2) * step
    y = (np.arange(8) - 4) * step
    k = 2 * np.pi * np.fft.fftfreq(count, step)
    offset = k / (2 * np.pi) - n_i * np.sin(tilt)
    amplitude = np.exp(-((offset / (0.5 * sine * n_i)) ** 2)) * (
        abs(offset) < sine * n_i
    )
    kz = np.sqrt(np.maximum((2 * np.pi * n_i) ** 2 - k**2, 0.0))
    amplitude = amplitude * np.exp(-1j * (k * x_f + kz * (plane + z_f)))
    c_i = kz / (2 * np.pi * n_i)
    if branch == "mirror":
        r_s, r_p = -np.ones_like(c_i), np.ones_like(c_i)
    else:
        c_t = np.sqrt((1 - (n_i / n_t) ** 2 * (1 - c_i**2)).astype(complex))
        r_s = (n_i * c_i - n_t * c_t) / (n_i * c_i + n_t * c_t)
        r_p = (n_t * c_i - n_i * c_t) / (n_t * c_i + n_i * c_t)
    # E_y of a TE wave; E_x of a TM wave, whose p_r has E_x = -c_i where p_i has c_i.
    reflected = (r_s if te else -r_p) * amplitude * np.exp(2j * kz * plane)
    values = np.zeros((len(y), count, 2), complex)
    values[..., 1 if te else 0] = amplitude @ np.exp(1j * np.outer(k, x))
    first = o.media.HomogeneousRegion("first", o.media.Dielectric(n_i))
    second = o.media.HomogeneousRegion("second", o.media.Dielectric(n_t))
    grid = o.sampling.CartesianGrid(x, y, Frame([0, 0, -plane]))
    incident = F.angular_spectrum(
        F.TransverseElectricField(grid, 1.0, values), first, evanescent="discard"
    )
    leg = o.transport.LatticeTransport(
        grid,
        o.system.DielectricInterface(S.Plane(), first, second),
        line_carrier(x_f, z_f),
        1.0,
        branch=branch,
        seed_stride=2,
    )
    carried, report = leg.carry(incident, method="direct")
    row = host(carried.values)[len(y) // 2, :, 1 if te else 0]
    exact = reflected @ np.exp(1j * np.outer(k, host(carried.grid.u)))
    bright = host(leg.lit).reshape(carried.grid.shape)[len(y) // 2] & (
        abs(exact) > 0.2 * abs(exact).max()
    )
    error = np.linalg.norm(row[bright] - exact[bright]) / np.linalg.norm(exact[bright])
    return error, report


BEAMS = {
    "mirror": dict(focus=2.0, branch="mirror"),
    "air to glass, TM": dict(focus=2.0, branch="reflected", te=False),
    "air to glass, TE, focal line on the way back": dict(
        focus=0.5, branch="reflected", te=True
    ),
    "glass to air, TE, total reflection at 55 deg": dict(
        focus=2.0, branch="reflected", n_i=1.5, n_t=1.0, sine=0.12, tilt=0.96
    ),
    "glass to air, TM, total reflection at 55 deg": dict(
        focus=2.0, branch="reflected", n_i=1.5, n_t=1.0, sine=0.12, tilt=0.96, te=False
    ),
}


@pytest.mark.parametrize("name", list(BEAMS))
def test_reflected_beam_tends_to_its_plane_wave_sum_as_the_system_grows(name):
    """The leg's error is the leading order's: it halves when the system doubles.

    Measured (sizes 1, 2, 4): 9.4, 4.8, 2.4 % for the mirror; 36, 19, 9.5 %
    with the focal line (kappa = 1; with exp(+i pi / 2) the error would tend
    to 2); 10, 5.0, 2.5 % under total reflection, where r turns with the
    angle (Goos-Haenchen) and |r| = 1.
    """
    errors, reports = zip(
        *(cylindrical_beam(size, **BEAMS[name]) for size in (1, 2, 4))
    )
    focal = "focal line" in name
    assert all(r.caustics == (1 if focal else 0) for r in reports)
    assert all(r.folded_relative_amplitude == 0 for r in reports)
    assert errors[1] < 0.6 * errors[0] and errors[2] < 0.6 * errors[1]
    assert errors[2] < (0.12 if focal else 0.03)


def test_reflected_point_focus_on_the_way_back_inverts_the_field():
    """A concave wavefront on a plane mirror: its focus lies between the mirror
    and the plane, kappa = 2, and the carried field tends to the exact one
    (not to minus it) as the system grows."""
    errors, factors = [], []
    for size in (0.5, 1.0, 2.0):
        plane, focus, sine = 93.0 * size, 53.0 * size, 0.3
        step = 0.4
        half = np.tan(np.arcsin(sine)) * (plane + focus) * 1.25 + 15
        count = int(2 * half / step) // 2 * 2
        axis = (np.arange(count) - count // 2) * step
        k = 2 * np.pi * np.fft.fftfreq(count, step)
        kx, ky = np.meshgrid(k, k)
        kappa = np.hypot(kx, ky)
        amplitude = np.exp(-((kappa / (np.pi * sine)) ** 2)) * (
            kappa < 2 * np.pi * sine
        )
        keep = amplitude > 1e-9
        kx, ky, amplitude = kx[keep], ky[keep], amplitude[keep]
        kz = np.sqrt((2 * np.pi) ** 2 - kx**2 - ky**2)
        spectrum = np.zeros((count, count), complex)
        spectrum[keep] = (
            amplitude
            * np.exp(-1j * kz * (plane + focus))
            * np.exp(1j * axis[0] * (kx + ky))
        )
        values = np.zeros((count, count, 2), complex)
        values[..., 0] = np.fft.ifft2(spectrum) * count**2
        grid = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, -plane]))
        incident = F.angular_spectrum(
            F.TransverseElectricField(grid, 1.0, values), AIR, evanescent="discard"
        )
        leg = o.transport.LatticeTransport(
            grid,
            o.system.DielectricInterface(S.Plane(), AIR, GLASS),
            F.Carrier.converging([0, 0, focus]),
            1.0,
            branch="mirror",
        )
        carried, report = leg.carry(incident)
        assert report.caustics == 2 and report.folded_relative_amplitude == 0
        # Exact: each plane wave's image, tangential E reversed, back on the plane.
        e = np.stack((amplitude, 0 * amplitude, -kx * amplitude / kz), -1)
        image = np.stack((-e[:, 0], -e[:, 1], -(kx * e[:, 0] + ky * e[:, 1]) / kz), -1)
        image = image * np.exp(1j * kz * (plane - focus))[:, None]
        out = carried.grid
        u, v = host(out.u), host(out.v)
        field = host(carried.values)
        strength = abs(field).sum(-1) * host(leg.lit).reshape(out.shape)
        bright = np.argwhere(strength > 0.2 * strength.max())
        chosen = bright[np.random.default_rng(0).choice(len(bright), 60, replace=False)]
        exact = np.stack(
            [np.exp(1j * (u[j] * kx + v[i] * ky)) @ image[:, :2] for i, j in chosen]
        )
        got = field[chosen[:, 0], chosen[:, 1]]
        errors.append(np.linalg.norm(got - exact) / np.linalg.norm(exact))
        factors.append(np.vdot(exact, got) / np.vdot(exact, exact))

    # Measured: 58, 29, 16 %; factors 0.87 - 0.47i, 0.96 - 0.22i, 0.99 - 0.13i.
    assert errors[1] < 0.6 * errors[0] and errors[2] < 0.6 * errors[1]
    assert errors[2] < 0.2 and factors[2].real > 0.95


def gaussian_on(axis, width, z, region=AIR):
    grid = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, z]))
    x, y = np.meshgrid(axis, axis)
    values = np.zeros(grid.shape + (2,), complex)
    values[..., 0] = np.exp(-(x**2 + y**2) / width**2)
    return grid, F.angular_spectrum(
        F.TransverseElectricField(grid, 1.0, values), region, evanescent="discard"
    )


def test_reflected_leg_reports_the_rays_that_do_not_come_back():
    """On a convex mirror of radius R the rays beyond R / sqrt(2) leave forward."""
    radius = 10.0
    axis = (np.arange(100) - 50) * 0.3
    grid, incident = gaussian_on(axis, 8.0, -1.0)
    leg = o.transport.LatticeTransport(
        grid,
        o.system.DielectricInterface(S.SphericalCap(radius), AIR, GLASS),
        F.Carrier.collimated([0, 0, 1]),
        1.0,
        branch="reflected",
    )
    _, report = leg.carry(incident)
    x, y = np.meshgrid(axis, axis)
    beyond = np.hypot(x, y) > radius / np.sqrt(2)
    assert not (leg.support.reshape(grid.shape) & beyond).any()
    power = (abs(host(incident.propagate(0.0).electric.values)) ** 2).sum(-1)
    share = np.sqrt(power[beyond].sum() / power.sum())
    assert share > 0.4 and report.unsupported_relative_amplitude >= share


def test_reflected_leg_refuses_a_fold_beyond_the_marginal_focus():
    """Spherical aberration of a concave mirror: rays that crossed the axis and
    rays that have not yet reach the same points of a plane between the
    marginal and the paraxial focus."""
    axis = (np.arange(100) - 50) * 0.4
    grid, incident = gaussian_on(axis, 15.0, -29.0)
    leg = o.transport.LatticeTransport(
        grid,
        o.system.DielectricInterface(S.SphericalCap(-60.0), AIR, GLASS),
        F.Carrier.collimated([0, 0, 1]),
        1.0,
        branch="mirror",
    )
    with pytest.raises(ValueError, match=r"indices \[0, 2\]\): the ray map folds"):
        leg.carry(incident)


def test_reflected_leg_samples_and_bands_in_the_incident_medium():
    # Total reflection in glass at 55 deg needs 0.27 lambda0 on the lattice.
    with pytest.raises(ValueError, match=r"reflected rays reach.*index 1.5"):
        cylindrical_beam(
            1, 2.0, "reflected", n_i=1.5, n_t=1.0, sine=0.12, tilt=0.96, step=0.4
        )
    count, step = 1000, 0.25
    x = (np.arange(count) - count // 2) * step
    y = (np.arange(8) - 4) * step
    grid = o.sampling.CartesianGrid(x, y, Frame())
    frequencies = 2 * np.pi * np.fft.fftfreq(count, step)

    def reflected(mode):
        kx = frequencies[mode]
        values = np.zeros((len(y), count, 2), complex)
        values[..., 1] = np.exp(1j * kx * x)[None]
        incident = F.angular_spectrum(
            F.TransverseElectricField(grid, 1.0, values), AIR, evanescent="discard"
        )
        leg = o.transport.LatticeTransport(
            grid,
            o.system.DielectricInterface(S.Plane(), AIR, GLASS),
            F.Carrier.collimated([kx, 0, np.sqrt((2 * np.pi) ** 2 - kx**2)]),
            1.0,
            branch="mirror",
        )
        return leg.reflect(incident, method="direct")

    outgoing, report = reflected(248)
    assert report.band_removed_relative_amplitude < 1e-12
    assert_allclose(abs(host(outgoing.propagate(0.0).electric.values)[..., 1]), 1)
    with pytest.raises(ValueError, match="grazing the reference plane"):
        reflected(249)
    with pytest.raises(ValueError, match="use transmit"):
        o.transport.LatticeTransport(
            grid,
            o.system.DielectricInterface(S.Plane(), AIR, GLASS),
            F.Carrier.collimated([0, 0, 1]),
            1.0,
        ).reflect(None)
    with pytest.raises(ValueError, match="branch"):
        o.transport.LatticeTransport(
            grid,
            o.system.DielectricInterface(S.Plane(), AIR, GLASS),
            F.Carrier.collimated([0, 0, 1]),
            1.0,
            branch="back",
        )


def sampled(leg, carried, count=150):
    """Bright lit points of a carried field: their positions (n, 3) and values (n, 2)."""
    out = carried.grid
    field = host(carried.values)
    strength = np.linalg.norm(field, axis=-1) * host(leg.lit).reshape(out.shape)
    bright = np.argwhere(strength > 0.2 * strength.max())
    chosen = bright[
        np.random.default_rng(0).choice(len(bright), min(count, len(bright)), False)
    ]
    u, v = host(out.u)[chosen[:, 1]], host(out.v)[chosen[:, 0]]
    points = host(out.frame.origin) + np.c_[u, v, 0 * u] @ host(out.frame.basis).T
    return points, field[chosen[:, 0], chosen[:, 1]]


def relative(got, exact):
    return np.linalg.norm(got - exact[:, :2]) / np.linalg.norm(exact[:, :2])


def external_reflection(radius):
    """A Gaussian beam of waist R / 5 on a glass sphere, reflected back to a plane
    R / 4 in front of it, against the sphere's Debye term 0 there."""
    from vecdiff.references import MieSphere
    from vecdiff.transport.meridian import BeamField

    plane = Frame([0, 0, -radius / 4])
    beam = F.PlaneBeam(
        F.profiles.Gaussian(radius / 5),
        F.polarization.x,
        wavelength0=1.0,
        region=AIR,
        frame=plane,
    )
    count = int(2 * 0.62 * radius / 0.4) // 2 * 2
    axis = (np.arange(count) - count // 2) * 0.4
    grid = o.sampling.CartesianGrid(axis, axis, plane)
    leg = o.transport.LatticeTransport(
        grid,
        o.system.DielectricInterface(S.SphericalCap(radius), AIR, GLASS),
        beam.carrier,
        1.0,
        branch="reflected",
    )
    carried, report = leg.carry(BeamField(beam, grid))
    points, got = sampled(leg, carried)
    sphere = o.volumes.SphereVolume([0, 0, radius], radius)
    exact = host(
        MieSphere(beam, sphere, GLASS)
        .evaluate(o.sampling.PointGrid(points), term=0, side="exterior")
        .electric.values
    )
    return relative(got, exact), report, (beam, sphere, points, exact)


def test_external_reflection_of_a_sphere_tends_to_debye_term_zero_as_it_grows():
    """Term 0 is (R22 - 1) / 2 h1: the reflected wave R22 h1 / 2 and minus the
    outgoing half of the incident field, which forms the shadow. A beam's
    plane waves all travel forward, so that half lives behind the sphere: on
    the plane in front it is the term 0 of an index-matched sphere, 1e-10 of
    the reflection here. Measured: 25, 13, 6.1 % (R = 10, 20, 40 lambda; 3.2 %
    at 80).
    """
    from vecdiff.references import MieSphere

    errors = []
    for radius in (10.0, 20.0, 40.0):
        error, report, reference = external_reflection(radius)
        assert report.caustics == 0 and report.unsupported_relative_amplitude < 1e-4
        errors.append(error)
        if radius == 10.0:
            beam, sphere, points, exact = reference
            matched = o.media.HomogeneousRegion("matched", o.media.Dielectric(1.0))
            shadow = host(
                MieSphere(beam, sphere, matched)
                .evaluate(o.sampling.PointGrid(points), term=0, side="exterior")
                .electric.values
            )
            assert np.linalg.norm(shadow) < 1e-8 * np.linalg.norm(exact)
    assert errors[1] < 0.6 * errors[0] and errors[2] < 0.6 * errors[1]
    assert errors[2] < 0.08


def internal_reflection(radius):
    """Refracted into a glass sphere toward its aplanatic point B = R / 1.5, then
    reflected once from inside: the two legs against Debye term 2 inside.

    The beam converges on A = 1.5 R; inside it diverges from B to the back,
    whose reflection diverges from 2 R: no caustic on the reflected leg, whose
    plane lies 0.05 R in front of the back vertex.
    """
    from vecdiff.references import MieSphere
    from vecdiff.transport import RefractedRays
    from vecdiff.transport.meridian import BeamField

    sine = 0.25
    beam = F.FocusedBeam(
        lambda x, y: np.clip(1 - (x**2 + y**2) / sine**2, 0, None) ** 3,
        F.polarization.x,
        focal_length=1.0,
        max_sine=sine,
        wavelength0=1.0,
        region=AIR,
        frame=Frame([0, 0, 1.5 * radius]),
        apodization="none",
    )
    half = np.tan(np.arcsin(sine)) * 2.6 * radius + 3
    count = int(2 * half / 0.4) // 2 * 2
    axis = (np.arange(count) - count // 2) * 0.4
    grid = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, -1.1 * radius]))
    front = o.transport.LatticeTransport(
        grid,
        o.system.DielectricInterface(
            S.SphericalCap(radius, Frame([0, 0, -radius])), AIR, GLASS
        ),
        beam.carrier,
        1.0,
    )
    inside, first = front.transmit(BeamField(beam, grid))
    count = int(2 * 0.35 * radius / 0.3) // 2 * 2
    axis = (np.arange(count) - count // 2) * 0.3
    back = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, 0.95 * radius]))
    leg = o.transport.LatticeTransport(
        back,
        o.system.DielectricInterface(
            S.SphericalCap(-radius, Frame([0, 0, radius])), GLASS, AIR
        ),
        RefractedRays(front.map, (grid.u, grid.v)),
        1.0,
        branch="reflected",
    )
    carried, report = leg.carry(inside)
    points, got = sampled(leg, carried)
    exact = host(
        MieSphere(beam, o.volumes.SphereVolume([0, 0, 0], radius), GLASS)
        .evaluate(o.sampling.PointGrid(points), term=2, side="interior")
        .electric.values
    )
    return relative(got, exact), report


def test_internal_reflection_in_a_sphere_tends_to_debye_term_two_as_it_grows():
    """Refraction in by the transmitted leg, one reflection by the reflected leg.

    Measured: 36, 19, 9.7 % (R = 10, 20, 40 lambda; 4.8 % at 80). The share
    of the field on the back plane that no refracted ray reaches, the focus's
    diffraction, falls alike (5.2, 2.8, 1.1 %).
    """
    errors, reports = zip(*(internal_reflection(r) for r in (10.0, 20.0, 40.0)))
    assert all(r.caustics == 0 and r.folded_relative_amplitude == 0 for r in reports)
    shares = [r.unsupported_relative_amplitude for r in reports]
    assert shares[1] < 0.7 * shares[0] and shares[2] < 0.7 * shares[1]
    assert errors[1] < 0.6 * errors[0] and errors[2] < 0.6 * errors[1]
    assert errors[2] < 0.12


def meridian_reflection(radius):
    from vecdiff.transport.meridian import BeamField, MeridianTransport

    plane = Frame([0, 0, -radius / 4])
    beam = F.PlaneBeam(
        F.profiles.Gaussian(radius / 5),
        F.polarization.x,
        wavelength0=1.0,
        region=AIR,
        frame=plane,
    )
    interface = o.system.DielectricInterface(S.SphericalCap(radius), AIR, GLASS)
    # The fan ends before the rays leave the sphere parallel to the plane.
    window = (30.0, 40.0)
    leg = MeridianTransport(
        interface,
        beam.carrier,
        beam.frame,
        1.0,
        offset=-radius / 4,
        window=window,
        branch="reflected",
        scale=radius / 2,
    )
    chart = o.sampling.PolarGrid(
        np.linspace(0, radius, 5), np.arange(4) * np.pi / 2, plane
    )
    field, report = leg.transmit(BeamField(beam, chart))
    return field, report, (beam, interface, window)


def ring_points(radius, depth, rings=40, azimuths=4):
    r, phi = np.meshgrid(
        np.linspace(0, radius, rings), 2 * np.pi * np.arange(azimuths) / azimuths
    )
    return o.sampling.PointGrid(
        np.c_[
            (r * np.cos(phi)).ravel(), (r * np.sin(phi)).ravel(), np.full(r.size, depth)
        ]
    )


def test_meridian_reflection_of_a_sphere_tends_to_debye_term_zero_as_it_grows():
    """E (with E_z) and H of the HankelField that travels back, against term 0.

    Measured on the bright rings: 29, 14, 7.2, 3.6 % (R = 10, 20, 40, 80).
    """
    from vecdiff.references import MieSphere

    errors = []
    for radius in (10.0, 20.0, 40.0, 80.0):
        field, report, (beam, _, _) = meridian_reflection(radius)
        assert report.azimuthal_orders == (-1, 1) and field.region == AIR
        points = ring_points(0.9 * radius, -radius / 4, 60, 6)
        got = field.evaluate(points)
        exact = MieSphere(beam, o.volumes.SphereVolume([0, 0, radius], radius), GLASS)
        exact = exact.evaluate(points, term=0, side="exterior")
        E, H = host(exact.electric.values), host(exact.magnetic)
        bright = np.linalg.norm(E, axis=-1) > 0.2 * np.linalg.norm(E, axis=-1).max()

        def misfit(a, b):
            return np.linalg.norm(a[bright] - b[bright]) / np.linalg.norm(b[bright])

        errors.append(misfit(host(got.electric.values), E))
        assert abs(misfit(host(got.magnetic), H) - errors[-1]) < 0.1 * errors[-1]
    assert all(b < 0.6 * a for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 0.05


def test_lattice_and_meridian_reflections_are_one_field_travelling_back():
    """The lattice leg's LatticeSpectrum and the meridian's HankelField agree on
    the plane and 3 lambda behind it, toward the source (2.5e-5 here)."""
    from vecdiff.transport.meridian import BeamField

    radius = 12.0
    field, _, (beam, interface, window) = meridian_reflection(radius)
    count = int(2 * 0.62 * radius / 0.3) // 2 * 2
    axis = (np.arange(count) - count // 2) * 0.3
    grid = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, -radius / 4]))
    leg = o.transport.LatticeTransport(
        grid, interface, beam.carrier, 1.0, window=window, branch="reflected"
    )
    spectrum, report = leg.reflect(BeamField(beam, grid))
    assert spectrum.direction == -1 and spectrum.region == AIR
    assert report.band_removed_relative_amplitude < 1e-4
    for depth in (-radius / 4, -radius / 4 - 3.0):
        points = ring_points(0.8 * radius, depth)
        a, b = field.evaluate(points), spectrum.evaluate(points)
        for x, y in (
            (a.electric.values, b.electric.values),
            (a.magnetic, b.magnetic),
        ):
            assert np.linalg.norm(host(x) - host(y)) < 1e-4 * np.linalg.norm(host(x))
