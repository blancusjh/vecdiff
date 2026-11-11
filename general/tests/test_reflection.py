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
