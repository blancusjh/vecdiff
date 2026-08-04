"""Geometry, transfer operator and reference-solver tests.

These pin the physics rather than the plumbing: the stigmatic condition, the
numerical inversion ``r -> rho``, Snell's law, and -- the one that matters most
-- Eq. (92), the per-channel flux balance between the two reference spheres,
which is what makes the geometric amplitude factor ``A(Q)`` non-negotiable.
"""

import numpy as np
import pytest
from scipy.special import j1

from vecdiff.CartesianSurfaces import CartesianSurface
from vecdiff.fresnel import FresnelOvoid
from vecdiff.geometry import (
    ReferenceSphere,
    TangentPlane,
    grazing_radius,
    image_sphere,
    incident_sphere,
    ray_geometry,
)
from vecdiff.reference.currents import PrescribedSphericalCapCurrents
from vecdiff.reference.stratton_chu import franz_integral_chunked
from vecdiff.transfer import (
    channel_transmittance,
    interface_operator,
    sphere_transfer_eigenvalues,
)

SYSTEMS = [
    (1.0, 1.5, -10.0, 6.0),
    (1.0, 2.4, -6.0, 3.6),
    (1.0, 1.5602, -4.0, 2.0),
    (1.5, 1.0, -3.0, 0.9),
]


def _surface(params):
    n0, ni, z0, zi = params
    return CartesianSurface(n0=n0, ni=ni, z0=z0, zi=zi)


def _pupil(surface, n=801, fraction=0.999):
    return np.linspace(0.0, fraction * surface.aperture_limit, n)


# ------------------------------------------------------------------ #
#  Geometry                                                            #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("params", SYSTEMS)
def test_stigmatic_condition_holds_on_the_pupil_grid(params):
    """Eq. (15): the optical path from A to A' is constant across the surface."""
    n0, ni, z0, zi = params
    surface = _surface(params)
    geom = surface.ray_geometry(_pupil(surface))

    expected = n0 * abs(z0) + ni * abs(zi)
    assert np.allclose(geom.optical_path, expected, rtol=0.0, atol=1e-12 * expected)


@pytest.mark.parametrize("params", SYSTEMS)
def test_rho_from_r_inverts_to_machine_precision(params):
    """The r -> rho inversion is numerical but exact; the analytic route is a quartic."""
    surface = _surface(params)
    r = np.linspace(0.0, 0.999999 * surface.r_fold, 4001)

    rho = surface.rho_from_r(r)

    assert np.max(np.abs(surface.r(rho) - r)) < 1e-12 * max(1.0, surface.r_fold)


@pytest.mark.parametrize("params", SYSTEMS)
def test_rho_from_r_masks_radii_past_the_fold(params):
    """Radii the surface never reaches have no preimage and must be reported."""
    surface = _surface(params)
    r = np.array([0.0, 0.5 * surface.r_fold, 1.5 * surface.r_fold])

    _, valid = surface.rho_from_r(r, return_valid=True)

    assert valid.tolist() == [True, True, False]


@pytest.mark.parametrize("params", SYSTEMS)
def test_snell_law_holds_pointwise(params):
    """Eq. (14), from the normal that RayGeometry builds out of the sag."""
    n0, ni, _, _ = params
    surface = _surface(params)
    geom = surface.ray_geometry(_pupil(surface))

    assert np.max(np.abs(n0 * geom.sin_t0 - ni * geom.sin_ti)) < 1e-10


@pytest.mark.parametrize("params", SYSTEMS)
def test_geometry_is_paraxially_trivial_on_axis(params):
    """A(0) = 1 and the meridional projection is unity: the axis is untouched."""
    surface = _surface(params)
    geom = surface.ray_geometry(np.array([0.0]))

    assert geom.A[0] == pytest.approx(1.0, abs=1e-14)
    assert (geom.cos_ai / geom.cos_a0)[0] == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("params", SYSTEMS)
def test_grazing_bounds_the_usable_aperture(params):
    """Past grazing the oval branch continues onto a sheet A no longer lights."""
    surface = _surface(params)
    limit = surface.aperture_limit

    assert 0.0 < limit <= surface.r_fold
    assert ray_geometry(surface, np.array([limit * 0.999])).valid[0]
    assert not ray_geometry(surface, np.array([limit * 1.01])).valid[0]


def test_grazing_radius_rejects_a_non_refracting_configuration():
    """Object and image on the same side is not a refracting branch; say so."""
    surface = CartesianSurface(n0=1.0, ni=1.5, z0=5.0, zi=10.0)

    with pytest.raises(ValueError, match="not a refracting configuration"):
        grazing_radius(surface)


def test_reference_surfaces_describe_the_two_spheres():
    surface = _surface(SYSTEMS[0])

    assert incident_sphere(surface) == ReferenceSphere(center_z=-10.0, radius=10.0)
    assert image_sphere(surface) == ReferenceSphere(center_z=6.0, radius=6.0)
    assert TangentPlane().z == 0.0


# ------------------------------------------------------------------ #
#  Fresnel and the transfer operator                                   #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("params", SYSTEMS)
def test_fresnel_energy_is_conserved(params):
    """Eq. (31): R_q + T_q = 1 for a lossless dielectric interface."""
    surface = _surface(params)
    r = _pupil(surface)
    fresnel = FresnelOvoid(ovoid=surface)

    rp, rs = fresnel.reflection_coefficients(r)
    Tp, Ts = fresnel.transmittances(r)

    assert np.max(np.abs(rp**2 + Tp - 1.0)) < 1e-12
    assert np.max(np.abs(rs**2 + Ts - 1.0)) < 1e-12


@pytest.mark.parametrize("params", SYSTEMS)
def test_flux_balance_between_the_reference_spheres(params):
    """Eq. (92) -- the note's own verification, and the test that pins A(Q).

    The power a ray tube carries across ``G`` is ``|E_G|^2 |z0|^2 dOmega_0`` and
    across ``G'`` it is ``|E_G'|^2 |zi|^2 dOmega_i``.  Their ratio must be the
    Fresnel power transmittance, channel by channel.  Only the exact ``A``
    makes that true.
    """
    n0, ni, _, _ = params
    surface = _surface(params)
    r = _pupil(surface)
    geom = surface.ray_geometry(r)

    lam_r, lam_phi, _ = sphere_transfer_eigenvalues(surface, r)
    Tp, Ts = channel_transmittance(surface, r)

    # Eq. (49): the solid-angle ratio between the two spheres.
    domega_ratio = (geom.l0**2 * geom.cos_ti) / (geom.li**2 * geom.cos_t0)
    sphere_ratio = (abs(geom.zi) ** 2 * domega_ratio) / abs(geom.z0) ** 2

    # The p channel is carried by the meridional amplitude, so undo the
    # projection that maps E_r on G to E_r on G'.
    lam_p = lam_r * geom.cos_a0 / geom.cos_ai

    flux_p = ni * lam_p**2 * sphere_ratio / n0
    flux_s = ni * lam_phi**2 * sphere_ratio / n0

    assert np.max(np.abs(flux_p - Tp)) < 1e-10
    assert np.max(np.abs(flux_s - Ts)) < 1e-10


@pytest.mark.parametrize("params", SYSTEMS)
def test_transversality_on_the_image_sphere(params):
    """Eqs. (72)-(74): the transferred field is transverse to u_i."""
    surface = _surface(params)
    r = _pupil(surface)
    geom = surface.ray_geometry(r)
    lam_r, _, lam_z = sphere_transfer_eigenvalues(surface, r)

    # E_G' . u_i = -sin(ai) E'_r + cos(ai) E'_z, with E'_r = lam_r E_r and
    # E'_z = lam_z E_r for a unit radial input.
    residual = -geom.sin_ai * lam_r + geom.cos_ai * lam_z

    assert np.max(np.abs(residual)) < 1e-12


@pytest.mark.parametrize("params", SYSTEMS)
def test_geometry_none_reproduces_the_bare_fresnel_weighting(params):
    """The legacy switch must be exactly t_p / t_s, with no factor sneaking in."""
    surface = _surface(params)
    r = _pupil(surface)
    fresnel = FresnelOvoid(ovoid=surface)
    tp, ts = fresnel.coefficients(r)

    lam_r, lam_phi, _ = sphere_transfer_eigenvalues(surface, r, geometry="none")

    assert np.allclose(lam_r, tp, atol=0.0, rtol=0.0)
    assert np.allclose(lam_phi, ts, atol=0.0, rtol=0.0)


def test_interface_operator_rejects_unknown_geometry_mode():
    surface = _surface(SYSTEMS[0])

    with pytest.raises(ValueError, match="geometry must be one of"):
        interface_operator(surface, np.array([0.1]), geometry="sometimes")


@pytest.mark.parametrize("params", SYSTEMS)
def test_operator_is_diagonal_in_the_local_frame(params):
    """The pointwise operator does not depend on the azimuthal structure.

    Applying it to a vortex ``exp(i m phi)`` must give the same answer as the
    Cartesian ``lam_plus``/``lam_minus`` specialisation, for every ``m`` -- that
    is what makes the derivation valid for arbitrary fields.
    """
    surface = _surface(params)
    r = _pupil(surface, n=201)
    phi = np.linspace(0.0, 2.0 * np.pi, 37)[:-1][:, None]

    op = interface_operator(surface, r)
    lam_r, lam_phi, _ = op.eigenvalues()
    lam_plus, lam_minus, _ = op.cartesian_weights()

    for m in (0, 1, 2, 5):
        Ex = np.exp(1j * m * phi) * np.ones_like(r)[None, :]
        Ey = np.zeros_like(Ex)

        c2, s2 = np.cos(2.0 * phi), np.sin(2.0 * phi)
        out_x = (lam_plus + lam_minus * c2) * Ex + (lam_minus * s2) * Ey
        out_y = (lam_minus * s2) * Ex + (lam_plus - lam_minus * c2) * Ey

        # The same thing computed through the local frame instead.
        Er = Ex * np.cos(phi) + Ey * np.sin(phi)
        Ephi = -Ex * np.sin(phi) + Ey * np.cos(phi)
        Er_out, Ephi_out = lam_r * Er, lam_phi * Ephi
        ref_x = Er_out * np.cos(phi) - Ephi_out * np.sin(phi)
        ref_y = Er_out * np.sin(phi) + Ephi_out * np.cos(phi)

        assert np.allclose(out_x, ref_x, atol=1e-12)
        assert np.allclose(out_y, ref_y, atol=1e-12)


# ------------------------------------------------------------------ #
#  The reference solver itself                                         #
# ------------------------------------------------------------------ #

def _converging_cap_focus(NA, R, *, n_theta=400, n_phi=96, obs=None):
    lam = 1.0
    n = 1.0
    k = 2.0 * np.pi * n / lam
    alpha = np.arcsin(NA)

    cap = PrescribedSphericalCapCurrents(
        radius=R, focus_z=0.0, n=n, wavelength=lam,
        amplitude=lambda th: np.ones_like(th), polarization=(1.0, 0.0, 0.0),
    )
    nodes, weights = np.polynomial.legendre.leggauss(n_theta)
    theta = 0.5 * alpha * (nodes + 1.0)
    w_theta = 0.5 * alpha * weights
    phi = np.arange(n_phi) * (2.0 * np.pi / n_phi)
    w_phi = 2.0 * np.pi / n_phi

    Q, J, M, dS = cap.currents_on_cap(theta[None, :], phi[:, None])
    W = (dS * w_theta[None, :] * w_phi).reshape(-1)

    if obs is None:
        obs = np.array([[0.0, 0.0, 0.0]])
    E = franz_integral_chunked(
        obs, Q.reshape(-1, 3), J.reshape(-1, 3), M.reshape(-1, 3), W, k, n, chunk=40_000
    )
    return E, k, R, alpha


def test_franz_integral_matches_the_debye_amplitude():
    """The exact solver must agree with Debye, and disagree only by O(NA^2).

    This is what licenses the reference: the residual is not solver error, it
    is the Debye approximation's own leading term, and it scales as NA^2 / 8.
    """
    ratios = {}
    for NA, R in ((0.4, 500.0), (0.2, 2000.0), (0.1, 8000.0)):
        E, k, R_cap, alpha = _converging_cap_focus(NA, R)
        debye = np.exp(1j * k * R_cap) * (-1j * k * R_cap * (1.0 - np.cos(alpha)))
        ratios[NA] = E[0, 0] / debye

    for NA, ratio in ratios.items():
        assert abs(ratio.imag) < 1e-3
        assert (1.0 - ratio.real) / NA**2 == pytest.approx(0.125, abs=0.006)


def test_franz_integral_reproduces_the_airy_pattern():
    """Shape check against the closed form, independent of the amplitude check."""
    NA, R, lam = 0.2, 2000.0, 1.0
    s = np.linspace(0.0, 6.0, 61) * lam
    obs = np.stack([s, np.zeros_like(s), np.zeros_like(s)], axis=-1)

    E, k, _, _ = _converging_cap_focus(NA, R, obs=obs)

    v = k * NA * s
    airy = np.where(v > 0.0, 2.0 * j1(np.where(v > 0.0, v, 1.0)) / np.where(v > 0.0, v, 1.0), 1.0)
    profile = np.abs(E[:, 0]) / np.abs(E[0, 0])

    assert np.max(np.abs(profile - np.abs(airy))) < 3e-3


def test_franz_integral_keeps_the_focus_linearly_polarized_at_low_na():
    E, _, _, _ = _converging_cap_focus(0.05, 32_000.0)

    assert abs(E[0, 1]) / abs(E[0, 0]) < 1e-12
    assert abs(E[0, 2]) / abs(E[0, 0]) < 1e-12


# ------------------------------------------------------------------ #
#  End to end: the package against the exact field                     #
# ------------------------------------------------------------------ #

def _uniform_pupil_field(surface, aperture, n_r=2001):
    from vecdiff import Field, Grid

    r = np.linspace(0.0, aperture, n_r)
    grid = Grid.from_polar(r, np.array([0.0]))
    return Field.from_cartesian(
        np.ones_like(r, dtype=complex), np.zeros_like(r, dtype=complex), grid, symmetric=True
    )


def test_package_reproduces_the_exact_focal_field():
    """The whole chain, default settings, against the Franz reference.

    This is the test the geometric factor and the pupil mapping exist for: at
    NA_i = 0.91 the corrected package matches an exact Maxwell field in
    absolute amplitude, phase and shape, while the old weighting is 30 percent
    low and misshapen.
    """
    from vecdiff.pupil_mapping import debye_prefactor
    from vecdiff.reference import focal_field_reference

    lam = 532e-6
    surface = CartesianSurface(n0=1.0, ni=1.5, z0=-10.0, zi=6.0)
    aperture = 0.999 * surface.aperture_limit
    k_i = 2.0 * np.pi * surface.ni / lam

    rho = np.linspace(0.0, 2.0, 41) * lam
    q = k_i * rho / abs(surface.zi)
    q[0] = 1e-9  # the propagator wants a strictly increasing q

    field = _uniform_pupil_field(surface, aperture)
    scale = debye_prefactor(surface, lam) / (2.0 * np.pi)
    Ex_full = field.propagate_through_diopter(surface.zi, surface, q).x[0] * scale
    Ex_legacy = (
        field.propagate_through_diopter(surface.zi, surface, q, geometry="none").x[0] * scale
    )

    obs = np.stack([rho, np.zeros_like(rho), np.full_like(rho, surface.zi)], axis=-1)
    _, E_ref = focal_field_reference(
        surface, lam, lambda r: np.ones_like(r),
        aperture=aperture, observation=obs, n_r=1200, n_phi=128, chunk=24_000,
    )
    Ex_ref = E_ref[:, 0]

    reference_profile = np.abs(Ex_ref) / np.abs(Ex_ref).max()
    full_profile = np.abs(Ex_full) / np.abs(Ex_full).max()

    assert abs(Ex_full[0] / Ex_ref[0]) == pytest.approx(1.0, abs=1e-5)
    assert abs(np.angle(Ex_full[0] / Ex_ref[0])) < 1e-3
    assert np.sqrt(np.mean((full_profile - reference_profile) ** 2)) < 1e-6

    # And the weighting the package used before is decisively worse.
    assert abs(Ex_legacy[0] / Ex_ref[0]) < 0.8


def test_hankel_and_fft_branches_agree():
    """Cross-check the two propagation paths -- a long-standing roadmap gap.

    They share no code beyond the transfer weights: one integrates over a
    non-uniform pupil abscissa with Simpson, the other warps onto a uniform
    grid and calls an FFT.
    """
    from vecdiff import Field, Grid

    lam = 532e-6
    surface = CartesianSurface(n0=1.0, ni=1.5, z0=-10.0, zi=6.0)
    aperture = 0.90 * surface.aperture_limit
    k_i = 2.0 * np.pi * surface.ni / lam

    rho = np.linspace(0.0, 3.0, 41) * lam
    q = k_i * rho / abs(surface.zi)
    q[0] = 1e-9
    hankel = _uniform_pupil_field(surface, aperture).propagate_through_diopter(
        surface.zi, surface, q
    ).x[0]

    n = 1024
    axis = np.linspace(-1.15 * aperture, 1.15 * aperture, n)
    grid = Grid.from_axes(axis, axis)
    mask = (grid.R <= aperture).astype(complex)
    field = Field.from_cartesian(mask, np.zeros(grid.shape, dtype=complex), grid, symmetric=False)
    kx = k_i * rho / abs(surface.zi)
    kgrid = Grid.from_axes(kx, np.array([-1e-9, 0.0, 1e-9]), domain="k")
    fft = field.propagate_through_diopter(
        surface.zi, surface, method="fft", wavelength=lam, kgrid=kgrid, output="k"
    ).x[1]

    assert abs(fft[0] / hankel[0]) == pytest.approx(1.0, abs=1e-3)
    profile_h = np.abs(hankel) / np.abs(hankel).max()
    profile_f = np.abs(fft) / np.abs(fft).max()
    assert np.sqrt(np.mean((profile_h - profile_f) ** 2)) < 1e-3


def test_plane_referenced_input_is_remapped_onto_the_sphere():
    """Eqs. (85)-(86): a field given on the tangent plane P is not a field on G."""
    from vecdiff import Field, Grid
    from vecdiff.geometry import TangentPlane
    from vecdiff.propagation import incident_field_on_sphere

    surface = CartesianSurface(n0=1.0, ni=1.5, z0=-10.0, zi=6.0)
    geom = surface.ray_geometry(np.linspace(0.0, 0.9 * surface.aperture_limit, 33))
    r_plane = geom.r_tan_obj

    grid = Grid.from_polar(r_plane, np.array([0.0]), reference=TangentPlane(0.0))
    field = Field.from_cartesian(
        np.ones_like(r_plane, dtype=complex), np.zeros_like(r_plane, dtype=complex),
        grid, symmetric=True,
    )

    on_sphere = incident_field_on_sphere(field, surface)

    # The samples land back on the surface radii they came from ...
    assert np.allclose(on_sphere.grid.r, geom.r, atol=1e-6)
    # ... carrying the 1 / |cos(alpha_0)| flux factor.
    assert np.allclose(np.abs(on_sphere.x[0]), 1.0 / np.abs(geom.cos_a0), rtol=1e-5)


def test_sphere_referenced_input_passes_through_untouched():
    from vecdiff import Field, Grid
    from vecdiff.geometry import ReferenceSphere, incident_sphere
    from vecdiff.propagation import incident_field_on_sphere

    surface = CartesianSurface(n0=1.0, ni=1.5, z0=-10.0, zi=6.0)
    r = np.linspace(0.0, 0.9 * surface.aperture_limit, 17)
    grid = Grid.from_polar(r, np.array([0.0]), reference=incident_sphere(surface))
    field = Field.from_cartesian(
        np.ones_like(r, dtype=complex), np.zeros_like(r, dtype=complex), grid, symmetric=True
    )

    assert incident_field_on_sphere(field, surface) is field

    wrong = Grid.from_polar(r, np.array([0.0]), reference=ReferenceSphere(center_z=-3.0, radius=3.0))
    stray = Field.from_cartesian(
        np.ones_like(r, dtype=complex), np.zeros_like(r, dtype=complex), wrong, symmetric=True
    )
    with pytest.raises(ValueError, match="not the incident sphere"):
        incident_field_on_sphere(stray, surface)


def test_pupil_mapping_round_trips():
    from vecdiff.pupil_mapping import (
        pupil_coordinate,
        radius_from_pupil_coordinate,
        resolve_mapping,
    )

    surface = CartesianSurface(n0=1.0, ni=1.5, z0=-10.0, zi=6.0)
    r = np.linspace(0.0, 0.99 * surface.aperture_limit, 501)
    geom = surface.ray_geometry(r)

    for mapping in ("identity", "sine", "tangent"):
        u = pupil_coordinate(geom, mapping)
        back, valid = radius_from_pupil_coordinate(surface, u, mapping)
        assert valid.all()
        assert np.max(np.abs(back - r)) < 1e-4 * surface.aperture_limit

    assert resolve_mapping(None, "full") == "sine"
    assert resolve_mapping(None, "none") == "identity"
    with pytest.raises(ValueError, match="mapping must be one of"):
        resolve_mapping("parabolic", "full")
