"""Properties that separate the two readings of the incidence cosine."""

import numpy as np
import pytest

from references.legacy.vecdiff import CartesianSurface
from references.legacy.vecdiff.spectral_interface import (
    PlaneWaveSet,
    flat_profile,
    fresnel_operator_rank,
    line_source,
    profile_from_surface,
    surface_field_local,
    surface_field_spectral,
    transmission,
)

N1, N2 = 1.0, 1.5
K1 = 2.0 * np.pi
POLS = ("TE", "TM")


def _surface():
    return CartesianSurface(n0=N1, ni=N2, z0=-40.0, zi=24.0)


def _pupil(n_x=1201, frac=0.70):
    surface = _surface()
    a = frac * surface.aperture_limit
    return profile_from_surface(surface, np.linspace(-a, a, n_x))


def _sources(offsets, n_k=601, sin_max=0.45):
    kx = np.linspace(-sin_max, sin_max, n_k) * K1
    amp = np.zeros(n_k, dtype=complex)
    for x_s in offsets:
        amp += line_source(kx, K1, x_s, -40.0)
    amp = amp * np.exp(-4.0 * (kx / (sin_max * K1)) ** 8)
    return PlaneWaveSet(kx=kx, amp=amp * (kx[1] - kx[0]), k=K1)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("polarization", POLS)
def test_spectral_channel_is_exact_on_a_plane(polarization):
    """On a plane the tangent-plane model *is* the rigorous solution."""
    profile = flat_profile(np.linspace(-30.0, 30.0, 2001))
    theta = np.radians([12.0, 35.0, 58.0])
    pw = PlaneWaveSet(kx=K1 * np.sin(theta), amp=np.ones(3, dtype=complex), k=K1)

    t, _ = transmission(np.cos(theta), N1, N2, polarization)
    exact = (pw.phasor(profile.x, profile.z) * t * pw.amp).sum(axis=1)
    psi, _ = surface_field_spectral(pw, profile, N1, N2, polarization)
    assert _rel(psi, exact) < 1e-13


@pytest.mark.parametrize("polarization", POLS)
def test_channels_agree_for_a_single_plane_wave(polarization):
    """One plane wave has one direction, so the two readings must coincide."""
    profile = _pupil(n_x=801)
    theta = np.radians(6.0)
    pw = PlaneWaveSet(
        kx=np.array([K1 * np.sin(theta)]), amp=np.ones(1, dtype=complex), k=K1
    )
    psi_s, _ = surface_field_spectral(pw, profile, N1, N2, polarization)
    psi_l, _, sin_i = surface_field_local(pw, profile, N1, N2, polarization)

    interior = np.abs(profile.x) < 0.9 * np.abs(profile.x).max()
    assert _rel(psi_l[interior], psi_s[interior]) < 1e-6

    # The phase gradient measures the angle to the local normal, not to the
    # axis: with the surface tilted by beta = arctan(dz/dx), that is
    # sin(theta + beta), and the two channels agree on it point by point.
    tilt = np.arctan(profile.dz_dx)
    assert np.allclose(sin_i[interior], np.sin(theta + tilt[interior]), atol=1e-5)


@pytest.mark.parametrize("polarization", POLS)
def test_spectral_is_linear_and_local_is_not(polarization):
    """The exact operator is linear; only one of the two channels is.

    This is the whole argument in one assertion: superposing two illuminations
    and transmitting is the same as transmitting each and superposing, and a
    model that violates it cannot be the transmitted field of anything.
    """
    profile = _pupil()
    a, b = _sources([-11.0]), _sources([11.0])
    both = PlaneWaveSet(kx=a.kx, amp=a.amp + b.amp, k=K1)

    sa, _ = surface_field_spectral(a, profile, N1, N2, polarization)
    sb, _ = surface_field_spectral(b, profile, N1, N2, polarization)
    sab, _ = surface_field_spectral(both, profile, N1, N2, polarization)
    assert _rel(sab, sa + sb) < 1e-13

    la, _, _ = surface_field_local(a, profile, N1, N2, polarization)
    lb, _, _ = surface_field_local(b, profile, N1, N2, polarization)
    lab, _, _ = surface_field_local(both, profile, N1, N2, polarization)
    assert _rel(lab, la + lb) > 1e-2


def test_local_direction_leaves_the_unit_disc_when_rays_overlap():
    """Two rays at a point give a phase gradient that is nobody's wavevector."""
    profile = _pupil()
    _, _, sin_one = surface_field_local(_sources([0.0]), profile, N1, N2, "TE")
    _, _, sin_two = surface_field_local(_sources([-11.0, 11.0]), profile, N1, N2, "TE")

    assert np.max(np.abs(sin_one)) < 1.0
    assert np.max(np.abs(sin_two)) > 1.0


@pytest.mark.parametrize("polarization", POLS)
def test_fresnel_kernel_is_low_rank(polarization):
    """``t(k_hat . n_hat)`` is a smooth function of a bilinear form, hence cheap.

    The spectral channel looks like a dense ``Q``-by-``kappa`` operator; this is
    the reason it is not, and the reason the sum over the spectrum costs a
    handful of transforms rather than one per spectral sample.
    """
    profile = _pupil()
    rank, singular = fresnel_operator_rank(
        _sources([-11.0, 11.0]), profile, N1, N2, polarization, tol=1e-10
    )
    assert rank < 20
    assert singular[0] > 0.0
