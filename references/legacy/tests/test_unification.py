"""The referee: the general spectral operator against the exact stigmatic chain.

vecdiff's classic path solves the Cartesian oval essentially exactly (it is
itself pinned against a Franz/Stratton-Chu Maxwell reference).  The general
operator of ``vecdiff.wave`` claims every smooth surface at leading order in
1/kR.  On the one surface both can do, they must agree — this is what licenses
trusting the general operator where no exact answer exists.
"""

import numpy as np
import pytest

from references.legacy.vecdiff import CartesianSurface
import references.legacy.vecdiff.wave as vw


@pytest.fixture(scope="module")
def oval():
    return CartesianSurface(n0=1.0, ni=1.5, z0=-30.0, zi=20.0)


@pytest.fixture(scope="module")
def verdict(oval):
    return vw.referee(oval, wavelength=1.0, n_rho=500, n_phi=32, m_max=2)


def test_oval_surface_reproduces_the_host_sag(oval):
    surf = vw.oval_surface(oval)
    r = np.linspace(0.0, 0.9 * oval.aperture_limit, 41)
    assert np.allclose(surf.sag(r), oval.sag(r), atol=1e-12)
    assert surf.max_radius == pytest.approx(oval.aperture_limit)
    # slope agrees with the host's ray geometry
    geom = oval.ray_geometry(r[1:])
    assert np.allclose(surf.dsag(r[1:]), geom.dz_dr, atol=1e-5)


def test_stigmatic_operator_reads_the_oval(oval):
    op = vw.stigmatic_operator(oval)
    assert op.n1 == pytest.approx(oval.n0)
    assert op.n2 == pytest.approx(oval.ni)
    assert op.aperture == pytest.approx(oval.aperture_limit)


def test_focal_profiles_agree(verdict):
    # normalized |Ex| focal-plane cuts, NA_i ~ 0.6
    assert verdict["profile_rms"] < 0.05


def test_focal_widths_agree(verdict):
    assert verdict["fwhm_general"] == pytest.approx(verdict["fwhm_exact"],
                                                    rel=0.15)


def test_longitudinal_channel_agrees(verdict):
    # |Ez|/|Ex| peak ratio: an independent cross-check — a different Hankel
    # order on the exact side, the surface integral on the general side.
    assert verdict["ez_ratio_general"] == pytest.approx(
        verdict["ez_ratio_exact"], rel=0.10)


def test_absolute_peak_amplitude_is_leading_order(verdict):
    # Same source normalisation on both sides; the general operator is
    # leading order in 1/kR, so amplitudes agree to tens of percent, not
    # orders of magnitude.
    ratio = np.abs(verdict["Ex_general"]).max() / np.abs(verdict["Ex_exact"]).max()
    assert 0.5 < ratio < 2.0
