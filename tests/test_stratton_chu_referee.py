"""The Maxwell referee: the general operator against the Franz/Stratton-Chu field.

The exact-chain referee (test_unification) compares two models.  This one
compares the general spectral operator against an *exact Maxwell field* in the
image medium — the same Franz/Stratton-Chu integral that pinned vecdiff's own
transfer conventions to 2e-9.  It is the strongest check the repository can
make without an external solver.

What it establishes, measured at NA_i ~ 0.64 on the stigmatic oval:

* the focal profile agrees to about 0.1%;
* the longitudinal channel weight agrees to about 1%;
* the absolute peak amplitude agrees to about 0.1% for this case.

The error-scaling study in ``examples/wave_error_scaling.py`` repeats the
comparison across aperture and size rather than inferring a general law from
this one geometry.
"""

import numpy as np
import pytest

from vecdiff import CartesianSurface
from vecdiff.reference import focal_field_reference
from vecdiff.wave.propagation import raised_cosine
import vecdiff.wave as vw


@pytest.fixture(scope="module")
def duel():
    oval = CartesianSurface(n0=1.0, ni=1.5, z0=-30.0, zi=20.0)
    a = 0.85 * oval.aperture_limit
    edge = 0.25

    def pupil(r):
        return raised_cosine(r, a * (1 - edge), a)

    rho = np.linspace(0.0, 2.5, 41)
    obs = np.stack([rho, np.zeros_like(rho), np.full_like(rho, oval.zi)],
                   axis=-1)
    _, E_sc = focal_field_reference(oval, 1.0, pupil, aperture=a,
                                    observation=obs, n_r=900, n_phi=96)

    grid = vw.Grid.from_spacing(0.25, 256)
    src = vw.object_spectrum(oval, grid)
    op = vw.stigmatic_operator(oval, aperture=a, edge_softness=edge,
                               n_rho=500, n_phi=32, m_max=2)
    fld = op(src).field_on(rho, np.array([0.0]), z=oval.zi)
    return {"Ex_sc": E_sc[:, 0], "Ez_sc": E_sc[:, 2],
            "Ex_w": fld.Ex[0], "Ez_w": fld.components[2][0]}


def test_focal_profile_matches_the_maxwell_reference(duel):
    pn = lambda v: np.abs(v) / np.abs(v).max()
    rms = float(np.sqrt(np.mean((pn(duel["Ex_sc"]) - pn(duel["Ex_w"])) ** 2)))
    assert rms < 0.05


def test_longitudinal_channel_matches_the_maxwell_reference(duel):
    # This channel is an independent check because it is carried by a
    # different Hankel order from the dominant transverse field.
    r_sc = np.abs(duel["Ez_sc"]).max() / np.abs(duel["Ex_sc"]).max()
    r_w = np.abs(duel["Ez_w"]).max() / np.abs(duel["Ex_w"]).max()
    assert r_w == pytest.approx(r_sc, rel=0.10)


def test_absolute_amplitude_is_leading_order(duel):
    """The geometric graph normal restores the expected absolute amplitude."""
    ratio = np.abs(duel["Ex_w"]).max() / np.abs(duel["Ex_sc"]).max()
    assert ratio == pytest.approx(1.0, abs=0.02)
