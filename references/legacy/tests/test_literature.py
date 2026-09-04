"""Validation against published results.

Two canonical references of vectorial focusing:

* Richards & Wolf, Proc. R. Soc. A 253, 358 (1959): the aplanatic focal field
  as the three integrals I0, I1, I2.  An independent theta-quadrature of those
  integrals is compared against the wave engine's Pupil field.
* Quabis, Dorn, Eberler, Gloeckl & Leuchs, Opt. Commun. 179, 1 (2000),
  "Focusing light to a tighter spot": half-maximum spot areas at NA = 0.9 and
  the radial-doughnut Ez channel (their Table 1; experimental confirmation
  Dorn et al., PRL 91, 233901 (2003)).

The published Table-1 values quoted here are the ones vecdiff's own
replication notebooks (docs/notebooks/tighter_spot) reproduce to 1-4%.
"""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import jv

import references.legacy.vecdiff.wave as vw


# ---------------------------------------------------------------- helpers
def rw_profiles(NA, n, ell, s, n_theta=4000):
    """Richards-Wolf / Quabis integrals by direct theta-quadrature."""
    alpha = np.arcsin(NA / n)
    theta = np.linspace(0.0, alpha, n_theta)
    u, c = np.sin(theta), np.cos(theta)
    lw = ell(u / (NA / n)) * np.sqrt(c)
    dt = theta[1] - theta[0]
    arg = 2 * np.pi * n * s[:, None] * u[None, :]
    J0, J1, J2 = jv(0, arg), jv(1, arg), jv(2, arg)
    tr = np.trapezoid
    return {
        "I0": tr(lw * u * (1 + c) * J0, dx=dt, axis=1),
        "I1": tr(lw * u**2 * J1, dx=dt, axis=1),
        "I2": tr(lw * u * (1 - c) * J2, dx=dt, axis=1),
    }


def _rms(a, b):
    a = np.abs(a) / np.abs(a).max()
    b = np.abs(b) / np.abs(b).max()
    return float(np.sqrt(np.mean((a - b) ** 2)))


# the doughnut width of Quabis Table 1: 90% power through the pupil
_T = brentq(lambda T: np.exp(-T) * (1.0 + T) - 0.1, 1.0, 20.0)
GAMMA_DOUGHNUT = _T / 2.0


def doughnut(v, phi):
    return v * np.exp(-GAMMA_DOUGHNUT * v**2)


def doughnut_annular(v, phi):
    return np.where(v >= 0.9, v * np.exp(-GAMMA_DOUGHNUT * v**2), 0.0)


def spot_area(NA, polarization, amplitude, component="total"):
    """Half-maximum area of the focal energy density, in lambda^2."""
    pup = vw.Pupil(na=min(NA, 0.9999), n=1.0, wavelength=1.0,
                   polarization=polarization, amplitude=amplitude)
    spec = pup.spectrum(vw.Grid.from_spacing(0.2, 512), edge_softness=1e-3)
    ax = np.linspace(-1.6, 1.6, 641)
    f = spec.field_on(ax, ax, 0.0)
    I = f.intensity if component == "total" else np.abs(f.components[2]) ** 2
    return float(np.sum(I >= 0.5 * I.max()) * (ax[1] - ax[0]) ** 2)


# ------------------------------------------------------- Richards & Wolf
class TestRichardsWolf:
    NA, n = 0.9, 1.0

    @pytest.fixture(scope="class")
    def cuts(self):
        grid = vw.Grid.from_spacing(0.25, 512)
        pup = vw.Pupil(na=self.NA, n=self.n, wavelength=1.0, polarization="x")
        spec = pup.spectrum(grid, edge_softness=1e-3)
        s = np.linspace(0.0, 2.0, 81)
        fx = spec.field_on(s, np.array([0.0]), 0.0)
        fy = spec.field_on(np.array([0.0]), s, 0.0)
        p = rw_profiles(self.NA, self.n, lambda v: np.ones_like(v), s)
        return s, fx, fy, p

    def test_ex_along_both_azimuths(self, cuts):
        s, fx, fy, p = cuts
        # Ex = I0 + I2 cos 2phi
        assert _rms(fx.Ex[0], p["I0"] + p["I2"]) < 2e-3
        assert _rms(fy.Ex[:, 0], p["I0"] - p["I2"]) < 2e-3

    def test_ez_profile_and_weight(self, cuts):
        s, fx, fy, p = cuts
        # Ez = -2i I1 cos phi
        assert _rms(fx.components[2][0], 2 * p["I1"]) < 2e-3
        ratio = np.abs(fx.components[2][0]).max() / np.abs(fx.Ex[0]).max()
        ratio_rw = (2 * p["I1"]).max() / np.abs(p["I0"] + p["I2"]).max()
        assert ratio == pytest.approx(ratio_rw, rel=1e-2)

    def test_high_na_longitudinal_fraction(self):
        # analytic 0.139 at NA = 1.2 in n = 1.6 (Richards-Wolf integrals)
        pup = vw.Pupil(na=1.2, n=1.6, wavelength=1.0, polarization="x")
        spec = pup.spectrum(vw.Grid.from_spacing(0.15, 256))
        x = np.linspace(-2.0, 2.0, 161)
        fr = spec.field_on(x, x, 0.0).component_fractions()
        assert fr["z"] == pytest.approx(0.139, rel=0.025)


# ------------------------------------------------------------ Quabis 2000
class TestQuabis2000:
    def test_ez_channel_areas_match_table_1(self):
        # published "Ez only" row for the radial doughnut (lambda^2)
        published = {0.7: 0.373, 0.9: 0.212}
        for NA, target in published.items():
            area = spot_area(NA, "radial", doughnut, component="z")
            assert area == pytest.approx(target, rel=0.08)

    def test_tighter_spot_at_high_na(self):
        """The headline result: the annular radial doughnut beats linear
        polarization at NA = 0.9 (published ratio 0.54)."""
        a_lin = spot_area(0.9, "x", None)
        a_rad = spot_area(0.9, "radial", doughnut_annular)
        assert a_rad < a_lin
        assert a_rad / a_lin == pytest.approx(0.54, abs=0.06)

    def test_dorn_2003_spot_area(self):
        """Dorn/Quabis/Leuchs PRL 91, 233901 configuration: annular radial
        doughnut at NA = 0.9, published (theory) area 0.166 lambda^2 —
        the repo's own Debye-Wolf replication lands at 0.175."""
        area = spot_area(0.9, "radial", doughnut_annular)
        assert area == pytest.approx(0.175, rel=0.05)
        assert area == pytest.approx(0.166, rel=0.10)
