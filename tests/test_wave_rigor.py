"""Physics-rigor tests of the general spectral operator.

Three constraints the theory imposes on any implementation:

* the planar-interface limit must reduce to the diagonal Fresnel multiplier,
  direction by direction and polarization by polarization;
* energy: the local Fresnel boundary map is lossless, while the radiated
  surface approximation is leading order in ``1/kR``; the tests quantify that
  flux budget and distinguish the normalized Franz measure from the bare
  amplitude transform;
* the discretisation (radial samples, azimuthal harmonics) must be converged.
"""

import numpy as np
import pytest

import vecdiff.wave as vw
from vecdiff.wave.interfaces import fresnel as fresnel_coeffs, reflect_field
from vecdiff.wave.propagation import _axisym_samples, _rim_tir_apodization
from vecdiff.wave.spectrum import AngularSpectrum


def _oblique_mode(grid, kx_frac, n, pol, wavelength=1.0):
    """A single oblique propagating plane-wave mode, 's' or 'p' polarized."""
    k = 2 * np.pi * n / wavelength
    kxs = 2 * np.pi * np.fft.fftfreq(grid.x.size, d=grid.dx)
    ix = int(np.argmin(np.abs(kxs - kx_frac * k)))
    kx = kxs[ix]
    kz = np.sqrt(k * k - kx * kx)
    area = (grid.x.size * grid.dx) * (grid.y.size * grid.dy)
    A = np.zeros((3, *grid.shape), dtype=complex)
    if pol == "s":
        A[1, 0, ix] = area
    else:
        A[0, 0, ix] = (kz / k) * area
        A[2, 0, ix] = -(kx / k) * area
    return AngularSpectrum(A, grid, wavelength, n, +1), kx / k


def _fresnel_analytic(cos_i, n1, n2):
    mu = n1 / n2
    cos_t = np.sqrt(1 - mu**2 * (1 - cos_i**2))
    ts = 2 * n1 * cos_i / (n1 * cos_i + n2 * cos_t)
    tp = 2 * n1 * cos_i / (n2 * cos_i + n1 * cos_t)
    return ts, tp


@pytest.mark.parametrize("pol", ["s", "p"])
@pytest.mark.parametrize("kx_frac", [0.0, 0.3, 0.6])
def test_planar_interface_reduces_to_fresnel(pol, kx_frac):
    """Field transmitted by a plane must be the analytic Fresnel amplitude."""
    grid = vw.Grid.from_spacing(0.25, 256)
    n1, n2 = 1.0, 1.5
    op = vw.InterfaceOperator(vw.Plane(), n1=n1, n2=n2, aperture=24.0,
                              edge_softness=0.15)
    spec, sin_i = _oblique_mode(grid, kx_frac, n1, pol)
    ts, tp = _fresnel_analytic(np.sqrt(1 - sin_i**2), n1, n2)

    c = grid.shape[0] // 2
    f_in = spec.field(0.0)
    f_out = op(spec).field(0.0)
    Ein = np.stack([f_in.Ex, f_in.Ey, f_in.components[2]])[:, c, c]
    Eout = np.stack([f_out.Ex, f_out.Ey, f_out.components[2]])[:, c, c]
    if pol == "s":
        ratio = abs(Eout[1] / Ein[1])
        t_ref = ts
    else:
        ratio = np.linalg.norm(Eout[[0, 2]]) / np.linalg.norm(Ein[[0, 2]])
        t_ref = tp
    # residual is finite-window diffraction; measured ~5e-3
    assert ratio == pytest.approx(t_ref, rel=1.5e-2)


@pytest.mark.parametrize("measure,closure_t", [
    # flat measure: transmitted flux overshoots slightly (+2% here); franz
    # measure: undershoots (-16% at this very steep surface, kR ~ 38, and
    # shrinking with angle range) — the two brackets of the leading-order
    # amplitude problem, documented rather than hidden.
    ("flat", 1.02),
    ("franz", 0.84),
])
def test_energy_flux_closure_on_a_curved_interface(measure, closure_t):
    """Document the leading-order flux budget and referee reflection locally."""
    n1, n2, ap = 1.0, 1.5, 4.5
    grid = vw.Grid.from_spacing(0.25, 256)
    surf = vw.Conic(radius=+6.0, conic=vw.stigmatic_conic_constant(n1, n2))
    pw = vw.plane_wave_spectrum(grid, wavelength=1.0, n=n1, polarization="x")
    common = dict(aperture=ap, edge_softness=0.25, n_rho=500, n_phi=32,
                  m_max=2, measure=measure)
    out_t = vw.InterfaceOperator(surf, n1=n1, n2=n2, mode="t", **common)(pw)
    out_r = vw.InterfaceOperator(surf, n1=n1, n2=n2, mode="r", **common)(pw)

    def flux(spec):
        mask = spec.grid.propagating(spec.k)
        dens = np.sum(np.abs(spec.A) ** 2, axis=0)
        area = (spec.grid.x.size * spec.grid.dx) * (spec.grid.y.size * spec.grid.dy)
        return float(spec.n * np.sum((spec.kz[mask] / spec.k) * dens[mask]) / area)

    # incident flux of the unit plane wave through the same apodized window
    smp = _axisym_samples(surf, ap, 500, 32)
    rho = smp["rho"]
    khat = np.zeros((3, smp["points"].shape[1]))
    khat[2] = 1.0
    coeffs = fresnel_coeffs(khat, smp["nhat"], n1, n2)
    vis = _rim_tir_apodization(rho, 32, ap, 0.25, coeffs, "t", n1, n2, 0.04)
    Fi = n1 * float(np.sum(vis**2 * rho) * (rho[1] - rho[0]) * 2 * np.pi)

    Ft, Fr = flux(out_t), flux(out_r)
    assert Ft / Fi == pytest.approx(closure_t, abs=0.06)
    # The local Fresnel reflection is the appropriate power reference for the
    # curved surface (not the normal-incidence value at every point).
    E = np.zeros_like(khat, dtype=complex)
    E[0] = 1.0
    Er, _, _ = reflect_field(E, khat, smp["nhat"], n1, n2)
    R_rho = np.sum(np.abs(Er) ** 2, axis=0).reshape(len(rho), 32).mean(axis=1)
    R_ref = float(np.sum(vis**2 * rho * R_rho) / np.sum(vis**2 * rho))
    if measure == "franz":
        assert Fr / Fi == pytest.approx(R_ref, rel=0.25)
    else:
        # The bare transform is retained as an amplitude comparison only.  It
        # has no obliquity normalization and must not be read as reflected power.
        assert Fr / Fi < 0.75 * R_ref


def test_polar_kernel_is_converged():
    """The default discretisation is converged: refining n_rho and m_max does
    not move the transmitted focal profile."""
    kappa = vw.stigmatic_conic_constant(1.5, 1.0)
    surf = vw.Conic(radius=-8.0, conic=kappa)
    grid = vw.Grid.from_spacing(0.25, 128)
    pw = vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.5, polarization="x")
    x = np.linspace(-2.0, 2.0, 81)

    def profile(n_rho, m_max):
        op = vw.InterfaceOperator(surf, n1=1.5, n2=1.0, aperture=6.0,
                                  n_rho=n_rho, m_max=m_max, n_phi=32)
        f = op(pw).field_on(x, np.array([0.0]), z=12.0)
        v = np.abs(f.Ex[0])
        return v / v.max()

    base = profile(400, 2)
    fine = profile(800, 4)
    assert float(np.sqrt(np.mean((base - fine) ** 2))) < 5e-3
