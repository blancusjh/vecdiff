"""Tests for the composable operator algebra, the general (NUFFT) surface
transform, and freeform 2D surfaces (vecdiff.wave)."""
import numpy as np
import pytest

import vecdiff.wave as vw


def _corr(a, b):
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


@pytest.fixture(scope="module")
def grid():
    return vw.Grid.from_spacing(0.28, 200)


# --------------------------------------------------------------- FreeSpace
def test_freespace_is_exact(grid):
    spec = vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.0)
    # a tilted-ish real spectrum: refract a plane wave first
    surf = vw.Conic(radius=-8.0, conic=vw.stigmatic_conic_constant(1.5, 1.0))
    s = vw.InterfaceOperator(surf, n1=1.5, n2=1.0, aperture=9.0, n_rho=300)(
        vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.5))
    x = np.linspace(-3, 3, 61)
    D = 4.0
    lhs = vw.FreeSpace(D)(s).field_on(x, x, 0.0).intensity
    rhs = s.field_on(x, x, D).intensity
    assert np.abs(lhs - rhs).max() / rhs.max() < 1e-9


def test_freespace_group_law(grid):
    surf = vw.Conic(radius=-8.0, conic=vw.stigmatic_conic_constant(1.5, 1.0))
    s = vw.InterfaceOperator(surf, n1=1.5, n2=1.0, aperture=9.0, n_rho=300)(
        vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.5))
    x = np.linspace(-3, 3, 61)
    g1 = vw.FreeSpace(1.5)(vw.FreeSpace(2.5)(s)).field_on(x, x, 0.0).intensity
    g2 = vw.FreeSpace(4.0)(s).field_on(x, x, 0.0).intensity
    assert np.abs(g1 - g2).max() / g2.max() < 1e-9


# ------------------------------------------- interface operator == source path
def test_interface_operator_matches_source(grid):
    surf = vw.Conic(radius=-8.0, conic=vw.stigmatic_conic_constant(1.5, 1.0))
    ref = vw.surface_spectrum(surf, grid, n1=1.5, n2=1.0, aperture=9.0,
                              polarization="x")
    op = vw.InterfaceOperator(surf, n1=1.5, n2=1.0, aperture=9.0, n_rho=600)
    out = op(vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.5))
    zs = np.linspace(8, 22, 43)
    assert abs(ref.best_focus(zs) - out.best_focus(zs)) < 0.5
    x = np.linspace(-3, 3, 81)
    zf = ref.best_focus(zs)
    assert _corr(ref.field_on(x, x, zf).intensity,
                 out.field_on(x, x, zf).intensity) > 0.99


def test_reflecting_interface_operator_matches_source(grid):
    """Both public paths must use the same radiating-half-space normal."""
    surf = vw.Conic(radius=+8.0, conic=-0.5)
    common = dict(n1=1.0, n2=1.5, aperture=4.0, mode="r", measure="franz",
                  n_rho=300, n_phi=32, n_kr=256, m_max=4)
    ref = vw.surface_spectrum(surf, grid, polarization="x", **common)
    out = vw.InterfaceOperator(surf, **common)(
        vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.0)
    )
    assert out.sigma == ref.sigma == -1
    assert np.linalg.norm(out.A - ref.A) / np.linalg.norm(ref.A) < 1e-8


def test_interface_freespace_shifts_focus(grid):
    surf = vw.Conic(radius=-8.0, conic=vw.stigmatic_conic_constant(1.5, 1.0))
    out = vw.InterfaceOperator(surf, n1=1.5, n2=1.0, aperture=9.0)(
        vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.5))
    zs = np.linspace(8, 22, 57)
    zf = out.best_focus(zs)
    D = 4.0
    zfD = vw.FreeSpace(D)(out).best_focus(zs - D)
    assert abs(zfD - (zf - D)) < 0.4


# --------------------------------------------------- NUFFT == polar (axisym)
def test_nufft_matches_polar(grid):
    pytest.importorskip("finufft")
    surf = vw.Conic(radius=-8.0, conic=vw.stigmatic_conic_constant(1.5, 1.0))
    pw = vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.5)
    a = vw.InterfaceOperator(surf, n1=1.5, n2=1.0, aperture=9.0, method="polar")(pw)
    b = vw.InterfaceOperator(surf, n1=1.5, n2=1.0, aperture=9.0, method="nufft")(pw)
    zs = np.linspace(8, 22, 43)
    za = a.best_focus(zs)
    assert abs(za - b.best_focus(zs)) < 0.5
    x = np.linspace(-3, 3, 81)
    assert _corr(a.field_on(x, x, za).intensity,
                 b.field_on(x, x, za).intensity) > 0.99


# ------------------------------------------------------- Freeform2D == polar
def test_freeform2d_matches_axisym(grid):
    pytest.importorskip("finufft")
    conic = vw.Conic(radius=-8.0, conic=vw.stigmatic_conic_constant(1.5, 1.0))
    ff = vw.Freeform2D(sag_fn=lambda x, y: conic.sag(np.hypot(x, y)), radius=9.0)
    pw = vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.5)
    a = vw.InterfaceOperator(conic, n1=1.5, n2=1.0, aperture=9.0, method="polar")(pw)
    b = vw.InterfaceOperator(ff, n1=1.5, n2=1.0, aperture=9.0, n_free=220)(pw)
    zs = np.linspace(8, 22, 43)
    za = a.best_focus(zs)
    assert abs(za - b.best_focus(zs)) < 0.6
    x = np.linspace(-3, 3, 81)
    assert _corr(a.field_on(x, x, za).intensity,
                 b.field_on(x, x, za).intensity) > 0.99


# ------------------------------------------------------------- composition
def test_two_surface_system_focuses(grid):
    n_g = 1.5
    front = vw.InterfaceOperator(
        vw.Conic(radius=+6.0, conic=vw.stigmatic_conic_constant(1.0, n_g)),
        n1=1.0, n2=n_g, aperture=5.5)
    # The field after the first curved interface has a dense spectrum.  The
    # current fast composition path reconstructs one local ray and is therefore
    # a stated geometrical-optics approximation, never the default Maxwell map.
    back = vw.InterfaceOperator(vw.Plane(), n1=n_g, n2=1.0, aperture=5.5,
                                incidence_model="local_ray")
    system = vw.System([front, vw.FreeSpace(8.0), back])
    assert len(system) == 3
    out = system(vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.0))
    z = np.linspace(2, 40, 120)
    k = int(np.argmax(out.focus_scan(z)))
    assert 0 < k < len(z) - 1                      # an interior focus
    assert out.transversality_residual() < 1e-6


def test_plane_wave_spectrum_polarization(grid):
    f = vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.0,
                               polarization="x").field(0.0)
    fr = f.component_fractions()
    assert fr["x"] > 0.999 and fr["y"] < 1e-6


def _single_s_mode(grid, ix, amplitude=1.0):
    area = (grid.x.size * grid.dx) * (grid.y.size * grid.dy)
    A = np.zeros((3, *grid.shape), dtype=complex)
    A[1, 0, ix] = amplitude * area
    return vw.AngularSpectrum(A, grid, wavelength=1.0, n=1.0)


def test_spectral_incidence_preserves_superposition():
    """A dielectric interface must be linear in the incident Maxwell field."""
    g = vw.Grid.from_spacing(0.35, 32)
    a = _single_s_mode(g, 2)
    b = _single_s_mode(g, 5, amplitude=0.7j)
    ab = vw.AngularSpectrum(a.A + b.A, g, wavelength=1.0, n=1.0)
    op = vw.InterfaceOperator(
        vw.Plane(), n1=1.0, n2=1.5, aperture=3.0, measure="flat",
        n_rho=40, n_phi=24, n_kr=32, m_max=8, max_spectral_modes=2,
    )
    lhs = op(ab).A
    rhs = op(a).A + op(b).A
    assert np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs) < 1e-13


def test_dense_spectrum_is_not_silently_reduced_to_one_ray():
    g = vw.Grid.from_spacing(0.4, 24)
    dense = vw.point_source_spectrum(g, distance=8.0)
    op = vw.InterfaceOperator(vw.Plane(), n1=1.0, n2=1.5,
                              aperture=2.0, max_spectral_modes=2)
    with pytest.raises(ValueError, match="incidence_model='local_ray'"):
        op(dense)


def test_reflection_reverses_propagation_sense():
    g = vw.Grid.from_spacing(0.4, 24)
    pw = vw.plane_wave_spectrum(g)
    reflected = vw.InterfaceOperator(
        vw.Plane(), n1=1.0, n2=1.5, mode="r", aperture=2.0,
        measure="flat", n_rho=32, n_phi=16, n_kr=24, m_max=2,
    )(pw)
    assert reflected.sigma == -pw.sigma
