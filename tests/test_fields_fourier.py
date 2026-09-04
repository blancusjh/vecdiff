import numpy as np
import pytest
from vecdiff import *
from vecdiff.fourier.cartesian import transform, inverse
from vecdiff.fourier.nufft import synthesize


@pytest.mark.parametrize("count", [15, 16])
@pytest.mark.parametrize("direction", [-1, 1])
def test_unknown_Ez_completion_and_fft_propagation(count, direction):
    grid = CartesianGrid.from_spacing(.31, count); x, y = grid.xy
    kx = 2*np.pi/(count*grid.dx); kz = direction*np.sqrt((2*np.pi)**2-kx*kx)
    ex = np.exp(1j*kx*x); field = TransverseElectricField(ex, np.zeros_like(ex), grid, PlaneDomain())
    assert field.Ez is None
    with pytest.raises(ValueError): field.norm2()
    completed = field.complete(direction=direction)
    assert type(completed) is ElectricField
    np.testing.assert_allclose(completed.Ez, -kx/kz*ex, atol=2e-14)
    out = propagate(field, .27*direction, direction=direction)
    np.testing.assert_allclose(out.components, completed.components*np.exp(1j*kz*.27*direction), atol=2e-14)


def test_supplied_zero_is_not_replaced():
    grid = CartesianGrid.from_spacing(.3, 12); x, _ = grid.xy
    e = np.exp(2j*np.pi*x/(12*grid.dx)); zero = np.zeros_like(e)
    f = ElectricField(e, zero, grid, PlaneDomain(), Ez=zero)
    with pytest.raises(ValueError, match="k dot E"): spectrum_of(f)


def test_evanscent_homogeneous_mode():
    grid = CartesianGrid.from_spacing(.1, 12); x, _ = grid.xy
    kx = 4*np.pi/(12*grid.dx); alpha = np.sqrt(kx*kx-(2*np.pi)**2)
    ey = np.exp(1j*kx*x)
    field = TransverseElectricField(np.zeros_like(ey), ey, grid, PlaneDomain())
    out = propagate(field, .3)
    np.testing.assert_allclose(out.Ey, ey*np.exp(-alpha*.3), atol=1e-14)


def test_physical_fft_origin_and_parseval():
    grid = CartesianGrid(np.arange(12)*.2+.37, np.arange(11)*.3-.17)
    rng = np.random.default_rng(42); e = rng.normal(size=grid.shape)+1j*rng.normal(size=grid.shape)
    a = transform(e, grid)
    np.testing.assert_allclose(inverse(a, grid), e, atol=3e-15)
    np.testing.assert_allclose(np.sum(abs(e)**2)*grid.dx*grid.dy, np.sum(abs(a)**2)/grid.period_area)


def test_nufft_matches_direct():
    pytest.importorskip("finufft")
    rng = np.random.default_rng(12); k = rng.normal(size=(43, 3)); p = rng.normal(size=(67, 3))
    a = rng.normal(size=(3, 43))+1j*rng.normal(size=(3, 43))
    np.testing.assert_allclose(synthesize(k, a, p, backend="nufft", eps=1e-12), synthesize(k, a, p), atol=1e-10)


def test_grid_rejects_nonuniform_coordinates():
    with pytest.raises(ValueError): CartesianGrid([0, 1, 2.1], [0, 1])


def test_rigid_frame_propagation():
    a = .3; r = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    grid = CartesianGrid.from_spacing(.32, 16); domain = PlaneDomain(Frame([.1, 0, .4], r))
    wave = plane_wave(r[:, 2], r[:, 0]); field = wave.field(domain, grid)
    out = propagate(field, .23)
    np.testing.assert_allclose(out.components, wave.field(out.domain, grid).components, atol=1e-14)
