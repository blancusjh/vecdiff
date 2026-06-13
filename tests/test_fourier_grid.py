import numpy as np

from vecdiff.fourier import FT, FT2, FT3, IFT, IFT2, IFT3, KGRID2
from vecdiff.grid import Grid
from vecdiff.longitudinal import generate_Ez_cartesian


def test_centered_1d_transform_handles_centered_samples():
    values = np.zeros(8)
    values[values.size//2] = 1.0

    spectrum = FT(values, physical=False)

    assert np.allclose(spectrum, np.ones_like(spectrum))


def test_centered_1d_inverse_reconstructs_samples():
    x = np.linspace(-1.0, 1.0, 16, endpoint=False)
    values = np.exp(-x**2) + 0.25j*x

    spectrum = FT(values, dx=x[1] - x[0])
    reconstructed = IFT(spectrum, dx=x[1] - x[0])

    assert np.allclose(reconstructed, values)


def test_grid_spacing_and_kgrid_match_numpy_fftfreq():
    x = np.linspace(-2.0, 2.0, 8)
    y = np.linspace(-1.0, 1.0, 6)
    X, Y = np.meshgrid(x, y)
    grid = Grid.from_cartesian(X, Y)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    assert grid.spacing() == (dx, dy)

    KX, KY = KGRID2(grid)
    expected_kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(x.size, d=dx))
    expected_ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(y.size, d=dy))

    assert np.allclose(KX[0], expected_kx)
    assert np.allclose(KY[:, 0], expected_ky)


def test_grid_from_spacing_builds_centered_grid():
    grid = Grid.from_spacing((4, 6), dx=0.5, dy=0.25)

    assert grid.shape == (4, 6)
    assert grid.spacing() == (0.5, 0.25)
    assert np.allclose(grid.X[0], [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
    assert np.allclose(grid.Y[:, 0], [-0.5, -0.25, 0.0, 0.25])


def test_ft2_returns_spectrum_and_dual_kgrid():
    x = np.linspace(-1.0, 1.0, 7)
    y = np.linspace(-2.0, 2.0, 5)
    X, Y = np.meshgrid(x, y)
    grid = Grid.from_cartesian(X, Y)
    values = X + 2.0 * Y

    spectrum, kgrid = FT2(values, grid)

    assert spectrum.shape == values.shape
    assert kgrid.domain == "k"
    assert kgrid.dual is grid


def test_ft2_accepts_dx_dy_without_explicit_grid():
    values = np.ones((5, 7))

    spectrum, kgrid = FT2(values, dx=0.2, dy=0.3)

    assert spectrum.shape == values.shape
    assert np.allclose(kgrid.dual.spacing(), (0.2, 0.3))


def test_ift2_reconstructs_ft2_samples():
    x = np.linspace(-1.0, 1.0, 16)
    y = np.linspace(-1.5, 1.5, 12)
    X, Y = np.meshgrid(x, y)
    grid = Grid.from_cartesian(X, Y)
    values = np.exp(-(X**2 + Y**2)) + 0.5j * X

    spectrum, kgrid = FT2(values, grid)
    reconstructed, reconstructed_grid = IFT2(spectrum, kgrid)

    assert reconstructed_grid is grid
    assert np.allclose(reconstructed, values)


def test_ift2_accepts_dx_dy_without_explicit_kgrid():
    values = np.exp(-np.linspace(-1.0, 1.0, 8)**2)[None, :]
    values = np.repeat(values, 6, axis=0)

    spectrum, _ = FT2(values, dx=0.1, dy=0.2)
    reconstructed, grid = IFT2(spectrum, dx=0.1, dy=0.2)

    assert np.allclose(grid.spacing(), (0.1, 0.2))
    assert np.allclose(reconstructed, values)


def test_ft2_non_physical_mode_matches_shifted_numpy_fft2():
    x = np.linspace(-1.0, 1.0, 6)
    y = np.linspace(-1.0, 1.0, 4)
    X, Y = np.meshgrid(x, y)
    grid = Grid.from_cartesian(X, Y)
    values = np.sin(X) + np.cos(Y)

    spectrum, _ = FT2(values, grid, physical=False)
    expected = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(values.astype(complex))))

    assert np.allclose(spectrum, expected)


def test_ft2_custom_kgrid_direct_matches_manual_sum():
    grid = Grid.from_spacing((5, 6), dx=0.2, dy=0.3)
    values = np.exp(-(grid.X**2 + grid.Y**2)) + 0.2j * grid.X
    kx = np.linspace(-1.7, 1.3, 4)
    ky = np.linspace(-0.9, 1.1, 3)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    kgrid = Grid.from_cartesian(KX, KY, domain="k")

    spectrum, out_grid = FT2(values, grid, kgrid=kgrid, method="direct")

    expected = np.empty((ky.size, kx.size), dtype=complex)
    dx, dy = grid.spacing()
    for iy, kval_y in enumerate(ky):
        for ix, kval_x in enumerate(kx):
            phase = np.exp(-1j * (kval_x * grid.X + kval_y * grid.Y))
            expected[iy, ix] = dx * dy * np.sum(values * phase)

    assert out_grid.domain == "k"
    assert out_grid.dual is grid
    assert np.allclose(spectrum, expected)


def test_ft2_custom_kgrid_zoom_matches_direct():
    grid = Grid.from_spacing((7, 8), dx=0.15, dy=0.25)
    values = np.exp(-(grid.X**2 + 0.5 * grid.Y**2)) + 0.1j * grid.Y
    kx = np.linspace(-2.0, 2.0, 9)
    ky = np.linspace(-1.5, 1.5, 6)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    kgrid = Grid.from_cartesian(KX, KY, domain="k")

    zoomed, out_grid = FT2(values, grid, kgrid=kgrid)
    direct, _ = FT2(values, grid, kgrid=kgrid, method="direct")

    assert out_grid.dual is grid
    assert np.allclose(zoomed, direct)


def test_ft2_custom_fft_kgrid_matches_standard_ft2():
    grid = Grid.from_spacing((7, 8), dx=0.15, dy=0.25)
    values = np.exp(-(grid.X**2 + grid.Y**2)) + 0.1j * grid.X
    standard, kgrid = FT2(values, grid)

    zoomed, out_grid = FT2(values, grid, kgrid=kgrid)

    assert out_grid is kgrid
    assert np.allclose(zoomed, standard)


def test_ft2_custom_kgrid_rejects_nonseparable_grid():
    grid = Grid.from_spacing((5, 6), dx=0.2, dy=0.3)
    values = np.ones(grid.shape)
    kx = np.linspace(-1.0, 1.0, 4)
    ky = np.linspace(-1.0, 1.0, 3)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    KX = KX + 0.01 * KY
    kgrid = Grid.from_cartesian(KX, KY, domain="k")

    with np.testing.assert_raises(ValueError):
        FT2(values, grid, kgrid=kgrid)


def test_ft2_zoom_method_requires_custom_kgrid():
    grid = Grid.from_spacing((5, 6), dx=0.2, dy=0.3)

    with np.testing.assert_raises(ValueError):
        FT2(np.ones(grid.shape), grid, method="zoom")


def test_kgrid2_accepts_shape_and_spacing():
    KX, KY = KGRID2(shape=(4, 6), dx=0.5, dy=0.25)

    expected_kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(6, d=0.5))
    expected_ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(4, d=0.25))
    assert np.allclose(KX[0], expected_kx)
    assert np.allclose(KY[:, 0], expected_ky)


def test_centered_3d_transform_handles_centered_samples():
    values = np.zeros((4, 4, 4))
    values[2, 2, 2] = 1.0

    spectrum = FT3(values, physical=False)

    assert np.allclose(spectrum, np.ones_like(spectrum))


def test_centered_3d_inverse_reconstructs_samples():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(5, 6, 7)) + 0.1j*rng.normal(size=(5, 6, 7))

    spectrum = FT3(values, dx=0.1, dy=0.2, dz=0.3)
    reconstructed = IFT3(spectrum, dx=0.1, dy=0.2, dz=0.3)

    assert np.allclose(reconstructed, values)


def test_generate_ez_cartesian_accepts_xy_grid_pair():
    x = np.linspace(-1.0, 1.0, 16, endpoint=False)
    X, Y = np.meshgrid(x, x)
    Ex = np.ones_like(X)
    Ey = np.zeros_like(X)

    Ez = generate_Ez_cartesian(Ex, Ey, (X, Y), wavelength=0.5)

    assert Ez.shape == Ex.shape
