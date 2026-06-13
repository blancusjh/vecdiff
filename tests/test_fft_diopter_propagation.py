import numpy as np

from vecdiff import CartesianSurface, Field, FieldCartesian, Grid
from vecdiff.coordinate_transformation import cartesian_to_circular
from vecdiff.fourier import FT2
import vecdiff.propagation as propagation


def _cartesian_grid(n=17, half_size=1.0):
    axis = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    return Grid.from_cartesian(X, Y)


def test_transverse_operator_matches_projector_identity(monkeypatch):
    grid = _cartesian_grid()
    rng = np.random.default_rng(123)
    Ex = rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)
    Ey = rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)
    field = FieldCartesian(Ex, Ey, grid=grid, symmetric=False)

    tp = 1.2 + 0.1 * grid.R
    ts = 0.7 + 0.05 * grid.R

    def fake_coefficients(R, diopter, support=None):
        return tp, ts

    monkeypatch.setattr(propagation, "fresnel_coefficients_on_grid", fake_coefficients)
    Ux, Uy = propagation.transverse_diopter_operator(field, object())

    rho_x = np.cos(grid.Phi)
    rho_y = np.sin(grid.Phi)
    phi_x = -np.sin(grid.Phi)
    phi_y = np.cos(grid.Phi)
    rho_dot_E = rho_x * Ex + rho_y * Ey
    phi_dot_E = phi_x * Ex + phi_y * Ey
    expected_x = tp * rho_x * rho_dot_E + ts * phi_x * phi_dot_E
    expected_y = tp * rho_y * rho_dot_E + ts * phi_y * phi_dot_E

    assert np.allclose(Ux, expected_x)
    assert np.allclose(Uy, expected_y)


def test_transverse_operator_reduces_to_scalar_limit(monkeypatch):
    grid = _cartesian_grid()
    rng = np.random.default_rng(321)
    Ex = rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)
    Ey = rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)
    field = FieldCartesian(Ex, Ey, grid=grid, symmetric=False)
    t = 0.8 + 0.2 * grid.R

    def fake_coefficients(R, diopter, support=None):
        return t, t

    monkeypatch.setattr(propagation, "fresnel_coefficients_on_grid", fake_coefficients)
    Ux, Uy = propagation.transverse_diopter_operator(field, object())

    assert np.allclose(Ux, t * Ex)
    assert np.allclose(Uy, t * Ey)


def test_fft_propagation_uses_centered_ft2(monkeypatch):
    grid = _cartesian_grid(n=16)
    Ex = np.exp(-(grid.X**2 + grid.Y**2))
    Ey = 0.25j * grid.X * Ex
    field = FieldCartesian(Ex, Ey, grid=grid, symmetric=False)
    diopter = CartesianSurface(n0=1.0, ni=1.5, z0=-10.0, zi=6.0)

    def fake_coefficients(R, diopter, support=None):
        return np.ones_like(R), np.ones_like(R)

    monkeypatch.setattr(propagation, "fresnel_coefficients_on_grid", fake_coefficients)
    out = field.propagate_through_diopter(diopter.zi, diopter, method="fft")
    expected_x, expected_grid = FT2(Ex, grid, physical=True)
    expected_y, _ = FT2(Ey, grid, physical=True)

    assert out.grid.domain == "k"
    assert out.grid.dual is grid
    assert np.allclose(out.grid.X, expected_grid.X)
    assert np.allclose(out.grid.Y, expected_grid.Y)
    assert np.allclose(out.x, expected_x)
    assert np.allclose(out.y, expected_y)


def test_fft_propagation_zero_padding_refines_k_grid(monkeypatch):
    grid = _cartesian_grid(n=16)
    Ex = np.exp(-(grid.X**2 + grid.Y**2))
    Ey = 0.25j * grid.X * Ex
    field = FieldCartesian(Ex, Ey, grid=grid, symmetric=False)
    diopter = CartesianSurface(n0=1.0, ni=1.5, z0=-10.0, zi=6.0)

    def fake_coefficients(R, diopter, support=None):
        return np.ones_like(R), np.ones_like(R)

    monkeypatch.setattr(propagation, "fresnel_coefficients_on_grid", fake_coefficients)
    unpadded = field.propagate_through_diopter(diopter.zi, diopter, method="fft")
    padded = field.propagate_through_diopter(diopter.zi, diopter, method="fft", pad_factor=3)

    assert padded.x.shape == (3 * grid.shape[0], 3 * grid.shape[1])
    assert padded.y.shape == padded.x.shape
    assert padded.grid.dual.shape == padded.x.shape
    assert np.isclose(padded.grid.dx, unpadded.grid.dx / 3.0)
    assert np.isclose(padded.grid.dy, unpadded.grid.dy / 3.0)


def test_fft_propagation_accepts_custom_kgrid(monkeypatch):
    grid = _cartesian_grid(n=16)
    Ex = np.exp(-(grid.X**2 + grid.Y**2))
    Ey = 0.25j * grid.X * Ex
    field = FieldCartesian(Ex, Ey, grid=grid, symmetric=False)
    diopter = CartesianSurface(n0=1.0, ni=1.5, z0=-10.0, zi=6.0)
    kx = np.linspace(-2.0, 2.0, 9)
    ky = np.linspace(-1.5, 1.5, 7)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    kgrid = Grid.from_cartesian(KX, KY, domain="k")

    def fake_coefficients(R, diopter, support=None):
        return np.ones_like(R), np.ones_like(R)

    monkeypatch.setattr(propagation, "fresnel_coefficients_on_grid", fake_coefficients)
    out = field.propagate_through_diopter(
        diopter.zi,
        diopter,
        method="fft",
        kgrid=kgrid,
    )
    expected_x, expected_grid = FT2(Ex, grid, kgrid=kgrid)
    expected_y, _ = FT2(Ey, grid, kgrid=kgrid)

    assert out.grid.domain == "k"
    assert out.grid.dual is grid
    assert out.x.shape == (ky.size, kx.size)
    assert np.allclose(out.grid.X, expected_grid.X)
    assert np.allclose(out.grid.Y, expected_grid.Y)
    assert np.allclose(out.x, expected_x)
    assert np.allclose(out.y, expected_y)


def test_fft_propagation_accepts_custom_kgrid_with_padding(monkeypatch):
    grid = _cartesian_grid(n=8)
    Ex = np.exp(-(grid.X**2 + grid.Y**2))
    Ey = 0.25j * grid.X * Ex
    field = FieldCartesian(Ex, Ey, grid=grid, symmetric=False)
    diopter = CartesianSurface(n0=1.0, ni=1.5, z0=-10.0, zi=6.0)
    kx = np.linspace(-2.0, 2.0, 9)
    ky = np.linspace(-1.5, 1.5, 7)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    kgrid = Grid.from_cartesian(KX, KY, domain="k")

    def fake_coefficients(R, diopter, support=None):
        return np.ones_like(R), np.ones_like(R)

    monkeypatch.setattr(propagation, "fresnel_coefficients_on_grid", fake_coefficients)
    out = field.propagate_through_diopter(
        diopter.zi,
        diopter,
        method="fft",
        kgrid=kgrid,
        pad_factor=2,
    )

    padded_grid = propagation._center_padded_grid(grid, 2)
    expected_x = np.pad(Ex, ((4, 4), (4, 4)), mode="constant")
    expected_y = np.pad(Ey, ((4, 4), (4, 4)), mode="constant")
    expected_x, expected_grid = FT2(expected_x, padded_grid, kgrid=kgrid)
    expected_y, _ = FT2(expected_y, padded_grid, kgrid=kgrid)

    assert out.grid.domain == "k"
    assert out.grid.dual.shape == padded_grid.shape
    assert out.grid.dual.spacing() == padded_grid.spacing()
    assert out.x.shape == (ky.size, kx.size)
    assert np.allclose(out.grid.X, expected_grid.X)
    assert np.allclose(out.grid.Y, expected_grid.Y)
    assert np.allclose(out.x, expected_x)
    assert np.allclose(out.y, expected_y)


def test_fft_propagation_rejects_invalid_pad_factor():
    grid = _cartesian_grid(n=8)
    field = FieldCartesian(np.ones(grid.shape), np.zeros(grid.shape), grid=grid, symmetric=False)
    diopter = CartesianSurface(n0=1.0, ni=1.5, z0=-10.0, zi=6.0)

    with np.testing.assert_raises(ValueError):
        field.propagate_through_diopter(diopter.zi, diopter, method="fft", pad_factor=0)

    with np.testing.assert_raises(TypeError):
        field.propagate_through_diopter(diopter.zi, diopter, method="fft", pad_factor=1.5)


def test_legacy_positional_call_still_uses_hankel_branch():
    r = np.linspace(0.0, 0.8, 12)
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    q = np.linspace(0.0, 4.0, 10)
    grid = Grid.from_polar(r, phi)
    diopter = CartesianSurface(n0=1.0, ni=1.5, z0=5.0, zi=10.0)
    field = FieldCartesian(np.exp(-r**2), np.zeros_like(r), grid=grid)

    out = field.propagate_through_diopter(diopter.zi, diopter, q)

    assert out.symmetry == "cartesian"
    assert out.grid.type == "polar"
    assert out.x.shape == (phi.size, q.size)
    assert out.y.shape == (phi.size, q.size)


def test_fft_propagation_converts_polar_grid_to_cartesian(monkeypatch, capsys):
    r = np.linspace(0.0, 0.8, 12)
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    grid = Grid.from_polar(r, phi)
    diopter = CartesianSurface(n0=1.0, ni=1.5, z0=-10.0, zi=6.0)
    field = FieldCartesian(np.exp(-r**2), np.zeros_like(r), grid=grid)

    def fake_coefficients(R, diopter, support=None):
        return np.ones_like(R), np.ones_like(R)

    monkeypatch.setattr(propagation, "fresnel_coefficients_on_grid", fake_coefficients)
    out = field.propagate_through_diopter(diopter.zi, diopter, method="fft")
    captured = capsys.readouterr()

    assert "A Cartesian Grid is preferable" in captured.out
    assert out.grid.type == "cartesian"
    assert out.grid.domain == "k"
    assert out.grid.dual.type == "cartesian"
    assert out.x.shape == (r.size, r.size)
    assert out.y.shape == (r.size, r.size)
    assert np.any(np.isclose(out.grid.dual.X[0], 0.0))
    assert np.any(np.isclose(out.grid.dual.Y[:, 0], 0.0))


def test_cross_component_is_governed_by_sin_two_phi(monkeypatch):
    grid = _cartesian_grid()
    Ex = np.exp(-(grid.X**2 + grid.Y**2))
    Ey = np.zeros_like(Ex)
    field = FieldCartesian(Ex, Ey, grid=grid, symmetric=False)
    tp = 1.0 + 0.1 * grid.R
    ts = 0.6 + 0.05 * grid.R

    def fake_coefficients(R, diopter, support=None):
        return tp, ts

    monkeypatch.setattr(propagation, "fresnel_coefficients_on_grid", fake_coefficients)
    _, Uy = propagation.transverse_diopter_operator(field, object())

    expected_y = 0.5 * (tp - ts) * np.sin(2.0 * grid.Phi) * Ex
    assert np.allclose(Uy, expected_y)

    y_axis = grid.Y[:, 0]
    axis_row = int(np.argmin(np.abs(y_axis)))
    assert np.allclose(Uy[axis_row], 0.0, atol=1e-14)


def test_right_circular_convention_matches_cartesian_components():
    grid = _cartesian_grid()
    Phi = np.exp(-(grid.X**2 + grid.Y**2))

    field = Field.from_circular(
        np.zeros_like(Phi),
        Phi,
        grid,
        symmetric=False,
    )
    EL, ER = cartesian_to_circular(field.x, field.y)

    assert np.allclose(field.x, Phi / np.sqrt(2.0))
    assert np.allclose(field.y, 1.0j * Phi / np.sqrt(2.0))
    assert np.allclose(EL, 0.0)
    assert np.allclose(ER, Phi)
