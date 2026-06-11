"""Longitudinal-field reconstruction from Maxwell transversality."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .coordinate_transformation import polar_grid_to_cartesian_grid
from .fourier import FT2, IFT2, KGRID2
from .grid import Grid

π = np.pi


def spacing_from_cartesian_grid(grid) -> tuple[float, float]:
    """Return uniform Cartesian spacings ``(dx, dy)`` from a grid."""
    if hasattr(grid, "X") and hasattr(grid, "Y"):
        x = np.asarray(grid.X, dtype=float)
        y = np.asarray(grid.Y, dtype=float)
    else:
        try:
            x, y = grid
        except (TypeError, ValueError) as exc:
            raise TypeError("grid must be a Grid instance or an (x, y) pair.") from exc
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y grids must have the same shape.")
    if x.ndim != 2:
        raise ValueError("x and y must be 2D arrays.")

    return _uniform_spacing(x, axis=1, name="x"), _uniform_spacing(y, axis=0, name="y")


def kz_angular_spectrum(KX, KY, wavelength, n=1.0,
                        direction="+z", include_evanescent=False):
    """Return the longitudinal angular wave number for an angular spectrum."""
    if direction not in {"+z", "-z"}:
        raise ValueError("direction must be '+z' or '-z'.")

    k = 2.0 * π * n / wavelength
    kt2 = KX**2 + KY**2
    kz2 = k**2 - kt2

    if include_evanescent:
        KZ = np.sqrt(kz2.astype(complex))
    else:
        KZ = np.empty_like(KX, dtype=complex)
        propagating = kz2 >= 0.0
        KZ[propagating] = np.sqrt(kz2[propagating])
        KZ[~propagating] = np.inf

    if direction == "-z":
        KZ = -KZ
    return KZ


def generate_Ez_cartesian(Ex, Ey, grid, wavelength, n=1.0,
                          method="exact", direction="+z",
                          include_evanescent=False):
    """Generate ``Ez`` from Cartesian-sampled transverse field components."""
    if direction not in {"+z", "-z"}:
        raise ValueError("direction must be '+z' or '-z'.")

    Ex = np.asarray(Ex, dtype=complex)
    Ey = np.asarray(Ey, dtype=complex)
    if Ex.shape != Ey.shape:
        raise ValueError("Ex and Ey must have the same shape.")
    if Ex.ndim != 2:
        raise ValueError("Ex and Ey must be 2D Cartesian arrays.")

    dx, dy = spacing_from_cartesian_grid(grid)
    EX = FT2(Ex, dx=dx, dy=dy)
    EY = FT2(Ey, dx=dx, dy=dy)
    KX, KY = KGRID2(Ex.shape, dx=dx, dy=dy)

    if method == "exact":
        KZ = kz_angular_spectrum(
            KX,
            KY,
            wavelength,
            n=n,
            direction=direction,
            include_evanescent=include_evanescent,
        )
    elif method == "paraxial":
        k = 2.0 * π * n / wavelength
        KZ = k if direction == "+z" else -k
    else:
        raise ValueError("method must be 'exact' or 'paraxial'.")

    with np.errstate(divide="ignore", invalid="ignore"):
        EZ = -(KX * EX + KY * EY) / KZ
    EZ = np.where(np.isfinite(EZ), EZ, 0.0)
    return IFT2(EZ, dx=dx, dy=dy)


def generate_Ez_field(field, wavelength, n=1.0, method="exact",
                      direction="+z", include_evanescent=False):
    """Generate ``Ez`` for a ``Field`` on its native grid."""
    if not hasattr(field, "grid"):
        raise TypeError("field must provide a grid attribute.")

    if field.grid.type == "cartesian":
        return generate_Ez_cartesian(
            field.x,
            field.y,
            field.grid,
            wavelength=wavelength,
            n=n,
            method=method,
            direction=direction,
            include_evanescent=include_evanescent,
        )

    if field.grid.type == "polar":
        Ex, Ey, cart_grid = _polar_field_to_cartesian(field)
        Ez = generate_Ez_cartesian(
            Ex,
            Ey,
            cart_grid,
            wavelength=wavelength,
            n=n,
            method=method,
            direction=direction,
            include_evanescent=include_evanescent,
        )
        return _cartesian_field_to_polar(Ez, cart_grid, field.grid)

    raise ValueError(f"Unsupported field grid type: {field.grid.type!r}.")


def _uniform_spacing(values: np.ndarray, axis: int, name: str) -> float:
    diffs = np.diff(values, axis=axis)
    if diffs.size == 0:
        raise ValueError(f"{name} grid axis must contain at least two samples.")

    spacing = float(np.mean(diffs))
    if not np.allclose(diffs, spacing):
        raise ValueError(f"{name} grid axis must be uniformly sampled.")
    if np.isclose(spacing, 0.0):
        raise ValueError(f"{name} grid spacing must be nonzero.")
    return spacing


def _polar_field_to_cartesian(field):
    r = np.asarray(field.grid.r, dtype=float)
    varphi = np.asarray(field.grid.varphi, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("polar grid must provide at least two radial samples.")

    half_size = float(np.max(r))
    n_cart = max(int(r.size), 2)
    x = np.linspace(-half_size, half_size, n_cart)
    X, Y = np.meshgrid(x, x, indexing="xy")
    cart_grid = Grid.from_cartesian(X, Y)

    Ex = _polar_component_to_cartesian(field.x, r, varphi, X, Y)
    Ey = _polar_component_to_cartesian(field.y, r, varphi, X, Y)
    return Ex, Ey, cart_grid


def _polar_component_to_cartesian(component, radial_axis, angular_axis, X, Y):
    component = np.asarray(component, dtype=complex)
    rho = np.sqrt(X**2 + Y**2)

    if component.ndim == 1:
        return np.interp(rho, radial_axis, component, left=component[0], right=0.0)

    if component.ndim == 2 and component.shape[0] == 1:
        row = component[0]
        return np.interp(rho, radial_axis, row, left=row[0], right=0.0)

    return polar_grid_to_cartesian_grid(
        component,
        radial_axis,
        angular_axis,
        X,
        Y,
        fill_value=0.0,
    )


def _cartesian_field_to_polar(component, cart_grid, polar_grid):
    component = np.asarray(component, dtype=complex)
    x_axis = np.asarray(cart_grid.X[0, :], dtype=float)
    y_axis = np.asarray(cart_grid.Y[:, 0], dtype=float)
    points = np.column_stack((polar_grid.Y.ravel(), polar_grid.X.ravel()))

    interpolator = RegularGridInterpolator(
        (y_axis, x_axis),
        component,
        bounds_error=False,
        fill_value=0.0,
    )
    return interpolator(points).reshape(polar_grid.X.shape)
