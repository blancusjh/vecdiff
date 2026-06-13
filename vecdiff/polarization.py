"""Polarization utilities for sampled :class:`vecdiff.fields.Field` objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .coordinate_transformation import polar_grid_to_cartesian_grid


@dataclass
class PolarizationData:
    """Polarization quantities sampled on a plane."""

    ex: np.ndarray
    ey: np.ndarray
    s0: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    s3: np.ndarray
    psi: np.ndarray
    chi: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray


def stokes_parameters(ex: np.ndarray, ey: np.ndarray) -> tuple[np.ndarray, ...]:
    """Compute Stokes parameters from complex Cartesian field components."""

    ex = np.asarray(ex, dtype=complex)
    ey = np.asarray(ey, dtype=complex)
    s0 = np.abs(ex) ** 2 + np.abs(ey) ** 2
    s1 = np.abs(ex) ** 2 - np.abs(ey) ** 2
    s2 = 2.0 * np.real(ex * np.conj(ey))
    s3 = -2.0 * np.imag(ex * np.conj(ey))
    return s0, s1, s2, s3


def ellipse_parameters(
    s0: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    s3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ellipse orientation ``psi`` and ellipticity angle ``chi``."""

    eps = np.finfo(float).eps
    psi = 0.5 * np.arctan2(s2, s1)
    chi = 0.5 * np.arcsin(np.clip(s3 / np.maximum(s0, eps), -1.0, 1.0))
    return psi, chi


def polarization_from_components(ex: np.ndarray, ey: np.ndarray) -> PolarizationData:
    """Compute polarization data from complex Cartesian components."""

    ex = np.asarray(ex, dtype=complex)
    ey = np.asarray(ey, dtype=complex)
    s0, s1, s2, s3 = stokes_parameters(ex, ey)
    psi, chi = ellipse_parameters(s0, s1, s2, s3)
    amplitude = np.sqrt(s0)
    phase = np.angle(ex)
    return PolarizationData(ex, ey, s0, s1, s2, s3, psi, chi, amplitude, phase)


def _cartesian_plot_mesh(half_size: float, n_img: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-half_size, half_size, n_img)
    return np.meshgrid(x, x)


def _polar_component_to_cartesian(component, radial_axis, angular_axis, xx, yy):
    component = np.asarray(component, dtype=complex)
    radial_axis = np.asarray(radial_axis, dtype=float)
    angular_axis = np.asarray(angular_axis, dtype=float)
    rho = np.sqrt(xx**2 + yy**2)

    if component.ndim == 1:
        return np.interp(rho, radial_axis, component, left=component[0], right=0.0)

    if component.ndim == 2 and component.shape[0] == 1:
        row = component[0]
        return np.interp(rho, radial_axis, row, left=row[0], right=0.0)

    return polar_grid_to_cartesian_grid(
        component,
        radial_axis,
        angular_axis,
        xx,
        yy,
        fill_value=0.0,
    )


def polarization_from_field(field) -> PolarizationData:
    """Compute polarization on the field's native sampling grid."""

    if not hasattr(field, "x") or not hasattr(field, "y"):
        raise TypeError("`field` must expose Cartesian components `x` and `y`.")
    return polarization_from_components(field.x, field.y)


def polarization_map_from_field(
    field,
    half_size: float | None = None,
    n_img: int = 500,
) -> tuple[np.ndarray, np.ndarray, PolarizationData]:
    """Sample a polar-grid ``Field`` on a Cartesian plane and compute polarization.

    Returns ``(xx, yy, pol)``.  For Cartesian-grid fields, the native grid is
    returned unchanged unless ``half_size`` is provided.
    """

    if not hasattr(field, "grid"):
        raise TypeError("`field` must provide a `grid` attribute.")

    if field.grid.type == "cartesian":
        if half_size is None:
            return field.grid.X, field.grid.Y, polarization_from_field(field)

        x = np.linspace(-half_size, half_size, n_img)
        xx, yy = np.meshgrid(x, x, indexing="xy")
        points = np.column_stack((yy.ravel(), xx.ravel()))
        x_axis = np.asarray(field.grid.X[0, :], dtype=float)
        y_axis = np.asarray(field.grid.Y[:, 0], dtype=float)

        interp_x = RegularGridInterpolator(
            (y_axis, x_axis),
            field.x,
            bounds_error=False,
            fill_value=0.0,
        )
        interp_y = RegularGridInterpolator(
            (y_axis, x_axis),
            field.y,
            bounds_error=False,
            fill_value=0.0,
        )
        ex = interp_x(points).reshape(xx.shape)
        ey = interp_y(points).reshape(xx.shape)
        return xx, yy, polarization_from_components(ex, ey)

    if field.grid.type != "polar":
        raise ValueError(f"Unsupported field grid type: {field.grid.type!r}.")

    radial_axis = np.asarray(field.grid.r, dtype=float)
    angular_axis = np.asarray(field.grid.varphi, dtype=float)
    if half_size is None:
        half_size = float(np.max(radial_axis))

    xx, yy = _cartesian_plot_mesh(float(half_size), int(n_img))
    ex = _polar_component_to_cartesian(field.x, radial_axis, angular_axis, xx, yy)
    ey = _polar_component_to_cartesian(field.y, radial_axis, angular_axis, xx, yy)
    return xx, yy, polarization_from_components(ex, ey)
