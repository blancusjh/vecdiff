"""Independent geometric limits for the ray-tube flux amplitude."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from vecdiff_general.propagation.transport_flux import ray_plane_flux


def lattice():
    return np.stack(
        np.meshgrid(np.linspace(-0.5, 0.5, 19), np.linspace(-0.4, 0.4, 17)), -1
    )


def test_plane_wave_on_nonplanar_chart_has_unit_factor():
    xy = lattice()
    h = 0.2 * xy[..., 0] ** 2 + 0.1 * xy[..., 1] ** 2
    gradient = xy * [0.4, 0.2]
    u = np.broadcast_to([0.3, -0.2, np.sqrt(0.87)], h.shape + (3,))
    result = ray_plane_flux(xy, h, gradient, u, np.zeros(h.shape + (3, 2)), plane_z=2)
    assert result.regular.all()
    assert_allclose(result.amplitude, 1, atol=1e-14)
    assert_allclose(
        result.plane_xy, xy + (2 - h)[..., None] * u[..., :2] / u[..., 2, None]
    )


def test_perspective_sphere_to_tangent_plane_recovers_notes_cosine():
    xy = lattice()
    radius = 2
    axial = np.sqrt(radius**2 - np.sum(xy**2, -1))
    h = radius - axial
    gradient = xy / axial[..., None]
    u = np.concatenate([-xy, axial[..., None]], -1) / radius
    du = np.zeros(h.shape + (3, 2))
    du[..., 0, 0] = -1 / radius
    du[..., 1, 1] = -1 / radius
    du[..., 2, :] = -xy / (radius * axial[..., None])
    result = ray_plane_flux(xy, h, gradient, u, du)
    assert result.regular.all()
    assert_allclose(result.amplitude, u[..., 2], rtol=1e-14)
    assert_allclose(result.plane_xy, xy / u[..., 2, None], rtol=1e-14)


@pytest.mark.parametrize("image_z,regular", [(0.5, True), (1.0, False), (1.5, False)])
def test_converging_bundle_detects_crossed_focus_even_positive_final_determinant(
    image_z, regular
):
    xy = lattice()
    h = np.zeros(xy.shape[:-1])
    v = np.concatenate([-xy, np.ones(h.shape + (1,))], -1)
    distance = np.linalg.norm(v, axis=-1)
    u = v / distance[..., None]
    dv = np.zeros(h.shape + (3, 2))
    dv[..., 0, 0] = -1
    dv[..., 1, 1] = -1
    du = (
        dv - u[..., :, None] * np.einsum("...i,...ij->...j", u, dv)[..., None, :]
    ) / distance[..., None, None]
    result = ray_plane_flux(xy, h, np.zeros_like(xy), u, du, plane_z=image_z)
    if regular:
        assert result.regular.all()
        assert_allclose(result.amplitude, 2, rtol=1e-14)
        assert_allclose(result.plane_xy, 0.5 * xy, atol=1e-14)
    else:
        assert not result.regular.any()
        assert result.caustic_crossed.all()
        assert np.isnan(result.amplitude).all()


def test_arbitrary_regular_bundle_conserves_projected_ray_tube_flux():
    xy = lattice()
    h = 0.1 * np.sum(xy**2, -1)
    gradient = 0.2 * xy
    v = np.concatenate([0.2 * xy, np.ones(h.shape + (1,))], -1)
    norm = np.linalg.norm(v, axis=-1)
    u = v / norm[..., None]
    dv = np.zeros(h.shape + (3, 2))
    dv[..., 0, 0] = 0.2
    dv[..., 1, 1] = 0.2
    du = (
        dv - u[..., :, None] * np.einsum("...i,...ij->...j", u, dv)[..., None, :]
    ) / norm[..., None, None]
    result = ray_plane_flux(xy, h, gradient, u, du, plane_z=1)
    assert result.regular.all()
    # This is the surface integral per chart area, not cos(theta) alone.
    surface_flux_measure = u[..., 2] - np.sum(u[..., :2] * gradient, -1)
    plane_flux_measure = result.amplitude**2 * u[..., 2] * result.area_jacobian
    assert_allclose(plane_flux_measure, surface_flux_measure, rtol=1e-14)


def test_rejects_nonunit_or_complex_eikonal():
    xy = np.zeros((1, 2))
    h = np.zeros(1)
    du = np.zeros((1, 3, 2))
    with pytest.raises(ValueError, match="unit vectors"):
        ray_plane_flux(xy, h, xy, np.array([[0, 0, 2.0]]), du)
    with pytest.raises(ValueError, match="finite and real"):
        ray_plane_flux(xy, h, xy, np.array([[0, 0, 1 + 1j]]), du)
