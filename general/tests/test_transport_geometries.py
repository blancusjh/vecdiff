"""Complete transport on nonspherical closed surfaces, independent dipole data."""

import numpy as np
import pytest
from vecdiff_general.comparisons.surface_radiation import SurfaceRadiation
from vecdiff_general.propagation.references.diagnostics import maxwell_residuals
from vecdiff_general.propagation.references.pure_surface_transport import (
    evanescent_spectrum,
)
from vecdiff_general.propagation.references.surface_transport import relative_fields
from vecdiff_general.propagation.references.transport_geometries import (
    AIR,
    SHAPES,
    add_fields,
    boundary,
    exact_field,
    receiver,
)


@pytest.mark.parametrize("shape", SHAPES)
def test_nonspherical_quadrature_obeys_the_divergence_theorem(shape):
    grid = boundary(shape, 24)
    normals = grid.normals.reshape(-1, 3)
    points = grid.points.reshape(-1, 3)
    weights = grid.weights.ravel()
    volumes = {
        "ellipsoid": 4 * np.pi * 0.65 * 0.45 * 0.55 / 3,
        "box": 8 * 0.6 * 0.5 * 0.55,
        "cylinder": np.pi * 0.55**2 * 1.2,
        # Integral r^3/3 over the unit sphere; odd angular moments vanish.
        "rippled": 4 * np.pi * 0.6**3 / 3 * (1 + 3 * (0.16**2 * 8 / 35 + 0.1**2 / 105)),
    }
    np.testing.assert_allclose(
        np.sum(normals * weights[:, None], axis=0), 0, atol=1e-13
    )
    np.testing.assert_allclose(
        np.einsum("pi,pj,p->ij", points, normals, weights),
        volumes[shape] * np.eye(3),
        atol=2e-13,
    )


def test_displaced_dipole_reference_satisfies_maxwell_equations():
    points = [[0.2, 0.1, 1.1], [-0.3, 0.2, 1.3]]
    coarse = maxwell_residuals(exact_field, points, wavelength0=1, index=1, step=0.005)
    fine = maxwell_residuals(exact_field, points, wavelength0=1, index=1, step=0.0025)
    assert max(fine.values()) < 3e-9
    assert fine["faraday"] < coarse["faraday"] / 10


@pytest.mark.parametrize("shape", SHAPES)
def test_complete_transfer_converges_to_exact_fields_on_a_curved_receiver(shape):
    probes = receiver()
    exact = exact_field(probes)
    errors = []
    for order in (12, 24):
        trace = exact_field(boundary(shape, order))
        result = SurfaceRadiation.from_field(trace, AIR).evaluate(probes)
        errors.append(max(relative_fields(result, exact).values()))
    assert errors[1] < 1e-10
    assert errors[1] < errors[0] / 10000


def test_full_weyl_transfer_on_an_asymmetric_rippled_boundary():
    radiation = SurfaceRadiation.from_field(exact_field(boundary("rippled", 24)), AIR)
    probes = receiver()
    propagating = radiation.angular_spectrum(
        polar_order=48, azimuthal_order=96
    ).evaluate(probes)
    evanescent = evanescent_spectrum(
        radiation, order=48, azimuths=96, cutoff=8
    ).evaluate(probes)
    complete = add_fields(propagating, evanescent)
    assert max(relative_fields(complete, exact_field(probes)).values()) < 1e-10
    assert max(relative_fields(propagating, exact_field(probes)).values()) > 0.01
