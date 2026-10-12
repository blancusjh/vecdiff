"""Q qualification with independently evaluated Maxwell traces and holdouts."""

import numpy as np
import pytest
import vecdiff_general as o
from vecdiff_general.propagation.references import DielectricBEM, MieSphere, SurfaceMesh
from vecdiff_general.propagation.references.diagnostics import maxwell_residuals
from vecdiff_general.propagation.references.surface_transport import (
    dipole,
    modal_case,
    problem,
    relative_fields,
    scattered,
    transfer_case,
)


@pytest.mark.parametrize("scale", [1, 532e-9])
@pytest.mark.parametrize("sag", [0, 0.7])
def test_independent_modal_traces_transfer_with_rotation_translation_and_si(scale, sag):
    # The input trace is evaluated directly from analytic waves, not from Q.
    # This also regresses cancellation in radiate's analytically transverse kernel.
    row = modal_case(scale=scale, sag=sag)
    assert max(row["forward"].values()) < 1e-10
    assert max(row["inverse_transfer"].values()) < 1e-9
    assert max(row["radiate_transfer"].values()) < 1e-10
    assert row["trace_report"]["magnetic_relative_residual"] < 1e-9


def test_dipole_reference_satisfies_maxwell_and_fourth_order_convergence():
    points = [[0.4, -0.3, 1.2], [-0.2, 0.1, 0.6]]
    coarse = maxwell_residuals(dipole, points, wavelength0=1, index=1, step=0.005)
    fine = maxwell_residuals(dipole, points, wavelength0=1, index=1, step=0.0025)
    assert max(fine.values()) < 1e-7
    assert fine["faraday"] < coarse["faraday"] / 10


def test_mie_cutoff_error_is_not_removed_by_spatial_refinement():
    source, sphere, glass = problem()
    evaluate = scattered(MieSphere(source, sphere, glass))
    base = transfer_case(evaluate)
    fine = transfer_case(evaluate, count=192)
    wider = transfer_case(evaluate, max_sine=0.99)
    for method in ("forward", "inverse_transfer", "radiate_transfer"):
        # This is a negative control: full Mie scattering is not in Q's
        # restricted angular band. Merely resolving the same band cannot fix it.
        assert 0.1 < max(base[method].values()) < 0.3
        np.testing.assert_allclose(
            list(base[method].values()), list(fine[method].values()), atol=2e-5
        )
        assert max(wider[method].values()) < 0.65 * max(base[method].values())
    assert base["trace_report"]["magnetic_relative_residual"] > 0.1


def test_bem_sphere_reference_and_transport_converge_toward_mie():
    source, sphere, glass = problem()
    mie = MieSphere(source, sphere, glass)
    probes = o.PointGrid([[0, 0, 4.3], [0.5, -0.5, 4.2], [-1, 1, 4.1]])
    expected = scattered(mie)(probes)
    mie_transfer = transfer_case(scattered(mie), length=16, count=64, max_sine=0.99)
    errors, differences = [], []
    for level in (1, 2):
        bem = DielectricBEM(source, SurfaceMesh.sphere(sphere, level), glass)
        evaluate = scattered(bem)
        errors.append(max(relative_fields(evaluate(probes), expected).values()))
        row = transfer_case(evaluate, length=16, count=64, max_sine=0.99)
        differences.append(
            max(
                abs(
                    row["inverse_transfer"][key] - mie_transfer["inverse_transfer"][key]
                )
                for key in ("E", "H")
            )
        )
    assert errors[1] < 0.04 and errors[1] < 0.3 * errors[0]
    assert differences[1] < differences[0] and differences[1] < 0.003


def test_backward_wave_exposes_the_difference_between_fit_and_radiation():
    source, _, _ = problem()
    axis = (np.arange(16) - 8) * 0.25
    plane = o.CartesianGrid(axis, axis)
    q = o.SurfaceTransport(plane, o.Plane(), source.region, 1)
    backward = o.PlaneWaveExpansion(
        [[0, 0, -2 * np.pi]], [[1, 0, 0]], [0, 0, 0], 1, source.region
    )
    trace = backward.evaluate(q.grid)
    _, report = q.inverse(trace)
    assert report.relative_residual < 1e-12
    np.testing.assert_allclose(report.magnetic_relative_residual, 2, atol=1e-12)
    # Both traces identify the backward branch; forward radiation cancels it.
    assert np.linalg.norm(q.radiate(trace).expansion.amplitudes) < 1e-12


@pytest.mark.parametrize("mismatch", ["wavelength", "surface"])
def test_radiate_rejects_trace_from_a_different_problem(mismatch):
    source, _, _ = problem()
    axis = (np.arange(16) - 8) * 0.25
    plane = o.CartesianGrid(axis, axis)
    q = o.SurfaceTransport(plane, o.Plane(), source.region, 1)
    trace = source.evaluate(q.grid)
    field = o.ElectricField(
        q.grid if mismatch == "wavelength" else plane.translated([0, 0, 1]),
        2 if mismatch == "wavelength" else 1,
        trace.electric.values,
    )
    with pytest.raises(ValueError, match=mismatch):
        q.radiate(o.ElectromagneticField(field, trace.magnetic))
