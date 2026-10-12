"""Unmodified exact Mie traces: distinguish full transfer from its projections."""

from vecdiff_general.propagation.references.pure_surface_transport import (
    complete_reference_case,
    global_case,
    total_field_box_case,
)


def test_complete_mie_transfer_requires_the_evanescent_weyl_sector():
    coarse = complete_reference_case(spectral_order=16)
    fine = complete_reference_case(spectral_order=48)
    assert max(fine["full_surface_integral"].values()) < 1e-12
    assert max(fine["propagating_plus_evanescent"].values()) < 1e-11
    assert max(coarse["propagating_plus_evanescent"].values()) > 1e-4
    # No aperture, no taper, no T: the missing sector alone causes this error.
    assert 0.07 < max(fine["propagating_only"].values()) < 0.09


def test_exact_total_mie_field_transfers_inside_a_source_free_box():
    # Unlike an outgoing scattering reconstruction, this directly transfers the
    # full total field from a closed boundary, without adding an incident field.
    coarse = total_field_box_case(32)
    fine = total_field_box_case(64)
    assert max(fine["error"].values()) < 1e-7
    assert max(fine["error"].values()) < max(coarse["error"].values()) / 1000


def test_global_exact_trace_audit_reports_total_and_scattered_scales():
    row = global_case(max_sine=0.99)
    for method in ("inverse", "radiate"):
        # Recorded limitation of a restricted propagating representation; this
        # tolerance is not a claim of high-accuracy full-wave continuation.
        assert 0.005 < max(row[method]["total"].values()) < 0.02
        assert max(row[method]["scattered_scale"].values()) > 0.05
