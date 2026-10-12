"""Public pipeline contracts and independently evaluated modal limits."""

import subprocess
import sys

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose


def setup(basis=np.eye(3), wavelength=1.0):
    axis = np.arange(-12, 12) * 0.4 * wavelength
    plane = o.CartesianGrid(
        axis, axis, o.Frame(np.array([2.0, -1.0, 0]) * wavelength, basis)
    )
    air = o.HomogeneousRegion("air", o.Dielectric(1))
    glass = o.HomogeneousRegion("glass", o.Dielectric(1.5))
    values = np.zeros(plane.shape + (2,), complex)
    x, y = np.meshgrid(axis, axis)
    values[..., 0] = np.exp(2j * np.pi * x / (len(axis) * 0.4 * wavelength))
    values[..., 1] = 0.3j * np.exp(-2j * np.pi * y / (len(axis) * 0.4 * wavelength))
    return (
        plane,
        air,
        glass,
        o.angular_spectrum(
            o.TransverseElectricField(plane, wavelength, values),
            air,
            evanescent="discard",
        ),
    )


def test_core_import_does_not_load_research_or_reference_solvers():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vecdiff_general as o; assert not any(n.startswith(('vecdiff_general.research','vecdiff_general.propagation.references')) for n in sys.modules); assert not any(hasattr(o,n) for n in ['propagate_interfaces','propagate_high_frequency','propagate_layers','LayerStack','HighFrequencySolution']); assert hasattr(o,'PropagationPipeline')",
        ],
        check=True,
    )


def test_empty_pipeline_is_homogeneous_transport():
    plane, air, glass, source = setup()
    result = o.PropagationPipeline().solve(source)
    target = plane.translated(3 * plane.frame.normal)
    expected = source.evaluate(target)
    actual = result.evaluate(target)
    assert result.spectrum is source
    assert result.interactions == ()
    assert_allclose(actual.electric.values, expected.electric.values, atol=2e-13)
    assert_allclose(actual.magnetic, expected.magnetic, atol=2e-15)


@pytest.mark.parametrize("wavelength", [1.0, 532e-9])
@pytest.mark.parametrize("angle", [0.0, 0.43])
def test_plane_pipeline_matches_exact_modal_transmission(wavelength, angle):
    c, s = np.cos(angle), np.sin(angle)
    basis = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    plane, air, glass, source = setup(basis, wavelength)
    interface = o.DielectricInterface(o.Plane(plane.frame), air, glass)
    pipeline = o.PropagationPipeline(
        [interface], transmission=o.DirectionResolvedFresnel(nodes=18, cutoff=0)
    )
    result = pipeline.solve(source)
    exact = o.scatter_plane(source, o.PlanarInterface(plane.frame, air, glass))
    target = plane.translated(2 * wavelength * plane.frame.normal)
    expected = exact.transmitted.evaluate(target)
    actual = result.evaluate(target)
    assert_allclose(
        actual.electric.values, expected.electric.values, atol=2e-11, rtol=2e-11
    )
    assert_allclose(actual.magnetic, expected.magnetic, atol=2e-13, rtol=2e-11)
    first = pipeline._operators
    pipeline.solve(source)
    assert pipeline._operators is first


def test_transparent_curved_interface_does_not_filter_or_reconstruct():
    plane, air, glass, source = setup()
    other = o.HomogeneousRegion("other_air", o.Dielectric(1))
    surface = o.SphericalCap(30, plane.frame)
    result = o.PropagationPipeline([o.DielectricInterface(surface, air, other)]).solve(
        source
    )
    assert_allclose(
        result.spectrum.expansion.amplitudes,
        source.expansion.amplitudes,
        atol=0,
        rtol=0,
    )
    assert result.spectrum.expansion.region == other


def test_zero_field_and_single_normal_mode_have_well_defined_angular_plans():
    plane, air, glass, _ = setup()
    boundary = o.DielectricInterface(o.Plane(plane.frame), air, glass)
    pipeline = o.PropagationPipeline([boundary])
    for value in [0.0, 1.0]:
        a = np.zeros(plane.shape + (2,), complex)
        a[..., 0] = value
        source = o.angular_spectrum(
            o.TransverseElectricField(plane, 1, a), air, evanescent="discard"
        )
        result = pipeline.solve(source)
        actual = result.evaluate(plane)
        expected = np.zeros(plane.shape + (3,), complex)
        expected[..., 0] = 0.8 * value
        assert_allclose(actual.electric.values, expected, atol=2e-13)


def test_explicit_scope_and_material_validation():
    plane, air, glass, source = setup()
    interface = o.DielectricInterface(o.Plane(plane.frame), air, glass)
    with pytest.raises(ValueError, match="edges and stops"):
        o.PropagationPipeline(o.OpticalSystem([interface], stops=[2]))
    wrong = o.DielectricInterface(o.Plane(plane.frame), glass, air)
    with pytest.raises(ValueError, match="source medium"):
        o.PropagationPipeline([wrong]).solve(source)
    tilted = o.DielectricInterface(o.Plane(o.Frame([0, 0, 0])), air, glass)
    with pytest.raises(ValueError, match="transverse origin"):
        o.PropagationPipeline([tilted]).solve(source)
