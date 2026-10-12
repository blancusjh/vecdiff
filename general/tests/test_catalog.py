"""Unit conversion, orientation and physical identity of the AL1225 prescription."""

import numpy as np
import pytest
import vecdiff_general as optics
from numpy.testing import assert_allclose


def test_al1225_dimensions_dispersion_and_paraxial_focus():
    system = optics.thorlabs_al1225()
    front, back = system.interfaces
    assert_allclose(system.apertures, (0.00625, 0.00625))
    assert_allclose(back.surface.frame.origin, [0, 0, 0.004])
    n = front.plus.material.n
    assert_allclose(n, 1.5111833144579327, rtol=1e-13)
    # Independent paraxial thick-lens formula: aspheric entrance / plane exit.
    efl = 1 / ((n - 1) * front.surface.curvature)
    assert_allclose(efl, 0.025, rtol=5e-5)
    assert_allclose(efl - 0.004 / n, 0.0224, atol=0.00005)
    assert front.surface.height(np.array([0.00625, 0])) < 0.004
    # Preserve the finite prescription; it must not silently become open faces.
    with pytest.raises(ValueError, match="finite material edges"):
        optics.PropagationPipeline(system)


def test_al1225_sag_sign_units_gradient_and_scaling():
    lens = optics.thorlabs_al1225()
    surface = lens.interfaces[0].surface
    xy_mm = np.array([[0, 0], [2, 3], [5.625, 0], [6.25, 0]])
    r = np.linalg.norm(xy_mm, axis=-1)
    # Independent evaluation in the source table's millimetres, through A16.
    sag_mm = r**2 / (12.78 * (1 + np.sqrt(1 - 0.4 * r**2 / 12.78**2)))
    sag_mm += (
        1.8429898e-6 * r**4
        - 3.8172252e-9 * r**6
        - 2.4345457e-11 * r**8
        + 3.1730496e-14 * r**10
        - 3.700237e-15 * r**12
        + 6.5107821e-17 * r**14
        - 4.9604147e-19 * r**16
    )
    xy = xy_mm * 1e-3
    assert_allclose(surface.height(xy), sag_mm * 1e-3, atol=1e-17)
    for direction in np.eye(2):
        h = direction * 1e-8
        finite_difference = (surface.height(xy + h) - surface.height(xy - h)) / 2e-8
        assert_allclose(surface.gradient(xy) @ direction, finite_difference, atol=1e-10)
    scaled = optics.thorlabs_al1225(scale=0.001)
    assert_allclose(
        scaled.interfaces[0].surface.height(xy * 0.001), surface.height(xy) * 0.001
    )
    assert_allclose(
        scaled.interfaces[0].surface.gradient(xy * 0.001), surface.gradient(xy)
    )
    assert_allclose(scaled.interfaces[1].surface.frame.origin, [0, 0, 4e-6])


@pytest.mark.parametrize(
    "kwargs", [{"scale": 0}, {"scale": np.nan}, {"wavelength0": 0}]
)
def test_al1225_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        optics.thorlabs_al1225(**kwargs)
