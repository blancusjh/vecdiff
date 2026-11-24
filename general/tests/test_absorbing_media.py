"""Media of complex refractive index: lossless or absorbing matter.

The fields vary as exp(-i omega t), so an absorbing medium has
n = n' + i kappa with kappa >= 0; a metal has a small n' and kappa of a few.
"""

import json

import numpy as np
import pytest
import vecdiff as o

M = o.media


@pytest.mark.parametrize("n", [1.5 + 0.02j, 0.05 + 3.3j, 2 + 4j, 1.0 + 1e-6j])
def test_an_absorbing_medium_holds_its_complex_index_and_permittivity(n):
    medium = M.Medium(n, "metal")
    assert medium.absorbing and not medium.lossless
    assert medium.n == n and isinstance(medium.n, complex)
    assert medium.n_real == n.real and medium.extinction == n.imag
    assert medium.relative_permittivity == n**2
    assert abs(medium.epsilon - M.EPSILON0 * n**2) <= 1e-15 * abs(medium.epsilon)
    # A metal's permittivity has a negative real part; its index stays n' > 0.
    assert (medium.epsilon.real < 0) == (n.imag > n.real)
    region = M.HomogeneousRegion("behind", medium)
    assert region.material is medium


def test_a_real_index_is_a_lossless_medium_held_as_a_float():
    for medium in (M.Medium(1.5), M.Dielectric(1.5), M.Medium(1.5 + 0j)):
        assert medium.lossless and not medium.absorbing
        assert type(medium.n) is float and medium.n == 1.5
        assert medium.extinction == 0 and medium.n_real == 1.5
    # A medium is its index and name, whichever class names it.
    assert M.Medium(1.5, "glass") == M.Dielectric(1.5, "glass")
    assert hash(M.Medium(1.5, "glass")) == hash(M.Dielectric(1.5, "glass"))
    assert M.Dielectric(1.5) != M.Dielectric(1.6)
    assert M.air.n == 1.0 and isinstance(M.air, M.Medium)


def test_a_dielectric_is_lossless_and_names_the_medium_that_absorbs():
    with pytest.raises(ValueError, match="media.Medium"):
        M.Dielectric(1.5 + 0.01j)


def test_a_negative_imaginary_part_is_refused_naming_the_time_convention():
    for make in (M.Medium, M.Dielectric):
        with pytest.raises(ValueError, match=r"exp\(-i omega t\).*gain"):
            make(1.5 - 0.01j)


@pytest.mark.parametrize("n", [0.0, -1.5, -0.1 + 2j, 0 + 3j])
def test_a_refractive_index_has_a_positive_real_part(n):
    with pytest.raises(ValueError, match="positive real part"):
        M.Medium(n)


def test_a_refractive_index_is_a_finite_number():
    with pytest.raises(TypeError, match="refractive index is a number"):
        M.Medium("glass")
    with pytest.raises(ValueError, match="finite"):
        M.Medium(complex(1, np.inf))
    # numpy scalars are numbers too.
    assert M.Medium(np.complex128(1.5 + 0.1j)).n == 1.5 + 0.1j
    assert type(M.Dielectric(np.float64(1.5)).n) is float


def test_regions_and_paths_accept_an_absorbing_medium():
    metal = M.Medium(0.05 + 3.3j, "silver")
    mirror = o.system.Reflection(o.surfaces.Plane(), M.air, behind=metal)
    system = o.system.OpticalSystem(mirror)
    assert system.elements[0].far.material == metal
    assert "0.05+3.3j" in o.system.describe(system.elements[0].far)


def test_a_material_distribution_holds_complex_indices_where_one_absorbs():
    from vecdiff.media import MaterialDistribution, MaterialRegion
    from vecdiff.volumes import SphereVolume

    metal = M.Medium(0.2 + 3j, "gold")
    ball = MaterialRegion("ball", SphereVolume([0.0, 0.0, 0.0], 1.0), metal)
    indices = MaterialDistribution(M.air, (ball,)).refractive_index(
        np.array([[0, 0, 0.0], [0, 0, 2.0]])
    )
    assert indices.dtype == complex and indices[0] == 0.2 + 3j and indices[1] == 1
    lossless = MaterialDistribution(M.air).refractive_index(np.zeros((1, 3)))
    assert lossless.dtype == float


def test_a_prescription_keeps_an_absorbing_index_as_n_and_kappa(tmp_path):
    metal = M.Medium(0.05 + 3.3j, "silver")
    surface = o.surfaces.SphericalCap(-0.1)
    system = o.system.OpticalSystem(
        o.system.Reflection(surface, M.Dielectric(1.0, "air"), behind=metal)
    )
    path = tmp_path / "mirror.json"
    o.io.save_system(system, path)
    regions = json.loads(path.read_text())["regions"]
    assert {"n": 0.05, "kappa": 3.3}.items() <= regions[1].items()
    assert "kappa" not in regions[0]
    loaded = o.io.load_system(path)
    assert loaded.elements[0].far.material == metal
