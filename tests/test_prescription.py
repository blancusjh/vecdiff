from pathlib import Path
import numpy as np
import pytest
from vecdiff import EvenAsphere
from vecdiff.IO import read_prescription, write_prescription

DATA = Path(__file__).parents[1]/"examples/data/US7557996.csv"


def test_duv_preserves_folded_encounters_and_asphere_roundtrip(tmp_path):
    s = read_prescription(DATA)
    assert len(s.encounters) == 48
    assert s.wavelength == pytest.approx(193.368e-6)
    assert s.image_z == pytest.approx(1423.0777)
    assert [e.number for e in s.encounters if e.interaction == "reflect"] == [7, 10]
    assert [e.direction for e in s.encounters[6:11]] == [1, -1, -1, -1, 1]
    # Negative spacing to a virtual stop does not reverse a forward beam.
    assert s.encounters[34].direction == 1
    assert s.encounters[-1].transmitted_medium.n == pytest.approx(1.59667693)
    with pytest.raises(ValueError, match="cannot discard mirrors or stops"): s.dielectric_assembly()
    dest = tmp_path/"roundtrip.csv"; write_prescription(s, dest)
    reread = read_prescription(dest)
    for a, b in zip(s.encounters, reread.encounters):
        assert a.interaction == b.interaction and a.direction == b.direction
        assert a.transmitted_medium == b.transmitted_medium
        np.testing.assert_allclose(a.surface.frame.origin, b.surface.frame.origin, atol=1e-12)
        if hasattr(a.surface, "sag"):
            r = np.linspace(0, a.semidiameter, 31)
            np.testing.assert_allclose(a.surface.sag(r), b.surface.sag(r), atol=1e-12)
            np.testing.assert_allclose(a.surface.slope(r), b.surface.slope(r), atol=1e-12)


@pytest.mark.parametrize("old,new,match", [
    ("36.232225000\n", "36.232225001\n", "vertex"),
    (",refractive,", ",unrecognized,", "surface_type"),
    (",SIO2,1.56078570,", ",SIO2,-1.0,", "positive"),
    (",88.996,", ",-88.996,", "semidiameter"),
])
def test_bad_prescription_fails_explicitly(tmp_path, old, new, match):
    dest = tmp_path/"bad.csv"
    source = DATA.read_text()
    assert old in source
    dest.write_text(source.replace(old, new, 1))
    with pytest.raises(ValueError, match=match): read_prescription(dest, vertex_tolerance_mm=1e-12)


def test_no_hidden_wavelength_selection(tmp_path):
    with pytest.raises(ValueError, match="tabulated"): read_prescription(DATA, wavelength_nm=248)
    p = tmp_path/"two.csv"
    lines = DATA.read_text().splitlines()
    p.write_text(lines[0]+",index_248_nm\n"+"\n".join(x+",1.5" for x in lines[1:]))
    with pytest.raises(ValueError, match="select wavelength"): read_prescription(p)
    assert len(read_prescription(p, wavelength_nm=193.368).encounters) == 48


def test_asphere_slope_and_stigmatic_path():
    s = EvenAsphere(-.1, -2.25, [1e-7, -1e-10])
    r = np.linspace(.1, 4, 17); h = 1e-4
    np.testing.assert_allclose(s.slope(r), (s.sag(r+h)-s.sag(r-h))/(2*h), rtol=1e-8)
    conic = EvenAsphere(-1/10., -2.25)
    r = np.linspace(0, 12, 101); z = conic.sag(r)
    np.testing.assert_allclose(1.5*z+np.hypot(r, 20-z), 20, atol=1e-13)
