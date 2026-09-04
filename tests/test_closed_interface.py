import numpy as np
import pytest
from vecdiff import Medium, Sphere, plane_wave, sample_surface, solve_closed_interface
from vecdiff.observables.electromagnetism import boundary_residuals


@pytest.fixture(scope="module")
def solutions():
    sphere = Sphere(.5)
    results = []
    for order in (8, 12):
        boundary = sample_surface(sphere, (-1, 1), (0, 2*np.pi), 2*order, 3*order)
        sources = sample_surface(sphere, (-1, 1), (0, 2*np.pi), order, 2*order)
        results.append(solve_closed_interface(plane_wave(), Medium(1.5), boundary, sources,
                                              inward_offset=.25, outward_offset=.25))
    return sphere, results


def test_held_out_all_four_boundary_conditions_converge(solutions):
    sphere, fields = solutions
    check = sample_surface(sphere, (-1, 1), (.023, 2*np.pi+.023), 27, 43)
    errors = []
    for field in fields:
        eo, ho = field.evaluate(check.points, region="exterior")
        ei, hi = field.evaluate(check.points, region="interior")
        errors.append(boundary_residuals(eo, ho, ei, hi, check.normals, Medium(), Medium(1.5),
                                         weights=check.weights, electric_scale=1., magnetic_scale=1.))
    for component in errors[0]:
        assert errors[1][component] < errors[0][component]/10
        assert errors[1][component] < 2e-3


def test_bulk_fields_match_independent_mie(solutions):
    pytest.importorskip("miepython")
    from references.mie import fields as mie
    sphere, fields = solutions
    q = sample_surface(sphere, (-1, 1), (.1, 2*np.pi+.1), 9, 13)
    for region, scale in [("exterior", 1.3), ("interior", .7)]:
        points = scale*q.points
        e, h = fields[-1].evaluate(points, region=region)
        em, hm = mie(points, .5)
        for a, b in [(e, em), (h, hm)]:
            assert np.linalg.norm(a-b)/np.linalg.norm(b) < 2e-5


def test_maxwell_curls_away_from_auxiliary_sources(solutions):
    field = solutions[1][-1]
    q = np.array([[.12, .23, .17], [-.17, .08, -.09]])
    step = 1e-5
    de, dh = [], []
    for axis in np.eye(3):
        ep, hp = field.evaluate(q+step*axis, region="interior")
        em, hm = field.evaluate(q-step*axis, region="interior")
        de.append((ep-em)/(2*step)); dh.append((hp-hm)/(2*step))
    def curl(d):
        return np.stack((d[1][:, 2]-d[2][:, 1], d[2][:, 0]-d[0][:, 2], d[0][:, 1]-d[1][:, 0]), axis=-1)
    e, h = field.evaluate(q, region="interior")
    np.testing.assert_allclose(curl(de), 2j*np.pi*h, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(curl(dh), -2j*np.pi*1.5**2*e, atol=2e-7, rtol=2e-7)


def test_explicit_domain_and_immutable_solution(solutions):
    field = solutions[1][-1]
    with pytest.raises(ValueError):
        field.evaluate([[0, 0, 0]], region="unknown")
    with pytest.raises(ValueError):
        field.coefficients[0] = 0


def test_diagnostics_use_json_serializable_python_scalars(solutions):
    import json
    field = solutions[1][-1]
    assert type(field.rank) is int
    json.dumps(dict(rank=field.rank, fit_residual=field.fit_residual,
                    condition_number=field.condition_number), allow_nan=False)
