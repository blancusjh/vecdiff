"""The optical system: an ordered path of encounters, and the regions it derives."""

import numpy as np
import pytest
import vecdiff as o

S, Y, Frame = o.surfaces, o.system, o.geometry.Frame
air, glass = o.media.air, o.media.Dielectric(1.5, "glass")
front, back = S.Plane.z(0.0), S.Plane.z(2.0)


def names(system):
    return [region.name for region in system.regions]


def test_a_lens_is_one_line_per_element_with_its_materials():
    lens = Y.OpticalSystem(
        Y.Stop(1.0, z=-1.0),
        Y.Refraction(front, air, glass),
        Y.Refraction(back, glass, air),
    )
    assert names(lens) == ["air", "air", "glass", "air 2"]
    stop, first, second = lens.elements
    assert stop.medium == lens.regions[0] and stop.aperture == 1.0
    assert first.before == lens.regions[1] and second.after.material == air
    assert Y.OpticalSystem([first, second]).regions == lens.regions[1:]


def test_a_ghost_path_crosses_the_regions_of_its_two_surfaces_again():
    ghost = Y.OpticalSystem(
        Y.Refraction(front, air, glass),
        Y.Reflection(back, glass, behind=air),
        Y.Reflection(front, glass, behind=air),
        Y.Refraction(back, glass, air),
    )
    assert names(ghost) == ["air", "glass", "glass", "glass", "air 2"]
    # Behind each reflection: the region on that surface's other side.
    assert ghost.elements[1].behind == ghost.regions[-1]
    assert ghost.elements[2].behind == ghost.regions[0]
    assert [e.branch for e in ghost.elements] == [
        "transmitted",
        "reflected",
        "reflected",
        "transmitted",
    ]


def test_light_leaving_a_ball_returns_to_the_region_it_came_from():
    ball = S.Sphere(5.0)
    path = Y.OpticalSystem(
        Y.Refraction(ball, air, glass),
        Y.Reflection(ball, glass, behind=air),
        Y.Refraction(ball, glass, air),
    )
    assert names(path) == ["air", "glass", "glass", "air"]
    assert path.regions[-1] == path.regions[0]


def test_a_mirror_keeps_the_light_in_its_region():
    mirror = Y.OpticalSystem(Y.Reflection(front, air))
    assert mirror.elements[0].branch == "mirror" and mirror.elements[0].far is None
    assert mirror.regions[0] == mirror.regions[1]


def test_named_regions_are_kept_and_checked():
    inside = o.media.HomogeneousRegion("inside the plate", glass)
    path = Y.OpticalSystem(Y.Refraction(front, air, inside), Y.Reflection(back, glass))
    assert path.regions[1] == inside == path.regions[2]
    with pytest.raises(ValueError, match="met there before"):
        Y.OpticalSystem(
            Y.Refraction(front, air, glass),
            Y.Reflection(back, glass, behind=air),
            Y.Reflection(front, glass, behind=o.media.Dielectric(1.3)),
        )


def test_adjacent_elements_must_agree_on_the_medium_between_them():
    with pytest.raises(ValueError, match="arrives\\s+through glass.*it is in"):
        Y.OpticalSystem(Y.Refraction(front, air, air), Y.Refraction(back, glass, air))
    with pytest.raises(TypeError, match="a path holds"):
        Y.OpticalSystem(Y.Refraction(front, air, glass), back)
    with pytest.raises(ValueError, match="medium= is"):
        Y.OpticalSystem(Y.Refraction(front, air, glass), medium=glass)
    with pytest.raises(TypeError, match="Dielectric"):
        Y.Refraction(front, 1.0, glass)
    with pytest.raises(ValueError, match="edge"):
        Y.Refraction(front, air, glass, edge=-1.0)


def test_light_cannot_cross_a_surface_it_met_as_a_mirror():
    with pytest.raises(ValueError, match="ideal mirror"):
        Y.OpticalSystem(
            Y.Refraction(front, air, glass),
            Y.Reflection(back, glass),
            Y.Reflection(front, glass, behind=air),
            Y.Refraction(back, glass, air),
        )


def test_the_axis_ray_finds_each_view_and_refuses_a_path_out_of_order():
    from vecdiff.propagation.path import orient

    axis = Frame([0, 0, -5.0])
    path = Y.OpticalSystem(
        Y.Refraction(front, air, glass),
        Y.Reflection(back, glass),
        Y.Refraction(front, glass, air),
    )
    met = orient(path.elements, axis, np.array([0, 0, 1.0]), 1.0)
    normals = [o.backend.to_numpy(e.surface.frame.normal)[2] for e in met.elements]
    assert normals == [1.0, 1.0, -1.0]
    assert [o.backend.to_numpy(a.normal)[2] for a in met.axes] == [1.0, 1.0, -1.0]
    with pytest.raises(ValueError, match="out of order"):
        orient(
            Y.OpticalSystem(
                Y.Refraction(back, air, glass), Y.Refraction(front, glass, air)
            ).elements,
            axis,
            np.array([0, 0, 1.0]),
            1.0,
        )
