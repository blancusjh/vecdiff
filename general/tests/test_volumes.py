"""Spatial membership under placement, curved faces and finite rim closure."""

import numpy as np
from numpy.testing import assert_array_equal
from vecdiff_general.geometry import Frame
from vecdiff_general.surfaces import Plane, SphericalCap
from vecdiff_general.volumes import SphereVolume, VolumeBetweenSurfaces


def test_volume_membership_is_preserved_by_rigid_placement():
    front = Plane()
    back = Plane(Frame(origin=[0, 0, 2]))
    volume = VolumeBetweenSurfaces(front, back, rim_radius=1)

    # Quarter-turn about y: local axial extent becomes global x extent.
    basis = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    frame = Frame(origin=[3, -2, 5], basis=basis)
    placed = VolumeBetweenSurfaces(
        Plane(frame),
        Plane(frame.translated(2 * frame.normal)),
        rim_radius=1,
    )

    local_points = np.array(
        [[0, 0, 1], [0, 0, 0], [0, 0, 2], [1, 0, 1], [1.1, 0, 1], [0, 0, 3]]
    )
    global_points = frame.origin + local_points @ frame.basis.T
    expected = [True, True, True, True, False, False]

    assert_array_equal(volume.contains(local_points), expected)
    assert_array_equal(placed.contains(global_points), expected)
    assert placed.contains(global_points[0])


def test_curved_faces_and_rim_bound_the_volume_independently():
    front = SphericalCap(radius=2)
    back = Plane(Frame(origin=[0, 0, 1]))
    volume = VolumeBetweenSurfaces(front, back, rim_radius=1)

    # The last position is outside both the rim and the spherical graph chart.
    # Membership must reject it without evaluating the graph past its equator.
    points = np.array([[0.8, 0, 0.1], [0.8, 0, 0.2], [0.8, 0, 1.1], [3, 0, 0.5]])

    assert_array_equal(volume.contains(points), [False, True, False, False])


def test_sphere_membership_includes_boundary_and_preserves_batch_shape():
    center = np.array([2, -3, 4])
    volume = SphereVolume(center, radius=1)
    offsets = np.array([[[0, 0, 0], [0, 1, 0]], [[1, 1, 0], [0, 0, -1]]])

    assert_array_equal(
        volume.contains(center + offsets),
        [[True, True], [False, True]],
    )
    assert volume.contains(center)
