import numpy as np

from vecdiff import FieldCartesian, Grid
from vecdiff.view import field_cartesian_maps


def test_field_cartesian_maps_accepts_cartesian_grid():
    axis = np.linspace(-1.0, 1.0, 7)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    grid = Grid.from_cartesian(X, Y)
    Ex = X + 1j * Y
    Ey = X - 1j * Y
    field = FieldCartesian(Ex, Ey, grid=grid, symmetric=False)

    map_x, map_y, extent, labels = field_cartesian_maps(field, half_size=1.0, n_img=7)

    assert labels == ("x", "y")
    assert extent == [-1.0, 1.0, -1.0, 1.0]
    assert np.allclose(map_x, Ex)
    assert np.allclose(map_y, Ey)
