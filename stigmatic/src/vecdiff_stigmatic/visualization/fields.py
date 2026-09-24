"""Plots of physical field samples; plotting never constructs optical coordinates."""

import numpy as np


def plot_components(field, *, quantity="abs2", figure=None):
    """Plot Ex, Ey, Ez on an embedded PlaneGrid; axes show local micrometres."""
    import matplotlib.pyplot as plt

    from ..sampling.grids import PlaneGrid

    if not isinstance(field.grid, PlaneGrid):
        raise TypeError("component images require a PlaneGrid")

    transforms = {
        "abs2": lambda values: np.abs(values) ** 2,
        "real": np.real,
        "imag": np.imag,
    }

    if quantity not in transforms:
        raise ValueError("quantity must be abs2, real, or imag")

    if figure is None:
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    else:
        axes = figure.subplots(1, 3)

    grid = field.grid
    du, dv = grid.spacing
    extent = (
        np.array(
            [
                grid.u[0] - du / 2,
                grid.u[-1] + du / 2,
                grid.v[0] - dv / 2,
                grid.v[-1] + dv / 2,
            ]
        )
        * 1e6
    )

    for component, axis in enumerate(axes):
        image = axis.imshow(
            transforms[quantity](field.values[..., component]),
            origin="lower",
            extent=extent,
        )
        axis.set(title=("Ex", "Ey", "Ez")[component], xlabel="u (µm)", ylabel="v (µm)")
        figure.colorbar(image, ax=axis, label="V²/m²" if quantity == "abs2" else "V/m")

    return figure, axes
