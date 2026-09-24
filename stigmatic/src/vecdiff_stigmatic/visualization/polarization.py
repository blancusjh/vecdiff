"""Polarization ellipse glyphs in the observation plane's declared basis."""

import numpy as np

from ..observables.polarization import ellipse_parameters


def plot_ellipses(field, *, stride=4, axis=None, relative_threshold=0.03):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    grid = field.grid

    if not hasattr(grid, "frame") or not hasattr(grid, "u"):
        raise TypeError("ellipse plots require an embedded plane grid")

    if int(stride) != stride or stride < 1:
        raise ValueError("stride must be a positive integer")

    if axis is None:
        _, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)

    orientation, ellipticity, amplitude = ellipse_parameters(
        field, grid.frame.basis[:, :2]
    )
    maximum = np.max(amplitude)
    width = 0.75 * stride * min(grid.spacing) * 1e6

    for row in range(0, len(grid.v), stride):
        for column in range(0, len(grid.u), stride):
            if amplitude[row, column] <= relative_threshold * maximum:
                continue

            ratio = abs(np.tan(ellipticity[row, column]))
            glyph = Ellipse(
                (grid.u[column] * 1e6, grid.v[row] * 1e6),
                width=width,
                height=max(width * ratio, width * 0.01),
                angle=np.degrees(orientation[row, column]),
                fill=False,
                edgecolor="C0" if ellipticity[row, column] >= 0 else "C3",
            )
            axis.add_patch(glyph)

    axis.set(
        xlim=(grid.u[0] * 1e6 - width, grid.u[-1] * 1e6 + width),
        ylim=(grid.v[0] * 1e6 - width, grid.v[-1] * 1e6 + width),
        xlabel="u (µm)",
        ylabel="v (µm)",
        aspect="equal",
        title="Polarization projected onto the observation plane",
    )

    return axis
