"""Small shared helpers to keep the runnable examples short and readable."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from vecdiff import FieldCartesian, Grid
from _output import example_output_dir, print_saved


def focal_field(field, scale):
    """Rescale a propagated field's grid to focal coordinates by ``scale``.

    Works for both Cartesian (FFT) and polar (Hankel) propagation grids. The
    caller supplies ``scale`` because the convention differs per propagation
    (e.g. ``zi / ni`` for the Hankel example vs ``zi / (2*pi*ni)`` elsewhere).
    """
    if field.grid.type == "cartesian":
        grid = Grid.from_cartesian(
            scale * field.grid.X, scale * field.grid.Y, domain="focal/lambda", dual=field.grid.dual
        )
    elif field.grid.type == "polar":
        grid = Grid.from_polar(scale * field.grid.r, field.grid.varphi)
        grid.domain = "focal/lambda"
    else:
        raise ValueError(f"Unsupported focal grid type: {field.grid.type!r}.")
    return FieldCartesian(field.x, field.y, grid=grid, symmetric=False)


def figure_saver(example_file, dpi=220):
    """Return ``save(obj, name)`` that writes a figure to the example's output dir.

    ``obj`` may be a Matplotlib ``Figure`` or an ``Axes`` (the polarization
    helpers return the axes), collapsing the repeated savefig/print boilerplate
    into a single call.
    """
    output_dir = example_output_dir(example_file)

    def save(obj, name):
        fig = obj.figure if isinstance(obj, Axes) else obj
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print_saved(path)

    return save
