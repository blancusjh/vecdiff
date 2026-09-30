"""Harmonic polarization animation with the documented exp(-i omega t) convention."""

import numpy as np

from .._arrays import positive_integer
from .polarization import (
    _phase_geometry,
    _phase_vectors,
    _plane_data,
    plot_polarization,
)


def animate_polarization(field, *, frames=48, phase=0.0, **kwargs):
    """Animate Re(E exp(-i phase)) inside the same ellipse map.

    Accepts plot_polarization options. The movie retains each local phasor's
    phase; electric arrows start at the fixed glyph centers, with their tips on
    the ellipses. No point markers are drawn. phase sets the initial phase
    in radians. Glyph lengths are scaled to the ellipse, not to an SI length.
    Returns a Matplotlib FuncAnimation suitable for to_jshtml() or save().
    """
    from matplotlib.animation import FuncAnimation

    frames = positive_integer(frames, "frames")
    axis = plot_polarization(field, phase=phase, **kwargs)
    bodies = next(
        (
            item
            for item in axis.collections
            if item.get_gid() == "polarization-ellipses"
        ),
        None,
    )

    if bodies is None:
        raise ValueError("animation requires a nonzero projected field")

    curves = np.asarray(bodies.get_segments())
    centers = (curves[:, 0] + curves[:, 32]) / 2
    radii = np.linalg.norm(curves[:, 0] - centers, axis=-1)
    electric, *_ = _plane_data(field, 0)
    vectors = _phase_vectors(electric, centers, radii)
    arrows = next(
        item for item in axis.collections if item.get_gid() == "polarization-rotation"
    )
    colors = bodies.get_colors().copy()
    colors[:, 3] = 1.0
    # Keep the readout outside the data area: an overlay hides outer glyphs.
    title = axis.get_title()
    phase_label = axis.title

    def update(index):
        current_phase = phase + 2 * np.pi * index / frames
        segments, heads, owners = _phase_geometry(
            vectors, centers, radii, current_phase
        )
        arrows.set_verts(heads)
        arrows.set_facecolors(colors[owners])
        phase_label.set_text(f"{title}\nωt = {current_phase / np.pi:.2f}π")

        return arrows, phase_label

    return FuncAnimation(axis.figure, update, frames=frames, interval=40, blit=True)
