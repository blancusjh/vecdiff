"""Harmonic polarization animation with the documented exp(-i omega t) convention."""

import numpy as np


def animate_polarization(field, *, stride=4, frames=48):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    grid = field.grid

    if not hasattr(grid, "frame") or not hasattr(grid, "u"):
        raise TypeError("animation requires an embedded PlaneGrid")

    projected = field.values @ grid.frame.basis[:, :2]
    sampled = projected[::stride, ::stride]
    u, v = np.meshgrid(grid.u[::stride] * 1e6, grid.v[::stride] * 1e6)
    figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
    quiver = axis.quiver(u, v, sampled[..., 0].real, sampled[..., 1].real)
    axis.set(xlabel="u (µm)", ylabel="v (µm)", aspect="equal")

    def update(index):
        values = np.real(sampled * np.exp(-2j * np.pi * index / frames))
        quiver.set_UVC(values[..., 0], values[..., 1])

        return (quiver,)

    return FuncAnimation(figure, update, frames=frames, interval=40, blit=True)
