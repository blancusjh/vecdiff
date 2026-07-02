import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.animation import FuncAnimation

from vecdiff import FieldCartesian, Grid, animate_harmonic_field, harmonic_field_frames


def _sample_field(n=17):
    axis = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(axis, axis)
    grid = Grid.from_cartesian(X, Y)
    # Right-circular field: Ey = i Ex, so the real vector rotates in time.
    amp = np.exp(-(X**2 + Y**2))
    return FieldCartesian(x=amp.astype(complex), y=1j * amp, grid=grid, symmetric=False)


def test_harmonic_field_frames_shapes_and_intensity():
    field = _sample_field()
    # A half_size forces resampling onto an n_img x n_img display mesh.
    xx, yy, ex, ey, intensity = harmonic_field_frames(field, half_size=0.9, n_img=24)
    assert xx.shape == yy.shape == ex.shape == ey.shape == intensity.shape == (24, 24)
    assert np.allclose(intensity, np.abs(ex) ** 2 + np.abs(ey) ** 2)


def test_animate_harmonic_field_returns_animation_with_frame_count():
    field = _sample_field()
    anim = animate_harmonic_field(field, n_img=24, n_frames=12, target_arrows=6)
    assert isinstance(anim, FuncAnimation)
    assert anim._save_count == 12


def test_harmonic_arrows_rotate_for_circular_light():
    # For a right-circular field the instantaneous quiver at a bright point must
    # rotate between frames rather than merely oscillate along a fixed line.
    field = _sample_field()
    anim = animate_harmonic_field(field, n_img=24, n_frames=8, target_arrows=6)
    q = anim._func(0)[0]
    u0, v0 = np.array(q.U), np.array(q.V)
    anim._func(2)  # advance a quarter of the cycle
    u1, v1 = np.array(q.U), np.array(q.V)
    # A pure oscillation keeps direction fixed (u,v) -> scalar*(u,v); rotation
    # makes the two frames linearly independent at the bright samples.
    bright = np.hypot(u0, v0) > 0.1 * np.hypot(u0, v0).max()
    cross = np.abs(u0 * v1 - v0 * u1)[bright]
    assert np.any(cross > 1e-6 * (np.hypot(u0, v0).max() ** 2))
