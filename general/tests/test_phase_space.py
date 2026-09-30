"""Phase-space atoms: exact homogeneous R against the lattice angular spectrum."""

import numpy as np
from scipy.special import erfc
from vecdiff_general.propagation.phase_space import AtomField

K = 2 * np.pi


def test_atom_propagation_matches_the_lattice_asm_through_a_focus():
    radius, focus = 200.0, 400.0

    def field(xy):
        r = np.hypot(xy[..., 0], xy[..., 1])
        values = np.zeros(xy.shape[:-1] + (3,), complex)
        values[..., 0] = (
            0.5
            * erfc((r - radius) / 10.0)
            * np.exp(-1j * K * (np.sqrt(r**2 + focus**2) - focus))
        )
        return values

    def gradient(xy):
        return -K * xy / np.sqrt(np.sum(xy**2, -1) + focus**2)[..., None]

    spacing, count = 0.8, 768
    axis = (np.arange(count) - count // 2) * spacing
    X, Y = np.meshgrid(axis, axis)
    plane = field(np.stack((X, Y), -1))[..., 0]
    k = 2 * np.pi * np.fft.fftfreq(count, spacing)
    KX, KY = np.meshgrid(k, k)
    kz = np.sqrt((K**2 - KX**2 - KY**2).astype(complex))
    atoms = AtomField.from_function(
        field,
        gradient,
        K,
        extent=radius + 40,
        tile=2 * np.sqrt(focus),
        step=2.0,
        size=64,
        tolerance=1e-9,
    )
    for z in (focus / 2, focus):
        exact = np.fft.fftshift(
            np.fft.ifft2(
                np.fft.fft2(np.fft.ifftshift(plane))
                * (KX**2 + KY**2 < K**2)
                * np.exp(1j * kz * z)
            )
        )
        near = (abs(X) <= 10) & (abs(Y) <= 10)
        ours = atoms.propagate(z).evaluate(np.column_stack((X[near], Y[near])))[:, 0]
        assert np.linalg.norm(ours - exact[near]) <= 1e-5 * np.linalg.norm(exact[near])
