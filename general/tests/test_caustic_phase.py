"""The root of the flux factor through a focus: 1 / sqrt(J) -> exp(-i pi kappa / 2) / sqrt(|J|).

A plane interface has an exact reference for any incident field: its plane
waves, each transmitted by its own Fresnel coefficient. A focus is placed
inside the incident segment of the leg, where the ray tube collapses, and the
carried field is compared with that reference on the virtual output plane.
"""

import numpy as np
import pytest
import vecdiff as o

bd, host = o.backend.bd, o.backend.to_numpy
F, S, P = o.fields, o.surfaces, o.propagation
Frame = o.geometry.Frame
AIR = o.media.HomogeneousRegion("air", o.media.Dielectric(1.0))
DENSE = o.media.HomogeneousRegion("dense glass", o.media.Dielectric(2.5))
STEP = 0.4


def line_carrier(depth):
    """Rays that converge on the line x = 0, z = -depth, parallel to y."""

    def toward(r):
        r = bd.asarray(r, float)
        d = bd.zeros(r.shape)
        d[..., 0], d[..., 2] = -r[..., 0], -depth - r[..., 2]
        return d

    def gradient(r):
        d = toward(r)
        return d / bd.linalg.norm(d, axis=-1, keepdims=True)

    def hessian(r):
        d = toward(r)
        length = bd.linalg.norm(d, axis=-1)
        s = d / length[..., None]
        projector = bd.eye(3) - s[..., :, None] * s[..., None, :]
        projector[..., 1, 1] -= 1.0
        return -projector / length[..., None, None]

    return F.Carrier(lambda r: -bd.linalg.norm(toward(r), axis=-1), gradient, hessian)


def modes(count, sine):
    """Lattice frequencies of ``count`` points and a Gaussian pupil cut at ``sine``."""
    k = 2 * np.pi * np.fft.fftfreq(count, STEP)
    return (
        k,
        lambda kappa: (
            np.exp(-((kappa / (np.pi * sine)) ** 2)) * (kappa < 2 * np.pi * sine)
        ),
    )


def fit(carried, exact):
    """Best complex factor between two fields and the relative misfit left after it."""
    factor = np.vdot(exact, carried) / np.vdot(exact, exact)
    return factor, np.linalg.norm(carried / factor - exact) / np.linalg.norm(exact)


def line_focus(plane, focus, sine=0.3):
    """Carried over exact for an s-polarised cylindrical wave focused on a line.

    The reference plane lies ``plane`` in front of the interface z = 0 and
    the focal line ``focus`` in front of it (behind it when negative).
    """
    count = int(2 * (1.6 * sine * plane + 20) / STEP) // 2 * 2
    x = (np.arange(count) - count // 2) * STEP
    y = (np.arange(8) - 4) * STEP
    k, pupil = modes(count, sine)
    k1, k2 = 2 * np.pi, 2 * np.pi * 2.5
    amplitude = pupil(np.abs(k))
    kz1 = np.sqrt(np.maximum(k1**2 - k**2, 0.0))
    kz2 = np.sqrt(k2**2 - k**2)
    waves = np.exp(1j * np.outer(k, x))
    t_s = 2 * kz1 / (kz1 + kz2)

    values = np.zeros((len(y), count, 2), complex)
    values[..., 1] = (amplitude * np.exp(1j * kz1 * (focus - plane))) @ waves
    grid = o.sampling.CartesianGrid(x, y, Frame([0, 0, -plane]))
    incident = F.angular_spectrum(
        F.TransverseElectricField(grid, 1.0, values), AIR, evanescent="discard"
    )
    leg = o.transport.LatticeTransport(
        grid,
        o.system.Refraction(S.Plane(), AIR, DENSE),
        line_carrier(focus),
        1.0,
    )
    carried, report = leg.carry(incident, method="direct")

    assert carried.grid.shape == grid.shape
    exact = (t_s * amplitude * np.exp(1j * (kz1 * focus - kz2 * plane))) @ waves
    row = host(carried.values)[len(y) // 2, :, 1]
    bright = host(leg.lit).reshape(grid.shape)[len(y) // 2] & (
        np.abs(exact) > 0.2 * np.abs(exact).max()
    )

    return fit(row[bright], exact[bright]) + (report,)


def test_tube_collapses_are_counted_along_each_segment_with_their_sign():
    def rays(carrier, plane, index=2.5):
        radii = bd.asarray([[0.5, 0.0], [3.0, 1.0], [-2.0, 4.0]])
        return o.transport.InterfaceMap(S.Plane(), carrier, 1.0, index, -plane).forward(
            radii
        )

    # A point focus 53 in front of the interface; its refracted rays diverge
    # from a virtual focus near 2.5 * 53 = 132 in front of it.
    point = F.Carrier.converging([0, 0, -53.0])
    inside = rays(point, 93.0)
    assert host(inside.valid).all()
    assert (host(inside.caustics) == 2).all()
    np.testing.assert_allclose(np.angle(host(inside.flux)), np.pi, atol=1e-15)

    # No focus between the plane and the interface.
    before = rays(F.Carrier.converging([0, 0, 40.0]), 40.0)
    assert host(before.valid).all()
    assert (host(before.caustics) == 0).all()
    assert (host(before.flux).imag == 0).all() and (host(before.flux).real > 0).all()

    # A plane beyond the virtual focus: the backward segment undoes both collapses.
    beyond = rays(point, 200.0)
    assert host(beyond.valid).all()
    assert (host(beyond.caustics) == 0).all()

    # A focal line collapses one direction: a quarter turn, exp(-i pi / 2).
    line = rays(line_carrier(50.0), 85.0)
    assert host(line.valid).all()
    assert (host(line.caustics) == 1).all()
    np.testing.assert_allclose(np.angle(host(line.flux)), -np.pi / 2, atol=1e-15)

    # Index-matched: the same plane, the same rays, nothing collapses in total.
    matched = rays(point, 93.0, index=1.0)
    assert (host(matched.caustics) == 0).all()
    np.testing.assert_allclose(host(matched.output), host(matched.start), atol=1e-12)
    np.testing.assert_allclose(host(matched.flux), 1.0, rtol=1e-12)


def test_line_focus_inside_the_leg_turns_the_field_by_minus_a_quarter_wave():
    """The quarter turn's sign: with exp(+i pi / 2) the factor would tend to -1.

    A point focus cannot tell the two signs apart; a focal line can. The
    carried field tends to the exact one as the system grows.
    """
    small, misfit_small, report = line_focus(85.0, 50.0)
    large, misfit_large, _ = line_focus(170.0, 100.0)

    assert report.caustics == 1 and report.folded_relative_amplitude == 0
    assert small.real > 0.85 and large.real > 0.95
    assert abs(large - 1) < 0.6 * abs(small - 1) < 0.2
    assert misfit_large < 0.6 * misfit_small


def test_virtual_plane_beyond_the_virtual_focus_undoes_the_quarter_wave():
    factor, _, report = line_focus(340.0, 100.0)

    assert report.caustics == 0
    assert report.folded_relative_amplitude < 2e-3
    assert abs(factor - 1) < 0.12


def test_leg_without_a_focus_is_unchanged():
    factor, misfit, report = line_focus(40.0, -50.0)

    assert report.caustics == 0
    assert abs(factor - 1) < 0.02 and misfit < 0.02


def test_point_focus_inside_the_leg_inverts_the_field():
    """Before the continued root this case returned -1 times the exact field."""
    plane, focus, sine = 93.0, 53.0, 0.5
    count = int(2 * 40.0 / STEP)
    axis = (np.arange(count) - count // 2) * STEP
    k, pupil = modes(count, sine)
    kx, ky = np.meshgrid(k, k)
    kappa = np.hypot(kx, ky)
    amplitude = pupil(kappa)
    keep = amplitude > 1e-9
    kx, ky, amplitude = kx[keep], ky[keep], amplitude[keep]
    k1, k2 = 2 * np.pi, 2 * np.pi * 2.5
    kz1, kz2 = np.sqrt(k1**2 - kx**2 - ky**2), np.sqrt(k2**2 - kx**2 - ky**2)

    # x-polarised transverse amplitudes; the lattice holds the modes exactly.
    spectrum = np.zeros((count, count), complex)
    spectrum[keep] = amplitude * np.exp(1j * kz1 * (focus - plane))
    shift = np.exp(1j * axis[0] * (kx + ky))
    spectrum[keep] *= shift
    values = np.zeros((count, count, 2), complex)
    values[..., 0] = np.fft.ifft2(spectrum) * count**2
    grid = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, -plane]))
    incident = F.angular_spectrum(
        F.TransverseElectricField(grid, 1.0, values), AIR, evanescent="discard"
    )
    carrier = F.Carrier.converging([0, 0, -focus])
    leg = o.transport.LatticeTransport(
        grid, o.system.Refraction(S.Plane(), AIR, DENSE), carrier, 1.0
    )
    carried, report = leg.carry(incident, method="direct")

    assert report.caustics == 2 and report.folded_relative_amplitude == 0

    # Exact: each plane wave by its own Fresnel coefficients, on the virtual plane.
    e = np.stack((amplitude, 0 * amplitude, -kx * amplitude / kz1), axis=-1)
    s_i = np.stack((kx, ky, kz1), axis=-1) / k1
    s_t = np.stack((kx, ky, kz2), axis=-1) / k2
    te = np.cross([0, 0, 1.0], s_i)
    size = np.linalg.norm(te, axis=-1, keepdims=True)
    te = np.where(size > 1e-12, te / np.where(size > 1e-12, size, 1), [1.0, 0, 0])
    p_i, p_t = np.cross(te, s_i), np.cross(te, s_t)
    c_i, c_t = s_i[:, 2], s_t[:, 2]
    t_s, t_p = 2 * c_i / (c_i + 2.5 * c_t), 2 * c_i / (2.5 * c_i + c_t)
    transmitted = (t_s * np.sum(e * te, axis=-1))[:, None] * te + (
        t_p * np.sum(e * p_i, axis=-1)
    )[:, None] * p_t
    transmitted = transmitted * np.exp(1j * (kz1 * focus - kz2 * plane))[:, None]

    out = carried.grid
    u, v = host(out.u), host(out.v)
    field = host(carried.values)
    strength = np.abs(field).sum(axis=-1) * host(leg.lit).reshape(out.shape)
    bright = np.argwhere(strength > 0.2 * strength.max())
    chosen = bright[np.random.default_rng(0).choice(len(bright), 40, replace=False)]
    exact = np.stack(
        [np.exp(1j * (u[j] * kx + v[i] * ky)) @ transmitted[:, :2] for i, j in chosen]
    )
    factor, misfit = fit(field[chosen[:, 0], chosen[:, 1]], exact)

    assert factor.real > 0.9 and abs(factor - 1) < 0.25
    assert misfit < 0.2


def test_leg_that_folds_under_the_field_is_refused():
    """Rays from both sides of a caustic reach the same output points."""
    axis = (np.arange(160) - 80) * 0.5
    grid = o.sampling.CartesianGrid(axis, axis, Frame([0, 0, -60.0]))
    x, y = np.meshgrid(axis, axis)
    values = np.zeros(grid.shape + (2,), complex)
    values[..., 0] = np.exp(-(x**2 + y**2) / 40.0**2)
    incident = F.angular_spectrum(
        F.TransverseElectricField(grid, 1.0, values), AIR, evanescent="discard"
    )
    glass = o.media.HomogeneousRegion("glass", o.media.Dielectric(1.5))
    leg = o.transport.LatticeTransport(
        grid,
        o.system.Refraction(S.Plane(), AIR, glass),
        F.Carrier.converging([0, 0, -30.0]),
        1.0,
    )

    with pytest.raises(ValueError, match="the ray map folds"):
        leg.carry(incident, method="direct")


def test_meridian_leg_says_when_the_carrier_focuses_inside_it():
    from vecdiff.transport.meridian import MeridianTransport

    interface = o.system.Refraction(S.Plane(), AIR, DENSE)

    with pytest.raises(ValueError, match="focuses inside this leg"):
        MeridianTransport(
            interface,
            F.Carrier.converging([0, 0, -53.0]),
            Frame(),
            1.0,
            offset=-93.0,
            scale=20.0,
        )
