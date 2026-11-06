"""Spin, helicity, canonical momentum and stress-tensor forces against closed forms."""

import numpy as np
import pytest
import vecdiff_general as o
from numpy.testing import assert_allclose
from vecdiff_general.media.medium import C0, EPSILON0, MU0
from vecdiff_general.propagation.references import MieSphere

bd, host = o.backend.bd, o.backend.to_numpy
obs = o.observables

AIR = o.media.HomogeneousRegion("air", o.media.Dielectric(1.0))
GLASS = o.media.HomogeneousRegion("glass", o.media.Dielectric(1.5))
rng = np.random.default_rng(11)
POINTS = o.sampling.PointGrid(rng.uniform(-1.0, 1.0, (40, 3)))


def densities(expansion, material):
    jet = obs.plane_wave_jet(expansion, POINTS)
    field = jet.field
    omega = 2 * np.pi * C0 / expansion.wavelength0
    return {
        "w": host(obs.energy_density(field, material)),
        "omega": omega,
        "S": host(obs.spin_density(field, material)),
        "h": host(obs.helicity_density(field, material)),
        "p_o": host(obs.canonical_momentum(jet, material)),
        "p_s": host(obs.spin_momentum(jet, material)),
        "P": host(obs.poynting(field)),
        "jet": jet,
    }


@pytest.mark.parametrize("sigma", [1, -1, 0])
def test_plane_wave_carries_spin_sigma_w_over_omega_and_momentum_n_w_over_c(sigma):
    s = np.array([0.3, -0.4, 0.866])
    s /= np.linalg.norm(s)
    u = np.cross(s, [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(s, u)
    k = 2 * np.pi * 1.5
    wave = o.fields.PlaneWaveExpansion(
        [k * s], [(u + 1j * sigma * v) / np.sqrt(2)], [0, 0, 0], 1.0, GLASS
    )
    d = densities(wave, GLASS.material)
    w, omega = d["w"][:, None], d["omega"]
    assert_allclose(d["S"], sigma * w * s / omega, atol=1e-14 * w.max() / omega)
    assert_allclose(d["h"], sigma * d["w"] / omega, atol=1e-14 * w.max() / omega)
    assert_allclose(d["p_o"], 1.5 * w * s / C0, rtol=1e-13)
    assert_allclose(d["p_s"], 0, atol=1e-13 * w.max() / C0)
    for part in ("electric", "magnetic"):
        assert_allclose(
            host(obs.spin_density(d["jet"].field, GLASS.material, part=part)),
            d["S"],
            atol=1e-14 * w.max() / omega,
        )
    origin = np.array([0.2, 0.1, -0.3])
    assert_allclose(
        host(obs.orbital_angular_momentum(d["jet"], GLASS.material, origin)),
        np.cross(host(POINTS.points) - origin, d["p_o"]),
        atol=1e-14 * abs(d["p_o"]).max(),
    )


@pytest.mark.parametrize("ratio", [1.2, 2.0])
@pytest.mark.parametrize("m", [0, 1j, -1j, 0.3 + 0.7j, 2.0])
def test_evanescent_wave_transverse_spin_and_super_momentum(ratio, m):
    """Bliokh, Bekshaev & Nori, Nat. Commun. 5, 3300 (2014), Eqs. (2)-(5).

    E = (x + m k/kz y - i kappa/kz z) exp(i kz z - kappa x): the canonical
    momentum exceeds the wavenumber, k_z > k; the transverse spin kappa/kz is
    independent of polarization; the transverse Poynting momentum is
    polarization dependent, sigma kappa k / kz.
    """
    k = 2 * np.pi
    kz = ratio * k
    kappa = np.sqrt(kz**2 - k**2)
    wave = o.fields.PlaneWaveExpansion(
        [[1j * kappa, 0, kz]], [[1, m * k / kz, -1j * kappa / kz]], [0, 0, 0], 1.0, AIR
    )
    d = densities(wave, AIR.material)
    sigma = 2 * np.imag(m) / (1 + abs(m) ** 2)
    scale = (d["w"] / d["omega"])[:, None]
    one = np.ones_like(d["w"])[:, None]
    tol = 1e-13 * scale.max()
    assert_allclose(d["p_o"], scale * k * one * [0, 0, ratio], atol=k * tol)
    assert_allclose(d["S"], scale * [0, kappa / kz, sigma * k / kz], atol=tol)
    assert_allclose(d["h"], sigma * scale[:, 0], atol=tol)
    assert_allclose(
        d["P"] / C0**2, scale * [0, sigma * kappa * k / kz, k**2 / kz], atol=k * tol
    )


def test_poynting_momentum_splits_into_canonical_and_spin_parts_at_a_tight_focus():
    """eps mu P = p_o + curl S / 2 pointwise, for each part, in a medium."""
    beam = o.fields.FocusedBeam(
        o.fields.profiles.Gaussian(30.0),
        o.fields.polarization.cartesian(1, 0.5 + 0.5j),
        focal_length=40.0,
        max_sine=0.9,
        wavelength0=1.0,
        region=GLASS,
    )
    jet = obs.plane_wave_jet(beam.plane_waves(2.0), POINTS)
    material = GLASS.material
    kinetic = material.epsilon * MU0 * host(obs.poynting(jet.field))
    for part in ("dual", "electric", "magnetic"):
        total = host(obs.canonical_momentum(jet, material, part=part)) + host(
            obs.spin_momentum(jet, material, part=part)
        )
        assert_allclose(total, kinetic, atol=1e-12 * abs(kinetic).max())
    assert np.linalg.norm(host(obs.spin_density(jet.field, material))) > 0
    with pytest.raises(ValueError, match="part"):
        obs.spin_density(jet.field, material, part="both")


def sphere_quadrature(radius, count):
    t, w = np.polynomial.legendre.leggauss(count)
    phi = 2 * np.pi * np.arange(2 * count) / (2 * count)
    sine = np.sqrt(1 - t**2)
    normals = np.stack(
        np.broadcast_arrays(
            sine[:, None] * np.cos(phi), sine[:, None] * np.sin(phi), t[:, None]
        ),
        axis=-1,
    )
    weights = radius**2 * w[:, None] * np.full(phi.shape, np.pi / count)
    return o.sampling.SurfaceGrid(radius * normals, normals, weights)


def test_source_free_fields_exert_no_force_or_torque_on_an_empty_volume():
    k = 2 * np.pi
    waves = o.fields.PlaneWaveExpansion(
        [[0, 0, k], [k * 0.6, 0, k * 0.8], [0, -k, 0]],
        [[1, 1j, 0], [0, 0.4, 0], [0.3j, 0, 1]],
        [0, 0, 0],
        1.0,
        AIR,
    )
    field = waves.evaluate(sphere_quadrature(1.3, 24))
    scale = 0.5 * EPSILON0 * 4 * np.pi * 1.3**2
    assert_allclose(obs.force(field, AIR.material), 0, atol=1e-13 * scale)
    assert_allclose(obs.torque(field, AIR.material), 0, atol=1e-13 * scale)


@pytest.mark.parametrize("polarization", [[1, 0, 0], [1, 1j, 0]])
def test_radiation_pressure_on_a_mie_sphere_matches_wiscombe(polarization):
    """Wiscombe MIEV0, m = 0.75, x = 10: Q_ext - g Q_sca = 2.232265 - 2.001164."""
    radius = 10 / (2 * np.pi)
    amplitude = np.asarray(polarization) / np.linalg.norm(polarization)
    wave = o.fields.PlaneWaveExpansion(
        [[0, 0, 2 * np.pi]], [amplitude], [0, 0, 0], 1.0, AIR
    )
    sphere = MieSphere(
        wave,
        o.volumes.SphereVolume([0, 0, 0], radius),
        o.media.HomogeneousRegion("m = 0.75", o.media.Dielectric(0.75)),
    )
    field = sphere.evaluate(sphere_quadrature(1.4 * radius, 72))
    intensity = 0.5 * EPSILON0 * C0
    efficiency = obs.force(field, AIR.material) * C0 / (intensity * np.pi * radius**2)
    assert_allclose(efficiency, [0, 0, 2.232265 - 2.001164], atol=2e-6)
    # Bohren & Huffman (4.62), (4.82) from the same coefficients, to the digit.
    a, b, n = sphere.a, sphere.b, np.arange(1, sphere.order + 1)
    asymmetry = np.sum(
        n[:-1] * (n[:-1] + 2) / (n[:-1] + 1) * (a[:-1] * a[1:].conj()).real
        + n[:-1] * (n[:-1] + 2) / (n[:-1] + 1) * (b[:-1] * b[1:].conj()).real
    ) + np.sum((2 * n + 1) / (n * (n + 1)) * (a * b.conj()).real)
    series = sphere.efficiencies()["extinction"] - 4 / 10**2 * asymmetry
    assert_allclose(efficiency[2], series, rtol=1e-9)
    # Lossless sphere: no torque, also for circular polarization.
    torque = obs.torque(field, AIR.material)
    assert_allclose(torque, 0, atol=1e-10 * intensity * radius**3 / C0)
