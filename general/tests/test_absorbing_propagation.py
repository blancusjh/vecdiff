"""Plane waves in an absorbing region: the angular spectrum with a complex index.

In a homogeneous medium of index n = n' + i kappa every plane wave
exp(i k . r) with k . k = (n k0)^2 solves Maxwell's equations; on a plane the
transverse wavevector is real and k_z = sqrt((n k0)^2 - k_t^2) takes the
root with Im k_z > 0, which decays along the propagation. E is transverse
bilinearly, k . E = 0 (not k . conj(E)), and H = k x E / (k0 Z0). The
representation is checked against the exact decaying plane wave, an
inhomogeneous oblique one, and a Gaussian beam's plane-wave integral
evaluated by direct quadrature.
"""

import numpy as np
import pytest
import vecdiff as o
from scipy.integrate import quad
from scipy.special import j0, j1
from vecdiff.media.medium import Z0
from vecdiff.observables.maxwell import plane_wave_jet

host = o.backend.to_numpy
F = o.fields
Frame = o.geometry.Frame
K0 = 2 * np.pi


def region(n):
    return o.media.HomogeneousRegion("lossy", o.media.Medium(n, "lossy"))


def uniform_lattice(count=16, step=0.25):
    axis = (np.arange(count) - count // 2) * step
    return o.sampling.CartesianGrid(axis, axis), axis


def points_on_axis(z):
    return o.sampling.PointGrid(np.c_[np.zeros_like(z), np.zeros_like(z), z])


@pytest.mark.parametrize("n", [1.5 + 0.02j, 0.05 + 3.3j, 2 + 4j, 1.0 + 1e-6j])
def test_a_plane_wave_in_an_absorbing_medium_decays_as_exp_minus_kappa_k0_z(n):
    """E = exp(i n k0 z) x: amplitude exp(-kappa k0 z), phase n' k0 z, and
    H / E = n / Z0, complex: H lags or leads E by arg n."""
    grid, _ = uniform_lattice()
    values = np.zeros(grid.shape + (2,), complex)
    values[..., 0] = 1.0
    spectrum = F.angular_spectrum(
        F.TransverseElectricField(grid, 1.0, values), region(n)
    )
    z = np.linspace(0.0, 2.0, 9) / max(n.imag, 1.0)
    field = spectrum.evaluate(points_on_axis(z), method="direct")
    E, H = host(field.electric.values), host(field.magnetic)
    exact = np.exp(1j * n * K0 * z)
    assert np.abs(E[:, 0] - exact).max() < 1e-13
    assert np.abs(np.abs(E[:, 0]) - np.exp(-n.imag * K0 * z)).max() < 1e-13
    assert np.abs(E[:, 1:]).max() == 0
    assert np.abs(H[:, 1] * Z0 - n * exact).max() < 1e-13 * abs(n)
    # By FFT, on the lattice moved along its normal.
    moved = spectrum.propagate(z[-1])
    assert np.abs(host(moved.electric.values)[..., 0] - exact[-1]).max() < 1e-13
    assert np.abs(
        host(moved.magnetic)[..., 1] * Z0 - n * exact[-1]
    ).max() < 1e-13 * abs(n)


def oblique_wave(n, mode, te):
    """An inhomogeneous wave of real transverse wavevector on the lattice:
    the spectrum's evaluation and the exact wave, built here."""
    grid, axis = uniform_lattice(32, 0.2)
    frequency = 2 * np.pi * np.fft.fftfreq(32, 0.2)
    kx, ky = frequency[mode[0]], frequency[mode[1]]
    kz = np.sqrt(complex((n * K0) ** 2 - kx**2 - ky**2))
    if kz.imag < 0:
        kz = -kz
    k = np.array([kx, ky, kz])
    s = np.array([-ky, kx, 0]) / np.hypot(kx, ky)
    # TM: k x s, transverse to k bilinearly, scaled to unit tangential length.
    e = s if te else np.cross(k, s) / np.linalg.norm(np.cross(k, s)[:2])
    x, y = np.meshgrid(axis, axis)
    values = e[:2] * np.exp(1j * (kx * x + ky * y))[..., None]
    spectrum = F.angular_spectrum(
        F.TransverseElectricField(grid, 1.0, values), region(n)
    )
    rng = np.random.default_rng(3)
    points = rng.uniform([-2, -2, 0], [2, 2, 1.5], (40, 3))
    field = spectrum.evaluate(o.sampling.PointGrid(points), method="direct")
    phase = np.exp(1j * points @ k)[:, None]
    return spectrum, field, e * phase, np.cross(k, e) / (K0 * Z0) * phase, k, e


@pytest.mark.parametrize("n", [1.5 + 0.02j, 0.05 + 3.3j, 2 + 4j])
@pytest.mark.parametrize("te", [True, False])
def test_an_oblique_wave_in_an_absorbing_medium_is_the_exact_inhomogeneous_wave(n, te):
    """Planes of constant phase (normal Re k) and of constant amplitude (normal
    Im k, along z) differ; E, H and the Maxwell residuals are those of the
    exact wave, with bilinear transversality."""
    spectrum, field, E, H, k, e = oblique_wave(n, (5, 3), te)
    E_got, H_got = host(field.electric.values), host(field.magnetic)
    assert np.abs(E_got - E).max() < 1e-12 * np.abs(E).max()
    assert np.abs(H_got - H).max() < 1e-12 * np.abs(H).max()
    # Re k leans away from the normal; Im k is the normal itself.
    assert np.linalg.norm(np.cross(k.real, k.imag)) > 0.1 * np.linalg.norm(
        k.real
    ) * abs(k.imag[2])
    assert abs(k @ e) < 1e-12 * np.linalg.norm(k)
    if not te:
        # Transverse bilinearly, not in the Hermitian sense.
        assert abs(k @ np.conj(e)) > 1e-3 * np.linalg.norm(k)
    jet = plane_wave_jet(
        spectrum.expansion, o.sampling.PointGrid(np.array([[0.3, -0.2, 0.4]]))
    )
    residuals = jet.bulk_residuals(n, np.ones(1))
    assert max(residuals.values()) < 1e-12


def gaussian_quadrature(n, waist, rho, phi, z):
    """E_x and E_z of the beam E_t = exp(-rho^2 / w^2) x on z = 0, by direct
    quadrature of its plane-wave integral (Hankel transforms of orders 0, 1):

    E_x = (1 / 2 pi) int A(q) J0(q rho) exp(i k_z z) q dq,
    E_z = -(i cos phi / 2 pi) int A(q) J1(q rho) exp(i k_z z) q^2 / k_z dq,

    A(q) = pi w^2 exp(-q^2 w^2 / 4), k_z = sqrt((n k0)^2 - q^2), Im k_z > 0.
    """

    def kz(q):
        root = np.sqrt(complex((n * K0) ** 2 - q * q))
        return root if root.imag >= 0 else -root

    def integral(function):
        top = 14 / waist
        real = quad(lambda q: function(q).real, 0, top, limit=400, epsabs=1e-15)[0]
        imag = quad(lambda q: function(q).imag, 0, top, limit=400, epsabs=1e-15)[0]
        return real + 1j * imag

    def spectrum(q):
        return (
            np.pi * waist**2 * np.exp(-((q * waist) ** 2) / 4) * np.exp(1j * kz(q) * z)
        )

    ex = integral(lambda q: spectrum(q) * j0(q * rho) * q) / (2 * np.pi)
    ez = integral(lambda q: spectrum(q) * j1(q * rho) * q * q / kz(q))
    return ex, -1j * np.cos(phi) * ez / (2 * np.pi)


@pytest.mark.parametrize(
    "n, depth",
    [(1.5 + 1e-3j, 6.0), (1.5 + 0.3j, 1.5), (0.2 + 3.0j, 0.15)],
)
def test_a_gaussian_beam_in_an_absorbing_medium_is_its_plane_wave_integral(n, depth):
    """Weak, strong and metallic absorption: the sampled beam's angular spectrum,
    propagated by FFT and summed at points, against direct quadrature of its
    plane-wave integral: measured 6e-15 of the peak at worst."""
    waist, step = 1.5, 0.15
    count = int(16 * waist / step) // 2 * 2
    axis = (np.arange(count) - count // 2) * step
    grid = o.sampling.CartesianGrid(axis, axis)
    x, y = np.meshgrid(axis, axis)
    values = np.zeros(grid.shape + (2,), complex)
    values[..., 0] = np.exp(-(x**2 + y**2) / waist**2)
    spectrum = F.angular_spectrum(
        F.TransverseElectricField(grid, 1.0, values), region(n), coefficient_rtol=0
    )
    rho = np.array([0.0, 0.6, 1.2, 2.0])
    phi = np.array([0.0, 0.4, 1.1, 2.5])
    points = np.c_[rho * np.cos(phi), rho * np.sin(phi), np.full(4, depth)]
    field = spectrum.electric(o.sampling.PointGrid(points), method="direct")
    got = host(field.values)
    exact = np.array(
        [gaussian_quadrature(n, waist, r, p, depth) for r, p in zip(rho, phi)]
    )
    peak = abs(exact[0, 0])
    assert np.abs(got[:, 0] - exact[:, 0]).max() < 1e-12 * peak
    assert np.abs(got[:, 2] - exact[:, 1]).max() < 1e-12 * peak
    # The FFT propagation agrees with the summed modes on the moved lattice.
    moved = spectrum.propagate(depth, magnetic=False)
    centre = host(moved.values)[count // 2, count // 2, 0]
    assert abs(centre - exact[0, 0]) < 1e-12 * peak
    # The beam decays faster than its axial plane wave: its oblique waves
    # travel farther per unit depth.
    assert abs(exact[0, 0]) < np.exp(-n.imag * K0 * depth)


def test_a_plane_wave_expansion_of_an_absorbing_region_checks_its_complex_dispersion():
    lossy = region(1.5 + 0.02j)
    k = np.array([[0.0, 0.0, (1.5 + 0.02j) * K0]])
    a = np.array([[1.0, 0, 0]])
    expansion = F.PlaneWaveExpansion(k, a, np.zeros(3), 1.0, lossy)
    field = expansion.evaluate(points_on_axis(np.array([0.0, 1.0])), method="direct")
    assert abs(host(field.electric.values)[1, 0] - np.exp(1j * k[0, 2])) < 1e-14
    with pytest.raises(ValueError, match="dispersion"):
        F.PlaneWaveExpansion(k.real, a, np.zeros(3), 1.0, lossy)
    # A plane transport between parallel planes keeps the exact wave.
    grid, _ = uniform_lattice()
    values = np.zeros(grid.shape + (2,), complex)
    values[..., 0] = 1.0
    spectrum = F.angular_spectrum(F.TransverseElectricField(grid, 1.0, values), lossy)
    moved = o.propagation.PlaneTransport().spectrum(spectrum, 0.7)
    E = host(moved.electric(points_on_axis(np.array([0.7])), method="direct").values)
    assert abs(E[0, 0] - np.exp(1j * (1.5 + 0.02j) * K0 * 0.7)) < 1e-13


def test_a_complete_electric_field_in_an_absorbing_region_is_checked_bilinearly():
    """A complete E (with its E_z) is accepted when k . E = 0, and one with
    k . conj(E) = 0 instead is refused."""
    n = 2 + 4j
    _, _, E, _, k, e = oblique_wave(n, (5, 3), te=False)
    grid, axis = uniform_lattice(32, 0.2)
    x, y = np.meshgrid(axis, axis)
    wave = np.exp(1j * (k[0] * x + k[1] * y))[..., None]
    spectrum = F.angular_spectrum(F.ElectricField(grid, 1.0, e * wave), region(n))
    assert spectrum.expansion.amplitudes.shape == (32 * 32, 3)
    # E_z completed with the conjugate k_z: transverse in the Hermitian sense.
    wrong = e.copy()
    wrong[2] = -(k[0] * e[0] + k[1] * e[1]) / np.conj(k[2])
    with pytest.raises(ValueError, match="transverse"):
        F.angular_spectrum(F.ElectricField(grid, 1.0, wrong * wave), region(n))


def test_an_absorbing_region_has_no_propagating_disk_to_keep():
    """Every plane wave decays there: 'discard' is refused, not guessed."""
    grid, _ = uniform_lattice()
    values = np.ones(grid.shape + (2,), complex)
    field = F.TransverseElectricField(grid, 1.0, values)
    with pytest.raises(o.media.AbsorbingMediumError, match="evanescent='retain'"):
        F.angular_spectrum(field, region(1.5 + 0.02j), evanescent="discard")
    spectrum = F.angular_spectrum(field, region(1.5 + 0.02j))
    with pytest.raises(o.media.AbsorbingMediumError, match="paraxial"):
        spectrum.propagate(1.0, model="fresnel")


def test_the_propagator_carries_light_through_an_absorbing_region_without_surfaces():
    """No element, or an index-matched one: the exact spectrum, evaluated on
    a plane the targets sample."""
    n = 1.5 + 0.02j
    lossy = region(n)
    grid, _ = uniform_lattice()
    values = np.zeros(grid.shape + (2,), complex)
    values[..., 0] = 1.0
    light = F.TransverseElectricField(grid, 1.0, values)
    system = o.system.OpticalSystem(medium=lossy)
    solution = o.propagation.Propagator(system)(light)
    field = solution.field(points_on_axis(np.array([0.0, 1.0])), method="direct")
    assert abs(host(field.electric.values)[1, 0] - np.exp(1j * n * K0)) < 1e-13
    matched = o.system.Refraction(
        o.surfaces.Plane(Frame([0, 0, 0.5])), lossy, lossy.material
    )
    solution = o.propagation.Propagator(matched)(light)
    field = solution.field(points_on_axis(np.array([1.0])), method="direct")
    assert abs(host(field.electric.values)[0, 0] - np.exp(1j * n * K0)) < 1e-13
