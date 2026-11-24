"""The trusted angular-spectrum path against the validating one it replaced.

``_reference`` is the former implementation: centered FFTs by explicit
shifts, frequency meshes, the public PlaneWaveExpansion and PlaneSpectrum
constructors, and propagation through magnetic_amplitudes.
"""

import math
import pickle
from dataclasses import replace

import numpy as np
import pytest
import vecdiff as o
from numpy.testing import assert_allclose
from vecdiff.fields import (
    ElectricField,
    PlaneSpectrum,
    PlaneWaveExpansion,
    TransverseElectricField,
    angular_spectrum,
)
from vecdiff.media.medium import Z0

WAVELENGTH = 0.5


def _centered(g, inverse=False):
    transform = np.fft.ifft2 if inverse else np.fft.fft2
    shifted = np.fft.ifftshift(g, axes=(0, 1))
    return np.fft.fftshift(transform(shifted, axes=(0, 1)), axes=(0, 1))


def _reference(field, region, direction, evanescent, rtol):
    grid = field.grid
    nx, ny = len(grid.u), len(grid.v)
    fx = np.fft.fftshift(np.fft.fftfreq(nx, grid.u[1] - grid.u[0]))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, grid.v[1] - grid.v[0]))
    fu, fv = np.meshgrid(2 * np.pi * fx, 2 * np.pi * fy, indexing="xy")
    coefficients = _centered(field.values) / (nx * ny)
    k = 2 * np.pi * region.material.n / field.wavelength0
    squared = fu**2 + fv**2
    kz = direction * np.sqrt((k**2 - squared).astype(complex))
    magnitude = np.linalg.norm(coefficients, axis=-1)
    roundoff = magnitude < rtol * magnitude.max()
    roundoff_rms = float(np.linalg.norm(coefficients[roundoff]))
    coefficients[roundoff] = 0
    grazing = np.abs(k**2 - squared) <= 64 * np.finfo(float).eps * k**2
    evanescent_rms = 0.0
    if evanescent == "discard":
        discarded = (squared > k**2) | grazing
        evanescent_rms = float(np.linalg.norm(coefficients[discarded]))
        coefficients[discarded] = 0
    if isinstance(field, TransverseElectricField):
        dot = fu * coefficients[..., 0] + fv * coefficients[..., 1]
        longitudinal = np.where(~grazing, -dot / np.where(grazing, 1, kz), 0)
        local = np.concatenate((coefficients, longitudinal[..., None]), axis=-1)
        coefficients = local @ grid.frame.basis.T
    frame = grid.frame
    wavevectors = (
        fu[..., None] * frame.u + fv[..., None] * frame.v + kz[..., None] * frame.normal
    )
    expansion = PlaneWaveExpansion(
        wavevectors.reshape(-1, 3),
        coefficients.reshape(-1, 3),
        grid.points[ny // 2, nx // 2],
        field.wavelength0,
        region,
    )
    return PlaneSpectrum(grid, expansion, roundoff_rms, evanescent_rms)


def _reference_propagation(spectrum, distance):
    expansion = spectrum.expansion
    occupied = np.any(expansion.amplitudes != 0, axis=-1)
    displacement = distance * spectrum.grid.frame.normal
    phase = np.zeros(len(occupied), complex)
    phase[occupied] = np.exp(1j * expansion.wavevectors[occupied] @ displacement)
    shape = spectrum.grid.shape + (3,)
    count = math.prod(spectrum.grid.shape)
    E = count * _centered((expansion.amplitudes * phase[:, None]).reshape(shape), True)
    H = expansion.magnetic_amplitudes * phase[:, None]
    return E, count * _centered(H.reshape(shape), True)


def _plane(nu, nv):
    c, s = np.cos(0.4), np.sin(0.4)
    tilt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ np.array(
        [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    )
    frame = o.geometry.Frame([0.3, -1.2, 7.0], tilt)
    step = WAVELENGTH / 4.3  # the lattice reaches beyond the light circle (n = 1.5)
    return o.sampling.PlaneGrid(
        1.1 + step * np.arange(nu), -0.4 + step * np.arange(nv), frame
    )


def _transverse(grid, seed=3):
    rng = np.random.default_rng(seed)
    U, V = np.meshgrid(grid.u - grid.u.mean(), grid.v - grid.v.mean())
    envelope = np.exp(-(U**2 + V**2) / 0.8**2) * np.exp(2.1j * U - 0.7j * V)
    noise = rng.normal(size=grid.shape + (2,)) + 1j * rng.normal(size=grid.shape + (2,))
    values = envelope[..., None] * np.array([1.0, 0.4j]) + 0.05 * noise
    return TransverseElectricField(grid, WAVELENGTH, values)


GLASS = o.media.HomogeneousRegion("glass", o.media.Dielectric(1.5))


@pytest.mark.parametrize("shape", [(24, 20), (15, 18)], ids=["even", "odd"])
@pytest.mark.parametrize("evanescent", ["retain", "discard"])
@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("kind", ["transverse", "complete"])
def test_trusted_path_matches_the_validating_one(shape, evanescent, direction, kind):
    grid = _plane(*shape)
    field, rtol = _transverse(grid), 64 * np.finfo(float).eps
    if kind == "complete":
        # A complete E of a few lattice modes, transverse to roundoff once
        # the FFT's roundoff bins are dropped.
        sparse = (
            _transverse(grid).values
            * (np.random.default_rng(5).random(grid.shape) < 0.05)[..., None]
        )
        field = angular_spectrum(
            TransverseElectricField(grid, WAVELENGTH, sparse),
            GLASS,
            direction=direction,
        ).propagate(0.0, magnetic=False)
        rtol = 1e-12
    expected = _reference(field, GLASS, direction, evanescent, rtol)
    actual = angular_spectrum(
        field, GLASS, direction=direction, evanescent=evanescent, coefficient_rtol=rtol
    )
    a, b = actual.expansion, expected.expansion
    decaying = np.abs(b.wavevectors.imag).max(axis=-1) > 0
    occupied = np.any(b.amplitudes != 0, axis=-1)
    if evanescent == "discard":
        assert expected.evanescent_removed_rms > 0 and not np.any(occupied & decaying)
    else:
        assert np.any(occupied & decaying)
    scale = np.abs(b.amplitudes).max()
    assert_allclose(a.wavevectors, b.wavevectors, rtol=1e-15, atol=0)
    assert_allclose(a.amplitudes, b.amplitudes, rtol=0, atol=1e-14 * scale)
    assert np.array_equal(a.amplitudes != 0, b.amplitudes != 0)
    assert np.array_equal(a.origin, b.origin)
    if evanescent == "discard":  # a decaying mode has Re k . normal = +-roundoff
        assert actual.direction == expected.direction == direction
    assert_allclose(
        actual.roundoff_removed_rms, expected.roundoff_removed_rms, rtol=1e-12
    )
    assert_allclose(
        actual.evanescent_removed_rms, expected.evanescent_removed_rms, rtol=1e-12
    )
    # What the trusted path built passes the public validation unchanged.
    public = PlaneSpectrum(
        actual.grid,
        PlaneWaveExpansion(
            a.wavevectors, a.amplitudes, a.origin, a.wavelength0, a.region
        ),
        actual.roundoff_removed_rms,
        actual.evanescent_removed_rms,
    )
    for array in (a.wavevectors, a.amplitudes, a.origin):
        assert not array.flags.writeable
    distance = direction * 0.3 * WAVELENGTH
    E, H = _reference_propagation(expected, distance)
    size = np.abs(E).max()
    for spectrum in (actual, public):
        result = spectrum.propagate(distance)
        assert_allclose(result.electric.values, E, rtol=0, atol=1e-13 * size)
        assert_allclose(result.magnetic, H, rtol=0, atol=1e-13 * size / Z0)
        alone = spectrum.propagate(distance, magnetic=False)
        assert np.array_equal(alone.values, result.electric.values)
    assert_allclose(
        actual.propagate(
            distance, model="fresnel" if evanescent == "discard" else "asm"
        ).electric.values,
        public.propagate(
            distance, model="fresnel" if evanescent == "discard" else "asm"
        ).electric.values,
        rtol=0,
        atol=1e-14 * size,
    )


def test_magnetic_field_is_formed_once_when_read_and_survives_copies():
    grid = _plane(16, 12)
    spectrum = angular_spectrum(_transverse(grid), GLASS)
    field = spectrum.propagate(0.2)
    assert isinstance(field, o.fields.ElectromagneticField)
    H = field.magnetic
    assert field.magnetic is H and not H.flags.writeable
    expected = _reference_propagation(spectrum, 0.2)[1]
    assert_allclose(H, expected, rtol=0, atol=1e-13 * np.abs(expected).max())
    fresh = spectrum.propagate(0.2)
    for copy in (pickle.loads(pickle.dumps(fresh)), replace(fresh)):
        assert np.array_equal(copy.magnetic, H)
    assert isinstance(fresh.electric, ElectricField)
    assert_allclose(fresh.poynting(), field.poynting(), rtol=0, atol=0)
