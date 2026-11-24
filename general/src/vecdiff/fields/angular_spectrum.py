"""Homogeneous vector angular spectra and branch-dependent Ez completion.

The material dispersion and longitudinal completion belong to propagation.
Fourier functions only transform arrays and construct frequency coordinates.
"""

import math

import numpy as np

from vecdiff._backend import bd
from vecdiff.fields.electromagnetic_field import (
    ElectricField,
    TransverseElectricField,
)
from vecdiff.fields.spectrum import (
    PlaneSpectrum,
    PlaneWaveExpansion,
    _check_transverse,
)
from vecdiff.fourier.cartesian import _centered, _frequency_axes
from vecdiff.media.medium import HomogeneousRegion
from vecdiff.sampling.grids import PlaneGrid


def _on_light_circle(transverse_squared, wavenumber):
    """Modes on the light circle to roundoff in k² - kappa².

    k_z = sqrt(k² - kappa²) of a lattice mode exactly on the circle is about
    1e-8 k, not zero, so the test is on its square. In an absorbing medium
    (complex k) no mode is on it: Im k² > 0.
    """
    return bd.abs(wavenumber**2 - transverse_squared) <= 64 * np.finfo(float).eps * (
        abs(wavenumber) ** 2
    )


def angular_spectrum(
    field,
    region,
    *,
    direction=1,
    evanescent="retain",
    coefficient_rtol=64 * np.finfo(float).eps,
):
    """Represent a plane field using homogeneous vector propagation modes.

    Two-component inputs prescribe local u/v components. Complete electric
    inputs are validated without projection. The selected material and branch
    determine k_normal and, when needed, the longitudinal electric component.

    'discard' removes modes outside the propagating disk and on its rim;
    'retain' keeps the decaying branch. In an absorbing region (complex
    index) every mode decays, k_z = sqrt(n² k0² - kappa²) with Im k_z > 0,
    and there is no disk: only 'retain' is taken. Coefficients below coefficient_rtol times the largest mode
    are zeroed before completion. Set this threshold to zero for strict
    retention. Both removed RMS amplitudes are reported in V/m.
    """

    if not isinstance(field, (ElectricField, TransverseElectricField)):
        raise TypeError("expected complete or transverse electric field")

    if isinstance(field, ElectricField) and not bd.all(field.valid):
        raise ValueError("an incident field cannot contain uncomputed samples")

    if not isinstance(field.grid, PlaneGrid):
        raise TypeError("FFT angular spectrum requires a PlaneGrid")

    if not isinstance(region, HomogeneousRegion):
        raise TypeError("region must be a HomogeneousRegion")

    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")

    if evanescent not in ("retain", "discard"):
        raise ValueError("evanescent must be 'retain' or 'discard'")

    if evanescent == "discard" and region.material.absorbing:
        from vecdiff.media.absorption import NOTE, AbsorbingMediumError, describe

        raise AbsorbingMediumError(
            f"in an absorbing medium, {describe(region)}, every plane wave decays "
            "(Im k_z > 0) and there is no propagating disk to keep: use "
            f"evanescent='retain' (see {NOTE})"
        )

    if not bd.isfinite(coefficient_rtol) or not 0 <= coefficient_rtol < 1:
        raise ValueError("coefficient_rtol must lie in [0, 1)")

    # The mathematical transform takes arrays and lattice parameters only.
    # Frequencies stay one-dimensional axes, broadcast over (v, u).
    grid = field.grid
    fx, fy = _frequency_axes(grid.coordinates)
    frequency_u = (2 * bd.pi * fx)[None, :]
    frequency_v = (2 * bd.pi * fy)[:, None]

    # The modal sum uses Fourier-series amplitudes, so divide the DFT by N.
    # The field constructors guarantee finite values.
    sample_count = math.prod(grid.shape)
    coefficients = _centered(field.values, scale=1 / sample_count)

    # The optical dispersion and chosen branch determine the normal frequency.
    wavenumber = 2 * bd.pi * region.material.n / field.wavelength0
    transverse_squared = frequency_u**2 + frequency_v**2
    normal_wavenumber = direction * bd.sqrt(
        (wavenumber**2 - transverse_squared).astype(complex)
    )

    # |c|² per mode, one component at a time (no (modes, 3) temporaries).
    power = sum(
        coefficients[..., j].real ** 2 + coefficients[..., j].imag ** 2
        for j in range(coefficients.shape[-1])
    )
    magnitude = bd.sqrt(power)
    roundoff = magnitude < coefficient_rtol * bd.max(magnitude)
    del magnitude
    roundoff_removed = float(bd.sqrt(bd.sum(power[roundoff])))
    removed = roundoff

    evanescent_removed = 0.0

    if evanescent == "discard":
        # A mode exactly on the light circle carries no power and has no
        # transverse completion; it leaves with the decaying modes.
        grazing = _on_light_circle(transverse_squared, wavenumber)
        discarded = (transverse_squared > wavenumber**2) | grazing
        evanescent_removed = float(bd.sqrt(bd.sum(power[discarded & ~roundoff])))
        removed = roundoff | discarded

    kept = ~removed
    bd.multiply(coefficients, kept[..., None], out=coefficients)
    # A kept mode whose |c|² is a normal float is nonzero; the others are
    # looked at below (none unless |c| leaves (1e-145, 1e145)).
    ordinary = (power > 1e-290) & (power < 1e290)

    # Complete only explicitly transverse inputs, in the plane's local basis.
    if isinstance(field, TransverseElectricField):
        grazing = _on_light_circle(transverse_squared, wavenumber)
        candidates = grazing & ~removed

        if bd.any(candidates) and bd.any(coefficients[candidates] != 0):
            raise ValueError(
                "populated grazing modes require a full vector polarization basis"
            )

        # Ez = -(k_u E_u + k_v E_v) / k_normal off the light circle, else 0;
        # then the global components, local (u, v, normal) times the basis
        # transposed. In place, one component at a time.
        cu, cv = coefficients[..., 0], coefficients[..., 1]
        scratch = bd.empty(grid.shape, complex)
        longitudinal = bd.multiply(frequency_u, cu)
        longitudinal += bd.multiply(frequency_v, cv, out=scratch)
        longitudinal /= bd.where(grazing, 1, -normal_wavenumber)
        longitudinal[grazing] = 0

        basis = grid.frame.basis
        coefficients = bd.empty(grid.shape + (3,), complex)
        for i in range(3):
            component = bd.multiply(cu, basis[i, 0], out=coefficients[..., i])
            component += bd.multiply(cv, basis[i, 1], out=scratch)
            component += bd.multiply(longitudinal, basis[i, 2], out=scratch)
        del cu, cv, longitudinal, scratch

    # Cartesian modes retain the original physical phase origin.
    frame = grid.frame
    wavevectors = bd.empty(grid.shape + (3,), complex)
    for i in range(3):
        component = bd.multiply(
            normal_wavenumber, frame.normal[i], out=wavevectors[..., i]
        )
        component += frequency_u * frame.u[i] + frequency_v * frame.v[i]
    del normal_wavenumber

    # Every piece is validated above or by construction: the lattice is the
    # grid's, the amplitudes transverse (completed, or checked here for a
    # complete input), and the origin the grid's centre.
    unusual = kept & ~ordinary
    occupied = kept & ordinary
    if bd.any(unusual):
        occupied[unusual] = bd.any(coefficients[unusual] != 0, axis=-1)
    if isinstance(field, ElectricField):
        # PlaneWaveExpansion's test, |k . a| <= 1e-10 |k| |a|, with |k|²
        # from the lattice and |a|² from |c|²; the unusual modes take its
        # underflow-safe form.
        k, c = wavevectors, coefficients
        error = bd.abs(
            k[..., 0] * c[..., 0] + k[..., 1] * c[..., 1] + k[..., 2] * c[..., 2]
        )
        size = transverse_squared + bd.abs(wavenumber**2 - transverse_squared)
        size = bd.sqrt(size * bd.where(ordinary, power, 0))
        if bd.any((error > 1e-10 * size) & ~unusual):
            raise ValueError("electric modal amplitudes are not transverse")
        _check_transverse(k[unusual], c[unusual])
    del transverse_squared, power

    expansion = PlaneWaveExpansion._trusted(
        wavevectors.reshape(-1, 3),
        coefficients.reshape(-1, 3),
        grid.center,
        field.wavelength0,
        region,
        occupied=occupied.reshape(-1),
    )

    return PlaneSpectrum._trusted(grid, expansion, roundoff_removed, evanescent_removed)
