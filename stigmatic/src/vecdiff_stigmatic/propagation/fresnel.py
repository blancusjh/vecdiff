"""Paraxial focal-plane Fourier propagation with physical output coordinates."""

import numpy as np

from .._arrays import positive
from ..fields import ElectricField, ElectromagneticField, TransverseElectricField
from ..fourier.cartesian import fft_modes, frequency_grid
from ..media.medium import Z0
from ..sampling.grids import PlaneGrid


def fresnel_focus(pupil: TransverseElectricField, focal_distance: float, material):
    """Ideal thin-lens Fourier focus of a planar pupil in a homogeneous medium.

    The input is the field immediately before the ideal lens. Its quadratic
    phase cancels the input Fresnel phase. The output includes the quadratic
    observation phase and first-order longitudinal E/H. This paraxial lens
    model is distinct from an actual Cartesian-oval/Debye calculation.
    """

    if not isinstance(pupil, TransverseElectricField):
        raise TypeError("Fresnel focusing requires two local components on a PlaneGrid")

    focal_distance = positive(focal_distance, "focal distance")
    grid = pupil.grid
    wavelength_in_material = pupil.wavelength0 / material.n
    wavenumber = 2 * np.pi / wavelength_in_material
    frequency_u, frequency_v = frequency_grid(grid.shape, grid.spacing)
    frequency_u = np.fft.fftshift(frequency_u)
    frequency_v = np.fft.fftshift(frequency_v)
    output_u = focal_distance * frequency_u[0] / wavenumber
    output_v = focal_distance * frequency_v[:, 0] / wavenumber
    output_grid = PlaneGrid(
        output_u, output_v, grid.frame.translated(focal_distance * grid.frame.normal)
    )
    observation_u, observation_v = np.meshgrid(output_u, output_v, indexing="xy")
    pupil_u, pupil_v = np.meshgrid(grid.u, grid.v, indexing="xy")
    cell_area = np.prod(grid.spacing) * np.prod(grid.shape)
    origin_phase = np.exp(-1j * (frequency_u * grid.u[0] + frequency_v * grid.v[0]))

    def transform(values):
        return (
            np.fft.fftshift(fft_modes(values), axes=(0, 1))
            * cell_area
            * origin_phase[..., None]
        )

    prefactor = np.exp(1j * wavenumber * focal_distance) / (
        1j * wavelength_in_material * focal_distance
    )
    quadratic_phase = np.exp(
        1j * wavenumber * (observation_u**2 + observation_v**2) / (2 * focal_distance)
    )
    field_factor = prefactor * quadratic_phase[..., None]
    transverse = field_factor * transform(pupil.values)
    derivative_u = (
        1j
        * wavenumber
        / focal_distance
        * (
            observation_u[..., None] * transverse
            - field_factor * transform(pupil_u[..., None] * pupil.values)
        )
    )
    derivative_v = (
        1j
        * wavenumber
        / focal_distance
        * (
            observation_v[..., None] * transverse
            - field_factor * transform(pupil_v[..., None] * pupil.values)
        )
    )
    electric_normal = 1j / wavenumber * (derivative_u[..., 0] + derivative_v[..., 1])
    magnetic_normal = (
        -1j
        / (2 * np.pi / pupil.wavelength0 * Z0)
        * (derivative_u[..., 1] - derivative_v[..., 0])
    )
    electric_local = np.concatenate((transverse, electric_normal[..., None]), axis=-1)
    magnetic_local = np.stack(
        (
            -material.n / Z0 * transverse[..., 1],
            material.n / Z0 * transverse[..., 0],
            magnetic_normal,
        ),
        axis=-1,
    )
    electric = electric_local @ grid.frame.basis.T
    magnetic = magnetic_local @ grid.frame.basis.T

    return ElectromagneticField(
        ElectricField(output_grid, pupil.wavelength0, electric), magnetic
    )
