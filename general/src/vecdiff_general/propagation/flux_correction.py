"""Local power correction followed by reconstruction into Maxwell spectra.

This optional nonlinear closure constrains transmitted power, not the complete
dielectric scattering problem. Every returned field is a Maxwell spectrum.
"""

from dataclasses import dataclass, replace

import numpy as np

from ..fields import ElectricField, ElectromagneticField
from ..fields.spectrum import PlaneSpectrum


@dataclass(frozen=True)
class FluxIterate:
    iteration: int
    spectrum: PlaneSpectrum
    local_factor: np.ndarray
    global_factor: float
    flux_rms: float
    power_error: float
    uncontrolled_fraction: float
    limited_fraction: float


def normal_flux(field):
    """Signed time-averaged E×H* flux through the field grid's normals."""
    return np.sum(field.poynting() * field.electric.grid.normals, axis=-1)


def refine_flux(
    transport,
    spectrum,
    target_flux,
    *,
    iterations=6,
    relaxation=0.5,
    max_log_step=0.2,
    flux_floor=1e-6,
    inverse_tolerance=1e-9,
    max_inverse_iterations=200,
    monotone=False,
    flux_tolerance=1e-6,
):
    """Yield baseline and trial spectra with locally corrected normal power.

    At iterate j, multiply the reconstructed surface E by
    a_j = (target_flux / actual_flux)**(relaxation/2), then apply Q^-1.
    Cap each log-amplitude step and leave dark or nonpositive-current-flux
    samples unchanged; their illuminated fraction is reported. Finally scale
    the *spectrum* by a constant to match integrated target power. E and H in
    each returned spectrum satisfy the homogeneous Maxwell equations.

    target_flux must use transport.grid's oriented normals and SI units. For
    an interface it is the normal flux of incident+reflected total fields,
    including interference, not incident flux alone. Only nonnegative target
    flux is supported in this forward-wave experiment. Radiation leaking into
    target-dark parts of the finite chart is included in both reported errors.

    With monotone=True, first normalize the starting spectrum to the target
    power, then stop before accepting a step that increases the flux RMS.
    Stop also when flux_tolerance is met. This prevents iteration drift; a
    stopped iteration is not necessarily a solution of the local constraint.

    This is an alternating correction/projection, not a proof of convergence.
    The reported local_factor is the current step, not a cumulative mapping:
    Q^-1 is nonlocal. Decreasing this objective establishes flux improvement
    only. Evaluate boundary jumps, independent field error, padding and grid
    refinement before interpreting an accepted step as improved accuracy.
    """
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if not np.isfinite(relaxation) or not 0 < relaxation <= 1:
        raise ValueError("relaxation must lie in (0, 1]")
    if not np.isfinite(max_log_step) or not 0 < max_log_step <= 1:
        raise ValueError("max_log_step must lie in (0, 1]")
    if not np.isfinite(flux_floor) or not 0 < flux_floor < 1:
        raise ValueError("flux_floor must lie in (0, 1)")
    if not np.isfinite(flux_tolerance) or not 0 < flux_tolerance < 1:
        raise ValueError("flux_tolerance must lie in (0, 1)")
    target = np.asarray(target_flux, dtype=float)
    weights = transport.grid.weights
    if target.shape != transport.grid.shape or not np.isfinite(target).all():
        raise ValueError("target flux must be finite and match the surface grid")
    peak = float(np.max(target))
    if peak <= 0 or np.any(target < -1e-12 * peak):
        raise ValueError(
            "the forward correction needs nonnegative, nonzero target flux"
        )
    target = np.maximum(target, 0)
    target_power = float(np.sum(weights * target))
    norm = float(np.sum(weights * target**2))
    illuminated = target > flux_floor * peak
    field = transport.forward(spectrum)  # Also validates spectrum compatibility.

    def result(iteration, local, global_factor, uncontrolled=0, limited=0):
        actual = normal_flux(field)
        return FluxIterate(
            iteration,
            spectrum,
            local,
            float(global_factor),
            float(np.sqrt(np.sum(weights * (actual - target) ** 2) / norm)),
            float(abs(np.sum(weights * actual) / target_power - 1)),
            float(uncontrolled),
            float(limited),
        )

    initial_scale = 1.0
    if monotone:
        power = field.power()
        if not np.isfinite(power) or power <= 0:
            raise ValueError("reconstructed field has nonpositive or nonfinite power")
        initial_scale = np.sqrt(target_power / power)
        spectrum = replace(
            spectrum,
            expansion=replace(
                spectrum.expansion,
                amplitudes=spectrum.expansion.amplitudes * initial_scale,
            ),
        )
        field = ElectromagneticField(
            ElectricField(
                transport.grid,
                transport.wavelength0,
                field.electric.values * initial_scale,
            ),
            field.magnetic * initial_scale,
        )
    previous = result(0, np.ones_like(target), initial_scale)
    yield previous
    for iteration in range(1, int(iterations) + 1):
        if monotone and previous.flux_rms <= flux_tolerance:
            break
        actual = normal_flux(field)
        controlled = illuminated & (actual > flux_floor * peak)
        increment = np.zeros_like(target)
        increment[controlled] = (
            relaxation / 2 * np.log(target[controlled] / actual[controlled])
        )
        limited = np.count_nonzero(abs(increment) > max_log_step) / illuminated.sum()
        local = np.exp(np.clip(increment, -max_log_step, max_log_step))
        uncontrolled = np.count_nonzero(illuminated & ~controlled) / illuminated.sum()
        trial = ElectromagneticField(
            ElectricField(
                transport.grid,
                transport.wavelength0,
                field.electric.values * local[..., None],
            ),
            field.magnetic * local[..., None],
        )
        spectrum, _ = transport.inverse(
            trial,
            tolerance=inverse_tolerance,
            max_iterations=max_inverse_iterations,
        )
        reconstructed = transport.forward(spectrum)
        power = float(np.sum(weights * normal_flux(reconstructed)))
        if not np.isfinite(power) or power <= 0:
            raise ValueError("reconstructed field has nonpositive or nonfinite power")
        global_factor = np.sqrt(target_power / power)
        spectrum = replace(
            spectrum,
            expansion=replace(
                spectrum.expansion,
                amplitudes=spectrum.expansion.amplitudes * global_factor,
            ),
        )
        # A constant factor commutes with Q, avoiding another synthesis.
        field = ElectromagneticField(
            ElectricField(
                transport.grid,
                transport.wavelength0,
                reconstructed.electric.values * global_factor,
            ),
            reconstructed.magnetic * global_factor,
        )
        candidate = result(iteration, local, global_factor, uncontrolled, limited)
        if monotone and candidate.flux_rms >= previous.flux_rms:
            break
        yield candidate
        previous = candidate
