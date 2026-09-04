"""Coherent round trips recover the complex Fabry–Pérot response."""
import numpy as np
import matplotlib.pyplot as plt
from vecdiff import Medium, LayerStack, plane_wave, propagate_layers, coherent_feedback
from ._report import main


def run():
    """Validate a lossless high-index slab against an independent Airy expression."""
    n, thickness = 4., 1.
    wavelengths = np.linspace(.85, 1.15, 401)
    stack = LayerStack((Medium(), Medium(n), Medium()), (thickness,))
    r = (n-1)/(n+1)
    transmission, analytical = [], []
    for wavelength in wavelengths:
        field = propagate_layers(plane_wave(wavelength=wavelength), stack)
        transmission.append(field.transmission[0, 1])
        phase = np.exp(2j*np.pi*n*thickness/wavelength)
        analytical.append((1-r*r)*phase/(1-r*r*phase*phase))
    error = float(np.max(abs(np.array(transmission)-analytical)))
    phase = np.exp(2j*np.pi*n*thickness)
    feedback = coherent_feedback(np.array([2/(1+n)]), lambda f: r*r*phase*phase*f, rtol=1e-12)
    solved = propagate_layers(plane_wave(), stack)
    feedback_error = float(abs(feedback.state[0]-solved.forward[1][0, 1]))
    assert error < 1e-12 and feedback_error < 1e-11
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    axes[0].plot(wavelengths, abs(np.array(transmission))**2, label="All round trips")
    axes[0].plot(wavelengths[::10], abs(np.array(analytical)[::10])**2, ".", label="Airy reference")
    axes[0].axhline((1-r*r)**2, ls=":", color="gray", label="One transmitted pass")
    axes[0].set(xlabel="Vacuum wavelength / slab thickness", ylabel="Power transmittance")
    axes[0].legend(fontsize=8)
    axes[1].semilogy(np.arange(1, len(feedback.residual_history)+1), feedback.residual_history)
    axes[1].set(xlabel="Round-trip iteration", ylabel="Relative fixed-point residual")
    z = np.linspace(0, thickness, 301)
    e, _ = solved.evaluate(np.column_stack((0*z, 0*z, z)), region=1)
    axes[2].plot(z, np.sum(abs(e)**2, axis=-1))
    axes[2].set(xlabel="z / wavelength", ylabel="Internal |E|²", title="Standing wave at resonance")
    return fig, dict(parameters=dict(index=n, thickness=thickness, wavelength_range=[.85, 1.15], count=401),
                     assumptions="Infinite parallel lossless layers; complex amplitudes, not intensities, are summed.",
                     max_complex_transmission_error=error, explicit_round_trip_error=feedback_error,
                     round_trip_iterations=feedback.iterations, residual_history=list(feedback.residual_history))


if __name__ == "__main__":
    main("cavity_resonance", run)
