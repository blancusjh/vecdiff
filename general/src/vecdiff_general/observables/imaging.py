"""Detector formation from actual source-dependent electromagnetic responses."""

import numpy as np

from vecdiff_general.observables.electromagnetism import coherent_sum, poynting


def detector_image(responses, amplitudes=None, *, coherent=False):
    """Normal detector flux in W/m², from explicitly supplied source responses.

    Coherent sources sum E/H first. Incoherent sources add power with weights
    |amplitude|². No spatial invariance or translated-PSF approximation is
    imposed: each response can come from its own propagation calculation.
    """
    responses = tuple(responses)

    if not responses:
        raise ValueError("at least one source response is required")

    grid = responses[0].grid

    if not hasattr(grid, "normals"):
        raise ValueError("a detector requires an oriented observation grid")

    amplitudes = (
        np.ones(len(responses))
        if amplitudes is None
        else np.asarray(amplitudes, dtype=complex)
    )

    if amplitudes.shape != (len(responses),) or not np.isfinite(amplitudes).all():
        raise ValueError("one finite source amplitude is required per response")

    if any(
        not np.array_equal(response.grid.points, grid.points) for response in responses
    ):
        raise ValueError("source responses must share detector positions")

    if coherent:
        flux = poynting(coherent_sum(responses, amplitudes))
    else:
        flux = sum(
            abs(amplitude) ** 2 * poynting(response)
            for amplitude, response in zip(amplitudes, responses)
        )

    return np.sum(flux * grid.normals, axis=-1)
