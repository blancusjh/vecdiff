# %% [markdown]
# # Many reflections are complex-amplitude feedback
# For a slab, the internal forward field satisfies F=b+RF, with
# R=r10 r12 exp(2 i kz d). Summing intensities loses the interference that makes
# a resonance. The iterative feedback API reports the actual equation residual;
# the layer API composes scattering amplitudes to sum all orders directly.
# %%
from pathlib import Path
import sys
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(root))
from vecdiff import Medium, LayerStack, plane_wave, propagate_layers
stack = LayerStack((Medium(), Medium(4), Medium()), (1.,))
solution = propagate_layers(plane_wave(), stack)
internal_E, internal_H = solution.evaluate([[0, 0, .5]], region=1)
# %%
from examples.cavity_resonance import run
figure, report = run()
display(figure)
{key: report[key] for key in ["max_complex_transmission_error", "explicit_round_trip_error", "round_trip_iterations"]}
# %% [markdown]
# ## Evanescent coupling is also a multiple-interface effect
# A finite low-index gap can transmit above the single-interface critical angle.
# Propagation factors in the scattering recursion only decay; a thick gap must
# not overflow because an algebraic transfer matrix contains growing factors.
# %%
from examples.frustrated_tir import run as gap_run
figure, gap_report = gap_run()
display(figure)
gap_report["limiting_values"], gap_report["max_flux_error"]
# %% [markdown]
# These are infinite parallel layers, not a general three-dimensional resonator.
# For high-Q structures, refine wavelength spacing as well as solver tolerances.
# A converged feedback equation does not certify an approximate round-trip map.
