# %% [markdown]
# # A closed sphere needs a self-consistent field
# The production closed-interface method fits E/H tangential continuity using
# auxiliary Maxwell dipoles. Its complex linear solve includes internal
# interactions implicitly. It does not call Mie or iterate the local Fresnel
# approximation. Mie is used here only as an independent reference.
#
# This is an experimental dense numerical method. Expect about a minute for the
# default scan on a laptop; memory and time increase rapidly with source count.
# %%
from pathlib import Path
import sys
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(root))
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
from examples.sphere_resonance import run
figure, report = run()
display(figure)
plt.close(figure)
report["max_bulk_error"], report["worst_case_refinement"]
# %% [markdown]
# Four distinct checks matter: fit residual, all four held-out Maxwell boundary
# jumps, E/H error against Mie, and refinement of the least accurate scan point.
# A small fit residual alone can miss a resonant mode. No phase or amplitude is
# fitted to the reference. The native wavelength grid is sparse: narrow peaks
# require further wavelength refinement, even when the linear solve converges.
#
# Auxiliary sources must lie outside the region where their expansion is used.
# Generic freeform containment is not inferred automatically. Change source
# offsets as well as node counts before using a new geometry or high-Q regime.
