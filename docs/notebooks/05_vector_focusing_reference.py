# %% [markdown]
# # Polarization and longitudinal focusing: a separate reference theory
# This notebook intentionally imports `references.richards_wolf`. It models an
# ideal sine-condition objective, not refraction through a specific dielectric
# lens. The reference uses the core ElectricSpectrum abstraction; the core never
# imports the reference theory.
# %%
from pathlib import Path
import sys
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(root))
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
from examples.vector_focus import run
figure, report = run()
display(figure)
plt.close(figure)
report
# %% [markdown]
# All pupil choices have the same magnitude and incident pupil power. No curve
# is normalized to its own peak. Radial polarization produces an axial electric
# component; azimuthal polarization has zero Ez in this ideal model. A vortex
# changes phase, not pupil power. The reported errors compare two quadratures.
#
# These electric norms do not by themselves establish resolution, lithographic
# performance, or an experimental replication. Such claims require a specified
# object, coherence model, detector response, and an appropriate reference.
