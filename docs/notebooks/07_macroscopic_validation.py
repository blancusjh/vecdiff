# %% [markdown]
# # Macroscopic spectral-method validation
# These are recorded runs of the main per-k Fresnel method at curvature radii
# 50, 100, and 200 vacuum wavelengths. The aperture radius is 0.6R and n=1→1.5.
# The azimuthal integral is evaluated analytically through Fourier–Bessel
# harmonics; no local-ray substitution or alternative boundary solver is used.
# %%
from pathlib import Path
import sys
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(root))
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
from benchmarks.plot_macroscopic import run
figure, report = run()
display(figure)
plt.close(figure)
[{k: r[k] for k in ("radius_over_wavelength", "reconstructed_boundary", "power_balance_error", "seconds", "process_peak_rss_mib")}
 for r in report["cases"] if r["refinement"] == "combined"]
# %% [markdown]
# Boundary residuals are not intensity/image errors. They include the local
# curved-interface approximation, the hard aperture, and omitted evanescent
# fields. The spectral continuation onto source surfaces is a diagnostic, not
# an exact singular boundary evaluation. Small global power imbalance does not
# imply small local boundary errors. This notebook plots committed measurements;
# reproduce them with `python -m benchmarks.macroscopic` from the checkout root.
