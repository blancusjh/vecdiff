# %% [markdown]
# # Repeated spectral encounters in an interface assembly
# `InterfaceAssembly` specifies interfaces and the media between them.
# `propagate_interfaces` constructs forward and backward per-k Fresnel maps,
# including their global propagation phases, and solves their coherent feedback.
# It does not require a user-written round-trip map or another field representation.
# %%
from pathlib import Path
import sys
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(root))
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
from examples.interface_assembly import run
figure, report = run()
display(figure)
plt.close(figure)
{k: v for k, v in report.items() if "error" in k or "residual" in k or "change" in k}
# %% [markdown]
# The left panel independently checks the constructed map against the planar
# layer recursion. The right is a finite-aperture curved-interface calculation.
# The last panel checks reconstructed boundary fields and exposes the remaining
# physical error. Its resonant response is a property of that explicitly truncated approximate
# model. Refine bandwidth, transverse period, surface quadrature, and wavelength
# spacing before physical interpretation. Converging the feedback equation alone
# does not establish the correct curved-boundary solution. Closed spheres, folded
# paths, and evanescent inter-surface coupling are outside this assembly API.
