# %% [markdown]
# # Per-wavevector refraction and reflection
# The plane interface is diagonal in tangential spatial frequency. The native
# Fresnel module constructs the s/p basis for each incident k, including normal
# incidence. Under total internal reflection, transmitted k is complex and the
# transmitted field is evanescent; it must not be discarded.
# %%
from pathlib import Path
import sys
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(root))
from vecdiff import Medium, Plane, DielectricInterface, plane_wave, interface_transform
result = interface_transform(plane_wave(), DielectricInterface(Plane(), Medium(), Medium(1.5)))
reflected_E, reflected_H = result.reflected.evaluate([[0, 0, 0]])
transmitted_E, transmitted_H = result.transmitted.evaluate([[0, 0, 0]])
# %%
from examples.plane_interface import run
figure, report = run()
display(figure)
{key: report[key] for key in ["max_boundary_residual", "max_flux_error"]}
# %% [markdown]
# The checks use reconstructed E and H, not just Fresnel coefficients. They test
# tangential E/H, normal D/B, and R+T=1 in both index directions and polarizations.
# The p-reflection minimum is Brewster incidence; the high-to-low plateau is TIR.
#
# H is Z0 times the SI magnetic field. Do not insert it unconverted into another
# library's SI Maxwell formula. Power transmittance is not generally |t|² when
# the media or propagation angles differ.
