# %% [markdown]
# # A converged integral is not a solved dielectric boundary
# An open curved interface uses the local tangent-plane Fresnel law for each
# incident spectral component. This defines equivalent currents. Their radiated
# fields satisfy homogeneous Maxwell equations away from sources, but the
# prescribed trace need not equal the reconstructed trace on the curved surface.
# %%
from pathlib import Path
import sys
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(root))
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
from examples.curved_interface import run
figure, report = run()
display(figure)
plt.close(figure)
report
# %% [markdown]
# The first panel checks quadrature convergence for a fixed physical aperture.
# The second separates the locally imposed conditions from a propagating-only
# reconstruction diagnostic. The latter includes finite-aperture effects and
# omitted evanescent content; it is not a pure measure of curvature error.
#
# Near the surface, direct Green integration needs singular/near-singular
# quadrature. Evaluating at a source node is not a valid boundary-limit test.
# The size sweep in `benchmarks.validate_physics` extends this diagnostic from
# sub-wavelength caps to R=50 vacuum wavelengths.
# %% [markdown]
# ## Non-axisymmetric surfaces are physical geometry, not a new field type
# %%
import numpy as np
from vecdiff import FreeformSurface, sample_surface
surface = FreeformSurface(lambda x, y: x*x/40+y*y/60,
                         lambda x, y: (x/20, y/30))
sampling = sample_surface(surface, (-2, 2), (-2, 2), 20, 20, periodic_v=False)
assert np.allclose(np.linalg.norm(sampling.normals, axis=-1), 1)
sampling.points.shape
# %% [markdown]
# This freeform can be passed to the same interface transformation. Its Cartesian
# chart requires Gauss quadrature on both coordinates, not a periodic polar rule.
