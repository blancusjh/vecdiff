# %% [markdown]
# # Macroscopic stigmatic field: a bounded local spectrum
# A 10 mm curvature-radius conic at 193.368 nm. All amplitudes originate in
# per-k Fresnel boundary traces. The acceleration expands the existing radiation
# locally about the observation centre; it does not supply an ideal-pupil field.
# The kernel error, quadrature error and unresolved dielectric-boundary error
# are distinct. See `docs/macroscopic_fields.md` for the derivation and bound.
# %%
from pathlib import Path
import sys
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').exists())
sys.path.insert(0, str(root))
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from examples.macroscopic_focus import run
figure, report = run()
display(figure)
plt.close(figure)
report
# %% [markdown]
# ## Independent scale and quadrature controls
# These committed measurements compare against radiation by the same Fresnel
# currents, not a globally solved dielectric boundary. Rerun with
# `python -m benchmarks.macroscopic_focus`. A small focal window is essential.
# %%
import json
import numpy as np
results = json.loads((root/'benchmarks/results/macroscopic_focus.json').read_text())
cases = results['cases']
fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog([r['radius_over_wavelength'] for r in cases], [r['relative_EH_kernel_error'] for r in cases], 'o-', label='Local spectrum vs full radiation')
ax.loglog([r['radius_over_wavelength'] for r in cases], [r['combined_quadrature_change'] for r in cases], 's-', label='Surface quadrature change')
ax.set(xlabel='Curvature radius / vacuum wavelength', ylabel='Relative complex E/H norm', title='Separate approximation and quadrature controls')
ax.legend(); ax.grid(alpha=.2)
display(fig); plt.close(fig)
cases
# %% [markdown]
# ## A real DUV prescription, imported without dropping encounters
# This demonstrates IO of all 48 encounters, including a folded branch and
# the stop. The present ordered dielectric propagator cannot execute this full
# prescription; parsing it does not supply mirror coating amplitudes or phases.
# %%
from vecdiff.IO import read_prescription
system = read_prescription(root/'examples/data/US7557996.csv')
print(f'{len(system.encounters)} encounters; wavelength={system.wavelength*1e6:.3f} nm; image z={system.image_z:.6f} mm')
print('Mirrors:', [e.number for e in system.encounters if e.interaction == 'reflect'])
fig,ax=plt.subplots(figsize=(12,4))
for encounter in system.encounters:
    surface=encounter.surface; radius=encounter.semidiameter
    r=np.linspace(-radius,radius,201)
    sag=surface.sag(abs(r)) if hasattr(surface,'sag') else r*0
    ax.plot(surface.frame.origin[2]+sag,r,color={'reflect':'crimson','stop':'green','refract':'steelblue'}[encounter.interaction],lw=1)
ax.set(xlabel='Global z (mm)',ylabel='Meridional radius (mm)',title='Imported DUV surface profiles: mirrors red, stop green (no ray trace)')
ax.set_aspect('equal'); ax.grid(alpha=.15)
display(fig);plt.close(fig)
