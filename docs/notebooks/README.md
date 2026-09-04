# New-API scientific notebooks

The six notebooks are paired with readable percent-format Python sources.
Edit the `.py` source, then run `python scripts/notebooks.py` to regenerate
the `.ipynb`. Stable cell IDs make changes reviewable. Committed notebooks have
no stale outputs; the execution runner stores fresh outputs under
`build/notebooks/` and fails on any cell exception.

| Notebook | Learning/validation goal |
| --- | --- |
| [01 Electric field](01_electric_field.ipynb) | Complete Ez and understand propagation/flux checks |
| [02 Fresnel boundaries](02_fresnel_boundaries.ipynb) | Reconstruct reflected/refracted fields and check Maxwell conditions |
| [03 Coherent resonances](03_coherent_resonances.ipynb) | Round trips, Airy amplitudes, standing waves, and evanescent gaps |
| [04 Curved-interface limits](04_curved_interface_limits.ipynb) | Separate quadrature accuracy from physical approximation; introduce freeform geometry |
| [05 Vector focusing reference](05_vector_focusing_reference.ipynb) | Polarization, vortices, and longitudinal fields without mixing reference theory into core physics |
| [06 Closed-sphere resonance](06_closed_sphere_resonance.ipynb) | Self-consistent boundary matching, Mie comparison, and convergence limits |

Install `.[notebooks,validation,nufft]` in this checkout and use that Python
kernel. Each notebook locates the repository when started from a subdirectory.
`python scripts/notebooks.py --check --execute` checks synchronization and runs
every cell in order. CI runs this command explicitly; a source-only check is
not reported as a successful Jupyter execution.
