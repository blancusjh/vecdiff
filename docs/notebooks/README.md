# New-API scientific notebooks

The seven notebooks are paired with readable percent-format Python sources.
Edit the `.py` source, then run `python scripts/notebooks.py --execute` to
regenerate and execute the `.ipynb`. Commit the executed notebooks with their
figures and numerical outputs. Stable cell IDs make source changes reviewable.
The runner saves both the source-tree notebook and an artifact copy under
`build/notebooks/`; it fails on cell errors or missing figures.

`python scripts/notebooks.py --check` verifies matching sources and completed
outputs without stripping results. Changed sources require execution; a normal
check never clears or rewrites existing outputs. CI independently re-executes
the suite and publishes its executed copies.

| Notebook | Learning/validation goal |
| --- | --- |
| [01 Electric field](01_electric_field.ipynb) | Complete Ez and understand propagation/flux checks |
| [02 Fresnel boundaries](02_fresnel_boundaries.ipynb) | Reconstruct reflected/refracted fields and check Maxwell conditions |
| [03 Coherent resonances](03_coherent_resonances.ipynb) | Round trips, Airy amplitudes, standing waves, and evanescent gaps |
| [04 Curved-interface limits](04_curved_interface_limits.ipynb) | Separate quadrature accuracy from physical approximation; introduce freeform geometry |
| [05 Vector focusing reference](05_vector_focusing_reference.ipynb) | Polarization, vortices, and longitudinal fields without mixing reference theory into core physics |
| [06 Spectral assembly](06_spectral_assembly.ipynb) | Construct repeated encounters and distinguish feedback convergence from curved-boundary accuracy |
| [07 Macroscopic validation](07_macroscopic_validation.ipynb) | Inspect recorded R=50–200 wavelength tests, power balance, and independent numerical controls |

Install `.[notebooks,validation,nufft]` in this checkout and use that Python
kernel. Each notebook locates the repository when started from a subdirectory.
`python scripts/notebooks.py --check --execute` checks synchronization and runs
every cell in order and requires embedded PNG figures in every executed notebook.
The notebooks explicitly enable inline rendering, including in headless CI.
CI runs this command explicitly; a source-only check is
not reported as a successful Jupyter execution.
