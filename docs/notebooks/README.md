# Optical application notebooks

Nine executed studies progress from a specified beam to image formation. Each
contains the physical problem, units and phasor convention, explicit API calls,
field maps, measured observables and checks of the numerical assumptions.
Figures remain embedded in the `.ipynb` files. The paired `.py` files are editable
cell sources, not separate implementations of the experiments.

| Study | Actual result | Physical scope |
| --- | --- | --- |
| [01 Gaussian beam](01_electric_field.ipynb) | Waist and output vector fields, meridional propagation, width, polarization, power and sampling checks | Main homogeneous Maxwell spectrum |
| [02 Planar reflection and refraction](02_fresnel_boundaries.ipynb) | Hot-colormap incident/reflected/transmitted maps, Brewster and critical angles, TIR, evanescent penetration and four boundary conditions | Main exact planar per-k Fresnel map, including finite beams |
| [03 Stigmatic refraction](03_stigmatic_refraction.ipynb) | Actual conic geometry, equal optical path, focal components, meridional field, spot widths, encircled normal flux and polarization | Main curved-surface physical optics; controlled local radiation expansion |
| [04 Sphere resonances](04_sphere_resonances.ipynb) | Wavelength scan, resonant/off-resonant E/H maps with sphere geometry, internal field enhancement, Mie convergence and main-method error maps | Independent Mie resonance reference versus the incomplete main single-encounter model; closed-sphere feedback remains pending |
| [05 Lithographic image formation](05_lithographic_image.ipynb) | Circuit mask, complex vector image, polarization, partial coherence, defocus and image convergence | Main Fresnel-current point response under an explicit local isoplanatic imaging assumption |
| [06 DUV projection reference](06_duv_projection_reference.ipynb) | Patent-system geometry and stored wavefront, vector PSF, meridional field, circuit image, TE/TM resolution and source/pixel checks | Separate pupil reference at one field point; not main-method propagation through 48 surfaces |

| [07 Curved boundary verification](07_curved_boundary_verification.ipynb) | Actual E-field maps, two-sided full-Green limits, quadrature/offset convergence and aperture controls | Pointwise physical acceptance and failure of the main finite-aperture model |
| [08 Field-dependent macroscopic optics](08_field_dependent_optics.ipynb) | Actual refraction/reflection, off-axis spots, meridional and polarization maps, coherent/incoherent three-source images | Direct single-surface response per source; no shift invariance; full instrument transport remains pending |

| [09 Macroscopic system transport](09_macroscopic_system_transport.ipynb) | Both curved faces, finite-conjugate recovery, displaced sources, image fields and measured speed/accuracy | Explicit high-frequency phase transport; preserves vector Fresnel laws and final diffraction |

The sphere and DUV reference studies are deliberately labeled. They provide
concrete target fields and useful comparisons; they do not turn the still-missing
closed-sphere and full-prescription propagation into completed capabilities.
Planar layers, generic feedback, curved-boundary failures and scale benchmarks
remain runnable in `examples/` and `benchmarks/`; they no longer occupy shallow
notebooks that merely print a benchmark dictionary.

Install `python -m pip install -e '.[notebooks,validation,nufft]'` in this checkout.
Run `python scripts/notebooks.py --execute` from the repository root to rebuild
and execute all nine notebooks. The command writes both the committed notebooks
and execution artifacts under `build/notebooks/`; plotting cells also save PNG
review figures under `build/notebook-review/`.

`python scripts/notebooks.py --check` verifies source synchronization, completed
cells, embedded figures and a fingerprint of numerical code and input data.
It never clears outputs. CI independently executes all notebooks and requires
fresh committed outputs; download and commit the executed artifact when inputs
change. The original README animations are preserved separately.
