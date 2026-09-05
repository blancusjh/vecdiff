# Scientific validation and acceptance limits

The suite separates four questions: does the algebra satisfy Maxwell's laws,
does a discretization converge, does the physical approximation apply, and
does an independent reference agree? One successful check cannot replace the
others. Finite tests cannot prove correctness in every geometry or resonance.

The per-k spectral interface transformation is the main method. The removed
auxiliary-source solver's results do not validate it. The [roadmap](future_implementations.md)
records the outstanding work on resonances and macroscopic optics.

## Conventions and boundary checks

**Pending:** general resonant behavior through coherent repeated encounters of
the main spectral method; verification of macroscopic elements and complete
systems; and any corrective implementation or additional macroscopic support
that verification shows to be necessary. Planar-layer tests and auxiliary-source
Mie comparisons do not close these items. See the [pending-work checklist](future_implementations.md#pending-work).

Time dependence is `exp(-i omega t)`, vacuum wavelength is explicit, and returned
H is `Z0 H_SI`. Media are homogeneous, isotropic, nonmagnetic, and lossless.
The source-free dielectric jumps are tangential E, tangential H, normal D, and
normal B. Tests reconstruct both output fields before checking them; they also
check dispersion, transversality, and normal Poynting-flux balance.

Boundary residuals are RMS values. For refinement studies the fixed scales are
incident electric amplitude and normalized magnetic amplitude. Normal D is
additionally divided by the larger relative permittivity. The closed-sphere
benchmark uses quadrature weights and the same held-out points at every order.

## Planes and coherent layers

Plane tests cover normal, oblique, Brewster, critical, near-grazing, circular,
equal-index, and total-internal-reflection regimes, including a tilted and
translated interface. The recorded 36-case plane sweep had maximum boundary
and relative flux errors of `1.16e-15` and `1.51e-15`.

Layer tests check all four conditions at every boundary in three- and
four-region stacks, both s/p polarizations, both index directions, and rotated
frames. Independent Airy complex amplitudes agree within `6e-16` in the cavity
example. Explicit coherent round trips agree with the all-orders slab field
within `2.4e-13`. Frustrated-TIR flux errors are below `6e-16`; a 1000-wavelength
evanescent gap checks overflow safety. Exact critical layers are unsupported
and rejected, not silently regularized. Generic feedback raises on failure.

## Curved open surfaces: an approximation, even after numerical convergence

Local Fresnel boundary traces satisfy the imposed conditions to roundoff by
construction. The Green propagator is independently compared with Stratton–Chu,
and E/H curl identities are tested away from sources. These checks do **not**
establish that the prescribed currents solve the curved dielectric boundary.

The cap benchmark evaluates reconstructed, propagating-only spectra at physical
surface points. This is a continuation diagnostic, not a singular on-surface
Green integral. Its error includes local physical optics, hard-aperture effects,
and omitted evanescent content. It must not be labeled pure curvature error.

With aperture `0.6 R`, observations through `0.3 R`, wavelength 1, and indices
1/1.5, the recorded fine-grid results are:

| R / wavelength | Et jump | Ht jump | Dn jump | Bn jump | Quadrature field change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | .199 | .325 | .159 | .275 | 6.0e-13 |
| 1 | .0982 | .305 | .163 | .154 | 1.1e-11 |
| 2 | .104 | .139 | .0746 | .114 | 7.0e-10 |
| 5 | .130 | .169 | .0910 | .145 | 1.8e-8 |
| 10 | .0648 | .115 | .0602 | .0814 | 4.1e-7 |
| 50 | .0313 | .0492 | .0260 | .0396 | 3.5e-6 |

These nonzero jumps remain after quadrature convergence. Macroscopic scale
does not automatically confer exact boundary accuracy.

## Closed objects and resonances

The original one-encounter sphere diagnostic remains a labeled failure case
in `benchmarks.validate_physics`: it misses exit interactions and feedback,
with 30–95% errors against Mie. It is not a production solver or compatibility
implementation.

General resonant behavior through the main spectral method is **pending**.
The physical map is now constructed for ordered z-graph assemblies, but its
curved-boundary accuracy and closed-body extension are not validated. The auxiliary-source solver and its benchmark results were removed;
no accuracy claims from that different method are retained as current validation.
Planar cavity and layer results remain valid only within their stated scope.

The [new spectral interaction and scale report](spectral_interactions.md) records
constructed-assembly tests, independent bandwidth/window/surface refinements,
and macroscopic cap runs through R=200 wavelengths. General resonance and full
macroscopic-system acceptance remain open because reconstructed boundary errors
are still nonzero at a scientifically significant level.

## Reproduction and scientific use

```bash
python -m pip install -e '.[test,notebooks,nufft,validation]' -c requirements-validation.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m benchmarks.validate_physics
python scripts/notebooks.py --check --execute
```

The pinned numerical environment targets Python 3.12. The general installation
supports Python 3.10+; CI also checks other supported versions without those
environment-specific pins. Notebook source synchronization is distinct from
Jupyter execution. CI executes the nine maintained application notebooks, checks embedded
PNG figures, and uploads execution artifacts. Executed outputs are also committed
in the source-tree notebooks. Seven maintained example workflows
are exercised by the test suite. Use the CI run for the revision being evaluated;
historical runs that included the removed solver are not current suite results.

Before using a new configuration: refine sampling and domain size; check all
four reconstructed boundary conditions and flux; refine wavelength spacing
near peaks; verify against
an independent solution where available. Preserve parameters, package versions,
raw residuals, and convergence history. Absorbing/magnetic/anisotropic media,
general multi-body scattering, singular surface quadrature, and universal
high-Q accuracy are not implemented or claimed.

## Two-sided curved-boundary limits

`benchmarks.curved_boundary_limits` replaces propagation-only continuation as the
acceptance diagnostic for its recorded spherical-cap cases. It evaluates full
near fields at $q\pm\delta\hat n$, refines target-centred source quadrature,
then extrapolates complex fields to the boundary using overlapping offset triples.
It records numerical convergence separately from four normalized physical
boundary residuals. Finite-aperture flat and equal-index controls, and localized
illumination, expose aperture truncation effects. No singular on-surface integral
or complete closed-body solver is claimed. See [the measured results](application_results.md).

`benchmarks.field_dependent_optics` checks distinct incident directions through
macroscopic refracting and reflecting surfaces without a shifted PSF. Local
expansion bounds and refined source quadrature are numerical checks of fixed
currents; they are deliberately separate from dielectric-boundary verification.
