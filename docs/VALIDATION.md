# Validation

Run the retained physical tests from the repository root:

```sh
.venv/bin/python -m pytest general/tests stigmatic/tests validation -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

The retained test suite covers exact planar boundary conditions,
Brewster/TIR behavior, homogeneous propagation, rotated and SI-scaled frames,
Q versus direct modal sums, inversion residuals, direction-resolved Fresnel,
coherent pipeline phase and superposition, native BEM/FEM references, Mie and
independent nonspherical pure-transfer checks. Dedicated tests for retired
solvers were removed; useful tests in mixed files were retained or migrated.

## Focal pipeline

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/python validation/pipeline_regression.py
```

This constructs a smoothed incident beam and a ball-lens path with radius
100λ, index 1.5 and a 280² grid spanning 130λ. The candidate uses only the
public propagation API. Saved reference fields are loaded after solving;
no reference-derived gain, phase, alignment or field enters the calculation.
The detector is the common saved plane at z = 146.9541667λ.

The measured relative complex E error against the directly transmitted Debye
p = 1 branch is approximately **6.195 × 10⁻⁵ (0.006195%)**. The former local
single-direction approximation gave 0.25895% on this case. These compare the
same physical branch; a full Mie field includes additional scattering.
The current library result also agrees with the saved direction-resolved
Fresnel focal E/H to approximately 2 × 10⁻¹³ relative error.

[pipeline_regression.json](../validation/reference_data/pipeline_regression.json)
records the latest solve/evaluation times, complex-field differences, modal
omission reports and separation ranks. End-to-end solve timing includes source
spectrum construction and operator preparation; output evaluation is timed
separately. It is one workstation run, not a throughput distribution.

The original performance experiment measured 217.48 s versus 40.65 s total
and 159.64 s versus 38.65 s on repeated use. Those are historical prototype
measurements, preserved in the `fresnel_performance_*` JSON files. They are not
substituted for the current public pipeline's measured time.

## Pure transfer

The test must start from an exact or independently converged field on the
source surface, transport that field, and compare it at the destination.
There is no Fresnel interaction or second scattering solve in this check.

```sh
.venv/bin/python -m vecdiff_general.propagation.references.pure_surface_transport \
  --output validation/reference_data/pure_surface_transport.json
.venv/bin/python -m vecdiff_general.propagation.references.surface_transport \
  --output validation/reference_data/surface_transport.json
.venv/bin/python -m vecdiff_general.propagation.references.transport_geometries \
  --help
```

`test_pure_surface_transport.py` checks exact Mie traces, a closed-surface
integral and an independent Weyl propagating-plus-evanescent quadrature.
Complete transfer can approach roundoff on these converged cases. That result
belongs to the complete reference integral, not to adding an option to the
current `SurfaceTransport`. Its finite FFT chart and forward propagating band
remain distinct sources of error. Total-field and scattered-field error scales
are reported separately.

`test_transport_geometries.py` tests ellipsoid, box, cylinder and asymmetric
rippled boundaries using analytic dipole fields and a curved receiver. The
reference module also provides BEM-derived nonspherical transfer and timing
studies. [transport_geometries.json](../validation/reference_data/transport_geometries.json)
records geometry, quadrature, errors, thread settings and repeated timings.

## Independent references

`propagation/references/` retains Mie and Debye sphere solutions, PMCHWT and
Müller BEM formulations, first-order FEM, MFS and a periodic Fourier–Maxwell
reference. BEM/FEM kernels are compiled from `general/native/`; they remain
outside the forward pipeline. Mesh refinement, boundary/flux diagnostics and
comparisons with Mie qualify the native solvers on their tested cases. A small
algebraic residual alone does not establish a converged physical reference.

[The archived synthetic study](../general/notebooks/archive/06_synthetic_reference_validation.ipynb)
compares the complete public `PropagationPipeline` against Mie and fixed-mesh
BEM for a sphere, then against BEM alone for plano-convex, biconvex, meniscus,
even-asphere, Cartesian-oval, astigmatic-freeform and two-freeform-face bodies.
Each method uses the same incident phase/amplitude, common observation plane
and shared focal/axial/supported meridional samples, without fitting fields.
Complex three-component E/H differences and transmitted-side interface traces
are reported, including full-vector/tangential comparisons, prescribed versus reconstructed pipeline traces and
the spatial offset between analytic faces and BEM facets. Timings include
input/geometry preparation, assembly/solve and complete field evaluation.

The pipeline uses explicitly continued open graph faces; Mie/BEM use closed
finite bodies with rims and reflections. Their differences include physical
model error, and the omitted meridional regions have no accuracy claim. The
sphere's outer 5% of radius is modified by its continuation. One fixed mesh
per body replaces development refinement sweeps in this notebook. The earlier
six-body study found **0.24–0.35%** focal E/H changes between its last two
meshes; those changes do not bound the remaining reference error, and the
added two-freeform-face body has no established refinement estimate. The
sphere BEM/Mie discrepancy is independently measured in the archived study.

The size study scales the sphere, Cartesian oval and two-freeform-face lens
through **R/λ = 3, 10, 30, 100** at fixed wavelength and indices. FFT charts
grow while resolving the configured angular band, and BEM keeps h/λ = 0.20.
Meshes above the native dense solver's 5,000-unknown capacity are explicitly
unavailable. Large-size BEM counts are area estimates, not measured solve
times; the reported storage is for one complex dense matrix, excluding
factorization workspace. Mie independently checks complex sphere E/H at every
size. Large nonspherical cases measure pipeline scaling without an independent
accuracy claim. Meridional probes remain sparse and do not establish a
spatially resolved error bound.
The Mie-only size-study interface probes coincide with actual Q-grid samples,
avoiding interpolation of rapidly varying complex traces. The smaller BEM
cases use bilinear interpolation to the reference facet probes, which adds
sampling uncertainty to those interface comparisons.

Large Q plans stream their Chebyshev terms instead of retaining rank-sized
dense arrays. The curved-Q regression checks the same complex E/H plane wave,
adjoint and band synthesis against the precomputed implementation, with and
without compact Bessel-factor caching.

Cross-library tests keep the stigmatic implementation independent and check
complex vector synthesis, Maxwell equations and the saved Green-integral case.

## Stored records and migration

The `focal_pipeline` and `focal_spectral_N280_a18_c1e-07_t3` NPZ/JSON files are
frozen reference and comparison data. The suffix `t3` describes the angular
partition; the ordinary Fresnel row is order zero. Older order/phase-space
records document historical experiments, not currently supported solvers.
Their former history-dependent harnesses have been retired. Notebooks 03–07
have been migrated to the supported API; their models and reference provenance
are documented in [Notebooks](NOTEBOOKS.md).

A complete local snapshot of the working tree before this cleanup is available
at Git ref `refs/snapshots/before-propagation-cleanup-20261008T020658Z`
(commit `4d07a9aa0094827e40d1fb5063d964c5d7742f59`). It includes the retired
implementations and their examples without keeping them in the public package.
The theory and numerical implementation are in the
[manuscript](direction_resolved_fresnel/direction_resolved_fresnel.pdf).

## Field/system/target operator and local branch

`general/tests/test_propagation_operator.py` checks exact planar Fresnel fields,
piecewise material-region evaluation, one-sided targets, curved and rotated
observation surfaces, automatic sampling, source/system binding, incremental
propagation and cache reuse, electric-only evaluation, and rejection of a
multi-direction field that violates the local eikonal criterion.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/python validation/propagation_branches.py
```

This independently compares both public-operator candidates with the stored
Debye and full-Mie fields for the same R/λ=100, n=1.5, centrally illuminated
sphere. The new local implementation estimates vector phase gradients; it is
not the historical Poynting-direction adapter. The measured focal E differences
are about 0.25915% (local) and 0.006195% (spectral) against Debye p=1, and
0.47264% / 0.41020% against full Mie. The spectral electric result agrees with
the saved implementation to about 2e-13 relative difference.

[The record](../validation/reference_data/propagation_branches.json) distinguishes
complete electric-output operator timings from the separate H validation;
[raw fields](../validation/reference_data/propagation_branches.npz) are retained.
These checks do not establish accuracy for large general geometries or supply
the missing independent BEM references from notebook 06.

## AL1225 lens study

The active [notebook 06](../general/notebooks/06_full_wave_reference_validation.ipynb)
now presents AL1225 catalog geometry and measured spectral pipeline fields for
scaled models at R/λ = 3, 10, 30, 100. Reproduce them with
`validation/al1225_benchmark.py`; recorded complex E/H arrays, observation
coordinates and timings are stored in `validation/reference_data/al1225_size_study.npz`.
Notebook reload time is not solver time. The fixed-resolution BEM capacity
limit is retained; when BEM is absent, no independent complex-field error is
reported. The actual physical-size lens at R/λ = 8012.82 remains uncomputed.

## Spatial coverage

`general/tests/test_coverage.py` checks reduced masks, complete point coverage,
cache reuse when toggling modes, and unchanged fields on common supported
points for both local and spectral transmission. Independent analytic checks
cover planar Fresnel one-sided limits and homogeneous waves across curved,
rotated equal-index boundaries. Invalid electric samples remain explicitly
masked and cannot become FFT incident data. These tests do not establish
full-wave accuracy of the modal extension near a refracting curved boundary.
AL1225 meridional fields are regenerated with complete coverage of the
continued-face model; the physical finite-rim exterior is not implemented.
