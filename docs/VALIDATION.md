# Validation

The method is checked against exact solutions of Maxwell's equations; the
other methods and the independent full-wave solvers are checked against the
same solutions and against each other. Nothing measured here enters a
calculation: no reference field, gain, phase or alignment is fitted.

## Running the tests

From the repository root:

```sh
.venv/bin/python -m pytest general/tests validation/tests -q
.venv/bin/python -m pytest -q              # also the rest of validation/
VECDIFF_SLOW=1 .venv/bin/python -m pytest general/tests/test_eikonal_regression.py -q
VECDIFF_BACKEND=CUDA .venv/bin/python -m pytest general/tests -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

`pytest.ini` registers two markers. `slow` marks the opt-in regression rows at
R = 1000 λ0 (`general/tests/test_eikonal_regression.py`), which run only with
`VECDIFF_SLOW=1`; `gpu` marks the rows that need CuPy and a CUDA device, which
are skipped without one. `VECDIFF_BACKEND=CUDA` runs the whole general suite
on the GPU. The fast suite (`general/tests validation/tests`) runs in about
two minutes on a laptop CPU.

A test that compares the method with an exact reference asserts a bound on
the error (the measured value plus 10–20 %, the measured value in a comment)
or that the error falls with size; it does not pin the error from both sides,
so it passes if accuracy improves. The slow regression module is a record of
the archived prototype's numbers and keeps its pins.

## The method

| What | Where | Against |
|---|---|---|
| Ray map, Jacobian, flux factor, refracted families | `test_eikonal.py`, `test_eikonal_sequences.py`, `test_closed_sphere.py` | complex-step and finite-difference derivatives; exact plane-interface transfer; the identity at matched indices |
| Caustic index through a focus inside the leg | `test_caustic_phase.py` | a focal line turns the field by a quarter wave, a point focus inverts it; folds refused |
| Reflection legs | `test_reflection.py` | each plane wave's boundary-value problem to 1e-13; focused beams; Debye terms 0 and 2 of a sphere |
| Meridian leg | `test_axisymmetric_eikonal.py` | the lattice leg on the same input; `MieSphere` first entry |
| Lattice leg, cylinder and oval | `test_cylinder_and_oval.py` | `InfiniteCylinder` first entry; the archive's general code |
| Sequences | `test_eikonal_sequences.py` | `LayeredSphere` first entry through two surfaces |
| Paths: stops, mirrors, folds, ghosts, a closed ball, a Cassegrain | `test_paths.py` | equivalent hard pupils, image beams, plane-wave sums, Debye terms 0 and 2 |
| Propagator and Solution | `test_propagator.py`, `test_propagation_contracts.py`, `test_systems_and_observables.py` | one spectrum per segment; E and H together; options refused |
| Targets and coverage | `test_propagation_operator.py`, `test_coverage.py` | planar Fresnel one-sided limits; homogeneous waves on curved and rotated targets |

The errors measured through these tests and the regression against the
archived prototype at R = 1000 λ0 are tabulated in the
[propagation guide](PROPAGATION.md#checks-of-the-method).

## Exact references: layered sphere and cylinder

`vecdiff.references` holds the exact solutions: `MieSphere`, `LayeredSphere`
and `InfiniteCylinder`, with their Debye series, under any propagating
incident field (plane waves, FFT spectra or beams).

`LayeredSphere(incident, volumes, interiors)` solves concentric spherical
layers (any number, real indices; `volumes` outermost first, `interiors[i]`
filling `volumes[i]` minus `volumes[i + 1]`) and `InfiniteCylinder(incident,
volume, interior)` a circular cylinder (`volumes.CylinderVolume`, normal and
oblique incidence, TE/TM coupled). Both take `incident=` any propagating field
as `MieSphere` does, and `evaluate(grid, side=, term=)` selects the total field
or one Debye term: `term=1` is the first entry (transmitted inward through
every interface without reflection, continued through the centre or the axis),
term p the waves that entered and met p − 1 reflections at any interface,
`term=0` the external reflection and diffraction outside, `("after", P)` the
rest. One layer reproduces `MieSphere`. These are the references for transport
through surface sequences (two aplanatic surfaces, the meniscus) and through
a non-spherical (astigmatic) surface.

They stay exact at size parameters near 10⁴: regular and Hankel functions
come from Miller's downward and the ratio upward recurrence with separate
logarithmic scales, the layered total field from the scattering coefficient
of everything inside each radius, and every one-way coefficient from the
Hankel Wronskian, so no nearly equal functions are subtracted (a direct
Hankel-basis Debye solve is ill-conditioned beyond the size parameter). The
cylinder expands a `FocusedBeam` on its own axial-wavenumber and azimuth
quadrature, exact within `reach` of the focus along the axis; points sharing
a position across the axis share the cylindrical-wave sums. Its total field
also holds the cylinder's sharp internal resonances, so its axial quadrature
converges only algebraically: for a hard pupil at 20 λ0 it changes by
1.3 × 10⁻³ from 62 to 1984 axial nodes and by 7 × 10⁻⁵ from 992 (smooth
pupil 5 × 10⁻⁶ and 1 × 10⁻⁷); refine `axial_nodes` before using it. The
Debye terms are smooth in h and converge spectrally. Inside a layer a Debye
term of a multipole that turns there grows beyond its turning point and
cancels only in the sum.

`test_layered_references.py` checks one layer and index-matched layers
against `MieSphere`, Debye terms summing to the total in every region,
interface continuity, Maxwell residuals, the optical theorem, a tunnelling
air core at 200 λ0, Bohren–Huffman coefficients at normal incidence, per-mode
energy at oblique incidence, and field values of the archived prototype for
both two-surface geometries and the cylinder. `test_momentum.py` checks the
radiation pressure on a Mie sphere against Wiscombe's MIEV0.
[`exact_references.py`](../general/examples/exact_references.py) runs a few
of these identities.

## Wave transport

These records check the other method, `alternatives.WavePropagator`, and
its pieces. Their numbers are carried from the runs recorded in
`validation/reference_data/`.

### Focal regression

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/python validation/pipeline_regression.py
```

This constructs a smoothed incident beam and a ball-lens path with radius
100λ, index 1.5 and a 280² grid spanning 130λ, carried by spectral wave
transport. Saved reference fields are loaded after solving;
no reference-derived gain, phase, alignment or field enters the calculation.
The detector is the common saved plane at z = 146.9541667λ.

The measured relative complex E error against the directly transmitted Debye
p = 1 branch is approximately **6.195 × 10⁻⁵ (0.006195%)**. The former local
single-direction approximation gave 0.25895% on this case. These compare the
same physical branch; a full Mie field includes additional scattering.
The current result also agrees with the saved direction-resolved
Fresnel focal E/H to approximately 2 × 10⁻¹³ relative error.

[pipeline_regression.json](../validation/reference_data/pipeline_regression.json)
records the latest solve/evaluation times, complex-field differences, modal
omission reports and separation ranks. End-to-end solve timing includes source
spectrum construction and operator preparation; output evaluation is timed
separately. It is one workstation run, not a throughput distribution.

The original performance experiment measured 217.48 s versus 40.65 s total
and 159.64 s versus 38.65 s on repeated use. Those are historical prototype
measurements, preserved in the `fresnel_performance_*` JSON files.

### Local and spectral treatments

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/python validation/propagation_branches.py
```

This compares both treatments with the stored Debye and full-Mie fields for
the same R/λ = 100, n = 1.5, centrally illuminated sphere. The local
treatment estimates vector phase gradients; it is not the historical
Poynting-direction adapter. The measured focal E differences are 0.25915 %
(local) and 0.006195 % (spectral) against Debye p = 1, and 0.47264 % /
0.41020 % against full Mie (notebook 07 reproduces these digits now). The
spectral electric result agrees with the saved implementation to about
2e-13 relative difference.

[The record](../validation/reference_data/propagation_branches.json) distinguishes
complete electric-output operator timings from the separate H validation;
[raw fields](../validation/reference_data/propagation_branches.npz) are retained.
These checks do not establish accuracy for large general geometries.
`general/tests/test_propagation_operator.py` checks the local treatment against
exact oblique Fresnel transmission and its refusal of counter-propagating
fields whose phases cancel.

### Pure transfer

The test must start from an exact or independently converged field on the
source surface, transport that field, and compare it at the destination.
There is no Fresnel interaction or second scattering solve in this check.

```sh
.venv/bin/python -m vecdiff_validation.pure_surface_transport \
  --output validation/reference_data/pure_surface_transport.json
.venv/bin/python -m vecdiff_validation.surface_transport \
  --output validation/reference_data/surface_transport.json
.venv/bin/python -m vecdiff_validation.transport_geometries --help
```

`validation/tests/test_pure_surface_transport.py` checks exact Mie traces, a
closed-surface integral and an independent Weyl propagating-plus-evanescent
quadrature. Complete transfer can approach roundoff on these converged cases.
That result belongs to the complete reference integral, not to adding an
option to the current `SurfaceTransport`. Its finite FFT chart and forward
propagating band remain distinct sources of error. Total-field and
scattered-field error scales are reported separately.

`validation/tests/test_transport_geometries.py` tests ellipsoid, box,
cylinder and asymmetric rippled boundaries using analytic dipole fields and a
curved receiver. The reference module also provides BEM-derived nonspherical
transfer and timing studies.
[transport_geometries.json](../validation/reference_data/transport_geometries.json)
records geometry, quadrature, errors, thread settings and repeated timings.

Large Q plans stream their Chebyshev terms instead of retaining rank-sized
dense arrays. The curved-Q regression checks the same complex E/H plane wave,
adjoint and band synthesis against the precomputed implementation, with and
without compact Bessel-factor caching.

### AL1225 lens study

[Notebook 06](../general/notebooks/06_full_wave_reference_validation.ipynb)
presents the AL1225 catalog geometry and spectral wave-transport fields for
scaled models at R/λ = 3, 10, 30, 100. Reproduce them with
`validation/al1225_benchmark.py`, which writes the complex E/H arrays,
observation coordinates and timings to
`validation/reference_data/al1225_size_study.npz`. That file is not in the
repository (it was never committed), so the notebook keeps the outputs of its
last run. Notebook reload time is not solver time. The fixed-resolution BEM
capacity limit is retained; when BEM is absent, no independent complex-field
error is reported. The physical-size lens at R/λ = 8012.82 remains
uncomputed.

### Spatial coverage

`general/tests/test_coverage.py` checks reduced masks, complete point coverage,
cache reuse when toggling modes, and unchanged fields on common supported
points for wave transport. Independent analytic checks cover planar Fresnel
one-sided limits and homogeneous waves across curved, rotated equal-index
boundaries. Invalid electric samples remain explicitly masked and cannot
become FFT incident data. These tests do not establish full-wave accuracy of
the modal extension near a refracting curved boundary.

## Full-wave references

The separate `vecdiff_validation` package in `validation/` holds PMCHWT and
Müller BEM formulations, first-order FEM, MFS, meshing (Gmsh) and a periodic
Fourier–Maxwell reference; its BEM/FEM kernels are compiled from
`validation/native/` (C and Fortran). The library never imports it. Mesh
refinement, boundary/flux diagnostics and comparisons with Mie qualify the
native solvers on their tested cases (`validation/tests/test_native_references.py`,
`test_full_wave_references.py`). A small algebraic residual alone does not
establish a converged physical reference. `vecdiff_validation.comparisons.surface_radiation`
provides independent prescribed-current Green/Weyl evaluation.

[The archived synthetic study](../general/notebooks/archive/06_synthetic_reference_validation.ipynb)
compares the complete wave transport (then `PropagationPipeline`, now
`alternatives.WavePropagator`) against Mie and fixed-mesh BEM for a sphere,
then against BEM alone for plano-convex, biconvex, meniscus, even-asphere,
Cartesian-oval, astigmatic-freeform and two-freeform-face bodies. Each method
uses the same incident phase/amplitude, common observation plane and shared
focal/axial/supported meridional samples, without fitting fields. Complex
three-component E/H differences and transmitted-side interface traces are
reported, including full-vector/tangential comparisons, prescribed versus
reconstructed traces and the spatial offset between analytic faces and BEM
facets. Timings include input/geometry preparation, assembly/solve and
complete field evaluation. The archive is not re-executed.

The wave transport uses explicitly continued open graph faces; Mie/BEM use
closed finite bodies with rims and reflections. Their differences include
physical model error, and the omitted meridional regions have no accuracy
claim. The sphere's outer 5% of radius is modified by its continuation. One
fixed mesh per body replaces development refinement sweeps in that notebook.
The earlier six-body study found **0.24–0.35%** focal E/H changes between
its last two meshes; those changes do not bound the remaining reference
error, and the added two-freeform-face body has no established refinement
estimate.

The size study scales the sphere, Cartesian oval and two-freeform-face lens
through **R/λ = 3, 10, 30, 100** at fixed wavelength and indices. FFT charts
grow while resolving the configured angular band, and BEM keeps h/λ = 0.20.
Meshes above the native dense solver's 5,000-unknown capacity are explicitly
unavailable. Large-size BEM counts are area estimates, not measured solve
times; the reported storage is for one complex dense matrix, excluding
factorization workspace. Mie independently checks complex sphere E/H at every
size. Large nonspherical cases measure scaling without an independent
accuracy claim. Meridional probes remain sparse and do not establish a
spatially resolved error bound.

Cross-library tests keep the stigmatic implementation independent and check
complex vector synthesis, Maxwell equations and the saved Green-integral case.
Those that call `vecdiff_stigmatic`, which lives in its own repository, are
skipped when it is not installed.

## Stored records

The `focal_pipeline` and `focal_spectral_N280_a18_c1e-07_t3` NPZ/JSON files are
frozen reference and comparison data. The suffix `t3` describes the angular
partition; the ordinary Fresnel row is order zero. Older order/phase-space
records document historical experiments, not currently supported solvers.

The retired implementations remain in the Git history. The theory of the
direction-resolved Fresnel treatment is in the
[manuscript](direction_resolved_fresnel/direction_resolved_fresnel.tex).
