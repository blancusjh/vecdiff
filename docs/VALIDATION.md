# Physical verification

The checks separate numerical convergence, correctness within a declared physical model, and preservation of selected legacy behavior. A local identity is not a substitute for testing the propagated solution.

The current suite contains **153 passing tests**. The **5 physical notebooks** execute with supporting assertions and 21 retained figures. Independent wheel verification installs and tests each distribution with the other package absent. Reports are in [validation/results](../validation/results/).

## Conventions and normalization

Positions and vacuum wavelength are metres; E is V/m and H is A/m. Time dependence is `exp(-i omega t)`. Public magnetic data is H, not Z0 H. Initial media are lossless, isotropic, nondispersive and nonmagnetic. Surface normals and material sides are explicit.

Boundary diagnostics report tangential E/H and normal D/B separately. Normalization uses a declared incident electric amplitude and material factors; a dark interference point is not allowed to silently change the scale. Power is the signed integral of `Re(E cross H*) / 2` over an oriented physical surface. Electric amplitude squared is not mislabeled SI intensity.

## Checks and references

| Area | Independent or physical reference |
|---|---|
| Architectural boundary | Fourier imports only mathematical dependencies; centered frequency meshes require numerical coordinates only |
| Requested interface operation | Transmission/reflection never call the other branch's coefficients; both preserve the field's grid and wavelength |
| Common Fourier interface | Cartesian/polar continuous normalization, analytic angular Gaussian modes 0,+1,-2, shifted charts, inverse reconstruction and shared-frequency agreement |
| Typed grids | Non-square Cartesian indexing, rotated/translated polar embedding, area weights and rejection of duplicate angular endpoints |
| Cartesian Fourier transform | Centered FFT2/IFFT2, independent Fourier sum, Parseval, impulses/constants, odd/even sizes and trailing vector/batch axes |
| Hankel transform | Analytic Gaussian pairs for orders zero and one; radial integration measure |
| Nonuniform synthesis | Direct complex E/H sum versus FINUFFT; physical phase origins |
| Homogeneous ASM | Known FFT-bin mode, tilted planes, explicit longitudinal completion, forward/backward evanescence and large global translation |
| Paraxial propagation | Measured complex-vector error against ASM as distance increases; separate approximate model label |
| Plane boundary | Reconstructed total E/H on both sides; all four Maxwell jump conditions |
| Fresnel power and polarization | Signed normal flux, Brewster cancellation, reflected p-basis sign, TIR phase/decay and zero normal transmitted flux |
| Coherent layers | Independent Airy formula over slab thickness, internal standing-wave E and constant net flux, thin/thick frustrated-TIR gap |
| Geometry and matter | Spherical/aspheric agreement, rotated volume membership, curved faces and rim, closed sphere boundaries, vacuum outside, prescription round trip |
| Surface radiation | Flat equal-index localized illumination, independent quadrature refinement, reactive full dyadic Green evaluation |
| Curved-interface limits | Local trace identity separated from independently radiated one-sided fields; target-centred quadrature refinement and offset extrapolation |
| Weyl representation | Angular refinement and far Green comparison for a localized control; observation-envelope rejection |
| Local spectrum | Absolute E/H error bounded against the same discrete Green currents on the declared observation ball |
| Multiple encounters | Analytic two-plane single-pass complex amplitude/phase, explicit reflected branches, finite-aperture rejection by infinite-plane models |
| High-frequency transport | Separate-mode linearity, per-face flux, inter-face ray-tube spreading and derivative-step refinement |
| Generic coherent feedback | Analytic geometric series and explicit discrete-equation residual; not a claim of solved curved-body scattering |
| Bulk Maxwell equations | Held-out real-space finite differences of both curl equations and both divergences, with approximately quadratic convergence |
| Stigmatic geometry | Constant optical path, Snell vectors, surface/reference-sphere Jacobians and supported higher/lower-index branches |
| Stigmatic transfer | Entrance/interface/exit flux, reflected contribution, geometric amplitude and four local boundary identities |
| Debye vector focus | Direct/Hankel/NUFFT E/H, radial/azimuthal refinement, polarization symmetries, and independent Bessel reduction of the longitudinal complex sign |
| Fresnel pupil focus | Analytic Gaussian width, physical output spacing and signed Parseval power |
| Observables | SI energy density, circular Stokes convention, coherent cancellation and analytic detector interference fringes |
| Legacy preservation | Saved geometry/Fresnel coefficients plus independent general-geometry Green radiation versus the retained reference |

## Historical physical qualification

The following measurements were recorded before replacement of the 13 original project notebooks. Those notebooks are now removed; the values and model parameters remain historical evidence, alongside the regression tests and saved legacy fixture. They are not measurements from the current five notebooks or uniform accuracy guarantees.

| Calculation | Measured result |
|---|---|
| Plane-interface angle sweep | Largest of four normalized boundary residuals: approximately 3.8e-15 |
| Curved probe quadrature refinement | Electric change between radial orders 112 and 144 at fixed offset: approximately 2.4e-13 V/m |
| Curved reconstructed boundary limit | Tangential E 0.00905; tangential H 0.0358; normal D 0.1475; normal B near roundoff |
| Two-face ray transport | Inter-face transmitted/incident power ratio: 0.9999999999999896 |
| Ray derivative refinement | Half-step relative electric-amplitude change: approximately 1.5e-12 |
| Cartesian oval | Maximum optical-path variation: approximately 6.9e-18 m |
| Debye Hankel/direct | Relative E/H discrepancies around 1e-15 at selected off-axis probes |
| Debye NUFFT/direct | Relative E/H discrepancies around 4.8e-13 at the same probes |
| Fresnel pupil | Output/input signed power ratio: 1.0000000000000095; analytic Gaussian profile within 2e-11 absolute normalized intensity |
| New general Green versus saved legacy Green | Relative complex vector E error: 3.4e-12 |
| Dedicated Debye versus new general Green | Relative complex vector E difference: 8.11e-5 |

The curved test uses a sphere of radius 10 vacuum wavelengths, a circular integration patch of radius R/2, x-polarized plane-wave illumination, n=1 to 1.5, and a probe at local x=0.2R, y=0. It extrapolates complex fields from offsets 0.008, 0.004, 0.002 and 0.001 wavelengths using a quadratic fit. The reported residual includes local physical-optics error and finite patch effects. It is not hidden by rescaling and is not asserted to vanish. Additional offset/quadrature controls are needed when changing this geometry.

The matched stigmatic case uses a 532 nm source, n=1 to 1.5, object at -10 mm, image at 6 mm, and 95% of the retained grazing aperture. The legacy reference used 320 radial and 96 azimuthal nodes. The new general path constructs its own oval, geometric incident eikonal, local Fresnel fields, and full Green radiation. The dedicated path constructs its own reference-sphere transfer and Debye field. No fitted gain, phase, or Ez sign is used.

The 8.11e-5 difference is consistent with the predeclared 1e-4 Debye/Green comparison budget. The general Green agreement preserves a prescribed-current physical-optics model; neither result establishes an exact Maxwell solution of the entire refracting body.

## Reproduce

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/notebooks.py
.venv/bin/python scripts/notebooks.py --check
.venv/bin/python scripts/verify_independent_wheels.py
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Wheel verification builds from clean temporary source copies, checks expected ontology modules in each archive, installs in isolated environments, asserts that the other library is absent, and runs the package tests. The optional NUFFT dependency is installed for those verification runs.

Normal tests use the saved legacy arrays and need no legacy installation. To regenerate that reference with an appropriately provisioned legacy interpreter:

```sh
python -B validation/export_legacy_stigmatic.py --source /path/to/stigmatic-7a1a235
```

The JSON beside the fixture records provenance, units, quadrature and the specific claim. `initial_qualification.json` is preserved as historical evidence of the earlier 47-test implementation; `current_qualification.json`, `independent_wheels.json`, and `notebooks.json` describe the expanded implementation.

## Limits of this evidence

Independent full-wave curved-body references, global closed-surface energy budgets, general reciprocity checks, material dispersion, and fully coupled coherent curved scattering remain further qualification tasks. Finite detector windows are not normalized to the incident power. A local-spectrum error bound concerns representation error relative to supplied currents, not the correctness of those currents as a dielectric solution. See [implementation status](IMPLEMENTATION_STATUS.md) for the supported regimes and planned extensions.

## Earlier implementation stages

The notebook references below describe the work at those stages. All 13 earlier notebooks were subsequently removed; use the current [catalog](NOTEBOOKS.md) for runnable physical examples.

## General volume separation and architecture review

This iteration ran 98 tests (67 general, 24 stigmatic and 7 external), reexecuted the changed general space–matter notebook, and verified the general wheel independently. Stigmatic source and notebooks were not changed or reexecuted. The general wheel must contain `volumes.py` and must not contain the removed `geometry/domains.py`.

The [general review](GENERAL_ARCHITECTURE_REVIEW.md) records uncovered API gaps separately from passing regression tests. Reproduce its observations with `.venv/bin/python validation/review_general_architecture.py`. Use `.venv/bin/python scripts/verify_independent_wheels.py --package general` to verify only the general distribution; without that option both are verified.

## Centered general Fourier interface

The general library now uses centered `FFT2`/`IFFT2` with NumPy normalization.
`frequency_grid((x, y))` returns centered frequencies in cycles per unit;
propagation explicitly converts to angular wavenumbers and modal amplitudes.
Coordinate-aware continuous transforms retain their integration measure and
phase offsets, with centered Cartesian output. The old general Fourier-series
helper names and `(shape, spacing)` signature are removed.

At the centered-Fourier stage, 106 tests passed (75 general, 24 stigmatic, 7 external). Additional checks
compare direct modal E/H with centered FFT propagation on shifted, tilted,
non-square planes with mixed odd/even dimensions, including zero distance.
The general Fourier notebook was reexecuted. Stigmatic files remain unchanged.

## Polar and nonuniform operational interface

At the polar/nonuniform stage, 119 tests passed: 88 general, 24 stigmatic and 7 external.
New checks exercise both Fourier signs in one, two and three dimensions with
trailing batch/component axes; direct exponential sums against FINUFFT; analytic
Gaussian forward/inverse integrals with explicit measures; polar versus direct
nonuniform integration; centered Cartesian versus nonuniform integration; and
first-axis Hankel forward/inverse pairs. Complex decay, zero growing terms and
required quadrature measures are also checked.

At that stage, the expanded Fourier notebook included executed `FT2`/`IFT2`, `HT`/`IHT`, and
`NUFFT` examples. Stigmatic implementations and notebooks were unchanged at that stage.

## FFT representation and aperture contracts

The current suite passes **131 tests**: 100 general, 24 stigmatic and 7 external.
Invalid FFT lattice/order/origin combinations are rejected at construction.
Valid forward/backward and evanescent fields retain direct/FFT E/H agreement.
The tests also respect the existing frame orthogonality tolerance.

Aperture checks reject conflicting quadrature nodes before any scattering,
including a conflict at a later interface. Local coordinates are checked on
translated/rotated plane and curved surfaces. Rim nodes and smaller patches
remain valid. Local incident-plus-reflected signed power matches transmitted
power, with quadrature objects and weights preserved. This is a local boundary
check, not a certification of global curved scattering or aperture coverage.

The Fourier and dielectric-boundary notebooks were reexecuted. The architecture
probe now reports rejection of its incompatible FFT and aperture inputs; other
recorded geometry/grid/material findings remain open.

## Current physical notebooks

All five execute without recorded cell errors. Notebook 05 now includes the GOTS surface and its stigmatic focus. Selected results for their declared parameters:

- Glass-to-air Brewster and critical angles: 33.69° and 41.81°; at 55°, electric-amplitude penetration length is 0.119 µm.
- The symmetric slab TE₀ example has effective index 1.482339.
- Enlarging the circular-aperture FFT window changes the displayed normalized far-field profile by at most 0.00245. This is a window check for the specified finite-edge transmission model.
- The lens's sampled axial wave-field peak is at 80.58 µm, compared with a 91.50 µm paraxial ray prediction. Refining surface quadrature from 32×64 to 48×96 nodes changes complex E at the fixed probes by 2.65e-14 relatively; this does not bound the high-frequency model error.

The notebook runner checks saved execution, retained figures and a fingerprint of notebook cell sources. It rejects stale outputs after notebook source edits. Library changes still require reexecution.

## GOTS surface and stigmatic imaging

The general surface now exposes explicit GOTS coefficients and parametric
coordinates, independently of the dedicated stigmatic implementation. Fifteen
additional tests check coefficient units, placement, parameter rejection,
constant optical path and Snell directions for both index orderings and three
length scales. The suite passes 146 tests (115 general, 24 stigmatic, 7 external).

For the earlier finite-conjugate notebook example (subsequently replaced by the matched lens), maximum optical-path error is 2.55e-14 vacuum
wavelengths, ray-direction error is 3.34e-16, and local signed interface-power
imbalance is 1.79e-16. Refining surface quadrature from 32×64 to 48×96 changes
complex E at fixed image probes by 7.24e-15 relatively. These are geometry,
local-interface and integration checks, not an exact closed-body Maxwell claim.

## Matched spherical and stigmatic lens comparison

Notebook 05 now compares two-face lenses with the same signed exit-vertex radius
(-42.560 µm), diameter (51.072 µm), central thickness (10.640 µm), materials and
incident beam. Their common paraxial image position is 95.760 µm. Meridional
and image-plane comparisons share coordinates and per-observable color scales.
The analytic collimated GOTS limit supplies the stigmatic exit face.

Seven additional tests exercise this limit, including convergence from finite
object distance, vertex curvature, Snell directions, optical path and invalid
conjugates/graph folds. The complete suite passes 153 tests (122 general,
24 stigmatic and 7 external). For the earlier smaller-aperture comparison, optical-path
error is 2.55e-14 wavelengths, ray-direction error 2.54e-16, local exit flux
imbalance 2.03e-16, and complex-field quadrature change 8.04e-15. These qualify
geometry and numerical integration within the declared physical-optics model.

## Wide-aperture demonstration

Notebook 05 doubles the shared aperture radius to 48 wavelengths and sets the
Gaussian waist to 48 wavelengths, illuminating the nonparaxial outer zones.
Both lenses have central thickness 20 wavelengths to preserve positive edge
thickness. The sphere retains 2.128 µm at the edge; marginal incidence is
36.87°, below the 41.81° critical angle. Its edge optical-path error relative
to the common geometrical image is -1.572 wavelengths.

At that fixed image plane, the on-axis electric norm squared is 118.27 V²/m²
for the spherical lens and 616.75 V²/m² for the stigmatic lens, a ratio of 5.215.
This is a same-input, same-plane comparison, not a normalized Strehl ratio or
a comparison of separately optimized focal planes.

The final quadrature uses 72×192 surface nodes, checked against 96×256.
At 25 probes spanning the meridional map, maximum complex-field differences
normalized to the largest reference amplitude are 3.81e-14 (sphere) and
9.50e-14 (stigmatic). The initial 48×128 rule failed the map-probe tolerance
for the stigmatic field and was replaced. Library source was unchanged.

## Operational interface propagation

The primary general path now composes planar ASM R, curved transport Q, local Fresnel T and a solved Q inverse. Green methods are isolated in `vecdiff_general.comparisons`. See [the operational method and measured checks](OPERATIONAL_PROPAGATION.md). Notebook 05 is executed with this path. Earlier current-based validation below/above applies only to the separate comparison model.
