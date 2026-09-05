# Future implementations: extend the spectral method

## Governing scope

The main method is the spectral transformation of an electric field at an
interface: retain each incident wavevector, construct its local Fresnel basis
and coefficients, form the reflected/refracted field, and propagate that field.
Future work must extend this method, not substitute another method and report
the substitute's accuracy as its own.

The immediate applications are **refractive DUV optical systems, refracting
telescopes, microscopes, and optical trains containing simple reflections**.
They do not inherently require EUV material physics. Modeling a lossless
monochromatic refractive element is within the present material scope; claiming
validated performance for a complete instrument requires more than that.

The auxiliary-source boundary solver and its dedicated validation artifacts
have been removed. It was a different numerical method, and its Mie comparisons
did not validate the spectral Fresnel method. Future work must not introduce
such a solver as a hidden fallback or a prerequisite for this roadmap.

## Pending work

Implemented progress is recorded in [spectral interactions and scale validation](spectral_interactions.md):
ordered assemblies now construct repeated per-k encounters, and axisymmetric
propagating fields can be evaluated with analytic azimuthal integration.
Caps at R=50,100,200 wavelengths have measured boundary and power residuals.
These changes do not close the physical-accuracy items below.

- [ ] **General resonant behavior: implementation and verification pending.**
  Extend the constructed ordered-interface maps to the remaining geometries.
  Verify reconstructed interior/exterior fields, Maxwell boundary
  conditions, power balance, encounter convergence, and resonance scans against
  independent references. Existing planar-layer resonances and the generic
  feedback iterator do not complete this task. Auxiliary-source results cannot
  close it.
- [ ] **Macroscopic elements: verification pending.** Benchmark the main spectral
  method for large apertures and curvature radii, multiple numerical apertures,
  and complete refractive prescriptions beyond the recorded cap cases. Establish field/image accuracy,
  reconstructed boundary residuals, flux, numerical convergence, runtime, and
  memory requirements. The radius-50-wavelength cap diagnostic is not sufficient.
- [ ] **Macroscopic support where inadequate: implementation pending, contingent
  on verification.** Correct or extend spectral propagation, interface coupling,
  sampling, and computational scaling wherever those tests expose inadequate
  handling. Rerun the same acceptance benchmarks before declaring support;
  neither large physical size nor a successful API call establishes it.

These are explicitly unfinished capabilities, not completed work awaiting only
documentation. Merging the refactor does not close these tasks or certify the
package for general resonant or macroscopic scientific workflows.

## Present capabilities versus application targets

| Capability or application | What exists | What remains |
| --- | --- | --- |
| Monochromatic lossless refraction | Real positive index, placed parametric surfaces, per-k Fresnel map, homogeneous propagation | Validate complete optical prescriptions and discretization choices |
| Refractive DUV projection | The wavelength and real-index element model can represent idealized transparent optics | Multi-element workflow, source/object model, image metrics, and independent system validation |
| Refracting telescopes and microscopes | Suitable field, surface, medium, and propagation building blocks | Curved surface-to-surface composition, apertures, field-of-view and polarization validation |
| Dielectric reflection and TIR | Reflected branch of the per-k interface transform | Validate folded multi-surface paths and repeated encounters |
| Ideal simple mirrors | Natural near-term boundary-law extension | Dedicated perfect-reflector law and public API; it is not currently provided by `DielectricInterface` |
| Parallel lossless multilayers | Coherent all-orders scattering recursion already implemented | Absorption, dispersive material data, and coatings attached to curved surfaces |
| Real metallic mirrors and EUV optics | Not supported by the current real-index material model | Complex constitutive response, absorbing coatings, and relevant small-scale field physics |

“Simple mirror” must identify its model: dielectric reflector, total-internal-
reflection surface, ideal perfect conductor, prescribed optical coating, or
real metal. Those are not interchangeable merely because each reflects light.

## 1. Complete optical workflows within the present material scope

### Physical composition and spectral propagation

Represent a lens as an assembly of interfaces bounding media; represent an
optical train as their physical arrangement. An aperture stop or mask is an
optical element. Sampling, quadrature tolerances, and iteration settings remain
separate from those physical configurations. An instrument name must not select
a different underlying propagation theory.

Extend and validate surface-to-surface propagation in the appropriate medium
and frame. Retain vector amplitudes, phase, polarization, and wavevector-specific
Fresnel transformations through each encounter. Do not infer one propagation
direction from the phase of a total interfering electric field.

Repeated reflections/refractions must be coherent sums of complex fields.
`coherent_feedback` supplies the numerical equation; `propagate_interfaces`
constructs the map for ordered, separated z-graph interfaces by composing the
main spectral transformations, with explicit branch orientation and medium.
Other geometries and evanescent coupling require further work.
Require both an iteration residual and convergence of the observable field.
Converging an approximate round-trip map does not establish exact curved
Maxwell boundary conditions.

### Ideal mirrors without a full metal model

Add an explicit perfect-electric-conductor boundary law as a useful idealization.
For a plane with real unit normal n and incident wavevector k_i, its vector map is

$$
\mathbf k_r=\mathbf k_i-2\mathbf n(\mathbf n\cdot\mathbf k_i),\qquad
\mathbf E_r=-\mathbf E_i+2\mathbf n(\mathbf n\cdot\mathbf E_i).
$$

Test reflected transversality, phase/polarization, unit reflected power,
vanishing tangential total E, and vanishing normal total B. Do not impose the
charge-free/current-free dielectric jump conditions on a perfect conductor:
surface charge and current are allowed. A local curved-mirror application must
retain its stated physical-optics approximation. This ideal boundary law does
not require pretending that a finite real refractive index represents a metal.

### Representative systems and acceptance criteria

| Workflow | Scientific purpose | Required comparisons |
| --- | --- | --- |
| Two-surface refractive lens | Entry, interior propagation, exit, and aperture effects | Field convergence, flux, paraxial limit, vector focal field |
| Two-lens refracting telescope | Afocal propagation, finite field angle, and polarization transport | Collimation/magnification in the geometric limit; finite-aperture diffraction |
| Refractive microscope objective and tube lens | High-NA vector focusing and image formation | Longitudinal/transverse fields, off-axis PSFs, throughput |
| DUV refractive relay/projection prescription | Object-to-image propagation at a specified wavelength/index set | Complex field, PSF, magnification/distortion, contrast versus spatial frequency |
| Folded optical train with ideal reflections | Non-collinear frames and reflection phase | Basis invariance, polarization, energy flux, independently checked propagation |

First use explicitly specified coherent sources and objects. Add incoherent or
partially coherent illumination as an explicit field-statistics model, not by
summing coherent amplitudes indiscriminately. MTF interpretation must state the
imaging/coherence assumptions. Independent Richards–Wolf calculations can test
an ideal-objective limit; they cannot replace a calculation through the actual
dielectric surfaces or certify an arbitrary microscope objective.

Each maintained example must state its prescription, wavelength, indices,
apertures, source, observations, numerical controls, normalization, and purpose.
Record errors in the actual field/image observable as well as local identities.
Do not label these application targets as completed system validations yet.

## 2. Scale the main spectral algorithm to macroscopic optics

Benchmark the actual per-k interface method at increasing aperture/curvature
scales and numerical apertures, with fixed physical configurations during each
convergence study. Separate surface quadrature, incident spectral support,
outgoing spectral quadrature, propagation window, and coherent encounter count.
Report wall time, peak memory, populated modes, surface points, boundary jumps,
flux, and complex-field/image errors. Dimensional size alone is not a benchmark.

The present curved-interface accumulation costs approximately O(K S) for K
incident modes and S surface points. FFT/NUFFT acceleration of propagation does
not remove that Fresnel-evaluation cost. Prioritize reusable interface geometry,
symmetry and harmonic structure, controlled operator compression, and factoring
known reference phase from the electric field where it reduces the bandwidth
of the remaining amplitude. Every compression/truncation needs an error check;
no hidden spectral filter or replacement by a total-field ray is acceptable.

Baseline evidence remains [the main-method cap diagnostic](validation.md): at
curvature radius 50 vacuum wavelengths its reconstructed boundary jumps were
approximately 2.6–4.9%, despite a quadrature field change of 3.5e-6. That diagnostic
also contains finite-aperture and omitted-evanescent effects. It is neither a
projection-system error nor proof that curvature alone causes the residual.
The new R=100 and R=200 wavelength cap runs are documented in
[spectral interactions and scale validation](spectral_interactions.md). They
still show percent-level boundary residuals and do not validate full instruments.

## 3. Extend constitutive media, mirrors, and coatings

1. **Dispersive, absorbing isotropic media.** Represent complex epsilon(omega)
   with explicit convention, units, valid spectral range, and provenance. Start
   with nonmagnetic media; do not imply arbitrary magnetic or anisotropic support.
   Update dispersion, H reconstruction, and spectral validation consistently,
   rather than only allowing a complex `n` in the material constructor.
2. **Complex-wavevector boundary laws.** Derive outgoing/decaying branch choices
   for the `exp(-i omega t)` convention, including evanescent incidence where
   needed. Preserve the proper algebra for complex polarization vectors. Test
   passive loss and near-critical behavior explicitly.
3. **Real metallic reflection.** Calculate phase, polarization, penetration,
   and absorption using material data. Check the perfect-conductor limit where
   applicable; do not replace a real metal with unit-magnitude coefficients.
4. **Absorbing multilayers and coatings.** Extend the existing stable planar
   scattering recursion; multilayers are not entirely missing today. Check
   energy flux and absorption with correct normalization, including local
   dissipation. Attach a coating to a physical surface separately from its
   quadrature. State when a local planar-coating approximation is valid.

For passive stacks with a lossless incident half-space, test the correctly
defined reflected, transmitted, and absorbed power balance. Do not generally
identify power transmittance with the squared electric transmission coefficient.
Use independently implemented planar calculations as references; none becomes
a dependency of the main interface transform.

## 4. EUV and wavelength-scale structure

EUV projection follows the material/coating work; it is not unlocked by changing
the wavelength alone. Real EUV optics require absorbing multilayer reflectors
and a wavelength-appropriate material model. See ASML's [lenses and mirrors](https://www.asml.com/en/technology/lithography-principles/lenses-and-mirrors)
for the distinction between refractive DUV optics and EUV reflective optics.

Sub-wavelength features, evanescent coupling, mask topography, and short-scale
surface structure need dedicated convergence and boundary validation. A curved
interface couples spatial spectral components; merely adding more local
Fresnel encounters does not prove that all such coupling is captured. Any
self-consistent extension should solve for the spectral field amplitudes using
spectral boundary operators, retain the planar Fresnel limit, and remain clearly
distinguished from the present local approximation. It must not silently switch
to auxiliary dipoles, Mie coefficients, or another representation.

Do not equate nanometre dimensions with a universal breakdown of classical
optics. Assess wavelength ratios and the material model. Nonlocal or microscopic
material physics, when required, is a separate extension with its own evidence.

## Validation and architectural rules that remain fixed

- Accuracy and runtime claims must name the algorithm that generated the field.
- Mie and other theories are independent references, not production fallbacks.
- Check reconstructed fields, not only the locally imposed Fresnel identities.
- Preserve electric-state/configuration/law/sampling separation; keep
  `interfaces/fresnel.py` and Fourier algorithms under `fourier/`.
- No backward-compatibility aliases, wrappers, or old implementation are added.
- Future items are not declared implemented until their API, tests, examples,
  physical limits, and reproducible validation are present.

## References for implementation work

- John D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley (1999), Chapters
  1 and 7: electromagnetic boundary conditions and plane-wave propagation.
- Joseph W. Goodman, *Introduction to Fourier Optics*, 3rd ed., Roberts & Company
  (2005): coherent/incoherent imaging and transfer-function assumptions.
- Steven J. Byrnes, *Multilayer optical calculations*, arXiv:1603.02720v5 (2020):
  [complex amplitudes, flux normalization, absorption, and branch choices](https://arxiv.org/abs/1603.02720).

This is a scope and implementation roadmap, not a change to the numerical code
and not a claim that a complete DUV instrument or ideal-mirror API is already
validated.

Macroscopic focal propagation and strict native prescription IO now have a measured implementation; see [macroscopic fields](macroscopic_fields.md). Extended surface-to-surface transport, folded propagation, global curved-boundary correction and complete DUV/Mie acceptance remain pending.

## Current recovery of field-dependent macroscopic use cases

Notebook 08 now computes distinct distant-source directions through macroscopic
refracting and dielectric-reflecting surfaces and combines the resulting fields
on shared physical image coordinates. No shift-invariant imaging approximation
is used there. The older specialized stigmatic mapping is an architectural
reference, not a restored compatibility API. The next acceptance target is
phase-aware extended surface-to-surface transport, first through a finite-conjugate
stigmatic pair and then an off-axis refractive prescription, with independent
source and destination sampling checks. Full-system circuit imaging must use
those transported fields rather than the current pupil-reference convolution.
These implementation and verification items remain open.
