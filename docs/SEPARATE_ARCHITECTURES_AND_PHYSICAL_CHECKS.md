**Separate stigmatic and general libraries: architecture and physical verification plan**

> Historical review: the general propagation path has since been replaced by
> [explicit R/Q/T composition](OPERATIONAL_PROPAGATION.md). Descriptions below
> of ray-tube or Green-based primary evaluation describe the earlier design.


Design proposal, 15 September 2026. This document supersedes the shared-runtime and stigmatic-specialization recommendations in `ARCHITECTURE_REVIEW.md`. The audit findings and numerical results in that earlier document remain useful. No repository implementation, branch, or Git reference has been changed.

**1. Separation is an architectural requirement**

Keep two independently installable and testable libraries: `vecdiff_stigmatic` and `vecdiff_general`. They can live in the same repository, with separate source trees, dependencies, examples, documentation, and tests. Neither imports the other. The stigmatic calculation must remain directly inspectable and runnable without the general solver installed.

Use the same written physical conventions and corresponding concepts, implemented independently for now. Even small geometry/field objects can be duplicated initially; avoiding premature coupling is more valuable than minimizing these few lines. A comparison program outside both libraries exchanges explicit physical arrays and metadata. It does not supply either library's optical kernels.

The names below are working names, not a request to create every class or directory immediately. The proposed future layout is:

```text
stigmatic/
    pyproject.toml
    src/vecdiff_stigmatic/
        geometry/         conjugates, Cartesian ovals, reference spheres
        grids/            embedded plane, sphere and pupil grids
        materials/        homogeneous dielectric data and interface sides
        fields/           electric samples and explicit two-component inputs
        sources/          reference-sphere illumination and polarization
        transfer/         ray bases, Fresnel factors, geometric amplitude
        focusing/         Debye, Fresnel, prescribed-current radiation
        numerics/         independent quadrature, Hankel and Fourier kernels
        system/           supported stigmatic elements and sequences
        observables/      power, polarization, focal measurements
        visualization/
        io/
    tests/
    examples/
    docs/

general/
    pyproject.toml
    src/vecdiff_general/
        geometry/         frames, surfaces, regions, body closures
        grids/            physical embeddings, measures, quadrature plans
        materials/        constitutive laws and spatial assignments
        fields/           electric samples and electromagnetic evaluations
        sources/          plane waves, spectra and declared approximate sources
        representations/  spectra, surface currents and eikonal phases
        propagation/      homogeneous ASM and paraxial Fresnel models
        scattering/       plane, local curved and layered boundary models
        transport/        explicit high-frequency ray-tube approximation
        solvers/          sequential, layered and coherent interactions
        numerics/         FFT, NUFFT, quadrature and linear algebra
        system/           prescriptions, bodies, interfaces, observations
        observables/
        visualization/
        io/
    tests/
    examples/
    docs/

validation/
    specifications/       physical conventions and benchmark definitions
    reference_data/       versioned arrays with provenance
    independent_models/   analytic/reference implementations
    comparisons/          readers for outputs of each library
```

This layout does not imply migrating or deleting the existing code during this session. Dependencies inside either library point from numerical applications toward geometry, fields, and constitutive data. Geometry does not import solvers; fields do not import a preferred propagation engine; visualization does not determine physical coordinates.

**2. Physical vocabulary and ownership**

The Space–Matter–Fields distinction should govern responsibilities. It does not require universal base classes called `Space`, `Matter`, or `PhysicalObject`.

| Object/concept | Responsibility |
|---|---|
| `Frame`, `Surface`, `Region` | Euclidean placement, parameterization, orientation and spatial membership. |
| `Grid` | Evaluation coordinates, embedding into physical space, topology, and declared measure when integration is supported. |
| `Material` | Constitutive response. Initially an isotropic, lossless, nonmagnetic dielectric with real positive refractive index. |
| `MaterialDistribution` | Assignment of materials to spatial regions, with an explicit background. |
| `Interface` | Oriented adjacency between two material regions. Its sides are geometric, not permanently incident/transmitted. |
| `ElectricField` | Electric phasor values on a particular physical grid at one vacuum wavelength. |
| `TransverseElectricField` | Optional two-component input with an explicitly declared transverse basis and meaning. It is not automatically a complete electromagnetic solution. |
| Field representation | Coefficients and validity information for evaluating a solution: spectrum, current distribution, or explicit eikonal phases. |
| `Solution` | Physical problem, branches/regions, chosen model, representations, and evaluation of E and H. |
| Numerical plan | Discretization, tolerances, transforms, quadrature and stable phase origins. |
| Observation/detector | Where the solution is sampled and what quantity is measured. |

**Geometry and grids.** A `PlaneGrid` owns its plane embedding, not just array spacings. A `SurfaceGrid` refers to immutable surface geometry and contains parameter coordinates, physical positions and consistently derived normals/weights. A point cloud can sample a field without pretending to define an integration measure. A pupil coordinate grid must specify whether its coordinates are physical distances, angles, direction cosines, or a chart mapped to a reference sphere.

One canonical Cartesian component convention keeps `Ex`, `Ey`, `Ez` unambiguous. Local spherical, cylindrical, surface-tangent and polarization components are explicitly transformed views. The local coordinate axes of a plane grid and the Cartesian components of a vector are separate concepts.

Changing the embedding while retaining values has a physical meaning that must be chosen. Passive coordinate conversion, rigidly moving a field, and resampling the same spatial solution are different operations. An immutable grid prevents a later frame edit from silently moving existing samples. Resampling returns both new values and their new grid; Fourier/Hankel focusing constructs the actual output coordinates from its wavelength, indices, distances, and transform convention.

**Matter.** `Material` describes glass; `Region(material=glass, ...)` describes where the glass is. The sequential notation material–surface–material is a useful prescription. A finite three-dimensional lens additionally needs explicit rim/side-wall geometry to form a closed body. Different surface radii do not determine a unique closure. An integration cutoff, an illumination mask, and a physical aperture/stop are distinct objects or parameters with distinct effects. A piecewise homogeneous material distribution is the initial supported model; writing n(r) does not by itself implement gradient-index propagation.

An interface can use `minus_region`, `plus_region`, and a normal from minus to plus. Incidence is determined at each encounter. Sampling exactly on a boundary requires an explicit side because the normal electric component can jump.

**Electric data.** The minimal complete field is conceptually:

```python
ElectricField(grid, wavelength0, values)  # values.shape == grid.shape + (3,)
# Ex, Ey and Ez are convenient views of this one canonical array.
```

There is no `medium`, duplicated `sampling`, or hidden refractive-index lookup inside it. Wavelength is explicit and positive, not silently defaulted to 1. Grid and values must be compatible and immutable or protected against stale derived state.

I recommend keeping a separate two-component input type instead of making `Ez=None` mean that a universal field method can always recover it. Its two components must be tied to a named orthonormal basis. In particular, tangent to a support surface, perpendicular to the optical axis, and perpendicular to a wavevector are different statements. No use of the name TE should conflate these meanings.

Plane completion requires the wavelength, the homogeneous region's dispersion relation, and a selected propagation branch. Given those, each mode can satisfy E_n = -(k_1 E_1 + k_2 E_2)/k_n when k_n is nonzero. Populated grazing modes require a different polarization representation or an explicit failure. Stigmatic geometric completion uses the local ray direction and belongs to that package's transfer model. Neither operation belongs in a context-free `ElectricField.compute_Ez()`.

Completion preserves the two prescribed components. Projection changes a supplied full vector to enforce a constraint. These are separate named operations. For k/k0=(0.6,0,0.8), completing Ex=1,Ey=0 gives (1,0,-0.75), whereas projecting (1,0,0) gives (0.64,0,-0.48). An explicitly zero component must never be interpreted as missing.

**Electric field versus solution.** E throughout a region determines H through Faraday's law when its spatial derivatives are available. E sampled on one surface does not generally do so. A forward wave and a standing-wave combination can have identical E on a plane and different H. Consequently, the solution retains modal directions, material context and one-sided traces. `solution.electric(grid, ...)` returns `ElectricField`; `solution.electromagnetic(grid, ...)` evaluates consistent E and H together. A stored H is a derived or jointly solved quantity, not a second independent source specification imposed without a consistency check.

Spectral, layered, current-based and high-frequency results may use different internal representations but follow this observation contract within their own library. Evaluation must reject points outside the representation's validity domain, or require an explicit alternate model. A representation-specific accuracy bound is labeled with precisely what it bounds.

**3. Physical conventions and observable quantities**

Use the time dependence exp(-i omega t), vacuum wavelength lambda0, k0=2 pi/lambda0, and physical SI E and H at the public interface. Lengths are converted to metres at public construction or supplied in metres; plots and prescriptions may use explicitly declared display/input units. Implicit mixing of mm geometry and metre wavelength is forbidden. Public H must not silently mean Z0 H, as some current internal code does.

For an initial nondispersive dielectric, epsilon=epsilon0 n^2 and mu=mu0. Within each source-free region:

```text
curl E =  i omega mu H
curl H = -i omega epsilon E
div(epsilon E) = 0
div(mu H) = 0
```

`div E = 0` and separate scalar Cartesian Helmholtz equations apply inside a homogeneous isotropic region. They are not the universal interface equations. For spatially varying epsilon and constant mu, the electric equation is curl curl E - omega^2 mu epsilon(r) E = 0.

For a homogeneous plane-spectrum branch, use kz=sqrt((n k0)^2-kx^2-ky^2), with the sign selected to propagate or decay into the appropriate half-space. Forward ASM uses exp(i kz d). Backward/returning branches get their own correctly oriented local references. Numerically unstable reversal of evanescent decay is not a universal reversible-propagation test.

Observable names should distinguish:

| Quantity | Meaning and required information |
|---|---|
| `electric_norm2` | E* dot E. It is not universally irradiance. |
| `poynting` | S = (1/2) Re(E cross H*), using SI fields. |
| `power(surface)` | Integral of signed S dot normal over a surface with a valid area measure. |
| `energy_density` | (epsilon |E|^2 + mu |H|^2)/4 for the initial lossless nondispersive model. |
| `energy(volume)` | Volume integral of energy density; it is not an integral of field intensity over a plane. |
| `polarization(basis)` | Jones/Stokes/ellipse quantities in a declared analysis basis, with explicit handedness/view direction. |

If wavelength-dependent material models are introduced, their energy-density expression needs the appropriate dispersion treatment; an n(lambda0) callback alone does not supply that. A monochromatic beam does not have a finite total pulse energy. Optional source power normalization is an explicit source operation, never a hidden correction after propagation.

These distinctions follow the Maxwell and Poynting framework described in [MIT's electrodynamic-fields treatment](https://web.mit.edu/6.013_book/www/chapter12/chap12.html).

**4. Stigmatic library: a direct physical calculation**

Its public vocabulary should remain close to the derivation in the supplied corrected refraction note: conjugate points, Cartesian oval, entrance reference sphere, exit reference sphere, ray bases, geometric transfer, and focal field.

The calculation has independently inspectable stages:

```text
Conjugates + material indices + aperture
    -> designed stigmatic surface and supported ray family
Source on entrance reference sphere G
    -> geometric propagation to the refracting surface
    -> reflected/transmitted Fresnel amplitudes
    -> geometric propagation to exit reference sphere G'
    -> chosen diffraction model
    -> ElectricField / electromagnetic evaluation on a physical output grid
```

Keep intermediate transfer data available: ray directions, bases, incidence angles, s/p coefficients, geometric amplitude, path length and Jacobians. This is essential for debugging. Avoid an opaque chain of operator overloads as the only public workflow.

For the supported real-object/real-image transmitted branch, the reference-sphere transfer in the note is

```text
E_G' = A(Q) [t_p E_theta0 theta_i + t_s E_phi phi]
A(Q) = |z0| l_i(Q) / (|zi| l_0(Q)).
```

Implement this vector expression as the readable reference. Compare it with cylindrical and Cartesian transfer matrices and with optimized kernels. Do not hide geometric amplitude inside a generic transmission coefficient. Keep optical path phase, Fresnel phase, vector-basis changes, and amplitude factors separately identifiable.

The user-supplied note's geometrical transfer and the thesis's paraxial focal reduction are not one approximation. Give focal models explicit names and assumptions:

| Model | Physical meaning |
|---|---|
| Vector Debye focusing | Angular diffraction integral associated with the reference-sphere/pupil model; preserve the fast Hankel and Fourier paths where applicable. |
| Paraxial Fresnel focusing | Quadratic-phase approximation and its physical input/output coordinate mapping. |
| Prescribed-current radiation | Full vector Green radiation from the chosen physical-optics currents; useful finite-distance reference for those currents. It does not independently solve the dielectric boundary problem. |

Hankel, direct angular integration, and FFT are numerical choices only when they evaluate the same integral with the same pupil mapping and measure. Perspective mapping r=|z0| tan(alpha0), sine-coordinate mapping, and physical propagation between a plane and a sphere are not interchangeable. Each mapping carries its own Jacobian and interpretation.

Initially validate the real conjugate domain the existing kernels actually support. Arbitrary signs of the axial coordinates do not automatically establish valid virtual-object/virtual-image ray branches. Virtual conjugates, mirrors, folds, caustics and total-internal-reflection encounters require their own derivations and applicability checks. Preserve existing examples and code while recording the limits; do not invent negative refractive indices to encode ray direction.

A supported multi-element stigmatic sequence must satisfy each element's reference-sphere and conjugacy assumptions. A one-way sequence returns the transmitted path and accounts for locally reflected power; it must not claim to include coherent internal returns that it has not solved.

**5. General library: geometry and material context with explicit models**

The general experiment specifies a system, source, wavelength and requested observation grids. Model selection is explicit. A solver returns an inspectable solution with a consistent observation interface, without exposing a different unrelated field class for every backend.

The physical layers are:

| Layer | Initial supported work and boundary of the claim |
|---|---|
| Homogeneous propagation | Full-vector plane spectrum and ASM; paraxial Fresnel as a separate approximation. Preserve correctly initialized longitudinal components and stable local phase origins. |
| Planar scattering | Per-mode vector Fresnel reflection/transmission, oblique planes, phase placement, TIR and transmitted evanescent waves. |
| Curved scattering | Local planar-interface approximation applied to each incident mode or explicitly supplied phase; keep returned approximate boundary traces and their model provenance. |
| Radiation | E/H evaluation from equivalent currents by direct Green quadrature, spectral synthesis, or a bounded local expansion. The current model and numerical evaluation error are distinct. |
| High-frequency transport | Separate phases, ray-tube Jacobians, geometric phase and Fresnel encounters. Explicit restrictions near caustics, grazing, vignetting and unsupported TIR. |
| Multiple interactions | Stable planar layer scattering first; supported curved-interface feedback with explicit geometry/branch restrictions and physical validation. |
| Observation | Common electric/electromagnetic evaluation, detector power, polarization and imaging quantities. |

The ordered forward chain “propagate, scatter, propagate” remains a useful solver, but the full physical assembly is a graph when reflection and coherent feedback matter. Keep `SequentialSolver`, a planar layered solver, and supported coherent-interaction solvers distinct. An iterative residual certifies convergence of the chosen discrete equations, not that those equations describe a complete Maxwell boundary problem.

Current limitations are part of the design: separated graph surfaces, propagating-mode-only coherent curved assembly, and restricted high-frequency surface families are not arbitrary closed-body scattering. Future boundary-integral or FEM reference work fits beside the current models; it is not an implicit capability of a class named `GeneralSolver`.

[Song et al.](https://doi.org/10.1088/2515-7647/ae0384) motivates an explicit full-vector projection and planar spectral-interface treatment. [Shi et al.](https://doi.org/10.1364/JOSAA.36.001252) distinguishes local planar-interface modeling and tests its limitations against full-wave calculations. The architectural inference is to keep projection, propagation, boundary approximation, and radiation separate and name their assumptions.

Macroscopic acceleration remains available inside the general library as a distinct model/representation choice. A local spectral expansion's bound can concern radiation from given discrete currents without bounding the approximation used to obtain those currents. A sum of incident modes must not silently become one phase inferred from the summed complex field.

**6. Validation has three separate questions**

Every major benchmark should answer: (a) did the numerical approximation converge, (b) does the converged field satisfy the physical problem, and (c) did the migration preserve the intended capability? These questions can have different answers. Fast and slow evaluators sharing the same approximate currents answer only part of the first question.

Each fixture records geometry, material sides, source including vector polarization, wavelength/units, phase origin, aperture meaning, observation coordinates, component basis, approximation, normalization, code revision and reference provenance. Compare complex E and H in a fixed convention. Do not fit an arbitrary gain or phase to make an acceptance test pass. A known analytic change of reference is converted explicitly; fitted phase/scale and normalized image profiles are secondary diagnostics only.

**A. Geometry, coordinates and representation**

Check surface tangents, normals, area Jacobians and oriented adjacency against independent geometry. Integrals should be invariant under a change of surface parameterization. Translate and rotate the entire physical problem and transform observations and vector components accordingly. Power is unchanged by these rigid motions. Verify chart poles and coordinate-axis degeneracies without confusing coordinate singularities with field singularities.

Check that grid changes produce correct coordinates and vector bases; no stale normal/weight caches remain. Compare direct Fourier sums, FFT/NUFFT and Hankel quadrature with explicit normalization and weight conventions. Distinguish continuous spectral densities from modal coefficients already multiplied by quadrature weights. Preserve stable phase anchoring under large translations. Fourier Parseval identities check transform normalization but do not alone certify electromagnetic power conservation.

**B. Bulk Maxwell equations**

For source-free homogeneous regions, evaluate all four Maxwell residuals using both modal identities and independent real-space differentiation at held-out points. Normalize curl residuals by the corresponding k-scaled field magnitude, with a declared floor near zeros. Refine the differentiation scale separately from the field discretization. Use complex bilinear k dot E=0 for individual modes, not a Hermitian dot product.

Analytic cases include oblique propagating waves, evanescent decay, forward/backward superpositions, and standing waves. Test superposition L(aE1+bE2)=aL(E1)+bL(E2) for fixed linear optical models, with correct coherent cross terms in observables. Check homogeneous propagation composition for supported forward branches; do not demand stable inversion of exponentially decaying evanescent waves.

A geometric/eikonal approximation generally has nonzero finite-wavelength Maxwell residuals. Study their asymptotic behavior; do not weaken the meaning of “exact” until that model passes by construction.

**C. Dielectric interface conditions**

For no free surface charge or current and a normal from side 1 to side 2, check:

```text
n cross (E2 - E1) = 0
n cross (H2 - H1) = 0
n dot (epsilon2 E2 - epsilon1 E1) = 0
n dot (mu2 H2 - mu1 H1) = 0.
```

E1 and H1 are the total fields, including incident and reflected waves and any coherent returns. E2 and H2 likewise include all relevant branches. Checking continuity of normal E itself would be the wrong dielectric test.

Perform two distinct checks. First verify the local Fresnel algebra. Then reconstruct the propagated total fields on both sides at held-out positions approaching the interface. Extrapolate the complex E/H values toward zero offset before taking residual norms. Use controlled singular/near-singular quadrature or trace jump formulas where appropriate; evaluating an ordinary singular kernel at its own source node is not a valid boundary test.

Refine surface quadrature and approach offsets independently. Report weighted L2 and maximum component residuals, a fixed incident-reference normalization, and local normalization with a stated floor. Report interior and edge regions separately. A low residual at one selected point is not a uniform surface guarantee.

Controls: zero index contrast without a physical stop, flat-interface limit, growing numerical integration support, localized illumination, and aperture-edge scans. They separate boundary approximation from missing support and edge modeling.

**D. Energy flow and reciprocity**

Use S=(1/2)Re(E cross H*) and properly oriented surface quadrature. For lossless planar interfaces, check R+T=1 for s and p, including the refractive-index and directional-cosine factors. Compare independent integrated incident, reflected and transmitted fluxes for finite beams. For a source-free lossless closed control surface, check integral S dot n dA=0. For finite observation windows include side flux, unapertured branches and omitted detector collection explicitly.

Power removed by a mask or stop must be recorded as removed power or handled by an explicit scattering/absorption model. Renormalizing the output cannot repair a failed energy test. Lossless separated branch bookkeeping should agree with the total-field control-surface result where both descriptions apply.

A single decaying evanescent plane wave has no net normal time-averaged flux in the lossless half-space, but a superposition across a gap can transmit power. Test frustrated total internal reflection with the complete E/H solution and interference terms. Standing-wave tests prevent inferring flux from |E|^2.

For reciprocal dielectric problems, compare appropriately normalized reciprocal source/receiver experiments. Compare scattering matrices in a consistent power-normalized propagating-mode basis, with phases and channel directions accounted for. A finite-aperture incomplete channel set need not itself be unitary. Do not apply a naive equal-E transmission symmetry across unequal-index media.

**E. Analytic planar and layered references**

Normal and oblique incidence; independent s and p amplitudes/phases; Brewster cancellation; matched indices; reversed incidence; translated/rotated interfaces; TIR phase and evanescent decay; limits toward critical incidence and grazing. Exact degeneracies should either have a well-defined treatment or be explicitly rejected, not patched by arbitrary epsilons.

For layers compare with independent analytic Airy/slab formulas, including all reflected orders, resonances, thick evanescent gaps, internal standing waves and boundary traces at every face. Stable scattering recursions must remain stable where naive transfer matrices overflow. Test homogeneous limits obtained by removing an index contrast or reducing a layer thickness.

**F. Stigmatic geometry and transfer**

Check the constant optical path for the supported real conjugate branch, Snell's law, ray intersections and normal orientation throughout the aperture. Quantify allowed aperture and proximity to grazing independently from the optical numerical aperture. Explicitly exclude unsupported virtual branches from claims.

Compare the reference vector transfer with its cylindrical and Cartesian matrix forms, including every complex component. Check local ray transversality, geometric longitudinal sign, polarization-basis orientation and on-axis limits. Independently integrate ray-tube power on G, the interface, and G', applying Fresnel splitting at the interface. This tests A(Q) and Jacobians without normalizing away errors.

Verify perspective/sine mapping measures and their physical meanings. A mapping invariant is not a proof that replacing a sphere by a plane is an exact wave propagation operation. Include multiple supported elements only with their own verified conjugacy conditions.

**G. Stigmatic focal diffraction**

Compare direct vector angular quadrature with the Hankel and FFT paths under identical model assumptions. Include x/y linear, circular, radial and azimuthal input polarization; azimuthal harmonics where supported; multiple NA values; focal and defocused planes. Check all complex Ex,Ey,Ez and derived H, absolute scale, phase and symmetry. Investigate the observed legacy helper Ez sign discrepancy before making it a reference.

Compare paraxial Fresnel and Debye calculations only in their common limit, tracking phase and amplitude error as NA changes. Compare Debye with finite-distance prescribed-current radiation over wavelength/geometrical-scale sweeps. Their disagreement outside an approximation's limit is not automatically a code defect. A reference ideal aplanatic objective is only a comparison to that optical model, not to every real refracting surface.

**H. General curved scattering and high-frequency transport**

Map boundary, field and flux error versus curvature/wavelength, incidence, NA, index contrast, illumination localization, aperture and observation distance. Include zero contrast and flat controls. Compare different surface parameterizations of the same physical geometry.

For transport, compare separately transported modes with direct Green propagation of the same preceding currents; then compare any single-phase compression against that multimode calculation. Refine the ray-tube derivative step separately from radiation quadrature. Record failure near caustics or shadow boundaries as a model limit, not a convergence success.

Comparison against the same currents establishes propagation accuracy only. To establish dielectric-scattering accuracy, use independent full-wave data for compatible geometries. A closed sphere permits a Mie reference once the implementation actually supports that closed-body problem; a current one-way open surface is not such a solver. Selected FEM/BEM reference cases would extend this to actual curved optical interfaces and internal resonances. Those are future reference tasks, not validations claimed to have been completed.

**I. Independent cross-library checks**

Describe the same supported stigmatic optical problem separately in each library. Match the source's actual complex vector field, reference surface, materials, aperture, physical coordinates and phase convention. Different objects both called “point source” are not automatically equivalent.

Compare full E/H, signed power, boundary residuals and focal observables over a parameter grid. Include both the already matched physical-optics-current case and comparisons where the models deliberately differ. Agreement between libraries is valuable preservation evidence; it is not independent proof of a boundary model both share conceptually.

**J. Observations, imaging and preserved uses**

Keep polarization maps, ellipse glyphs, field animation and component plots independent of propagation internals. Validate analytic linear/circular/elliptical inputs and the chosen view/basis convention. At high NA, transverse Stokes plots describe a declared projection; they do not replace the full 3D field.

Preserve coherent field summation and incoherent intensity/power addition as different experiments. A PSF/convolution model needs an explicitly stated isoplanatic approximation. Source-dependent imaging through a real system remains available separately. Verify physical magnification, focal grid coordinates, detector orientation/area and power collection. FWHM and normalized images are useful observables after complex-field correctness is established.

**7. Metrics, acceptance and existing evidence**

Use separate dimensionless weighted errors for E and H, and report a combined error only after scaling by a stated impedance. Do not add quantities of incompatible units. Use absolute-error measures with a fixed incident scale where a reference component vanishes. Each dataset specifies whether statistics are pointwise, weighted spatial norms, surface maxima or global power balances.

Initial numerical targets for well-conditioned complex128 benchmarks can be:

| Check | Initial target, to calibrate for conditioning |
|---|---|
| Analytic modal/vector algebra | Relative or appropriately scaled absolute residual around 1e-12. |
| FFT/direct finite-mode synthesis and homogeneous identities | Around 1e-10 after matching conventions and support. |
| Reference quadrature/differentiation for a 1e-3 physical target | Numerical uncertainty no more than 1e-5 in the same normalized observable. |
| Approximate curved/transport models | A benchmark-specific error budget over a declared parameter domain; for a first demanding qualification set, propose 1e-3 field, flux and boundary targets separately. Cases that fail remain documented failures or a narrower validity domain. |

These are proposed qualification targets, not claims that every current model meets them. A looser application-specific target can be legitimate when stated before evaluation. Never raise a threshold merely because the result failed. Numerical uncertainty, source approximation, boundary modeling, propagation approximation and detector truncation require separate estimates; a spectral truncation bound does not cover all five.

Already measured in this audit:

| Evidence | Consequence for this plan |
|---|---|
| Six snapshot test suites pass in complete environments (128, 176, 196, 129, 145 and 176 tests). | Useful preservation baselines, not a physical accuracy certificate. |
| Matched modern and legacy Green fields differ by approximately 5e-15 to 1.5e-12 over four wavelength cases. | Strong evidence that the particular prescribed-current stigmatic calculation is reproducible by the general engine. Keep independent implementations and compare through fixtures. |
| Legacy Debye comparison helper has a longitudinal sign discrepancy; vector error stays near 40%, despite excellent normalized transverse image agreement. | Resolve convention in independent reference tests; do not infer vector accuracy from image shape. This does not indict every production stigmatic routine. |
| R/lambda=10 boundary probe: curved truncated plane illumination 5.95%, flat control 7.62%, localized illumination 0.110% on the fixed reference scale (0.208% locally). | Converged quadrature is insufficient. Test support/edge effects and reconstructed boundary conditions independently; these numbers are not uniform error bounds. |
| At R/lambda=100 and 200, multimode transport errors are 0.0883% and 0.0442%, while a single-phase approximation gives 25.9% and 13.0%, at five observation points. | Keep phase decomposition explicit. Do not infer one ray direction from an interference field. |
| A translated-plane propagation overflow in the ontology code is absent in the macroscopic branch. | Preserve local phase references and add coordinate-translation stability cases. |

Raw results and scripts remain in this audit directory. Most proposed acceptance checks above have not yet been executed as a comprehensive qualification suite.

**8. Implementation sequence for a later coding session**

1. Freeze preservation specifications, without deleting branches. For stigmatic behavior use `7a1a235` and its unchanged copy in combined `b12a262`; use `b12a262` for corrected older general comparisons and `5b8e6f3` for current general/macroscopic capabilities. These are locally available revision anchors, not a claim about unfetched remote changes.
2. Settle the written conventions and correct the independent reference discrepancies first. Build benchmark data with all three complex E components, H, physical coordinates and power. Keep old behavior data and physically corrected references distinguishable.
3. Create independent project skeletons only when implementation is authorized. Establish the grid/field/material ownership rules in each, with independent installation and no cross-import dependency.
4. Build one complete example in each package. Stigmatic: reference sphere -> transfer -> focal field -> power and polarization. General: embedded plane source -> ASM -> dielectric plane -> reflected/transmitted observation -> power. Run analytic and preservation checks before broad migration.
5. Migrate stigmatic focal methods, pupils and useful visualization workflows within the stigmatic project. Resolve the Ez helper convention through the independent angular-integral reference. Keep geometric, Debye and Fresnel assumptions visible.
6. Migrate general layers, currents, curved interfaces, stable spectra and macroscopic transport incrementally. Preserve each capability's actual validity restrictions and rerun the audit's boundary and transport controls. Do not erase methods to make the API look uniform.
7. Qualify supported optical regimes with independent reference problems, boundary checks and power budgets. Add imaging and multi-element examples only with their documented assumptions and verified grids. Publish a capability/accuracy matrix alongside examples.
8. Consider sharing code only in a separate future decision, after independent implementations expose truly stable common contracts and debugging no longer benefits from duplication. No unification or branch retirement is part of this plan.

Readability rules throughout: equations close to their implementations; meaningful geometric names; explicit units, shapes, bases, branch directions and approximation assumptions; small functions corresponding to derivation steps; separate source construction, propagation and plotting; immutable inputs and inspectable diagnostics. Avoid deep inheritance, mutable basis caches, hidden normalization and a single magical propagation method that silently changes the physical model.

The first deliverable is therefore two comparable, independent optical calculations with clear physical contracts and meaningful failure diagnostics. Generality will grow through validated models, while the stigmatic calculation remains easy to read, debug and use.
