**vecdiff: repository audit and proposed physical architecture — 15 September 2026**

> Historical review: the general propagation path has since been replaced by
> [explicit R/Q/T composition](OPERATIONAL_PROPAGATION.md). Descriptions below
> of ray-tube or Green-based primary evaluation describe the earlier design.


My recommendation is to retain the macroscopic branch's working numerical developments, restore the stigmatic workflow as an explicit specialization, and give both a shared vocabulary for geometry, materials, fields, and observations. The next step should be a small, behavior-preserving architectural slice, preceded by correcting the validation blind spots described below. A wholesale rewrite or branch merge would make it harder to know what was preserved.

This is an analysis artifact. No repository file, branch, worktree configuration, or Git reference was changed. Existing uncommitted changes were present before the audit; the final porcelain status matches the initial status. All numerical work used Git archive snapshots in this directory. Branch conclusions refer to the locally available remote-tracking references; no fetch was performed. Notebook purposes and sources were inventoried; I did not rerun every historical notebook or certify every documented application.

**What exists and where to preserve it**

| Line of development | Inspected revision | Finding |
|---|---|---|
| `claude/transfer-matrix-geometric-factor-664h9g` | `7a1a235` | Latest dedicated stigmatic branch among the available references. Cartesian-oval geometry, reference spheres, geometric amplitude factor, sine pupil mapping, Hankel/FFT paths, polarization views and animation. Also contains a useful cylindrical TE/TM spectral-versus-local-incidence experiment. |
| `claude/repository-theory-94zcp4` | `4669440` | Earlier general `vecdiff.wave` implementation alongside the old API. Useful operator composition, pupil, wavefront, and imaging concepts, but contains physical defects corrected later. |
| Last combined implementation before the breaking refactor | `b12a262` | Particularly valuable preservation anchor: the dedicated stigmatic implementation is unchanged from `7a1a235`, while the general implementation has later normal-orientation, per-mode superposition, reflection, TIR-phase, and transversality corrections. |
| `refactor/physical-ontology` | `2701457` | General field/spectrum, placed parametric surfaces, exact plane boundaries and coherent layers, local curved Fresnel traces, Green radiation, diagnostics. It is not restricted to stigmatic geometry. It removed the former stigmatic API and many application workflows. |
| `feature/spectral-interactions` | `b5d93c3` | Adds coherent interactions for ordered separated graph surfaces, cylindrical synthesis, and scale/convergence studies. These numerical additions are also present in the macroscopic line. |
| `feature/macroscopic-spectral-fields` | `5b8e6f3` | Richest current numerical implementation: stable plane propagation, bounded local radiation spectra, prescriptions, aspheres, near-surface quadrature, field-dependent studies, and explicit high-frequency transport. Strongest starting point for future implementation, after separating its concepts more clearly. |
| `main` | local `96038de`, tracked remote `8fe2145` | Ontology generation. The remote's additional commit removes workflow/configuration documentation, not new optics. |
| Older roadmap and merge branches | `2ee5743`, `bca5755` | Ancestors of the later implementation; preserve their design notes, particularly virtual-conjugate/sign debt. They are not additional independent current engines. |

The physical, spectral, and main branch tips contain separate GitHub Actions removal commits. Those tips therefore are not all literal ancestors of the macroscopic tip, even though their preceding numerical developments are inherited. Branch count overstates the number of independent implementations. The local `remove-actions-*` branches are checkout/maintenance work, not optical families. No deletion is recommended during this design phase.

Two premises need correction. `ElectricField` still exists in `macroscopic/vecdiff/fields/electric_field.py`, with the same basic fields as the ontology branch. The issue is that it is one of several unrelated public return representations, and still couples a field sample to `sampling`, `domain`, and `medium`. Also, the ontology refactor contains `FreeformSurface`, `Sphere`, `SphericalCap`, placed frames, and general surface radiation. Its regression was removal of useful specialized workflows, not restriction of all physics to Cartesian ovals.

The migration document explicitly records removal of the stigmatic wrappers, older visualization/animation, several polarization studies, two-diopter imaging, and lithography examples. The architectural test even forbids selected historical class names. Those are previous design choices, not requirements that should overrule preserving the capabilities you value.

**Tests and numerical experiments performed**

| Snapshot | Final suite result |
|---|---:|
| Stigmatic `7a1a235` | 128 passed |
| Earlier general `4669440` | 176 passed |
| Last combined `b12a262` | 196 passed |
| Ontology `2701457` | 129 passed |
| Spectral interactions `b5d93c3` | 145 passed |
| Macroscopic `5b8e6f3` | 176 passed |

The stigmatic suite ran on system Python 3.14.7. The complete other suites used the existing Python 3.12.12 environment at `/private/tmp/vecdiff-notebooks-venv`, which includes FINUFFT, miepython, and notebook dependencies. Initial runs without those dependencies produced two optional skips for ontology, notebook collection errors for the newer branches, and two missing-FINUFFT failures for the earlier general branch. Those were environment limitations; the complete runs above resolved them. Historical plotting suites emit warnings. Raw logs are retained alongside this report.

Passing suites are useful preservation evidence, but their assertions do not establish equal physical accuracy. Many historical comparisons use normalized magnitudes, peak ratios, or broad tolerances.

I ran a matched Cartesian-oval comparison with `n1=1`, `n2=1.5`, object `z=-10`, image `z=6`, aperture `0.95` times the old grazing limit, and image NA `0.8001855`. Both calculations used the same global coordinates, source amplitude and phase, aperture, and 61 focal observations over `x/lambda0` from -2 to 2. Surface quadrature was refined from 160×48 to 320×96 nodes.

The modern route was `CartesianOval → EikonalElectricField → propagate_high_frequency` for one encounter, followed by full `SurfaceRadiation.evaluate`. The source was deliberately matched to the old `PhysicalOpticsCurrents`; it was not substituted with the differently polarized point source from the newer finite-conjugate example. This comparison tests agreement for the same physical-optics source and Fresnel-current model. It does not turn that source model into an exact finite-wavelength Maxwell point source or certify a dielectric boundary solution.

| `lambda0`, in the geometry's length unit | Modern vs old Green: relative complex E difference | Legacy Debye helper: normalized `abs(Ex)` RMS difference | Relative complex E difference after reversing only helper Ez, as a diagnostic |
|---:|---:|---:|---:|
| 0.5 | 5.01e-15 | 1.516e-3 | 7.511e-2 |
| 0.1 | 6.52e-15 | 6.509e-5 | 1.521e-2 |
| 0.01 | 7.00e-14 | 6.518e-7 | 1.521e-3 |
| 0.000532 | 1.51e-12 | 1.845e-9 | 8.094e-5 |

The unchanged legacy Debye comparison helper has a longitudinal sign discrepancy: its `Ez` approaches minus the Green result. The measured vector difference remains about 40.4% at the macroscopic end, despite the excellent transverse magnitude agreement. `stigmatic/examples/reference_vs_model.py:76` and `theory/vecdiff/wave/stigmatic.py:151` use `+1j * prefactor * h1`. For the converging-ray convention in their own Green reference, the azimuthal integral `integral cos(phi) exp(-i a cos(phi)) dphi = -2 pi i J1(a)` gives the opposite sign. This finding concerns those comparison helpers; the production Hankel method generally returns only the transverse pair, so it would be inaccurate to label every old propagation path as having the same Ez bug.

The last column is an audit-only sign diagnostic, not a source-code modification or a fitted amplitude/phase correction. The remaining discrepancy decreases with wavelength, consistent with the Debye approximation approaching full radiation for the prescribed currents. Even `Ex` alone has a complex relative difference of 7.41e-5 in the final case, whereas its normalized magnitude profile differs by 1.84e-9. Historical approximately 1e-9 profile claims must not be restated as approximately 1e-9 absolute complex-vector accuracy.

See [the comparison figure](matched_oval.png), [exportable PDF](matched_oval.pdf), and [raw results](matched_oval_results.json). The modern and old Green agreement demonstrates a concrete route to sharing implementation beneath a retained stigmatic API.

I also reran the older `wave.referee` comparison. At `4669440`, normalized transverse profile RMS is 1.7765% and the peak ratio is 0.92705. At corrected combined revision `b12a262`, they improve to 0.1401% and 0.998985. The raw complex Ex/Ez comparison to its Debye helper is still contaminated by the helper's longitudinal convention. This is why `b12a262` is a better preservation source than the older theory branch and why normalized-shape tests alone cannot settle equivalence.

Three independently rerun, existing two-sided boundary diagnostics on the macroscopic code give:

| Case, `R/lambda0=10`, aperture `R/2`, probe `rho/R=0.2` | Largest of four boundary residuals |
|---|---:|
| Curved surface, plane illumination, indices 1 → 1.5 | 5.9506% |
| Flat truncated control, plane illumination, indices 1 → 1.5 | 7.6179% |
| Curved surface, localized 121-mode illumination, indices 1 → 1.5 | 0.11022% |

These are amplitude residuals normalized to the defined incident reference amplitude, not image errors or global bounds. The localized case is 0.20847% when normalized to the local incident amplitude at that probe. Quadrature changes were approximately 5e-13 and changes between offset extrapolations approximately 5.6e-6. Thus numerical convergence can coexist with a significant physical boundary mismatch. The flat control and the repository's equal-index control show that aperture truncation and omitted boundary information cannot be conflated with curvature alone. The recorded broader probe scan also loses uniform 1% local accuracy near the outer probe; I did not rerun that entire scan.

Two rerun high-frequency transport cases compared with full Green propagation from the preceding face give:

| Curvature radius / wavelength | 625 separately transported phases: complex E/H error | One Gaussian eikonal: complex E/H error |
|---:|---:|---:|
| 100 | 0.088265% | 25.9021% |
| 200 | 0.044170% | 12.9950% |

The direct-current controls converged below 2.1e-13. These are five-point inter-interface checks, not full-system error bounds. They justify keeping the macroscopic work, while treating single-phase compression as a separate approximation. A field does not become accurately representable by one phase merely because the software accepts an `EikonalElectricField`.

A final numerical preservation check matters architecturally: a Gaussian sampled on a plane at `z=50`, with wavelength `.532` and spacing `.08`, overflows during a forward step on the ontology branch because it rereferences evanescent coefficients to the global origin. The macroscopic branch propagates it successfully. Local phase origins are numerical representation data that must survive this redesign.

**Physical contracts to settle before class design**

Your Space–Matter–Fields distinction is a sound organizing principle. It needs several precise qualifications so that the abstraction remains valid beyond one-way homogeneous propagation.

1. In a source-free dielectric the general electric constraint is `div(D)=div(epsilon E)=0`. `div(E)=0` holds inside a homogeneous isotropic region, not across a spatially varying dielectric. Absence of free charge does not exclude bound polarization charge. With constant nonmagnetic permeability, the general monochromatic wave equation is `curl curl E - omega^2 mu0 epsilon(r) E = 0`. The independent scalar-component Helmholtz equations are the homogeneous-region reduction. See the [MIT treatment of spatially varying dielectric response](https://web.mit.edu/6.013_book/www/chapter6/6.7.html).
2. For the `exp(-i omega t)` convention, `H = curl(E)/(i omega mu0)` when E is known as a spatial solution with the required derivatives. Values of E on a single surface are not enough to obtain those derivatives in the general case. My explicit counterexample has identical E on `z=0` for a forward wave and an equal-amplitude counterpropagating pair, but different H and normal power flux. The library must retain propagation branches or equivalent trace/solution information for standing waves and resonances.
3. Recovering `Ez = -(kx Ex + ky Ey)/kz` is a plane-spectrum completion with a selected dispersion relation and propagation branch. It does not apply pointwise to an arbitrary curved surface. Grazing populated modes can make completion nonunique. Two coefficients per mode remain sufficient, but one global missing-z component is not a universal parametrization.
4. The correct homogeneous ASM multiplier is `exp(i kz d)`, with `kz = sqrt(k^2-kx^2-ky^2)` and the appropriate outgoing/decaying branch. The final square-root expression in your sketch is missing the squared transverse ratio. Equivalently it is `exp(i k d sqrt(1-(|k_perp|/k)^2))`. Fresnel propagation replaces kz by its paraxial expansion and must be labeled approximate.
5. Scalar propagation of Cartesian components is valid in a homogeneous region if the initial vector spectrum already satisfies Maxwell constraints. Linearity alone does not establish the reduction through spatially varying media or boundaries. There is no additional polarization mixing required in homogeneous space merely because the field has high NA; the initialization and boundary transformation must be consistent.
6. Completion and projection are different physical choices. For `k/k0=(.6,0,.8)` and prescribed `Ex=1,Ey=0`, completion gives `(1,0,-.75)`. Applying the transverse projector `I-kk^T/k^2` to the supplied three-vector `(1,0,0)` gives `(.64,0,-.48)`. Both are transverse, but they are different fields. An explicit zero must not mean an unspecified component.
7. Your two supplied papers support separate laws: Song gives a full-vector projector and planar k-space interface formulation; Shi separates incident propagation, the local planar-interface approximation, and outgoing propagation, and examines multiple-interaction/resonance limitations. Neither makes arbitrary curved scattering an exact local Fresnel problem. The inference for this API is to expose `complete`, `project`, `propagate`, and `scatter` separately. Sources: [Song et al.](https://doi.org/10.1088/2515-7647/ae0384), [Shi et al.](https://doi.org/10.1364/JOSAA.36.001252). The supplied local PDFs were read, including Song's published-version discussion of projection and planar validity.
8. A radiation formula can satisfy homogeneous Maxwell equations exactly away from its sources while radiating approximate dielectric boundary data. Local Fresnel identities, quadrature convergence, and global dielectric accuracy must remain separate claims.

The matched-source experiment and the standing-wave/projection probes are recorded in `physical_probes.json` and the scripts accompanying this report. A rotated-plane probe also confirms the existing transverse constructor means *global Ex/Ey*, not the two components tangent to the supporting plane: a normally propagating wave on a plane whose normal is global x encounters the global-kz completion restriction. This is a representation limitation to document and redesign, not a reason to weaken Maxwell checks.

**Proposed ownership of concepts**

| Concept | Owns | Does not own |
|---|---|---|
| Geometry: `Frame`, `Surface`, `Region` | Euclidean placement, shape, orientation, membership and topology | Optical wavelength, field samples, Fresnel coefficients |
| `Grid` | Sampling coordinates and their embedding into physical space; declared component frame; integration measure when supplied | Refractive index, field amplitudes, solver selection |
| `Material` | Constitutive response, initially real positive `n(lambda0)` | A particular field or mesh |
| `MaterialDistribution` / optical assembly | Assignment of materials to regions, background material, interface adjacency | Field coefficients or numerical tolerances |
| `Interface` | Oriented boundary between two regions and a boundary law | A fixed quadrature resolution or one permanently designated incident wave |
| `ElectricField` | Electric phasor samples on one Grid and vacuum wavelength | Medium ownership, a second sampling object, hidden propagation assumptions |
| Field representation | Spectral amplitudes, local phase origins, propagation branches, region of validity; optional cached evaluation data | New constitutive physics implied by a different class name |
| Propagation/scattering model | Applicable physical law, declared assumptions, and context connecting fields to material regions | Source/object choices disguised as a numerical backend |
| Numerical plan | FFT/NUFFT/direct evaluation, quadrature, tolerances, truncation controls | Changing the physical approximation silently |
| Optical system | Geometry, materials, apertures, and possible encounters | The wavelength of every future experiment or a mandatory single path |
| Experiment/solution | Source, wavelength, system, requested observations, method and diagnostics | Pretending a converged discretization proves the approximation exact |

There need not be heavyweight classes literally called `Space`, `Matter`, and `Fields`. Their separation should be visible in ownership and function contracts. Small concrete classes and a few protocols are sufficient.

A material and its spatial assignment are distinct: glass is constitutive data; a lens is a region occupied by that glass. This realizes your statement that the medium depends on position without making every sampled field duplicate a medium value. A homogeneous spectrum may still need its resolved dispersion relation and validity region as part of the *solution/representation context*. Removing `field.medium` must not erase the information required to compute k, H, and flux correctly.

An interface's two sides should be named geometrically, for example `minus_region` and `plus_region`. Incidence is chosen at an encounter; it is not an intrinsic property of the surface. That will simplify reverse encounters and later reflections, instead of repeatedly constructing reversed copies of `DielectricInterface`.

An evaluation exactly on a dielectric boundary also needs an explicit side or trace: normal E can jump there. Region selection should not depend on a floating-point coordinate comparison that silently chooses one material. This belongs in the solution/trace context, not a duplicated mutable material attribute on every field array.

An ordered `material — surface — material` prescription is an excellent optical-system builder. It is not by itself a complete three-dimensional material distribution. Finite surface patches require a declared rim/side-wall closure to delimit a lens body; different radii can admit different closures. A background material is explicit. A numerical integration radius and a physical stop must also be distinct. Otherwise a change in quadrature silently changes the optical system. Start with a well-defined two-face body builder and keep sequential prescriptions usable without claiming they are universal solid geometry.

**ElectricField and Grid, concretely**

The user-facing sampled container can stay close to your sketch:

```python
@dataclass(frozen=True)
class ElectricField:
    grid: Grid
    Ex: Array
    Ey: Array
    wavelength: float  # vacuum wavelength, explicitly supplied
    Ez: Array | None = None
```

This is proposed API, not an implementation. Global Cartesian components are a reasonable canonical storage convention. `grid` owns where samples live; the field should not separately own `domain` and `sampling`. The current constructor only checks component shapes and even accepts `object()` as the domain, so the intended association is not enforced.

A Grid should bind its chart/axes to a geometry and provide physical positions. A `PlaneGrid` can have regular x/y axes and a Frame; a `SurfaceGrid` can have `(u,v)` nodes and an embedding `r(u,v)`. Quadrature weights include the appropriate Jacobian and measure. Tensor-product sampling in parameters does not automatically qualify for a Cartesian FFT. Keep point sampling and integration capability distinguishable; an arbitrary point cloud does not have an intrinsic area weight.

Grid and geometry immutability avoid stale normals and weights. The current sampled arrays are protected, but many Surface objects remain mutable. Coordinate relabeling, rigid placement, resampling, pupil mapping, and physical propagation must be separate operations. For example, moving a grid to a new plane is not sufficient to propagate the field; interpolating onto a different grid is not a Fresnel boundary transformation. A sine-mapped pupil is a change of integration coordinates with a Jacobian, not an unlabeled physical planar field.

I would not introduce a broad `TransverseElectricField` inheritance hierarchy yet. Use either a named constructor for a missing component on an explicitly declared plane, or a small `TangentialTrace` value type when the known data are the two components tangent to a surface. If the convenience name is retained, it must not mean TE polarization and must declare its component basis and completion assumptions. `Ez=None` remains unknown; `Ez=0` is supplied data.

Keep Cartesian, circular, and cylindrical component conversions as views or explicit basis transformations of the same field. They should not create independently mutable field identities with stale cached copies of components. Preserve the useful old constructors as conveniences if desired.

Internally, keep several representations because they solve different numerical problems: a sampled plane, a weighted plane-wave sum, equivalent surface currents, a local expansion, and a smooth-phase envelope. Give them one small evaluation contract, such as `solution.sample(grid) -> ElectricField`, and explicit validity metadata. `LayeredElectricField`, `AssemblyElectricField`, `ElectricSpectrum`, and `HighFrequencyResult` currently expose different access patterns. A common result/sample interface can restore conceptual unity without forcing expensive conversion to a raster between encounters.

For quantities:

- `magnitude_squared()` or `norm2()` can mean `|Ex|^2+|Ey|^2+|Ez|^2`. If a component is unknown, demand an explicit transverse-only quantity or completion.
- `poynting()` requires a Maxwell solution or H; it must not guess H from an arbitrary plane trace.
- Measured normal intensity is detector-dependent: `S dot n_detector`. Integrated optical power is a surface integral.
- In a nondispersive lossless medium, time-averaged energy density is `(epsilon |E|^2 + mu0 |H|^2)/4`, and energy requires a volume. Integrating `|E|^2` over a focal plane is not an energy measurement.
- Prefer SI H at the public boundary, or give the existing `Z0 H_SI` normalization an unambiguous name. Current functions call it H, relying on docstrings to carry a crucial convention.

Wavelength belongs to the monochromatic experiment/field, not a mutable global setting. An experiment may provide an explicit default. Length units and phasor convention must be consistent across geometry, grids, IO, and observables.

**Propagation and specialization**

Use common inputs and results, with explicit physical models:

```text
geometry + material distribution + source
                  |
           chosen optical model
                  |
       representation and numerical plan
                  |
           solution.sample(grid)
                  |
     ElectricField + derived observables
```

There are at least four distinct model families worth keeping:

1. Homogeneous angular-spectrum propagation and exact infinite planar Fresnel scattering, with stable coherent planar layers. This is the best-validated foundation.
2. General curved-interface physical optics: per-incident-mode Fresnel transformation, equivalent-current radiation, and explicitly constrained coherent composition. Preserve linear superposition. Do not infer one direction from an unresolved coherent sum.
3. Smooth-phase/high-frequency inter-interface transport: separate supplied phases, ray-tube spreading, and final diffraction. Its rejection of caustics, critical transmission, and unsupported vignetting is preferable to silently using the approximation outside its domain. Preserve it, but expose its approximation in the method choice and result.
4. Stigmatic transfer and Debye focusing: explicit object/image conjugates, reference spheres, amplitude/Jacobian factors, and fast Hankel or FFT realizations. Keep this as a supported specialization with the familiar workflows. It can share geometry, Fresnel laws, fields, and observation interfaces with the general route without becoming a second independent ontology.

The first implementation need not add a generic meta-operator framework. Plain functions or small model objects can express these maps. Physical model selection and numerical backend selection should be separate: choosing NUFFT must not imply discarding evanescent content or replacing many phase components with one ray.

The current macroscopic assembly solver is for separated forward-oriented z-graph envelopes, a fixed transverse spectral lattice, and modes propagating in every region. Its coherent solve is useful, but it does not provide arbitrary closed-body scattering. The high-frequency path supports placed axisymmetric graphs and planes and forward transmission; it is not currently a general freeform folded-system engine. Prescription IO can describe mirrors/stops that the selected solver rejects. Preserve that distinction and make solver applicability queryable before an expensive run.

Virtual conjugates and mirrors should eventually use explicit physical ray orientation and boundary laws. Do not import the old design note's negative-index mirror convention as constitutive physics. The unimplemented sign cases are valuable recorded debt, not established functionality. The newer `CartesianOval` also explicitly limits its constructor to real object/real image design.

A possible package organization, kept deliberately modest:

```text
vecdiff/
    geometry/        # frames, surfaces, regions, closures
    grids/           # embedded grids and quadrature
    materials/       # constitutive data and spatial assignments
    fields/          # ElectricField, bases, explicit traces
    representations/ # spectra, currents, phase envelopes, local expansions
    propagation/     # homogeneous ASM and paraxial Fresnel
    interfaces/      # dielectric/reflector laws
    solvers/         # layers, curved physical optics, high-frequency systems
    stigmatic/       # conjugates, sphere transfer, Debye, familiar workflows
    systems/         # bodies, prescriptions, apertures, experiments
    observables/     # polarization, Poynting flux, image quantities
    visualization/   # plots and harmonic animation
    numerics/        # Fourier/Hankel/NUFFT and linear algebra
    io/              # external formats and units
references/          # independent validation implementations
```

This tree is a dependency proposal, not a requirement to create every directory immediately. In particular, avoid placing a class in `fields` solely because its name includes ElectricField; solver state and a field value have different responsibilities.

**What should survive the redesign**

| Area | Preserve | Clarify or improve |
|---|---|---|
| Stigmatic geometry and transfer | Cartesian-oval design, reference spheres, aperture/grazing budget, geometric factor, pupil Jacobian, FFT/Hankel alternatives | Complex Ez convention, physical output coordinates, normalization, real/virtual domain restrictions |
| Polarization and visualization | Basis conversions, Stokes/ellipse views, component maps, animation, radial/azimuthal/circular inputs | Derive from one canonical field; distinguish a selected transverse plane from 3-D polarization |
| General boundary physics | Per-mode vector Fresnel, exact planar phase placement, TIR, bilinear complex-k transversality | Separate immutable interface geometry from encounter orientation and model applicability |
| Layers and feedback | Stable all-orders scattering recursion, evanescent gaps, convergence reporting | Keep explicit branch/region context; never reconstruct standing-wave H from a single E trace |
| Numerical kernels | Physical Fourier normalization, arbitrary origins, harmonic tail checks, direct/NUFFT comparisons | Do not conflate spectral coefficients including weights with continuous densities |
| Macroscopic methods | Local phase anchoring, bounded local spectra, target-centred quadrature, ray-tube transport, prescription IO | Uniform field/result sampling API; approximation and validity expressed explicitly |
| Imaging | Complex vector response, coherent/incoherent source handling, source-dependent calculations | Distinguish local isoplanatic convolution from propagation through the real system; preserve examples with their actual scope |
| Validation | Independent references, Maxwell/flux checks, numerical refinements, failure cases | Test complex components and supported behavior; remove architecture assertions that merely mandate retiring old names |

The old code is more expansive and often easier to follow at the formula level, but has mutable state, implicit reference assumptions, and duplicated representations. The new code has stronger checks and clearer module intent, but compresses equations into dense statements and exposes many unrelated representation types. For example, `high_frequency.py` puts most of a geometric transport algorithm in short identifiers, packed statements, and positional tuples. Improve readability by naming geometrical quantities, documenting shapes and units at boundaries, and separating steps corresponding to derivation equations. Do not replace every mathematical `k` with prose; distinguish conventional symbols from opaque variables whose roles change within a function.

**Proposed implementation sequence, for a later authorized session**

1. **Record preservation contracts.** Use `b12a262` and `5b8e6f3` as numerical anchors. Select a manageable set of representative workflows: stigmatic x/radial/circular focal fields, general plane/curved incidence, layered/TIR behavior, two-surface transport, and polarization visualization. Store physical inputs, absolute complex E/H, flux, and validity assumptions. The design milestone is a capability matrix, not removing branches.
2. **Resolve the reference discrepancies.** Add narrowly targeted complex-Ez comparisons and correct the comparison helpers' convention in a later code change. Separate normalized-shape, absolute-amplitude, phase, source approximation, and boundary-accuracy claims. Reconcile scalar/reference meanings before using any branch as the arbiter of all others.
3. **Agree the minimal contracts.** Decide Grid embedding, canonical component basis, vacuum wavelength and units, material/region ownership, explicit completion versus projection, and what information a solution retains to derive H. A short runnable API example should express the same problem using the general and stigmatic paths. Do not begin with a wholesale file move.
4. **Implement one vertical slice.** PlaneGrid + ElectricField without medium/sampling duplication → homogeneous ASM → exact planar interface → output Grid → flux. Reuse the existing kernels. Acceptance: old and new complex fields agree, tilted/translated grids behave correctly, evanescent forward propagation stays stable, and the API never infers a missing propagation branch silently.
5. **Restore the stigmatic specialization on the shared objects.** Keep its conjugate design, pupil construction, fast focal evaluation, and visual workflows. Match the physical-optics source between the two routes and compare every complex component over numerical aperture and wavelength/curvature scale. Carry forward documented limitations instead of restoring overbroad claims.
6. **Adapt the general curved and macroscopic engines.** Give spectra, currents, layered solutions, and high-frequency results a shared observation contract. Preserve local phase anchors and bounded validity. Reproduce the boundary and high-frequency controls in this audit before accepting the migration. Add one general two-surface displaced-source example alongside the stigmatic example.
7. **Move useful experimental workflows into supported modules selectively.** Imaging, detector choices, source coherence, and polarization visualization deserve explicit modules. Keep closed-body resonances, arbitrary folds, and unimplemented material models clearly outside the validated domain until independently established. Only after capability equivalence is demonstrated should branch retirement be discussed.

For future coding I would start a new branch from the macroscopic line and port the selected stigmatic workflows from the combined anchor deliberately. I would not merge old branches wholesale: their code represents incompatible public models, and a successful Git merge would not establish physical equivalence. This session created no such branch and performed no port.

The first design decision I recommend adopting is: **ElectricField is sampled electric data on one physically embedded Grid; material belongs to the spatial system; a solution representation supplies the extra information needed for propagation and derived electromagnetic quantities.** That preserves the clarity of your proposed object while avoiding the false assumption that an arbitrary E trace determines every other field quantity.

**Reproducing the audit**

`code_inventory.json` records files, definitions, and notebook topics across the snapshots; `branch_inventory.json` records inspected branch hashes and ancestry. The scripts in this directory use snapshots, not the working checkout. Scripts intentionally prepend their selected snapshot to `sys.path`.

Use the existing interpreter with `PYTHONDONTWRITEBYTECODE=1`, `MPLBACKEND=Agg`, `MPLCONFIGDIR` pointing to this directory's `mpl`, and `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. Run `export_stigmatic.py` before `compare_modern_stigmatic.py`, then `plot_audit.py`. `old_general_referee.py` accepts `theory` or `combined`; `rerun_boundary_controls.py`, `rerun_transport.py`, `physical_probes.py`, and `phase_origin_probe.py <ontology|macroscopic>` produce the other results. Suite commands are `python -m pytest -q -p no:cacheprovider` from the relevant snapshot.

These experiments are sufficient to guide the proposed architecture and identify concrete preservation risks. They are not a full independent validation of every surface, source, notebook, optical instrument, or material regime represented in the repository.
