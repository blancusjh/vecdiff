# Propagation

The public algorithms are reached through `vecdiff.propagation`;
the package top level re-exports nothing.

| Module | Main API | Operation |
|---|---|---|
| `transport/` | `InterfaceMap`, `LatticeTransport`, `MeridianTransport`, `TransportReport` | Carry a field across a surface along its carrier's rays |
| `propagation/homogeneous.py` | `propagate`, `PlaneTransport` | Evolution in a homogeneous medium |
| `propagation/propagator.py` | `Propagator`, `Solution` | The method: light through a system, E and H on any target |
| `fields/` | `Spectrum` and its kinds | The field of one region, as plane waves referred to a plane |
| `alternatives/wave_propagator.py` | `WavePropagator` | Wave transport through a system, the same `Solution`, for checks |
| `alternatives/surface_transfer.py` | `SurfaceTransport` | Evaluate and reconstruct fields on a graph surface |
| `alternatives/surface_interaction.py` | `scatter_plane`, `SpectralFresnel`, `LocalFresnel` | Reflection and transmission at fixed boundaries |
| `references/` | `MieSphere`, `LayeredSphere`, `InfiniteCylinder` | Independently check physical and numerical accuracy |

## Homogeneous propagation

`angular_spectrum(field, region)` constructs a `PlaneSpectrum`. ASM propagation
uses exact homogeneous dispersion. It can retain evanescent modes, subject to
the chosen branch and propagation distance. Paraxial Fresnel propagation uses
an approximate longitudinal phase. `PlaneTransport.spectrum` preserves a modal
representation for subsequent operations; `spectrum.propagate` evaluates E/H
on the translated FFT plane; `magnetic=False` (also for `propagate`) returns E
alone without forming H.

## Surface transfer

`SurfaceTransport(plane, surface, region, wavelength0)` constructs a finite-chart
Q. `forward(spectrum)` evaluates the homogeneous field on the surface. The
Chebyshev/Bessel phase separation preserves each retained mode's longitudinal
phase. Compact bandwidth-adapted plans accelerate the direction-resolved
interaction without changing those modal phases.
When dense precomputed terms would exceed 64 MiB per plan, the same expansion
is streamed. Compact Bessel factors have a separate 512 MiB per-plan limit;
full-Q factors are computed lazily and reused across E/H operations, avoiding
unused plans and repeated special-function evaluation on large charts. Band plans
remain subject to their aggregate cache budget. The angular
band, separation rank and tolerance are unchanged. This controls plan storage
as charts grow; arithmetic work still grows with grid size and separation rank.

There are two distinct reconstruction operations:

- `inverse(trace)` fits the surface-tangential electric field in physical
  surface L2 using the declared forward angular band. It returns the recovered
  spectrum and an `InverseReport`, including the magnetic mismatch.
- `radiate(trace)` reconstructs outgoing modes from both tangential E and H by
  surface equivalence. Finite chart coverage, quadrature and spectral truncation
  determine its error. This is the reconstruction `WavePropagator` uses.

Neither operation is obtained by replacing the sag with its negative. The
least-squares inverse is also different from the adjoint used in radiation.
The current Q retains forward propagating modes within `max_sine`; it does not
inherit the complete evanescent spectrum of the independent reference integrals.

## Surface interaction

`scatter_plane` solves a single planar dielectric boundary mode by mode,
returning reflected and transmitted expansions. It retains transmitted
evanescent fields under total internal reflection; incident evanescent waves
and exact grazing/critical limits require another treatment.

`SpectralFresnel.transmit` applies ordinary tangent-plane Fresnel
transmission separately to incident angular components before summing them
coherently. It interpolates the directional response, while Q retains each
mode's exact phase. It does not infer a single direction from the summed field.
The result contains the E/H trace and a report of retained modes, omitted modal
L2 and separation ranks.

For a controlled comparison, `SpectralFresnel(coefficients="unity")`
sets the electric amplitudes to `t_s = t_p = 1` at each interaction. The default
is `coefficients="fresnel"`. Both choices retain Snell refraction, polarization
rotation and vector E/H transport. The unity choice isolates Fresnel weighting;
it is neither an energy-conserving dielectric boundary model nor a separate
scalar Helmholtz solver. [Notebook 05](../general/notebooks/05_single_lens_lithography.ipynb)
runs this vector unity-transmission control and Fresnel transmission through
the same public pipeline and high-NA immersion interface. An additional
polarization control preserves incident angular power. It compares
dense-pattern contrast, occupied output NA, detector throughput and focal
polarization at identical input power, with sampling, padding and angular
interpolation checks.

Defaults are 18 interpolation nodes per axis in the central patch, 6 in outer
patches, transverse direction limit 0.65 and relative modal cutoff 1e-7.
These are validated benchmark settings, not universal error bounds. Complex
transmitted directions are retained by the local symbol under TIR. The
back-facing continuation is part of the forward approximation; shadowing and
returning branches are not solved.

## The propagator and its solution

`propagation.Propagator(system)` is the method ([eikonal transport](#eikonal-transport)).
Called with the light it returns a `Solution`; the system owns geometry and
media (optionally the incident field), the call owns the light, and the
Solution answers every observation.

```python
import vecdiff as vd

propagator = vd.propagation.Propagator(system)  # planes=, fans=, realization=, lattice=
solution = propagator(beam)                     # or (sampled_field, carrier=c)
print(solution)                                 # one line per refracting surface
focus = solution.field(z=f, window=(-a, a, -a, a))     # E and H together
meridian = solution.electric(y=0, window=(-a, a, z0, z1), side="plus")
S = vd.observables.poynting(focus)
I = focus.irradiance()                          # S . n on the target plane, W/m²
```

- **Light.** A beam from `fields` (`PlaneBeam`, `FocusedBeam`) brings its own
  carrier. A sampled field (`ElectricField`, `TransverseElectricField`) or a
  `fields.Spectrum` takes `carrier=` (a `fields.Carrier`) at the call; without
  one the propagator raises and names `alternatives.WavePropagator`, the
  transport of fields without a carrier. A carrier is needed only when a
  surface refracts. The first interface supplies the incident region; for an
  empty system give `OpticalSystem(incident_region=air)`. A system may hold
  `incident_field`, so `propagator()` takes it.
- **Options.** `planes` and `fans` (one per surface, [below](#reference-planes-and-fans)),
  `realization` ("auto", "meridian", "lattice", [below](#rotationally-symmetric-systems-one-meridian))
  and `lattice` (a `CartesianGrid` a beam is carried on). Nothing else is
  accepted; an unknown keyword raises.
- **Solution.** `spectra` holds one `fields.Spectrum` per region, the
  incident light first; `spectrum` is the last. `reports` holds one
  `transport.TransportReport` per refracting surface ([reports](#reports)), and
  `legs` the prepared legs. `field(target, ...)` returns an
  `ElectromagneticField`, `electric(target, ...)` E alone (no magnetic modes
  are formed), and `coverage(target, ...)` the target's grid, the region of
  each point, the computed points and the sampling used.
- **Caching.** A call carries the light across every surface. Its legs depend
  on the geometry, the carrier and the incident lattice or beam, not on the
  field: the propagator keeps them, and a call with the same ones reuses them.
  It keeps no previous Solution.

Targets are `z=f`, `y=a`, `x=b` (global coordinate planes, with charts (y,z),
(x,z) and (x,y)), any placed `GraphSurface` (sampled with `.sample(u=..., v=...)`
or automatically), or a sampled `PointGrid`, `CartesianGrid` or `SurfaceGrid`.
E and H stay global Cartesian components. Each point takes the spectrum of
its region between the surfaces; boundary samples need `side="minus"` or
`side="plus"`, one-sided reconstructed fields, not the raw Fresnel trace.
Uncomputed samples are NaN in E and H and marked by `.valid`; they cannot be
used as incident FFT data. This is the forward propagating-band model:
neither reflected paths nor evanescent near fields are supplied by
evaluating nearer a surface.

Every Spectrum has `region`, `wavelength0`, `grid` (its chart) and `plane`
(that chart's Frame), `direction` (+1 forward along the normal, -1 back),
`evaluate(grid)` for E and H, `electric(grid)`, `propagate(distance)` (the
field on its chart moved along the normal) and `relabeled(region)`. The kinds
are `PlaneSpectrum` (an FFT lattice), `LatticeSpectrum` (a lattice leg's
output), `HankelSpectrum` (a meridian leg's output) and `BeamSpectrum` (a beam
referred to a chart). Every sum, on a Spectrum or a Solution, takes the same
options: `method` ("auto", "nufft", "direct"), `tolerance`, `chunk_size` and
`threads`; anything else raises.

### Optional spatial coverage

`field(..., coverage="reduced")` is the default. It excludes the height bands occupied by
curved interfaces over the sampled source chart, retaining regions on either
side of their enclosing planes. `coverage="complete"` evaluates all target
points using the forward spectrum of their graph-defined material region.
There is no interpolation across the excluded bands. Equal-index bookkeeping
interfaces do not create gaps. Planar interfaces have zero-width height bands;
both modes agree there, with the chosen one-sided boundary convention.
`solution.coverage(target, coverage=...)` reports the mode, computed mask,
region indices and `extended` mask (points evaluated beyond reduced
coverage). The `propagation/coverage.py` module owns classification, masks
and modal evaluation.

**Complete means spatial coverage of the chosen forward model.** In the curved
height bands this is an extension of its retained propagating modes; it does
not restore evanescent fields, guarantee agreement with Fresnel surface traces,
or certify near-interface Maxwell accuracy. Direct near-surface reconstruction
and independent curved-interface accuracy remain separate validation work.
Finite rims and stops are still rejected in both modes. In a continued-face
lens model, points outside the nominal rim follow the continued graphs; they
are not silently relabeled as the exterior of a finite lens.

### Automatic target sampling

`z=f`, `y=a`, `x=b`, and unsampled placed surfaces are sampled automatically.
The default window is the projection of the finite input chart and optical path,
including one input-chart width beyond the last surface for longitudinal views.
It is a documented geometrical window, not an energy enclosure or focus search.
Use `window=(umin, umax, vmin, vmax)` for detector crops, especially off-axis foci.
The target coordinate is never changed to maximize intensity.

For planes, spacing is at most 0.4 lambda/n_max, resolving the retained
propagating complex-field band. Curved charts use a pilot estimate of chart
stretch; highly varying curvature needs explicit sampling and convergence
checks. `shape=(Nv, Nu)` or a sampled surface overrides automatic resolution.
Automatic requests above two million points ask for an explicit window or
shape rather than silently coarsening. Sampling choices are not physical
accuracy estimates; `solution.coverage(target).sampling` records them.

### Wave transport: local and spectral treatments

`alternatives.WavePropagator(system, transmission="spectral" | "local",
max_sine=0.85, transport_tolerance=1e-10)` is the other method, kept for
checks: it carries a sampled field or a `PlaneSpectrum` exactly to and from
each surface (`SurfaceTransport`) with a Fresnel treatment between, needs no
carrier, and returns the same `Solution` (its `reports` are the treatment's,
its `legs` the incident and transmitted surface transports). A beam is
refused: sample it on a `CartesianGrid`.

`SpectralFresnel` retains the existing angular-spectrum algorithm.
Advanced settings can still be supplied as a treatment object, for example
`transmission=SpectralFresnel(nodes=18, coefficients="unity")`.

`LocalFresnel` estimates a local incident wavevector internally:

    k_eff = Im(sum_c conj(E_c) grad(E_c)) / sum_c |E_c|²

The three Cartesian derivatives of E are synthesized alongside E in one
batched Q evaluation (`SurfaceTransport.electric_jet`). They are derivatives
of the incident homogeneous field, not finite differences of wrapped surface
phase. The estimator equals the scalar phase gradient for a common phase with
real vector envelope. General vector polarization and multiple interfering
branches can violate that assumption. The direction is normalized, its
Fresnel dyadic acts pointwise, and outgoing H is completed internally for
surface-equivalence reconstruction.

The local report contains the intensity/area-weighted eikonal-magnitude residual
and longitudinal amplitude removed by the inferred polarization projection.
The default validity threshold is 0.1; exceeding it raises an error without an
automatic switch to spectral. These diagnostics do not certify accuracy or
uniquely detect all multi-branch fields. Non-forward/grazing local incidence
in illuminated samples also raises. Samples below 1e-6 of peak electric
amplitude are zeroed; their aggregate relative surface amplitude is reported.
The default incident angular limit is 0.65 per chart direction, relative modal
cutoff 1e-7, further restricted to Q's declared band. Omitted modal L2 is reported.
Both branches operate within the current forward propagating-band model.

Charts remain ordered, nonintersecting graphs with the input frame orientation
and transverse origin. Finite edges and stops are rejected. Wave transport
does not include returning branches or a finite-body rim solver. See the
[runnable example](../general/examples/wave_transport.py) and
[validation records](VALIDATION.md).

## Eikonal transport

`propagation.Propagator(system)` carries fields with the eikonal–angular-spectrum method of the manuscript *Vectorial field transport through general curved interfaces*: vecdiff is meant for macroscopic systems. Between surfaces the field travels as an exact angular spectrum. Across each surface it is carried along its **carrier**, the ray family it declares. The method computes the **first entry**: the wave transmitted through every surface without reflection. [Notebook 08](../general/notebooks/08_eikonal_angular_spectrum_transport.ipynb) shows it on three geometries against their exact references.

### The method in brief

A point P of a reference plane in front of a surface follows its carrier ray to the surface, refracts there by the vector Snell law, and continues to P′ on the same geometric plane in the transmitted medium. This plane–surface–plane map Φ, with signed legs ℓ_i and ℓ_t, carries the field as

    E′(Φ(P)) = g exp(i k0 ΔS) [t_s E_s s + t_p E_p p_t + t_p E_par s_t],
    ΔS = n_i ℓ_i + n_t ℓ_t,   g = sqrt(s_iz c_t / (c_i s_tz J)),   J = det DΦ.

- **Continued dyadic.** The bracket is the continued dyadic (Eq. 4.3), and E_par is the part of the field along the carrier. Keeping E_par makes an index-matched surface exactly the identity.
- **Exact limits.** The transfer is exact for a plane wave on a plane interface.
- **What it is.** It is the formulation's leading-order model: it does not solve the curved-boundary Maxwell problem, and its error falls as 1/(R/λ0).
- **Carried plane components.** They are re-expanded in the transmitted medium and continue as an exact angular spectrum to the next surface or the observation points.

`transport.InterfaceMap` computes Φ, ΔS, g and DΦ; `transport.RayMap` holds the rays.
- **DΦ is analytic (Appendix A).** It comes from the surface Hessian and the carrier's direction Jacobian; no spherical focus is assumed.
- **A focus inside the leg.** Where the ray tube has collapsed between the two planes, J has passed through zero and the root in g continues through it as 1/√J → exp(−iπκ/2)/√|J|. κ, `RayMap.caustics`, counts the transverse directions in which the tube collapsed: one at a focal line, two at a point focus. It is counted on each straight segment from the tube's own cross-section, forward on a real segment and backward on a virtual one (a reference plane beyond the surface). This is the continuation the angular spectrum itself produces through a focus; the error of the leg still falls as 1/size, with a larger constant than without a focus.
  - The lattice leg carries such rays. Where the map folds, rays from both sides of a caustic reach the same points: the rays of one caustic index carry the field, the others carry none and are reported, and a leg that would lose more than 1 % of its amplitude that way is refused.
  - The meridian leg carries only rays that met no caustic and says so; a beam focused in front of its first surface needs `lattice=`.
- **Intersection.** Each ray–surface intersection starts from the height under P, then the osculating sphere at the vertex, then a march along the ray, and finishes with Newton's method. A ray that Newton loses from behind a surface (from a rim-depth plane) is retried from the osculating sphere.

### Carriers

- The method needs the incident field's carrier, and transmits with the carrier dyadic.
  - A beam from `fields` (`FocusedBeam`, `PlaneBeam`) can be the light itself, and brings `beam.carrier`. It is carried on one meridian, or on a lattice with `lattice=grid`.
  - A sampled field or spectrum takes `carrier=` at the call: a `fields.Carrier` (`Carrier.converging(focus)`, `diverging`, `collimated`, or a general optical path with gradient and Hessian).
  - Without a carrier the propagator raises and names `carrier=` and `alternatives.WavePropagator`.
- **Wave transport** ([above](#wave-transport-local-and-spectral-treatments)) is a separate class with its own options; none of them is accepted by `Propagator`.

### Reference planes and fans

- **`planes=(offset, ...)`**, one per surface, places each plane along the axis from its surface vertex (default: the vertex planes).
  - A plane at the depth where the marginal ray meets the surface shortens the legs.
  - That lowers a hard pupil's error: for the aplanatic sphere at 1000 λ0, 0.1418 % (vertex plane) against 0.01524 % (rim plane).
- **`fans=(fan, ...)`**, one per surface, weights the carried rays:
  - None carries every supported ray;
  - `(start, end)` rolls them off between those incidence angles, in degrees;
  - a callable receives the `RayMap` (its `label` is the ray's point on the first plane, so a fan can roll off along the input radius).
- **Fans and hard pupils.** A hard pupil's result depends on the fan, so report it with results. A hard cut at the fan's edge diffracts: 0.14182 % on the vertex plane and 0.01729 % on the rim plane, against 0.14177 % and 0.01524 % tapered. A taper that closes before the surface stops supporting the carrier removes that edge.

### The lattice leg

`transport.LatticeTransport` is one leg on a Cartesian lattice. It is what a sampled field, a beam given `lattice=` and every non-axisymmetric system use.
- It fills the output lattice through the inverse map: a forward table, then damped Newton steps with DΦ. The incident field is evaluated exactly at each preimage, never interpolated.
- The output lattice keeps the input spacing. It grows, up to four times per axis, until it holds the image of every ray the window weights. Refraction toward a closer image spreads P′ beyond the input footprint, and cutting it there diffracted spuriously.
- It re-expands within sin < 0.995 of the transmitted medium, as the manuscript does. Nearer the light circle E_z = −κ·E/k_z amplifies what the lattice leaves there.
- A fan whose edge rays leave almost parallel to the plane is refused with a request for a window, before its image fills memory.
- **The leg streams its lattices.** Rays are traced, inverted and carried one block of rows at a time: `block_size` points, 2^20 by default; on the CPU backend, blocks of 2^18 run on up to eight threads.
  - It keeps only the preimages, the lit and supported masks and the carried field, in host memory. Before, whole-lattice ray maps took about 1.2 kB a point.
  - Blocks change nothing but roundoff.
  - The incident field is read on the reference plane by two-dimensional sums, its plane waves' normal phase folded into their amplitudes.
- **A beam on a lattice.** `Propagator(system, lattice=grid)` carries a beam on the lattice leg, `grid` (a `CartesianGrid`) setting its spacing and extent.
  - The beam is read exactly at the preimages, so no sampled copy of the input lattice is held.
  - The result is a `LatticeSpectrum`: two transverse coefficients per mode in host memory, in place of a `PlaneSpectrum`'s wavevector and three amplitudes. Its modes are formed and summed one block of rows at a time, by a 2D type-3 transform when the targets lie on one plane.
  - A sampled field still gives a `PlaneSpectrum`.
  - The archive's cylinder at R = 1000 λ0 (an 11852² lattice, 140 million points) runs this way in double precision: 13 GB of host memory and an 8 GB GPU.

### Sequences of surfaces

Eikonal transport crosses any number of refracting surfaces, as the formulation prescribes:
1. a leg;
2. the exact angular spectrum;
3. the full field on the next reference plane;
4. a leg along the refracted carrier;
5. and so on.

After a refraction the carrier is the family of refracted rays (Appendix B), `transport.RefractedRays`.
- Each ray keeps its label, its point on the first reference plane.
- Its surface point and refracted direction (Eq. 3.6) continue in a straight line to the next plane, together with their Appendix A derivatives along the labels.
- Nothing is evaluated off the rays or inverted, so the next map's Jacobian, flux factor and inverse stay analytic, with J = det(dP′/d label) / det(dP/d label).

How the two legs chain, and the bookkeeping:
- **Index-matched surfaces** between legs are the identity; the family passes through them.
- **The lattice leg** reads the previous leg's output lattice and inverts the composed map from the first plane's labels.
- **The meridian leg** reads the previous leg's Hankel field. Its intermediate spectrum resolves the depth and radius of the next plane, so the handoff's κ-aliasing rule holds by construction.
- **Each interface reports its own unsupported amplitude:** the incident field outside the rays that survive it. No amplitude threshold decides a ray.
- **Fans:** a later fan stays within the labels the first leg carried.

### Surfaces

Eikonal transport needs a surface's Hessian, and every built-in surface declares one.

| Surface | Kind | Notes |
|---|---|---|
| `Plane`, `SphericalCap`, `EvenAsphere`, `AxisymmetricSurface` (`curvature_function=`) | graph | surfaces of revolution |
| `CartesianOval` | graph | stigmatic for a conjugate pair; real or virtual conjugates (object behind the vertex, image in front); Hessian by implicit differentiation |
| `FreeformSurface` | graph | `hessian_function=` |
| `CylindricalCap` | graph | front half of a circular cylinder along the frame's v, for \|x\| < \|R\| |
| `Sphere` | closed, implicit | the `SphericalCap` of the same radius closed into a ball |
| `Cylinder` | closed, implicit | closes `CylindricalCap` |
| `CartesianOvoid` | closed, implicit | closes the oval into its ovoid of revolution |

- **Graphs end at their rim.** A `SphericalCap` ends at its equator, so rays that would enter a ball beyond it are not carried. That limits a hard pupil whose edge tails reach the equator (the aplanatic pair's marginal ray meets its inner cap exactly there).
- **Closed surfaces** (`surfaces.ImplicitSurface`, F(Q) = 0) meet rays beyond their graph chart.
  - `implicit(points)` gives F, its gradient (the normal from the minus into the plus side, equal to the graph normal where both exist) and its Hessian.
  - `crossing(points, directions)` gives the signed leg to where a ray crosses into the plus side.
  - `InterfaceMap` traces them with the same Appendix A formulas.
  - Regions, charts, targets and coverage keep the graph, so a closed surface works wherever its cap does: on the lattice, on the meridian and in sequences.
  - Where the cap suffices, both map the same.
  - The archive's sphere fans reach 80°: `fans=((72, 80),)`.

### Rotationally symmetric systems: one meridian

If every refracting surface is a surface of revolution about the beam's axis (coaxial `Plane`, `SphericalCap`, `Sphere`, `EvenAsphere`, `AxisymmetricSurface`, `CartesianOval` or `CartesianOvoid`) and the carrier is symmetric about it, then the map is radial. The dyadic then acts on (E_rho, E_phi, E_z) by a matrix of the radius alone.

- **One meridian per azimuthal order.** Each order m of the beam is carried along one meridian, by the same `InterfaceMap` and dyadic as the lattice leg.
- **Hankel spectra.** E_x ± i E_y of order m leave as Hankel spectra of orders m ± 1, and E_z follows from transversality.
- **Cost.** Circular, linear, radial and azimuthal states are one or two orders. The cost grows linearly with the aperture instead of quadratically: the manuscript's aplanatic sphere at 1000 λ0 takes 1.1–1.4 s (the archived prototype: 64 s).

`realization=` chooses how the legs are carried:
- **"auto" (default):** the meridian for beams without a lattice (and for a `HankelSpectrum`), the lattice otherwise.
- **"meridian":** the meridian for either. A sampled field is read on rings about its grid's centre; its weakest orders are dropped while together they stay below 1e-5 (counted as unsupported); a field with more than 1e-2 of non-axisymmetric content is refused.
- **"lattice":** the lattice; a beam then needs `lattice=` (see [the lattice leg](#the-lattice-leg)).
- **No symmetry:** a beam in a system without the symmetry (a cylinder, a decentred surface, an asymmetric carrier) and without a lattice is refused, with the reason.

How the meridian is sampled:
- **Input plane:** Gauss–Legendre panels at most 10 λ0 wide. Each node's preimage is found by Newton steps with DΦ, and the beam is evaluated exactly there from its direction density, one Hankel sum per order.
- **Transmitted spectrum:** sampled in the polar angle of its plane waves, up to grazing. E_z's obliquity and the phase k_z z stay smooth there, so no spectral cut-off is imposed. Its panels resolve every phase that the carried extent and the evaluated region can produce.
- **Fast Hankel sum:** `fourier.polar.hankel_sum`, and `HT`/`IHT(method="fast")`.
  - Where tx ≥ z0 ≈ 40, the kernel is its ten-term Hankel asymptotic expansion, each term one type-3 FINUFFT sharing a plan across orders.
  - Where tx < 2 z0, the kernel is interpolated at Chebyshev nodes (kernel compression).
  - It agrees with direct sums to 1e-12–1e-15.

### Reflection legs

A leg can carry the reflected wave instead of the transmitted one: `LatticeTransport(..., branch="reflected")` or `branch="mirror"`, and the same for `MeridianTransport`. The propagator does not compose them yet.
- **Map.** s_r = s − 2 c_i n, back to the same geometric plane in the incident medium, with ΔS = n_i (ℓ_i + ℓ_r) and signed legs. `InterfaceMap(..., branch="reflected")` gives it, with ds_r = ds − 2 (dc_i n + c_i dn) from the same Appendix A quantities.
- **Flux.** g = √(s_iz / (|s_rz| J)) e^{−iπκ/2}. The power R I c_i dσ leaves the surface at c_r = c_i and I |s_z| dA is constant along each segment, so |r|² is the dyadic's and the area ratio g's. J is measured in the plane's own axes on both sides: a plane mirror at normal incidence has J = +1. The mirror reverses the tube's handedness about its direction and the sign of s_z together, so J still changes sign once per collapsed direction, and κ is counted on both segments as for refraction.
- **Dyadic.** r_s E_s s + r_p E_p p_r + r_p E_∥ s_r with p_r = s × s_r, r_s = (n_i c_i − n_t c_t)/(n_i c_i + n_t c_t), r_p = (n_t c_i − n_i c_t)/(n_t c_i + n_i c_t): tangential E and H are continuous with the transmitted wave. Beyond the critical angle c_t = i|c_t| and |r| = 1. `"mirror"` is the ideal mirror, r_s = −1, r_p = 1. Real indices only.
- **Output.** `reflect()` re-expands in the incident medium, travelling back (k_z < 0 in the surface frame); `LatticeSpectrum` and the meridian's `HankelSpectrum` carry that sign (`Spectrum.direction`). Every report and refusal uses the outgoing medium.
- **Checks** (`general/tests/test_reflection.py`): plane waves against each wave's boundary-value problem, to 1e-13 (R + T = 1, total reflection, ideal mirror); focused beams against their plane-wave sums, with a focal line (κ = 1) and a point focus (κ = 2) on the way back; a glass sphere against Debye term 0 outside and, with a refraction first, term 2 inside. The errors fall as 1/size:

| Case | 10 λ | 20 λ | 40 λ | 80 λ |
|---|---|---|---|---|
| Sphere, external reflection, lattice leg (term 0) | 25 % | 13 % | 6.1 % | 3.2 % |
| Same, meridian leg | 29 % | 14 % | 7.2 % | 3.6 % |
| Sphere, refraction and one internal reflection (term 2) | 36 % | 19 % | 9.7 % | 4.8 % |

### Reports

`solution.reports` holds one `transport.TransportReport` per refracting surface, and `print(solution)` gives one line for each:
- `unsupported_relative_amplitude`: the incident amplitude on rays the surface does not support (misses, rear or grazing incidence, total reflection, or outside the fan);
- `dark_samples`: output samples no ray reaches;
- `newton_residual`: the inverse map's residual;
- `azimuthal_orders`: the orders carried on the meridian (empty for the lattice leg);
- `caustics`: the caustic index κ of the carried rays;
- `folded_relative_amplitude`: the share of the carried amplitude on rays of another caustic index, beyond a fold of the map, which carry no field;
- `undersampled_relative_amplitude` (lattice leg): the share on rays that leave the surface steeper than the output lattice samples. The carried field turns at n_t k0 s_t across the plane, so the spacing must stay below λ0 / (2 n_t sin θ_t), whatever the input needed;
- `band_removed_relative_amplitude` (lattice leg): the share in propagating plane waves beyond sin = 0.995 of the transmitted medium, which the re-expansion removes.

A leg is refused, with the reason, when the folded, undersampled or band-removed share passes 1 %. Options that no spectrum has are refused on every path.

### Exact references

`references` holds the oracles. Each is one physical problem that takes `incident=` any propagating field (plane waves, FFT spectra or beams); see also [validation](VALIDATION.md#layered-sphere-and-infinite-cylinder).
- **`MieSphere`**: a homogeneous sphere.
- **`LayeredSphere`**: concentric layers.
- **`InfiniteCylinder`**: a circular cylinder at normal and oblique incidence.

`evaluate(grid, side=, term=)` gives the total field or one Debye term:
- `term=1` is the first entry, the quantity the method models;
- `term=p` is the waves that met p − 1 reflections;
- `term=0` is the external reflection and diffraction outside;
- `("after", P)` is the rest.

All stay exact at size parameters near 10⁴: log-scaled Bessel functions and Wronskian-reduced coefficients replace the ill-conditioned direct Hankel-basis solve.

### Regression against the handoff

Raw complex-field L2 errors in percent: on the 5 λ0 focal disk (161 radii, Eq. 8.2) for the spheres, on the 3.5 λ0 disk (140² grid) for the cylinder; n = 1 → 1.5 unless stated; R = 1000 λ0 unless stated. The handoff's numbers come from the archived prototype with its 80° fans. vecdiff uses the archive's fans: tapers 50°→56° for the aplanatic cap (identical to 80° for this sphere), 72°→80° on closed spheres, and 34°→38° for the meniscus's second surface. The opt-in module `test_eikonal_regression.py` (`VECDIFF_SLOW=1`) pins the rows marked †.

| Case | Handoff | vecdiff |
|---|---|---|
| Aplanatic sphere (A = 1.5R, u_max 0.4), hard, vertex plane, first / full | 0.142 / 0.386 (Table 1: 0.1421 / 0.3874) | 0.1417746 / 0.3864750 † |
| Same, smooth pupil | 0.009 | 0.0085149 † |
| Same, rim-depth plane | 0.0153 | 0.0152384 † |
| Non-aplanatic sphere (A = 2R), hard, first / full | 4.115 / 4.143 (Table 1: 4.127 / 4.155) | 4.11451 / 4.14281 (closed `Sphere`) † |
| Aplanatic error law | 1.42/(R/λ0) from R ≈ 150 | error × R: 152.2 at 100, 146.1 at 154, 141.7 from 1259 to 3000 |
| Two aplanatic surfaces, hard, vertex / rim planes | 0.883 / 0.0795 | 0.88324 / 0.07967 (closed spheres) † |
| Same, smooth, vertex / rim | 0.0191 / 0.0205 | 0.019095 / 0.020461 (closed spheres); 0.01914 / 0.02049 with caps † |
| Meniscus, hard, vertex / rim | 0.0260 / 0.0114 | 0.02603 † / **0.0031975** (corrected, see below) |
| Meniscus, smooth, vertex / rim | 0.00192 / 0.00104 | 0.001924 / 0.0010382 † |
| Cylinder (u_max 0.25), hard, R = 100 / 300 / 1000 | 0.193 / 0.0642 / 0.0192 | 0.19298 † / 0.06420 † / 0.01918 |
| Same, smooth | 0.0552 / 0.0183 / 0.0055 | 0.055186 † / 0.01834 † / 0.00550 |
| Index-matched second surface, smooth | 1.8e-5 | 3.9e-5 |
| Oval at 300 λ0: 2D code vs axisymmetric code | 0.115 | 8.0e-5 (lattice vs meridian, 6778²) †; not reproduced, see below |

Rows that differ, and why:
- **Meniscus, hard, rim planes: the handoff's 0.0114 is the archive's κ aliasing.**
  - The archive's intermediate spectrum on rim planes used the factor-2 rule (extent + 1.5|d|), which ignores how far out the next plane is read. Here the starts reach 12 R1, so the spectrum is under-resolved at 1000 λ0.
  - With factor 3 the archive itself gives 0.0031970 %, against vecdiff's 0.0031975 %. vecdiff's panels resolve the next plane's radius by construction, and its plane panels are converged.
  - At 100 λ0 the archive is converged at factor 2, and the two agree to 0.3 %.
  - This is the pitfall the handoff names ("2.6 % on the meniscus rim planes"), at a larger size.
- **Index-matched second surface: 3.9e-5 against 1.8e-5.** The meridian leg returns the single-surface field up to the field beyond the second fan; that field is the whole difference.
- **Table 1 against the handoff.** The manuscript's Table 1 differs from the archive by 0.3 % (4.127 against 4.115). vecdiff reproduces the archive with the archive's plane quadratures (30 743 nodes for the non-aplanatic sphere).
- **The sphere rows need the closed sphere.** With a `SphericalCap` the hard-pupil rows whose edge tails pass the cap's equator are not reproduced: the non-aplanatic sphere gives 4.036 %, the aplanatic pair 5.38 %. The archive with its fans cut at the same equator gives the same numbers, so the whole gap is the cap.
- **Oval: the archive's 0.115 % is not reproduced, and the gap is the archive's.**
  - vecdiff's lattice and meridian legs (6778², the archive's fans) differ by 8.0e-5 %, and all of it is their two fans: the meridian run with the lattice's fan differs from the meridian with its own radial taper by the same 8.0e-5 %. With equal fans the two legs agree to about 1e-8.
  - The fans' effect falls as the hard pupil's shadow fades: 0.2498 % at R = 30, 0.0017 % at 100, 8.0e-5 % at 300.
  - The archive's own two codes differ by 0.2424, 0.056 and 0.115 % at those radii. The rise between 100 and 300 is not a fan effect: the archive's general code carries about 0.1 % of its own at these sizes (where exactly was not traced).
  - vecdiff's meridian against the archive's axisymmetric code: 0.0038 % (hard) and 3e-11 (smooth) at 300 λ0.
- **Runtimes.**
  - Spheres at 1000 λ0 (i7-12650H, one thread): 0.8–4.1 s per row on the meridian, against 64 s (one sphere) to 390–650 s (two surfaces) for the archived code.
  - Cylinder, both pupils, streamed lattice on the CUDA backend: 9.8 + 1.1 s at 100 λ0 (1.6 GB host / 2.8 GB GPU), 52 + 19 s at 300, 728 + 391 s at 1000 (13.0 / 8.1 GB, measured with about 40 % contention).
