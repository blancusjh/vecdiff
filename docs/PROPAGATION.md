# Propagation

The public algorithms are reached through `vecdiff_general.propagation`;
the package top level re-exports nothing.

| Module | Main API | Operation |
|---|---|---|
| `homogeneous.py` | `propagate`, `PlaneTransport` | Evolution in a homogeneous medium |
| `surface_transfer.py` | `SurfaceTransport` | Evaluate and reconstruct fields on a graph surface |
| `surface_interaction.py` | `scatter_plane`, `SpectralFresnel`, `LocalFresnel` | Reflection and transmission at fixed boundaries |
| `pipeline.py` | `PropagationPipeline` | Compose transport and interactions |
| `references/` | `MieSphere`, `LayeredSphere`, `InfiniteCylinder`, `DielectricBEM`, other reference solvers | Independently check physical and numerical accuracy |

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
  determine its error. This is the reconstruction used by the pipeline.

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

## Propagation operator

The operator takes an incident electric field, an optical system, and a target.
The system owns geometry and media, optionally the incident field. It does not
own observations. Geometry and sampling use the existing surface/grid classes.

The default transport is `transport="eikonal"` ([below](#eikonal-transport)):
vecdiff is meant for macroscopic systems. It needs the incident field's
carrier, passed as `carrier=` or brought by a beam from `fields`; without one
the pipeline raises and asks for it. `transport="wave"` keeps the exact
transport to and from each surface with spectral (its default), local, carrier
or another Fresnel transmission, and is the transport of fields without a
carrier. Combinations that do not belong together (eikonal transport with a
spectral or local Fresnel treatment, reference planes or fans with wave
transport) raise at construction.

```python
from vecdiff_general import propagation, surfaces

propagate = propagation.PropagationPipeline(
    system, transport="wave", transmission="local", coverage="complete"
)  # transmission can also be "spectral"; coverage can be "reduced"
focal = propagate(incident_field, z=f)
meridian = propagate(incident_field, y=0, window=(-a, a, z0, z1), side="plus")
detector = surfaces.Plane.z(f).sample(u=x_coordinates, v=y_coordinates)
focal = propagate(incident_field, detector)
```

The return value is the existing `ElectricField` (`.Ex`, `.Ey`, `.Ez`, `.grid`).
Its read-only `.valid` mask identifies computed samples. Uncomputed samples
are NaN, not zero field; they cannot be used as incident FFT data.
`TransverseElectricField` input needs no H or eikonal argument. The first
interface supplies its incident medium; for an empty system specify
`OpticalSystem(incident_region=air)`. A system may hold `incident_field`, allowing
`propagate(z=f)` or `propagate(target)`. `PlaneSpectrum` input remains accepted.
`Plane.x(b)`, `Plane.y(a)` and `Plane.z(f)` are global coordinate planes, with
charts (y,z), (x,z), and (x,y). Other `GraphSurface` objects retain their placed
frames and can be sampled with `.sample(u=..., v=...)`. Sampled `PointGrid`,
`CartesianGrid` and `SurfaceGrid` targets are accepted directly. E components
remain global Cartesian components, irrespective of target orientation.

A call prepares only interfaces needed to reach its target regions. Repeated
calls with the same immutable source object reuse source completion, operators,
and intermediate spectra; `clear_cache()` explicitly discards these. Targets
crossing interfaces select the forward spectrum of each material region.
Boundary samples require `side="minus"` or `side="plus"`; these are reconstructed
one-sided fields, not the raw Fresnel trace. This is the existing forward
propagating-band model: neither missing reflected paths nor evanescent near
fields are supplied by evaluating nearer a surface.

### Optional spatial coverage

`coverage="reduced"` is the default. It excludes the height bands occupied by
curved interfaces over the sampled source chart, retaining regions on either
side of their enclosing planes. `coverage="complete"` evaluates all target
points using the forward spectrum of their graph-defined material region.
There is no interpolation across the excluded bands. Equal-index bookkeeping
interfaces do not create gaps. Planar interfaces have zero-width height bands;
both modes agree there, with the chosen one-sided boundary convention.

Coverage is independent of `transmission="local"` or `"spectral"`. Changing
`propagate.coverage` reuses the same cached source and interface results.
`propagate.last_coverage` reports the mode, computed mask, region indices and
`extended` mask (points evaluated beyond reduced coverage). The separate
`propagation/coverage.py` module owns classification, masks and modal evaluation.

**Complete means spatial coverage of the chosen forward model.** In the curved
height bands this is an extension of its retained propagating modes; it does
not restore evanescent fields, guarantee agreement with Fresnel surface traces,
or certify near-interface Maxwell accuracy. Direct near-surface reconstruction
and independent curved-interface accuracy remain separate validation work.
Finite rims and stops are still rejected in both modes. In a continued-face
lens model, points outside the nominal rim follow the continued graphs; they
are not silently relabeled as the exterior of a finite lens.

`last_solution` exposes reusable spectra and interface reports. Its
`field(target, ...)` returns E; the existing `solve(source)` and
`solution.evaluate(grid)` APIs remain available for explicitly requested E/H
beyond the final surface envelope. `last_sampling` records target bounds,
spacing and shape. Electric-only observation does not compute magnetic modes.

### Automatic target sampling

`z=f`, `y=a`, `x=b`, and unsampled placed surfaces default to `sampling="auto"`.
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
accuracy estimates.

### Local and spectral treatments

`SpectralFresnel` retains the existing angular-spectrum algorithm.
`DirectionResolvedFresnel` is a compatibility alias for exactly the same class.
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
and transverse origin. Finite edges and stops are rejected. The pipeline does
not include returning branches or a finite-body rim solver. See the
[runnable example](../general/examples/propagation_pipeline.py) and
[validation records](VALIDATION.md).

### Carrier-directed transmission

`transport="wave", transmission="carrier"` keeps the exact surface transport but takes the
local incident direction from the field's declared carrier (a
`fields.Carrier`, for example `beam.carrier`), passed with the field:

    pipeline = PropagationPipeline(system, transport="wave", transmission="carrier")
    field = pipeline(incident, z=focus, carrier=beam.carrier)

Each surface sample is transmitted by the continued dyadic of the
eikonal–angular-spectrum formulation,

    E_t = t_s E_s s + t_p E_p p_t + t_p E_par s_t,

where E_par is the part of the field along the carrier. Keeping that part
makes an index-matched interface exactly invisible. Nothing is inferred from
the field, so the operation is linear for a fixed carrier. Samples whose
carrier meets the surface from behind, grazes it or is totally reflected
carry no field; their share is reported.

## Eikonal transport

`PropagationPipeline(system)`, whose default is `transport="eikonal"`, replaces the exact legs
to and from each surface by the plane–surface–plane map along the carrier
(`propagation.EikonalTransport`). A point P of the reference plane follows its
carrier ray to the surface, refracts by the vector Snell law, and is
extrapolated to P′ on the same geometric plane in the transmitted medium.
The signed legs give the optical path ΔS = n_i ℓ_i + n_t ℓ_t, and the field
is carried as

    E′(Φ(ρ)) = g exp(i k0 ΔS) [t_s E_s s + t_p E_p p_t + t_p E_par s_t],
    g = sqrt(s_iz c_t / (c_i s_tz J)),  J = det DΦ.

`propagation.InterfaceMap` computes Φ, ΔS, g and DΦ. DΦ is analytic: it comes
from the surface Hessian (`GraphSurface.hessian`, declared by every built-in
surface; `FreeformSurface` takes a `hessian_function`) and the carrier's
direction Jacobian. No spherical focus is assumed. The output lattice is
filled through the inverse map, damped Newton steps from a forward table, so
the incident field is evaluated exactly at each preimage and never
interpolated. The carried plane components are then re-expanded in the
transmitted medium.

Options and reports:

- `reference_planes=(offset, ...)` places each plane along the axis from its
  surface vertex (default: the vertex planes).
- The output lattice keeps the input spacing. It grows, up to four times per
  axis, until it holds the image of every supported ray, because refraction
  can spread the image beyond the input footprint.
- `EikonalTransport(..., window=...)` rolls the carried field off. It takes
  either `(start, end)` incidence angles in degrees or a callable of the
  `RayMap`, such as a reference fan. Hard-edged inputs depend on this choice;
  report it with results. The pipeline passes `reference_fans=(fan, ...)`,
  one per surface, to its legs.
- `EikonalReport` gives the incident amplitude on rays the chart does not
  support (misses, rear or grazing incidence, total reflection, folds), the
  dark output samples, and the inverse-map residual.

The transfer is the formulation's leading-order model. It is exact for a
plane wave on a plane interface and the identity for index-matched media; it
does not solve the curved-boundary Maxwell problem.

### Sequences of surfaces

Eikonal transport crosses any number of refracting surfaces, as the
formulation prescribes: leg, exact angular spectrum, the full field on the
next reference plane, leg with the refracted carrier, and so on. After a
refraction the carrier is the family of refracted rays (Appendix B),
`propagation.eikonal.RefractedRays`: each ray keeps its label, its point on the
first reference plane, and its surface point and refracted direction (Eq. 3.6)
are continued in a straight line to the next plane together with their
Appendix A derivatives along the labels. Nothing is inverted off the rays, so
the next map's Jacobian, flux factor and inverse stay analytic:
J = det(d P' / d label) / det(d P / d label). Index-matched surfaces between
are the identity; the family passes through them.

- The lattice leg reads the previous leg's output lattice and inverts the
  composed map from the first plane's labels; the meridian leg does the same
  on one meridian, its intermediate spectrum resolving the depth and radius
  of the next plane (the κ-aliasing rule of the handoff holds by construction).
- Each interface reports its own unsupported amplitude: the incident field
  outside the rays that survive it. No amplitude threshold decides a ray.
- A ray Newton loses from the height under its start, as behind a surface
  from a rim-depth plane, is retried from the osculating sphere and then from
  a march along the ray.
- Carrier transmission (wave transport) still crosses one refracting surface.

### Rotationally symmetric systems

A beam from `fields` is also an incident field of eikonal transport:

    pipeline = PropagationPipeline(system, transport="eikonal")
    field = pipeline(beam, points)          # carrier: beam.carrier by default

When every refracting surface is a surface of revolution about the beam's
axis (`Plane`, `SphericalCap`, `EvenAsphere`, `AxisymmetricSurface`,
`CartesianOval`, coaxial) and the carrier is symmetric about it, the map is
radial and the dyadic acts on (E_rho, E_phi, E_z) by a matrix of the radius
alone. Each azimuthal order m of the beam is then carried along one meridian,
by the same `InterfaceMap` and dyadic as the lattice leg, and E_x ± i E_y of
order m leave as Hankel spectra of orders m ± 1; E_z follows from
transversality. Circular, linear, radial and azimuthal states are one or two
orders. The cost grows linearly with the aperture: the aplanatic sphere of the
manuscript at R = 1000 λ0 takes seconds, where its lattice would not fit in
memory.

- The input plane is sampled by Gauss-Legendre panels (71 nodes per 15
  transmitted wavelengths), each node's preimage found by Newton steps with
  the analytic DΦ; the beam is evaluated exactly there from its direction
  density, one Hankel sum per order.
- The transmitted spectrum is sampled in the polar angle of its plane waves,
  up to grazing: E_z's obliquity and the phase k_z z stay smooth there, so no
  spectral cutoff is imposed. Its panels resolve every phase the carried
  extent and the evaluated region can produce, and are formed on first
  evaluation. The Hankel sums use the fast transform of `fourier.polar`.
- The fan ends where the surface stops supporting the carrier or the window
  closes. A `SphericalCap` is a graph and ends at its equator; rays that would
  enter a full ball beyond it are not carried. A hard pupil's tails reach
  there, and a hard cut at the fan's edge diffracts: a window that closes
  before the equator, such as `reference_fans=((50, 56),)` for the
  manuscript's aplanatic sphere, removes that edge.
- `EikonalReport.azimuthal_orders` lists the carried orders (empty for the
  lattice leg); its unsupported amplitude is the incident field outside the
  carried fan.
- `axisymmetric=None` (default) takes the meridian for beams and the lattice
  for sampled fields. `True` takes the meridian for either: a sampled field
  is read on rings about its grid's centre, its weakest orders are dropped
  while together they hold below 1e-5 of it (counted as unsupported), and a
  field beyond 1e-2 of non-axisymmetric content is refused. `False` takes the
  lattice; a beam must then be sampled on a `CartesianGrid` first. A beam in a
  system without the symmetry is refused with the reason.
