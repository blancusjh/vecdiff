# Propagation

The public algorithms live under `vecdiff_general.propagation` and are also
available through the package's top-level exports.

| Module | Main API | Operation |
|---|---|---|
| `homogeneous.py` | `propagate`, `PlaneTransport` | Evolution in a homogeneous medium |
| `surface_transfer.py` | `SurfaceTransport` | Evaluate and reconstruct fields on a graph surface |
| `surface_interaction.py` | `scatter_plane`, `SpectralFresnel`, `LocalFresnel` | Reflection and transmission at fixed boundaries |
| `pipeline.py` | `PropagationPipeline` | Compose transport and interactions |
| `references/` | `MieSphere`, `DielectricBEM`, other reference solvers | Independently check physical and numerical accuracy |

## Homogeneous propagation

`angular_spectrum(field, region)` constructs a `PlaneSpectrum`. ASM propagation
uses exact homogeneous dispersion. It can retain evanescent modes, subject to
the chosen branch and propagation distance. Paraxial Fresnel propagation uses
an approximate longitudinal phase. `PlaneTransport.spectrum` preserves a modal
representation for subsequent operations; `spectrum.propagate` evaluates E/H
on the translated FFT plane.

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

```python
propagate = PropagationPipeline(
    system, transmission="local", coverage="complete"
)  # transmission can also be "spectral"; coverage can be "reduced"
focal = propagate(incident_field, z=f)
meridian = propagate(incident_field, y=0, window=(-a, a, z0, z1), side="plus")
detector = Plane.z(f).sample(u=x_coordinates, v=y_coordinates)
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
