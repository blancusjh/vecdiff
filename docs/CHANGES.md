# Changes since the PR stack

What changed between commit `ac10128` ("Document the eikonal-angular-spectrum
transport, and show it in notebook 08") and the current library, for a reader
who knew the API of that time. Built from `git log ac10128..HEAD` and
`git diff ac10128`.

## Names

| Before | Now | Note |
|---|---|---|
| package `vecdiff_general` | `vecdiff` | `import vecdiff as vd` |
| `propagation.PropagationPipeline(system, ...)` | `propagation.Propagator(system, *, planes, fans, realization, lattice)` | the method only |
| `PropagationPipeline(..., transport="wave", transmission=, max_sine=, transport_tolerance=)` | `alternatives.WavePropagator(system, *, transmission, max_sine, transport_tolerance)` | wave transport, for checks |
| `pipeline(incident_field, target, ..., carrier=)`, `pipeline.solve(source, carrier=)` | `Propagator(system)(light, carrier=None)` -> `Solution`, then `solution.field(target, ...)` | one call per light, any number of targets |
| `propagation.PropagationSolution` | `propagation.Solution` | `spectra`, `spectrum`, `elements`, `reports`, `legs`, `field`, `electric`, `coverage` |
| `PropagationSolution.interactions` | `Solution.reports` | |
| `PropagationSolution.evaluate(grid)` | `Solution.field(grid)` (E and H) or `Solution.electric(grid)` | |
| `reference_planes=` | `planes=` | one offset per element |
| `reference_fans=` | `fans=` | one fan per element |
| `axisymmetric=None / True / False` | `realization="auto" / "meridian" / "lattice"` | |
| `incident_region=` | `OpticalSystem(..., medium=)` | |
| `coverage=` on the pipeline | `Solution.field(..., coverage=)` | |
| `field(..., sampling="auto")` | removed | the sampling is `window=` and `shape=` |
| `propagation.EikonalTransport` | `transport.LatticeTransport` | |
| `AxialTransport` (`propagation._axisymmetric`) | `transport.MeridianTransport` | |
| `propagation.EikonalReport` | `transport.TransportReport` | new fields `caustics`, `folded_`, `undersampled_`, `band_removed_relative_amplitude` |
| `propagation.InterfaceMap`, `propagation.RayMap`, `RefractedRays` | `transport.InterfaceMap`, `transport.RayMap`, `transport.RefractedRays` | `InterfaceMap(..., branch=)`; `RayMap.caustics`; `RayMap.flux` is complex |
| `propagation.eikonal.ray_weights` | `transport.ray_weights` | also `transport.carried`, `fresnel`, `check_window` |
| `propagation.CarrierFresnel`, `CarrierTransmissionReport` | removed | the dyadic is `transport.carried`, applied by the legs |
| `propagation.DirectionResolvedFresnel` | `alternatives.SpectralFresnel` | the alias is gone |
| `propagation.SurfaceTransport`, `InverseReport`, `SpectralFresnel`, `LocalFresnel`, `InterfaceSolution`, `scatter_plane` | `alternatives.*` | |
| `LatticeSpectrum` (`propagation._lattice`) | `fields.LatticeSpectrum` | |
| `HankelField` | `fields.HankelSpectrum` | now has `propagate` |
| `BeamField` | `fields.BeamSpectrum` | |
| (none) | `fields.Spectrum` | the interface of all four kinds |
| `fields.EikonalField` | `vecdiff_validation.eikonal_field` | validation only |
| `LocalSpectrum`, `HalfSpaceSpectrum` | `vecdiff_validation.local_spectra` | validation only |
| `propagation.references.MieSphere`, `LayeredSphere`, `InfiniteCylinder` | `references.MieSphere`, `LayeredSphere`, `InfiniteCylinder` | |
| `propagation.references.DielectricBEM`, `DielectricFEM`, `SurfaceMesh`, `TetrahedralMesh`, `surface_mesh`, `tetrahedral_mesh`, `boundary_meridian` | `vecdiff_validation.*` | separate package, Fortran extension |
| `system.DielectricInterface(surface, minus, plus)` | `system.Refraction(surface, before, after, edge=None)` | |
| `OpticalSystem(interfaces, apertures=, stops=, incident_field=, incident_region=)` | `OpticalSystem(*elements, medium=None)` | elements `Refraction`, `Reflection`, `Stop` |
| apertures (material edges) | `Refraction(..., edge=)`, `Reflection(..., edge=)` | |
| (none) | `system.Reflection(surface, medium, behind=None)`, `system.Stop(radius, frame, z=)` | Fresnel reflection or ideal mirror; opaque screen with a hole |
| (none) | `media.air`, `media.vacuum` | elements and beams take a material or a region |
| `fields.profiles.Uniform`, `Gaussian` | also `Annulus`, `Zones`, products `a * b` | profiles declare their `edges` |
| `observables.coherent_sum`, `energy_density`, `power`, `poynting` | also `irradiance`, `stokes`, `stokes_components`, `ensemble_stokes`, `ellipse_parameters`, `jones_components`, `spin_density`, `helicity_density`, `canonical_momentum`, `spin_momentum`, `orbital_angular_momentum`, `force`, `torque`, `ElectricFieldJet`, `plane_wave_jet` | |
| (none) | `ElectromagneticField.irradiance()`, `.valid` | |
| `backend=` (summation) | `method=` | renamed before `ac10128`; a leftover `backend=` is now refused on every path |
| `surfaces.Sphere` etc. met from inside | `ImplicitSurface.far_side()`, `GraphSurface.reversed()`, `surfaces.ReversedSurface` | the propagator chooses the view (`propagation.path`) |

## Behaviour

- **One method.** `Propagator` is the eikonal–angular-spectrum transport and
  nothing else; wave transport is `alternatives.WavePropagator`, which refuses
  edges, stops and reflections. An unknown keyword raises.
- **`Solution.field` returns E and H** (an `ElectromagneticField`); the
  pipeline's `field` returned E. `Solution.electric` returns E alone.
- **Coverage default "complete"** (the pipeline's was "reduced"): points in
  the band of heights a curved surface spans take the continued spectrum.
- **`method="auto"`** is the default of every plane-wave sum
  (`PlaneWaveExpansion.evaluate` and the references defaulted to "direct"):
  FINUFFT when it is installed and cheaper by estimate, the direct sum for
  small or complex sums.
- **A call solves every element.** The propagator carries the light across
  the whole path on each call, reusing prepared legs; it keeps no previous
  Solution.
- **Stops and edges are carried.** A `Stop` is an encounter of its own;
  rays meeting a face beyond its `edge` leave the path and are reported.
  The pipeline rejected both.
- **The default fan.** `fans=None` carries every ray the encounter supports.
  On the meridian a surface that never stops carrying the carrier (a plane
  under a collimated beam) ends the fan at twice the beam's footprint; the
  pipeline raised there. After a stop the next leg carries the light the
  edge diffracts, 6 λ0/n + 3 √(λ0 d/n) beyond it.
- **Rays the fan cuts count as unsupported** on the lattice, as on the
  meridian (notebook 08's cylinder now reports 7.91e-02 instead of 4.58e-02
  for the same field).
- **The caustic index.** Through a focus inside a leg the flux factor
  continues as exp(−iπκ/2)/√|J|; the leg took the positive root whenever
  J > 0 (a point focus came out with the wrong sign) and dropped rays past a
  focal line. The meridian refuses a carrier that focuses inside its leg.
- **Refusals.** A lattice leg is refused when more than 1 % of the carried
  amplitude is folded (beyond a fold of the map), undersampled (rays steeper
  than the output lattice samples) or band-removed (beyond sin = 0.995 of the
  outgoing medium). The meridian refuses light that crosses the axis between
  legs, a folding map, and a sampled field with more than 1e-2 beyond 16
  azimuthal orders.
- **Automatic focal planes.** Without `window=`, a target plane is sampled
  over the footprint of the rays that reach it.
- **Beam quadratures** break at the profile's edges (stops, annuli, phase
  belts).
- **References** compute their radial functions by recurrence: the Mie field
  of notebook 08 (R = 100 λ0) takes 1.8 s instead of 82.6 s.
- **Packaging.** `vecdiff` is pure Python (setuptools, NumPy and SciPy); the
  full-wave solvers, meshing and the Fortran extension are the separate
  `vecdiff_validation` package.

## Not supported

- Absorbing or dispersive media within a calculation (`Dielectric` takes a
  real, positive index), coatings and thin films, partial coherence and
  broadband light (one wavelength per call; incoherent sums are the user's).
- Evanescent near fields of a surface, rim scattering of a finite body, and
  light that leaves the given path (another ghost is another system).
- Analysis outputs: PSF, MTF, Strehl ratio, Zernike or OPD maps, paraxial
  data, focus search.
- Closed surfaces (`Sphere`, `Cylinder`, `CartesianOvoid`) still subclass their
  caps, and JSON prescriptions refuse them and freeforms.
- The meridian realization needs coaxial surfaces of revolution and a
  symmetric carrier; everything else (tilts, decentres, folds, cylinders)
  needs `lattice=` or a sampled field.
- Wave transport (`alternatives`) carries refractions of unbounded faces
  only.
