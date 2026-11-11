# Architecture

`vecdiff` and `vecdiff_stigmatic` are independent distributions. They
share no runtime optical kernel; external validation compares explicit arrays.

| Responsibility | Location | Meaning |
|---|---|---|
| Array device | `backend.py` | NumPy on the CPU or CuPy on a CUDA GPU, one global switch |
| Coordinate placement | `geometry/` | Right-handed frames and global coordinates |
| Shape | `surfaces/`, `volumes.py` | Geometry, normals, membership and area Jacobians |
| Discretization | `sampling/` | Physical points, FFT lattices and quadrature measures |
| Matter | `media/` | Constitutive data and spatial assignment |
| Fields | `fields/` | Electric samples, E/H traces, beams and their carriers; `Spectrum`, the plane-wave field of a region, in four kinds (plane, lattice, Hankel, beam) |
| Optical prescription | `system.py` | Oriented surface/media adjacency and sequences |
| Numerical transforms | `fourier/` | FFT, Hankel, angular harmonics and NUFFT |
| Transport across a surface | `transport/` | The ray map, its Jacobian and caustic index (`rays.py`); the continued Fresnel dyadic (`dyadic.py`); the leg on a lattice (`lattice.py`) and on a meridian (`meridian.py`); its report |
| Homogeneous propagation | `propagation/homogeneous.py` | ASM and paraxial Fresnel evolution |
| The method | `propagation/propagator.py` | `Propagator(system)(light)` -> `Solution`: spectra per region, reports per surface, E and H on targets; legs reused |
| Wave transport | `alternatives/wave_propagator.py` | `WavePropagator`, the same Solution without a carrier, for checks |
| Exact solutions | `references/` | Mie, layered sphere and cylinder with their Debye series |
| Other methods | `alternatives/` | Surface transfer Q with least-squares inversion; exact planar scattering; spectral and local Fresnel treatments |
| Independent solvers | `validation/` (package `vecdiff_validation`) | BEM, FEM, MFS and prescribed-current comparisons, outside the library |
| Measurement | `observables/`, `visualization/` | Physical diagnostics, detectors and plots |

Public objects are reached through the namespace of what they are:
`surfaces.SphericalCap`, `fields.angular_spectrum`, `system.OpticalSystem`,
`propagation.Propagator`. The package top level holds only these
namespaces and re-exports nothing. `visualization` and `references`
are imported explicitly, so importing the package loads neither plotting nor
reference solvers.

The private `_directions.py` and `_fresnel.py` modules contain shared algebra
for the surface-interaction methods in `alternatives/`. `vecdiff_validation.comparisons.surface_radiation` provides
independent prescribed-current Green/Weyl evaluation. Reference solvers and
comparison tools are not imported or invoked by the propagation.

Lengths and vacuum wavelengths are metres; E and H are SI phasors with time
convention `exp(-i omega t)`. A field owns its sampling grid, wavelength and
values. Material context belongs to a region or solution representation.
Constructing an angular spectrum completes explicitly transverse plane data
using the specified material and branch. It is a field-representation operation;
propagating that spectrum is a separate homogeneous-medium operation.

`OpticalSystem` describes geometry and matter, and can optionally hold the incident field. Targets belong to propagation calls, not the system. Its optional aperture/stop data
can describe a reference geometry even when a particular solver cannot handle
it. The propagators reject finite edges and stops explicitly.

The general package no longer includes the layer-stack/admittance solver,
feedback solver, experimental phase-space method, ideal-imaging model or
specialized rim coupling. Their runtime wrappers were retired with them. The notebooks were migrated
to supported methods, with changes of physical model stated explicitly. The dedicated stigmatic package retains its own documented models.

## Field → system → target

`propagation.Propagator(system)` is called with the light (and its carrier,
if the light is not a beam) and returns a `Solution`, whose `field` takes a
target surface/grid or coordinate-plane shorthand and returns E and H
together (`electric` gives E alone). `alternatives.WavePropagator(system,
transmission="local" | "spectral")` returns the same `Solution` by wave
transport, without a carrier. Every spectrum in a Solution is a
`fields.Spectrum`. `targets.py` resolves placement/sampling without introducing a second
surface or field ontology. `GraphSurface.sample` produces the existing grids.
`LocalFresnel` owns phase-gradient estimation; `SpectralFresnel` owns directional
spectral interpolation. Both reuse the same R/Q and reconstruction machinery.
A Solution holds the regional spectra, so successive targets need no further
transport; the propagators keep their prepared legs across calls.
