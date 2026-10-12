# Architecture

`vecdiff_general` and `vecdiff_stigmatic` are independent distributions. They
share no runtime optical kernel; external validation compares explicit arrays.

| Responsibility | Location | Meaning |
|---|---|---|
| Coordinate placement | `geometry/` | Right-handed frames and global coordinates |
| Shape | `surfaces/`, `volumes.py` | Geometry, normals, membership and area Jacobians |
| Discretization | `sampling/` | Physical points, FFT lattices and quadrature measures |
| Matter | `media/` | Constitutive data and spatial assignment |
| Fields | `fields/` | Electric samples, E/H traces, modal and eikonal representations |
| Optical prescription | `system.py` | Oriented surface/media adjacency and sequences |
| Numerical transforms | `fourier/` | FFT, Hankel, angular harmonics and NUFFT |
| Homogeneous propagation | `propagation/homogeneous.py` | ASM and paraxial Fresnel evolution |
| Surface transfer | `propagation/surface_transfer.py` | Q, least-squares inversion and surface-equivalence reconstruction |
| Surface interaction | `propagation/surface_interaction.py` | Exact planar scattering; spectral and local Fresnel treatments |
| Composition | `propagation/pipeline.py` | Field/system/target operator with reusable forward states |
| Reference solutions | `propagation/references/` | Mie, Debye, BEM, FEM and independent transfer checks |
| Measurement | `observables/`, `visualization/` | Physical diagnostics, detectors and plots |

The private `_directions.py` and `_fresnel.py` modules contain shared algebra
for the surface-interaction methods. `comparisons/surface_radiation.py` provides
independent prescribed-current Green/Weyl evaluation. Reference solvers and
comparison tools are not imported or invoked by the propagation pipeline.

Lengths and vacuum wavelengths are metres; E and H are SI phasors with time
convention `exp(-i omega t)`. A field owns its sampling grid, wavelength and
values. Material context belongs to a region or solution representation.
Constructing an angular spectrum completes explicitly transverse plane data
using the specified material and branch. It is a field-representation operation;
propagating that spectrum is a separate homogeneous-medium operation.

`OpticalSystem` describes geometry and matter, and can optionally hold the incident field. Targets belong to propagation calls, not the system. Its optional aperture/stop data
can describe a reference geometry even when a particular solver cannot handle
it. The current forward pipeline rejects finite edges and stops explicitly.

The general package no longer includes the layer-stack/admittance solver,
feedback solver, experimental phase-space method, ideal-imaging model or
specialized rim coupling. Their runtime wrappers were retired with them. The notebooks were migrated
to supported methods, with changes of physical model stated explicitly. The dedicated stigmatic package retains its own documented models.

## Field → system → target

`PropagationPipeline(system, transmission="local" | "spectral")` is callable
with an existing incident field and a target surface/grid or coordinate-plane
shorthand. It returns the existing `ElectricField`; no H or eikonal input is
required. `targets.py` resolves placement/sampling without introducing a second
surface or field ontology. `GraphSurface.sample` produces the existing grids.
`LocalFresnel` owns phase-gradient estimation; `SpectralFresnel` owns directional
spectral interpolation. Both reuse the same R/Q and reconstruction machinery.
Prepared source and regional spectra are cached for successive target requests.
The lower-level solve/E-H evaluation API remains for observables and validation.
