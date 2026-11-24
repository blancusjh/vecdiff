# Architecture

The library is the method: a field carried along the rays of its carrier
across each surface and re-expanded in plane waves between surfaces.
Everything else either describes the physical situation the method is
applied to, measures its result, checks it, or is infrastructure. The
namespaces say which, in this order:

| Responsibility | Location | Meaning |
|---|---|---|
| Space | `geometry/` | Right-handed frames and global coordinates |
| Matter | `media/` | Materials (`Dielectric`, `air`, `vacuum`) and the regions of space they fill (`HomogeneousRegion`); `MaterialDistribution` for piecewise n(r) |
| Shape | `surfaces/`, `volumes.py` | Surfaces with normals, Hessians and area Jacobians (graph caps, closed implicit bodies, reversed views); volumes for the references |
| The optical system | `system.py` | The ordered path of encounters (`Refraction`, `Reflection`, `Stop`); the regions it crosses, derived from it |
| Light | `fields/` | Beams from a profile and a polarization, sampled fields, the carrier; `Spectrum`, the plane-wave field of a region, in four kinds (plane, lattice, Hankel, beam) |
| Transport across one surface | `transport/` | The ray map, its Jacobian and caustic index (`rays.py`); the continued Fresnel and reflection dyadics (`dyadic.py`); the leg on a lattice (`lattice.py`) and on a meridian (`meridian.py`); its report (`report.py`) |
| Propagation through the system | `propagation/` | `Propagator(system)(light)` -> `Solution` (`propagator.py`); the path as the light meets it (`path.py`); plane waves in a homogeneous medium (`homogeneous.py`); targets and coverage (`targets.py`, `coverage.py`) |
| Measurement | `observables/` | Energy flow, power, irradiance, polarization, momentum and spin, forces, detectors |
| Exact solutions | `references/` | Mie, layered sphere and cylinder with their Debye series |
| Other methods | `alternatives/` | `WavePropagator` (wave transport, the same `Solution` without a carrier), `SurfaceTransport` with least-squares inversion and radiation, `SpectralFresnel`, `LocalFresnel`, exact planar scattering |
| Numerical infrastructure | `fourier/`, `sampling/`, `backend.py` | FFT, Hankel, angular harmonics and NUFFT; grids and quadrature measures; NumPy on the CPU or CuPy on a CUDA GPU |
| Tools | `visualization/`, `io/`, `catalog.py` | Plots; JSON prescriptions of a path; catalog lenses |
| Independent solvers | `validation/` (package `vecdiff_validation`) | BEM, FEM, MFS, meshing and prescribed-current comparisons, outside the library |

Public objects are reached through the namespace of what they are:
`surfaces.SphericalCap`, `fields.angular_spectrum`, `system.OpticalSystem`,
`propagation.Propagator`. The package top level holds only these
namespaces and re-exports nothing. `visualization` and `references`
are imported explicitly, so importing the package loads neither plotting nor
reference solvers. `vecdiff_stigmatic`, the earlier library for stigmatic
optics, is an independent distribution in its own repository; it shares no
runtime optical kernel with `vecdiff`, and external validation compares
explicit arrays.

## Units

Lengths are in one unit, that of the vacuum wavelength `wavelength0`. E and H are phasors in V/m and
A/m with time convention `exp(-i omega t)`, so the Poynting vector is in
W/m². Quantities that integrate over an area or involve the frequency
(power, momentum, spin, force) are SI when lengths are metres. The JSON
prescriptions (`io`), the catalog, the Cartesian-oval parameters and the
plotting helpers (which label micrometres) assume metres.

## Field → system → target

`OpticalSystem` is the ordered path of the light's encounters and the
regions of space it crosses, which it derives from the path. The light and
the targets belong to the call, not to the system.

`propagation.Propagator(system)` is called with the light (and its carrier,
if the light is not a beam) and returns a `Solution`. The propagator traces
the light's axis through the path (`propagation.path`), so that every leg
meets its surface along the propagation, and carries the light across every
element with a `transport` leg; between legs the field is a `Spectrum` of
the segment's region. The `Solution` holds one spectrum per segment, one
`TransportReport` per carried encounter and the prepared legs; its `field`
takes a target surface or grid, or a coordinate-plane shorthand, and returns
E and H together (`electric` gives E alone). `targets.py` resolves placement
and sampling without introducing a second surface or field ontology;
`coverage.py` assigns each point to the segments that reach it.

Legs depend on the geometry, the carrier and the incident lattice or beam,
not on the field: the propagator keeps them across calls with the same ones.
A Solution holds the regional spectra, so successive targets need no further
transport.

`alternatives.WavePropagator(system, transmission="local" | "spectral")`
returns the same `Solution` by wave transport, without a carrier: it carries
refractions of unbounded faces only and refuses edges, stops and
reflections. `LocalFresnel` owns phase-gradient estimation; `SpectralFresnel`
owns directional spectral interpolation; both reuse the same surface transfer
and reconstruction. The private `alternatives/_directions.py` and
`_fresnel.py` modules hold their shared algebra. `system.PlanarInterface` is
the planar boundary of `alternatives.scatter_plane`, not an element of a path.

Constructing an angular spectrum (`fields.angular_spectrum`) completes
explicitly transverse plane data using the specified material and branch.
It is a field-representation operation; propagating that spectrum is a
separate homogeneous-medium operation (`propagation.homogeneous`). Surface
geometry and adjacent materials are specified independently of the
interaction model.

Reference solvers and comparison tools are never imported or invoked by the
propagation. `vecdiff_validation.comparisons.surface_radiation` provides
independent prescribed-current Green/Weyl evaluation.

The general package no longer includes the layer-stack/admittance solver,
feedback solver, experimental phase-space method, ideal-imaging model or
specialized rim coupling; they remain in the Git history. The stigmatic
package, in its own repository, retains its own documented models.
