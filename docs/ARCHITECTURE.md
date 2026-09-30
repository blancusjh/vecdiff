# Implemented architecture

The general and stigmatic libraries are independent implementations of corresponding physical concepts. Independence applies to imports, installation, tests, and debugging. There is no shared runtime optical kernel. External comparisons exchange explicit physical arrays and units.

## Space, matter, fields

| Responsibility | Objects | Ownership and behavior |
|---|---|---|
| Mathematical geometry | `Frame` | Coordinate placement and right-handed bases; no material or field dependencies. |
| Spatial extent | `SphereVolume`, `VolumeBetweenSurfaces` in `volumes.py` | Closed spatial regions and membership queries, independently of matter. |
| Surface geometry | `Plane`, `SphericalCap`, `EvenAsphere`, `FreeformSurface`, `AxisymmetricSurface`, `CartesianOval` | Shape, analytic normal, area Jacobian. Design indices on an oval determine its shape; they do not assign matter in space. |
| Discretization | `PointGrid`, `PlaneGrid`, `SurfaceGrid`, surface quadrature functions | Physical coordinates and integration measure. Sampling resolution is not duplicated in a field. |
| Constitutive matter | `Dielectric`, `HomogeneousRegion`, `MaterialRegion`, `MaterialDistribution`, `LayerStack` | Initial lossless isotropic nonmagnetic materials; region identity, piecewise spatial index, ordered layers. |
| Electric field values | `ElectricField`, `TransverseElectricField` | Vacuum wavelength, physical grid, immutable samples. Complete Cartesian vector versus explicitly incomplete plane input. |
| Solution representations | `PlaneWaveExpansion`, `PlaneSpectrum`, `LocalSpectrum`, `EikonalField` | Modal branch or declared phase, local origin, material context and representation validity. These are not extra state hidden in `ElectricField`. |
| Evaluated electromagnetic data | `ElectromagneticField` | E and consistent SI H evaluated on the same physical support. |
| Optical encounters | `DielectricInterface`, `PlanarInterface`, `OpticalSystem`, `StigmaticSystem` | Adjacent regions or conjugates; physical prescription. Geometry alone has no transmission law. |
| Evolution | `propagation/` algorithms and returned solutions | Apply a declared model and preserve inspectable incident, reflected, transmitted and intermediate results. |
| Measurement | `observables/`, `visualization/` | Poynting flux, signed power, energy density, coherent detector formation and clearly labeled plots. |

A sampled E field cannot, by itself, recover a unique H or Ez in every geometry. Fourier longitudinal completion is a homogeneous planar operation with a chosen propagation branch. Across matter boundaries the correct source-free constraint is `div(epsilon E) = 0`. Energy density requires material information; total energy additionally requires a volume measure. The current library exposes local energy density and oriented surface power instead of guessing a volume from a plane grid.

## General source map

```text
vecdiff_general/
    volumes.py                      sphere and region between two faces with a rim
    geometry/
        frames.py                   global placement and local axes
    surfaces/
        surface.py                  graph-surface contract, plane, callable freeform
        axisymmetric.py             radial profile and signed spherical cap
        asphere.py                  conic plus even radial powers
        cartesian_oval.py            general placed oval geometry
    sampling/
        grids.py                    CartesianGrid, PolarGrid and explicit point/surface data
        surface_sampling.py         Gauss/azimuth surface integration
        near_surface.py             target-centred quadrature for one-sided limits
    media/
        medium.py                   dielectric constitutive data and region identity
        distribution.py             material assignment as a function of position
        layers.py                   homogeneous slabs and interface positions
    fields/
        __init__.py                 public sample types
        electromagnetic_field.py     ElectricField, transverse input, evaluated E/H
        spectrum.py                 plane-wave, FFT, local and half-space spectra
        eikonal.py                  declared spatial phase and transverse envelope
    interfaces/
        dielectric_interface.py     oriented adjacency of two material regions
        optical_system.py           ordered forward encounters
        directions.py               Snell and reflection directions without fields
        fresnel.py                  separate transmission/reflection operators
        layer_admittance.py         fixed tangential basis for coherent layers
    fourier/
        cartesian.py                centered FFT2/IFFT2 and frequency_grid((x, y))
        polar.py                    FT2/IFT2, HT/IHT and angular harmonics
        nufft.py                    NUFFT sums, weighted integrals and safe phases
    propagation/
        angular_spectrum.py         optical dispersion, branch and Ez completion
        propagation.py              homogeneous ASM and paraxial Fresnel evolution
        interface_transform.py      exact single-plane solution and total fields
        layered_propagation.py      stable all-orders coherent planar recursion
        operators.py                planar R, curved Q/inverse, pointwise Fresnel T
        high_frequency.py           repeated R Q^-1 T Q R and outgoing ASM
        interface_assembly.py       forward sequence and coherent plane prescription
        multiple_scattering.py      explicit linear feedback equation with residual
        imaging.py                  ideal aplanatic pupils, focus and scalar images
        references/                 native BEM/FEM, Mie, Debye terms, meshing,
                                    periodic-layer reference, benchmarks
    comparisons/
        curved_interface.py         comparison-only Fresnel currents
        surface_radiation.py        comparison-only dyadic Green and local/Weyl tools
    observables/
        electromagnetism.py         SI Poynting, power, energy and coherent sum
        polarization.py             Jones projections, Stokes and ellipse parameters
        diagnostics.py              all four interface jump residuals
        imaging.py                  coherent/incoherent detector flux
    visualization/
        fields.py                   Cartesian component maps on physical planes
        polarization.py             projected polarization ellipse maps
        animation.py                exp(-i omega t) polarization animation
    io/
        prescription.py             versioned explicit-SI JSON prescriptions
```

## Dedicated stigmatic source map

The stigmatic package has its own `geometry/frames.py`, `sampling/grids.py`, `media/medium.py`, electric/electromagnetic fields, Fourier operations, observables and visualization files. Its specific objects are:

```text
vecdiff_stigmatic/
    surfaces/cartesian_oval.py       constant-OPL vertex-connected branch
    geometry/rays.py                 ray directions, reference measures and A
    interfaces/optical_system.py     materials, real conjugates and ray construction
    interfaces/directions.py         Snell and reflection geometry
    interfaces/fresnel.py            separate transmission/reflection operators
    fields/electromagnetic_field.py                complete/transverse E and evaluated SI E/H
    fields/sources.py                entrance-sphere geometric illumination
    fields/transfer_result.py        reference and physical-interface E/H, power budget
    propagation/interface_transform.py  entrance G → interface → exit G′
    propagation/debye.py             direct, Hankel or NUFFT vector focusing
    propagation/fresnel.py           separate ideal paraxial pupil model
```

This preserves the dedicated ray and reference-sphere workflow. It does not route stigmatic evaluation through the general engine. The general package has its own oval geometry and can reproduce the physical-current calculation, enabling a useful independent comparison.

## Dependency direction

Geometry and materials do not import propagation. Grids depend on geometry; electric samples depend on their grid. Solution representations explicitly own the context needed for propagation. Interface laws are reusable without assigning a field's spatial phase. Propagation assembles those laws, Fourier operations, and representations. Observables consume evaluated E/H. Plotting imports Matplotlib only when called.

Small `evaluate`/`propagate` convenience methods delegate to the responsible operation module. Importing a field type does not execute a transform or mutate its grid. Public constructor arrays are copied and made read-only where the objects represent immutable input data.

## Model selection is explicit

| Requested calculation | Entry point | Physical contract |
|---|---|---|
| Homogeneous parallel planes | `angular_spectrum(...).propagate(..., model="asm")` | Exact homogeneous modal evolution for the retained periodic discretization. |
| Paraxial free propagation | `model="fresnel"` | Approximate phase; not an exact Maxwell spatial solution. |
| One infinite plane interface | `scatter_plane` | Exact Fresnel scattering for supported incident modes; reflected and evanescent transmitted fields retained. |
| Parallel coherent layers | `propagate_layers` / `propagate_coherent_planes` | All orders, internal standing waves and evanescent gaps; stable face-anchored amplitudes. |
| General graph interfaces | `propagate_interfaces(source, system, grid)` | Explicit R Q^-1 T Q R; outgoing homogeneous ASM evaluation. |
| Comparison: known surface currents | `comparisons.SurfaceRadiation.evaluate` | Full dyadic Green integral of those currents; quadrature and source-model errors are separate. |
| Comparison: local observation ball | `comparisons.SurfaceRadiation.local_spectrum` | Conservative absolute E/H bounds against the same discrete currents. |
| Macroscopic optical path | `propagate_high_frequency` | Explicit R/Q/T operators with a solved Q inverse; propagating forward WKB branches and Cartesian sampling. |
| Dedicated stigmatic focus | `transfer` then `DebyeFocus` | Geometric reference-sphere transfer and homogeneous vector Debye field. |
| Ideal paraxial pupil focus | `fresnel_focus` | Thin-lens Fourier model, physical output coordinates, first-order longitudinal fields. |

`coherent_feedback` is a reusable linear-equation solver with an algebraic residual, not an already qualified arbitrary curved-body scattering solver. No API claims a closed-body Maxwell solution merely because a geometrical volume can be defined.

## Readability contract

Four spaces, conventional operator spacing, descriptive physical names, one logical operation per statement, and blank lines between validation, geometry, phase, and amplitude steps are enforced and reviewed. Transform normalization, units, frame conventions and approximation domains are documented beside the relevant routines.

## Mathematical transforms and physical operators

`fourier/` is a mathematical package. Its routines accept arrays, sample counts,
spacings, frequencies and quadrature weights. They do not import fields, grids,
materials, interfaces or propagation. In general, `frequency_grid((x, y))` receives numerical coordinates and needs no
`PlaneGrid`; its centered frequencies are cycles per unit. The optical angular-spectrum construction, material dispersion,
propagation branch and longitudinal completion belong to
`propagation/angular_spectrum.py`; ASM/Fresnel evolution belongs to
`propagation/propagation.py`. Tests enforce the Fourier dependency boundary.

An interface transformation is an operator on electric data at a fixed surface.
`fresnel_transmission_matrix` constructs T; `fresnel_reflection_matrix` constructs R.
Their coefficient functions are independent. `apply_fresnel(E, T)` acts on arrays;
`transmit(field, ...)` and `reflect(field, ...)` return an `ElectricField` on the
same grid with the same vacuum wavelength. No aggregate Fresnel result or
coefficient class is required. A complete scattering calculation can explicitly
request both operations to retain both physical branches and their power budget.

Directions are computed separately. A ray calculation calls Snell geometry,
without inventing zero electric data to obtain a direction. Under TIR the
transmitted direction is complex and has bilinear norm one. The interface
operator retains that complex direction; normal incidence uses a regular basis.
The library normal points into transmitted matter, so its incidence cosine is
`dot(u_i, normal)`. A normal pointing toward incident matter must be reversed.

## Unified Fourier API and explicit sampling charts

Both libraries provide `fourier.transform` and `fourier.inverse`, with explicit
`mode="cartesian"` or `mode="polar"`. They use the same continuous transform
normalization. General now exposes centered `FFT2`/`IFFT2` with NumPy
normalization; its propagation layer explicitly converts to and from modal
amplitudes. General Cartesian continuous transforms are also centered. The
stigmatic implementation and its existing ordering remain unchanged. See
[Fourier conventions](FOURIER_AND_GRIDS.md) before exchanging raw spectra.

`CartesianGrid` declares array order `(v,u)` and numerical axes `(u,v)`.
`PolarGrid` declares array order `(radius,azimuth)` and numerical axes `(r,phi)`.
Both map every index to a global Cartesian point through their physical frame.
Their `coordinate_system`, `array_order`, `coordinates`, `points` and integration
weights are explicit. `PlaneGrid` is a compatibility alias for `CartesianGrid`,
not a second implementation. Polar sampling does not change the electric-vector
basis: complete E is still global Cartesian, and transverse input uses the
plane's local u/v basis.

The Fourier API receives `grid.coordinates` and `grid.coordinate_system`, not the
Grid object. This retains a mathematical Fourier package while removing shape
inference from applications. Polar inverse integration uses finite radial
frequency support and quadrature; it is an approximation to the continuous
inverse, not an artificial exact matrix inversion. See [FOURIER_AND_GRIDS.md](FOURIER_AND_GRIDS.md).

Each `fields/` directory now has exactly four files including `__init__.py`.
General: `electromagnetic_field.py`, `spectrum.py`, `eikonal.py`. Stigmatic: `electromagnetic_field.py`,
`sources.py`, `transfer_result.py`. Polarization calculations are observables.

## General geometry and spatial volumes

In the general library, `geometry/` is a mathematical toolbox. It currently
contains `Frame`; generic curve, revolution, normal and Jacobian operations are
next candidates for this layer. They are not yet all implemented.

`volumes.py` contains `SphereVolume` and `VolumeBetweenSurfaces`. The latter is
restricted to coaxial graph faces with a shared orientation and cylindrical rim.
It has no material, wavelength or field. `MaterialRegion` in
`media/distribution.py` associates a spatial volume with a dielectric, and
`MaterialDistribution` evaluates the resulting piecewise refractive index.

## General polar and nonuniform interface

General polar operations use `FT2(g, coordinates, frequencies)` and `IFT2(G, ...)`,
with radial `HT(g, r, q)` / `IHT(G, q, r)` in the same file. Nonuniform operations
use the numerical `NUFFT(g, x, q)` sum with an explicit exponent sign. The common
continuous `transform` / `inverse` interface also accepts `mode="nonuniform"`
with required integration weights. These functions contain no optical objects.
Inverse polar and nonuniform integrals approximate the continuous inverse;
they do not promise exact inversion of arbitrary discrete quadratures.

## Validated representations and aperture bounds

`PlaneSpectrum` combines a Cartesian grid with an expansion only if its
transverse wavevectors match the centered FFT frequencies and array order.
The expansion's phase origin must equal the central physical grid point.
General plane-wave collections continue to use `PlaneWaveExpansion`.

Sequential interface propagation checks each quadrature's local radial nodes
against the corresponding prescribed aperture before scattering. The radius
bounds numerical support without defining a stop model. Node containment,
complete aperture coverage and quadrature accuracy are separate claims.

See [propagation](PROPAGATION.md) for the equations, inverse, sampling contracts, edges and stops, and limitations.

Independent full-wave comparisons live in `propagation/references`: our Fortran
PMCHWT/RWG dielectric BEM (with a second-kind Müller alternative tested by
Buffa–Christiansen functions) and Nédélec vector FEM/PML, plus analytical Mie spheres.
Python owns validated meshes, units and solve orchestration; the native core
assembles operators and reconstructs fields. They are
explicit imports and are never selected by primary propagation. See
[validation](VALIDATION.md) and notebook 06 for their
scope, independent physical checks, execution timings and convergence limits.
