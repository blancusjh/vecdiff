# General architecture review — 2026-09-16

> Historical review: the general propagation path has since been replaced by
> [explicit R/Q/T composition](OPERATIONAL_PROPAGATION.md). Descriptions below
> of ray-tube or Green-based primary evaluation describe the earlier design.


This review concerns the general library. The dedicated stigmatic library is
unchanged during general-library refinement. Spatial volumes and the Fourier
interface have been reorganized. FFT representation validation and aperture
node checks are now implemented; the remaining geometry, grid and topology
changes below are proposals.

## Implemented boundary

- `geometry/`: mathematical coordinate placement, currently `Frame`.
- `surfaces/`: placed surface shapes, their normals and area Jacobians.
- `volumes.py`: closed spatial extent, without material or field data.
- `media/distribution.py`: assignment of dielectric material to spatial extent.
- `sampling/`: evaluation coordinates and integration measures.
- `fields/`: field values and explicit solution representations.
- `interfaces/`: material adjacency and local optical transformations.
- `propagation/`: evolution under a stated physical model.
- `fourier/`: mathematical array transforms only.

Volumes are mathematical objects too. Their separate location expresses their
role in describing spatial extent; it does not make them material objects.
`SphereVolume` and the renamed `VolumeBetweenSurfaces` live together in one file.
The latter describes coaxial graph faces closed by a cylindrical rim, not an
arbitrary solid. Membership includes boundaries; intersecting faces are detected
only at queried interior coordinates. This is not a global validity certificate.

`LensVolume` and `geometry.domains` have been removed from the general API;
callers use `VolumeBetweenSurfaces` from the package root or `volumes`.

## Reproduced gaps

The [probe script](../validation/review_general_architecture.py) records its
observations in [JSON](../validation/results/general_architecture_review.json).
These probes document problems; they do not enshrine incorrect behavior as tests.

### 1. FFT representation contract — fixed

The original probe accepted four repeated modes as a four-cell FFT lattice and
produced a **4 V/m** disagreement at zero propagation distance.
[PlaneSpectrum](../general/src/vecdiff_general/fields/spectrum.py) now validates
centered transverse frequencies and their array ordering, including empty bins.
It requires the central physical grid point as its exact reference origin.
Frequency comparisons account for numerical roundoff and the accepted frame's
orthogonality error. Arbitrary or reanchored expansions remain valid
`PlaneWaveExpansion` objects; constructing an inconsistent FFT representation
raises rather than silently reordering or rephasing coefficients.

Regression checks cover repeated/permuted modes, a reanchored expansion that
preserves the physical modal field, and agreement between direct and FFT E/H
for both propagation directions and evanescent modes on a placed plane.

### 2. Aperture node bounds — fixed; coverage remains explicit

The original probe accepted **96 nodes** outside a 1 µm prescribed aperture.
[propagate_interfaces](../general/src/vecdiff_general/propagation/interface_assembly.py)
now checks all supplied aperture-bounded quadratures before the first interface
transformation. Radius is measured in each surface's local chart, with a
coordinate-roundoff allowance. Out-of-bounds nodes raise an indexed error;
no nodes are clipped and no weights are renormalized.

`OpticalSystem.apertures` denotes circular support radii, in metres. A smaller
integration patch is permitted, rim nodes are included, and `None` imposes no
radius bound. Node containment does not certify quadrature coverage or weights.
An absorbing stop and its diffraction still require an explicit physical model.
Tests cover early rejection at a later encounter, placed plane/curved surfaces,
rim nodes, smaller patches and preservation of local interface flux.

### 3. Surface discretization loses its coordinate chart

[sample_surface](../general/src/vecdiff_general/sampling/surface_sampling.py)
constructs a radial/azimuthal chart and then returns only Cartesian positions,
normals and area weights. The physical point at each array entry remains known,
but the chart axes and array-order declaration are lost. Conversely,
[SurfaceRadiation](../general/src/vecdiff_general/propagation/surface_radiation.py)
rejects `PolarGrid` because it requires the concrete `SurfaceGrid` class, even
when all required geometrical data are available.

**Next change:** retain typed chart coordinates and their embedding when creating
surface grids. Let current integration accept a validated oriented-surface grid
contract: positions, shape, unit normals and physical area weights. Keep
chart coordinates, vector-component bases and integration measures explicit.
Do not infer Cartesian versus polar from array shape. Handle zero area weights
at a polar axis deliberately rather than applying an unrelated positivity rule.

### 4. Shared material boundaries are treated as volume overlap

Two adjacent closed volumes both contain their shared face.
[MaterialDistribution](../general/src/vecdiff_general/media/distribution.py)
therefore raises an overlap error at that face, even though their interiors are
disjoint. Rejecting ambiguous index assignment is preferable to guessing, but
the present API cannot express the two one-sided material limits there.

**Next change:** distinguish interior membership from boundary membership and
associate each interface with its adjacent regions. Preserve rejection of
overlapping interiors. Provide explicit side selection at interfaces rather
than choosing a material by tuple order. Connect the spatial material assignment
to the region identities used by propagation so they cannot contradict each other.

### 5. A nonsmooth axis is silently assigned a smooth normal

[AxisymmetricSurface](../general/src/vecdiff_general/surfaces/axisymmetric.py)
returns a zero gradient at radius zero even for the conical profile `z(r) = r`.
It consequently returns normal `[0, 0, 1]` at an apex where a unique normal is
undefined.

**Next change:** establish regularity requirements for the axis, reject undefined
normals, and distinguish chart singularities from physical singularities. Test
smooth spherical/parabolic poles and a deliberately nonsmooth conical apex.

## Architecture to build next

1. **Mathematical geometry.** Add a small `geometry/curves.py` for curve evaluation
   and derivatives, and `geometry/revolution.py` for revolving a meridian
   `(r(t), z(t))`, tangent vectors, normals and area Jacobians. These functions
   receive mathematical data, without fields, wavelength, refractive index or
   propagation methods. A graph `z(r)` is a supported specialization. General
   parameterization should permit vertical meridians and overhangs without
   pretending every surface is a graph. Define orientation and pole behavior.

2. **Surface and grid composition.** Placed surfaces apply frames to mathematical
   shapes. Grids retain a typed chart and map every array entry to a physical
   position. Quadrature uses the corresponding measure. Field constructors
   should continue to own values, wavelength and grid, without duplicated medium
   or sampling state. Implement this with a few contracts and concrete types,
   not one file per abstraction.

3. **Material topology and optical prescriptions.** Unify spatial material
   assignment and interface adjacency without placing matter inside geometry.
   `OpticalSystem` currently describes an ordered forward encounter path, not a
   whole closed-body system. `LayerStack` belongs with optical arrangements once
   that contract is settled, rather than constitutive material definitions.
   A single surface-plus-adjacency interface can represent planes and curved
   faces; solver selection can retain an exact planar specialization.

4. **Operators and representations.** Preserve independently callable Fresnel
   transmission and reflection. Higher-level `scatter_surface` currently always
   constructs both branches; offer branch-specific operations without forcing
   unnecessary work. Likewise, `.electric()` currently computes H as well in
   several representations. Keep result records only where they hold meaningful
   state needed to inspect a solution. Validate representation domains, including
   `HalfSpaceSpectrum.direction` and its finite boundary coordinate. An evaluated
   E/H pair is data; matching shapes alone cannot certify Maxwell consistency.

5. **Observables and readability.** Make the detector's orientation and measure
   explicit when combining fields; equal point coordinates alone do not establish
   equal quadrature or surface normals. Keep physical stages separated by blank
   lines, use descriptive intermediate quantities, and replace `object`
   annotations with small, useful contracts. Retain the four-file field package
   and the consolidated polar Fourier module.

With the representation and aperture-node checks delivered, the next steps
are mathematical geometry and chart-preserving grids, followed by material
topology and system composition. Do not reorganize the stigmatic library during
these steps.

## Physical acceptance checks

| Change | Required evidence |
|---|---|
| Curve revolution and surface placement | Analytic sphere/cylinder normals and areas, meridian reparameterization invariance, rigid-motion covariance, explicit treatment of poles and undefined normals |
| Typed surface grids | Array entry → chart → physical point agreement; constant-function area integrals; agreement between Cartesian and polar quadrature under refinement |
| Material topology | Interior/background assignment, shared-face one-sided material queries, rejection of true overlap, agreement between interface adjacency and spatial material limits |
| Field representations | Zero-distance identity; FFT versus direct complex E/H; origin and rotation covariance; correct spatial-validity rejection |
| Planar interfaces | All four source-free Maxwell jump conditions and signed normal flux; Brewster, normal incidence and TIR controls; independent transmission/reflection calculations |
| Curved propagation | Reconstructed one-sided fields under independent quadrature and offset refinement; report remaining boundary error rather than checking only prescribed local traces |
| Complete lossless bodies | Outward flux of total E/H on a closed enclosing surface, interference included; convergence with enclosure and discretization; independent full-wave reference before claiming an exact dielectric solver |
| Finite patches and forward paths | Separate uncollected flux, truncation error and omitted return paths; never force finite-detector power to match input by normalization |

Existing checks remain preservation anchors. In particular, the reported curved
physical-optics example has a normal-D residual near **0.1475** despite converged
quadrature at the tested offsets. Renaming modules cannot remove that model
error. Fully coupled curved dielectric scattering remains a separate solver
and validation task; the generic feedback equation alone does not supply it.
