# The method: eikonal–angular-spectrum transport

vecdiff is one method, the eikonal–angular-spectrum transport of the
manuscript *Vectorial field transport through general curved interfaces*.
Across each surface the light meets, its field is carried along the rays of
its **carrier** (the ray family it declares) from a reference plane in front
of the surface to the same plane on the side the light leaves into; between
surfaces it is an exact superposition of plane waves. The transfer across a
surface is the leading-order solution of Maxwell's equations in the size of
the surface relative to the wavelength: it is exact for a plane wave on a
plane interface, the identity for index-matched media, and its error falls
as 1/(R/λ0) on curved surfaces. The method is meant for macroscopic systems.

`propagation.Propagator(system)(light)` is the method. This guide describes
it in the order of the library: the system, the leg that crosses one
surface, the two ways a leg is realized, sequences, reports, results, and
the exact references it is checked against. [Other methods](#other-methods),
kept for checks, come last.

| Namespace | Main objects | What they are |
|---|---|---|
| `system` | `OpticalSystem`, `Refraction`, `Reflection`, `Stop` | The path of encounters the light meets |
| `fields` | `PlaneBeam`, `FocusedBeam`, `Carrier`, `Spectrum` and its kinds | The light, its carrier, the plane-wave field of a region |
| `transport` | `InterfaceMap`, `RayMap`, `RefractedRays`, `LatticeTransport`, `MeridianTransport`, `TransportReport` | The field carried across one surface along its carrier's rays |
| `propagation` | `Propagator`, `Solution`, `CoverageReport`; `PlaneTransport`, `propagate` | The method through the whole path; plane waves between surfaces |
| `references` | `MieSphere`, `LayeredSphere`, `InfiniteCylinder` | Exact solutions with their Debye terms, to check against |
| `alternatives` | `WavePropagator`, `SurfaceTransport`, `SpectralFresnel`, `LocalFresnel`, `scatter_plane` | Other methods, for checks |

Lengths are in one unit, that of `wavelength0`; the examples use metres or
wavelengths. E is in V/m and H in A/m (H = E/Z0 for a plane wave in vacuum),
so the Poynting vector is in W/m²; powers and momenta are SI when lengths are
metres. Time dependence is exp(−iωt).

## In one call

A beam converging toward a glass sphere of radius 30 wavelengths, carried
into it and compared, on a line through the aplanatic image point, with the
sphere's exact first entry (the wave that crossed the surface once):

```python
import numpy as np
import vecdiff as vd
from vecdiff.references import MieSphere

air, glass = vd.media.air, vd.media.Dielectric(1.5, "glass")
R = 30.0  # in wavelengths: lengths are in the unit of wavelength0
beam = vd.fields.FocusedBeam(  # converging toward z = 1.5 R at NA 0.4
    vd.fields.profiles.Uniform(0.4),
    vd.fields.polarization.cartesian(1, 1j),
    focal_length=1.0,
    max_sine=0.4,
    wavelength0=1.0,
    region=air,
    frame=vd.geometry.Frame([0, 0, 1.5 * R]),
)
sphere = vd.surfaces.SphericalCap(R, vd.geometry.Frame([0, 0, -R]))
system = vd.system.OpticalSystem(vd.system.Refraction(sphere, air, glass))

solution = vd.propagation.Propagator(system, fans=((50.0, 56.0),))(beam)
print(solution)  # one line per carried element: what it did not carry

x = np.linspace(-5, 5, 81)
line = vd.sampling.PointGrid(np.c_[x, 0 * x, np.full(81, R / 1.5)])
field = solution.field(line)  # E and H together

inside = vd.media.HomogeneousRegion("glass", glass)  # references take regions
mie = MieSphere(beam, vd.volumes.SphereVolume([0, 0, 0], R), inside)
exact = mie.evaluate(line, side="interior", term=1)
error = np.linalg.norm(field.electric.values - exact.electric.values)
print(f"relative error {error / np.linalg.norm(exact.electric.values):.4f}")
```

It prints a relative error of 0.0226 (measured now; the whole script, Mie
reference included, runs in 0.2 s on a laptop). The report line says that
9.7 % of the incident amplitude lies on rays the fan does not carry (beyond
56° incidence). [Notebook 08](../general/notebooks/08_eikonal_angular_spectrum_transport.ipynb)
shows the same geometry at 100 λ0, its 1/R error law, two surfaces in
sequence and a cylinder.

## The system: a path of encounters

`system.OpticalSystem(*elements)` is the ordered path the light follows. Its
elements are what the light meets, in order:

```python
import vecdiff as vd

air, glass = vd.media.air, vd.media.Dielectric(1.5, "glass")
Frame, Plane = vd.geometry.Frame, vd.surfaces.Plane
curved = vd.surfaces.SphericalCap(-25e-3, Frame([0, 0, 3e-3]))
lens = vd.system.OpticalSystem(
    vd.system.Stop(4e-3, z=-0.5e-3),  # an opaque screen with a hole
    vd.system.Refraction(Plane.z(0), air, glass),
    vd.system.Refraction(curved, glass, air, edge=6e-3),  # the face ends at 6 mm
)
paraboloid = vd.surfaces.EvenAsphere(-1 / 10e-3, conic=-1.0)
mirror = vd.system.OpticalSystem(vd.system.Reflection(paraboloid, air))  # ideal
front, back = Plane.z(0.0), Plane.z(1e-3)
ghost = vd.system.OpticalSystem(  # the same two surfaces, met four times
    vd.system.Refraction(front, air, glass),
    vd.system.Reflection(back, glass, behind=air),  # Fresnel reflection
    vd.system.Reflection(front, glass, behind=air),
    vd.system.Refraction(back, glass, air),
)
print([region.name for region in ghost.regions])
```

The last line prints `['air', 'glass', 'glass', 'glass', 'air 2']`: the
light arrives in air, stays in the glass through both reflections, and
leaves into the air beyond the back face, a region distinct from the air in
front.

- **`Refraction(surface, before, after, edge=None)`** carries the light across
  a surface into the medium beyond; **`Reflection(surface, medium, behind=None)`**
  turns it back into its own medium: with `behind` the Fresnel reflection of
  that boundary (ghosts, internal and total internal reflection), without it
  an ideal mirror. **`Stop(radius, frame | z=)`** is an opaque screen across the
  path, placed in space by its frame, with a circular hole on its origin.
- **Media and regions.** A medium (`media.Dielectric`, `media.air`,
  `media.vacuum`) is what fills space; a region (`media.HomogeneousRegion`) is a
  part of space with its medium. The light between two encounters is a
  *segment* of the path and lies in one region; several segments can share one
  (in front of a mirror, inside a plate). The system derives the regions: at a
  Reflection or a Stop the light stays where it is, at a Refraction it crosses
  to the surface's other side, into the region already met there when the
  surface was met before. `system.regions` holds the region of each segment,
  `system.elements` the elements with their media resolved to regions; a
  region may be named in place of a medium wherever its identity matters.
  Adjacent elements must agree on the medium between them.
- **A stop and an edge.** A stop absorbs what meets its screen outside the
  hole; it is carried as an encounter of its own, so the field beyond it is the
  cut field, which diffracts from the cut, and the next leg carries the light
  its edge diffracts. A face's `edge` is where the material ends: rays meeting
  the surface beyond it miss the face and leave this path (reported as
  unsupported), so the path's field ends there too; the light passing beside
  the face is another path, and the edge itself scatters nothing.
- **Orientation follows the light.** A surface may face either way and be met
  more than once. The propagator traces the light's axis ray through the path
  (`propagation.path`) and meets every surface along its normal: against its
  normal it meets the surface's reverse (`GraphSurface.reversed()`, frame
  u, −v, −n, graph −h(x, −y); `SphericalCap(R)` reversed is `SphericalCap(−R)`),
  and from inside a closed surface its far side (`ImplicitSurface.far_side()`;
  for a `Sphere`, `Sphere(−R)` charted from the far vertex). Each leg's axis is
  the light's axis turned over as often as the light was reflected.
  `Solution.elements` holds the elements as they were met.
- **Only the light of the path.** The method computes the light that follows
  the given path: transmitted where it refracts, reflected where it reflects.
  Another ghost, or the light reflected at a surface the path crosses, is
  another path, built from the same surfaces.
- **Prescriptions.** `io.save_system` / `io.load_system` write and read a path
  (refractions, reflections, stops of planes, spherical caps and even
  aspheres) as JSON in metres; closed surfaces and freeforms are refused.

## The transport leg

A leg carries the field across one encounter. The two realizations below
share its map and its dyadic.

### The plane–surface–plane map

A point P of a reference plane in front of a surface follows its carrier ray to the surface, refracts there by the vector Snell law, and continues to P′ on the same geometric plane in the transmitted medium. This plane–surface–plane map Φ, with signed legs ℓ_i and ℓ_t, carries the field as

    E′(Φ(P)) = g exp(i k0 ΔS) [t_s E_s s + t_p E_p p_t + t_p E_par s_t],
    ΔS = n_i ℓ_i + n_t ℓ_t,   g = sqrt(s_iz c_t / (c_i s_tz J)),   J = det DΦ.

- **Continued dyadic.** The bracket is the continued dyadic (Eq. 4.3), and E_par is the part of the field along the carrier. Keeping E_par makes an index-matched surface exactly the identity.
- **Exact limits.** The transfer is exact for a plane wave on a plane interface.
- **What it is.** It is the formulation's leading-order model: it does not solve the curved-boundary Maxwell problem, and its error falls as 1/(R/λ0).
- **Carried plane components.** They are re-expanded in the transmitted medium and continue as an exact angular spectrum to the next surface or the observation points.
- **DΦ is analytic (Appendix A).** It comes from the surface Hessian and the carrier's direction Jacobian; no spherical focus is assumed.
- **Intersection.** Each ray–surface intersection starts from the height under P, then the osculating sphere at the vertex, then a march along the ray, and finishes with Newton's method. A ray that Newton loses from behind a surface (from a rim-depth plane) is retried from the osculating sphere.

`transport.InterfaceMap(surface, carrier, n_incident, n_transmitted, offset=0.0, branch="transmitted")`
computes Φ, ΔS, g and DΦ; its `forward(points)` returns a `transport.RayMap`
of the rays labelled by their points on the plane (output point, path,
flux, Jacobian, directions, surface point, caustic index).
[`one_transport_leg.py`](../general/examples/one_transport_leg.py) builds
one by hand.

### The flux factor through a focus: the caustic index

Where the ray tube has collapsed between the two planes, J has passed through
zero and the root in g continues through it as 1/√J → exp(−iπκ/2)/√|J|. κ,
`RayMap.caustics`, counts the transverse directions in which the tube
collapsed: one at a focal line, two at a point focus. It is counted on each
straight segment from the tube's own cross-section, forward on a real segment
and backward on a virtual one (a reference plane beyond the surface). This is
the continuation the angular spectrum itself produces through a focus; the
error of the leg still falls as 1/size, with a larger constant than without a
focus.

- The lattice leg carries such rays. Where the map folds, rays from both sides of a caustic reach the same points: the rays of one caustic index carry the field, the others carry none and are reported, and a leg that would lose more than 1 % of its amplitude that way is refused.
- The meridian leg carries only rays that met no caustic on the leg, and refuses a carrier that focuses inside it: a beam focused between its reference plane and its surface needs `lattice=`, or a reference plane past the focus.

In [`one_transport_leg.py`](../general/examples/one_transport_leg.py) a
collimated beam refracts into a glass cap of radius 100 λ0 and the reference
plane lies 400 λ0 behind the vertex, 100 λ0 past the focus: κ = 2 and
g = −2.9994 for a paraxial ray (measured now), the area ratio 3 of the tube
times exp(−iπ).

### Refraction, reflection and the ideal mirror

The leg carries the element it is given: `Refraction(surface, before, after)`
the transmitted wave, `Reflection(surface, medium, behind=b)` the Fresnel
reflection, `Reflection(surface, medium)` the ideal mirror. Only the direction
law and the dyadic differ:

- **Map.** s_r = s − 2 c_i n, back to the same geometric plane in the incident medium, with ΔS = n_i (ℓ_i + ℓ_r) and signed legs. `InterfaceMap(..., branch="reflected")` gives it, with ds_r = ds − 2 (dc_i n + c_i dn) from the same Appendix A quantities.
- **Flux.** g = √(s_iz / (|s_rz| J)) e^{−iπκ/2}. The power R I c_i dσ leaves the surface at c_r = c_i and I |s_z| dA is constant along each segment, so |r|² is the dyadic's and the area ratio g's. J is measured in the plane's own axes on both sides: a plane mirror at normal incidence has J = +1. The mirror reverses the tube's handedness about its direction and the sign of s_z together, so J still changes sign once per collapsed direction, and κ is counted on both segments as for refraction.
- **Dyadic.** r_s E_s s + r_p E_p p_r + r_p E_∥ s_r with p_r = s × s_r, r_s = (n_i c_i − n_t c_t)/(n_i c_i + n_t c_t), r_p = (n_t c_i − n_i c_t)/(n_t c_i + n_i c_t): tangential E and H are continuous with the transmitted wave. Beyond the critical angle c_t = i|c_t| and |r| = 1. `"mirror"` is the ideal mirror, r_s = −1, r_p = 1. Real indices only.
- **Output.** A reflected leg re-expands in the incident medium, travelling back (k_z < 0 in the surface frame); `LatticeSpectrum` and the meridian's `HankelSpectrum` carry that sign (`Spectrum.direction`). Every report and refusal uses the outgoing medium.

### Carriers

- The method needs the incident field's carrier, and transmits with the carrier dyadic.
  - A beam from `fields` (`FocusedBeam`, `PlaneBeam`) can be the light itself, and brings `beam.carrier`: converging to the focus, or collimated along the axis.
  - A sampled field or spectrum takes `carrier=` at the call: a `fields.Carrier` (`Carrier.converging(point)`, `diverging(point)`, `collimated(direction)`, or a general path with its gradient and Hessian).
  - Without a carrier the propagator raises and names `carrier=` and `alternatives.WavePropagator`.
- After the first refraction the carrier is the family of refracted rays (`transport.RefractedRays`, [sequences](#sequences-of-surfaces)).

### Surfaces

The leg needs a surface's Hessian, and every built-in surface declares one.

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

## Reference planes, fans and stops

- **`planes=(offset, ...)`**, one per element, places each plane along the propagation from its surface vertex (default: the vertex planes; a stop's on its screen).
  - A plane at the depth where the marginal ray meets the surface shortens the legs.
  - That lowers a hard pupil's error: for the aplanatic sphere at 1000 λ0, 0.1418 % (vertex plane) against 0.01524 % (rim plane).
- **`fans=(fan, ...)`**, one per element, weights the carried rays:
  - None carries every supported ray, within a stop's hole or a face's edge;
  - `(start, end)` rolls them off between those incidence angles, in degrees;
  - a callable receives the `RayMap` (its `label` is the ray's point on the first plane, so a fan can roll off along the input radius).
- **The default fan.** With `fans=None` a leg carries every ray the surface supports. On the meridian a fan that the surface never closes (a plane under a collimated beam) ends where the incident beam does, twice its footprint on the plane; a stop or an edge ends it at its radius. After a stop the next leg also carries the light the stop's edge diffracts, 6 λ0/n plus three Fresnel widths √(λ0 d / n) beyond the edge on a plane at distance d.
- **Fans and hard pupils.** A hard pupil's result depends on the fan, so report it with results. A hard cut at the fan's edge diffracts: 0.14182 % on the vertex plane and 0.01729 % on the rim plane, against 0.14177 % and 0.01524 % tapered. A taper that closes before the surface stops supporting the carrier removes that edge.

The numbers in this section are carried from the regression table
[below](#regression-against-the-handoff).

## Two realizations: the meridian and the lattice

`Propagator(system, realization=...)` chooses how every leg is carried:

- **"auto" (default):** the meridian for a beam without `lattice=` (and for a `HankelSpectrum`), the lattice for a sampled field, a lattice spectrum or a beam given `lattice=`.
- **"meridian":** the meridian for either. A sampled field is read on rings about its grid's centre; its weakest orders are dropped while together they stay below 1e-5 (counted as unsupported); a field with more than 1e-2 of its amplitude beyond 16 orders is refused.
- **"lattice":** the lattice; a beam then needs `lattice=` (a `CartesianGrid`).
- **No symmetry:** a beam in a system without the symmetry (a cylinder, a decentred or tilted surface, a fold mirror, an asymmetric carrier) and without `lattice=` is refused, with the reason.

### The meridian: systems of revolution

If every surface is a surface of revolution about the beam's axis (coaxial `Plane`, `SphericalCap`, `Sphere`, `EvenAsphere`, `AxisymmetricSurface`, `CartesianOval` or `CartesianOvoid`) and the carrier is symmetric about it, then the map is radial. The dyadic then acts on (E_rho, E_phi, E_z) by a matrix of the radius alone. `transport.MeridianTransport` is that leg.

- **One meridian per azimuthal order.** Each order m of the beam is carried along one meridian, by the same `InterfaceMap` and dyadic as the lattice leg.
- **Hankel spectra.** E_x ± i E_y of order m leave as Hankel spectra of orders m ± 1, and E_z follows from transversality. The output is a `fields.HankelSpectrum`.
- **Cost.** Circular, linear, radial and azimuthal states are one or two orders. The cost grows linearly with the aperture instead of quadratically: the manuscript's aplanatic sphere at 1000 λ0 takes 1.1–1.4 s (the archived prototype: 64 s).
- **Mirrors on the axis.** A reflection on the axis turns the axis over (frame u, −v, −n) and the harmonics with it: order m becomes −m, E_φ and E_z change sign.
- **Refused:** light that focuses between two legs (inside a ball, before its far side) reaches the next plane across the axis, which the meridian does not carry: it is refused with that reason, and the lattice carries it. A map that folds or crosses the axis inside the carried fan is refused too.

How the meridian is sampled:
- **Input plane:** Gauss–Legendre panels at most 10 λ0 wide. Each node's preimage is found by Newton steps with DΦ, and the beam is evaluated exactly there from its direction density, one Hankel sum per order.
- **Transmitted spectrum:** sampled in the polar angle of its plane waves, up to grazing. E_z's obliquity and the phase k_z z stay smooth there, so no spectral cut-off is imposed. Its panels resolve every phase that the carried extent and the evaluated region can produce.
- **Fast Hankel sum:** `fourier.polar.hankel_sum`, and `HT`/`IHT(method="fast")`.
  - Where tx ≥ z0 ≈ 40, the kernel is its ten-term Hankel asymptotic expansion, each term one type-3 FINUFFT sharing a plan across orders.
  - Where tx < 2 z0, the kernel is interpolated at Chebyshev nodes (kernel compression).
  - It agrees with direct sums to 1e-12–1e-15.

### The lattice: any geometry

`transport.LatticeTransport(plane, encounter, carrier, wavelength0, window=None)`
is one leg on a Cartesian lattice. It is what a sampled field, a beam given
`lattice=` and every system without rotational symmetry use.
- It fills the output lattice through the inverse map: a forward table, then damped Newton steps with DΦ. The incident field is evaluated exactly at each preimage, never interpolated.
- The output lattice keeps the input spacing. It grows, up to four times per axis, until it holds the image of every ray the window weights. Refraction toward a closer image spreads P′ beyond the input footprint, and cutting it there diffracted spuriously.
- It re-expands within sin < 0.995 of the transmitted medium, as the manuscript does. Nearer the light circle E_z = −κ·E/k_z amplifies what the lattice leaves there.
- A fan whose edge rays leave almost parallel to the plane is refused with a request for a window, before its image fills memory.
- **The leg streams its lattices.** Rays are traced, inverted and carried one block of rows at a time: `block_size` points, 2^20 by default; on the CPU backend, blocks of 2^18 run on up to eight threads.
  - It keeps only the preimages, the lit and supported masks and the carried field, in host memory.
  - Blocks change nothing but roundoff.
  - The incident field is read on the reference plane by two-dimensional sums, its plane waves' normal phase folded into their amplitudes.
- **A beam on a lattice.** `Propagator(system, lattice=grid)` carries a beam on the lattice leg, `grid` (a `CartesianGrid`) setting its spacing and extent.
  - The beam is read exactly at the preimages, so no sampled copy of the input lattice is held.
  - The result is a `LatticeSpectrum`: two transverse coefficients per mode in host memory, in place of a `PlaneSpectrum`'s wavevector and three amplitudes. Its modes are formed and summed one block of rows at a time, by a 2D type-3 transform when the targets lie on one plane.
  - A sampled field gives a `PlaneSpectrum`.
  - The archive's cylinder at R = 1000 λ0 (an 11852² lattice, 140 million points) runs this way in double precision: 13 GB of host memory and an 8 GB GPU.
- **Placement.** A sampled field's lattice must share the surfaces' orientation and transverse origin.

## Sequences of surfaces

The method crosses any number of encounters, as the formulation prescribes:
1. a leg;
2. the exact angular spectrum;
3. the full field on the next reference plane;
4. a leg along the refracted (or reflected) carrier;
5. and so on.

After a refraction the carrier is the family of refracted rays (Appendix B), `transport.RefractedRays`.
- Each ray keeps its label, its point on the first reference plane.
- Its surface point and refracted direction (Eq. 3.6) continue in a straight line to the next plane, together with their Appendix A derivatives along the labels.
- Nothing is evaluated off the rays or inverted, so the next map's Jacobian, flux factor and inverse stay analytic, with J = det(dP′/d label) / det(dP/d label).

How the legs chain, and the bookkeeping:
- **Index-matched surfaces** without an edge are the identity; the family passes through them.
- **The lattice leg** reads the previous leg's output lattice and inverts the composed map from the first plane's labels.
- **The meridian leg** reads the previous leg's Hankel spectrum. Its intermediate spectrum resolves the depth and radius of the next plane, so the handoff's κ-aliasing rule holds by construction.
- **Each encounter reports its own unsupported amplitude:** the incident field outside the rays that survive it. No amplitude threshold decides a ray.
- **Fans:** a later fan stays within the labels the first leg carried.
- **A call solves every element.** The propagator carries the light across the whole path on every call; it reuses the prepared legs (which depend on the geometry, the carrier and the incident lattice or beam, not on the field), and keeps no previous Solution.

## Reports and refusals

`solution.reports` holds one `transport.TransportReport` per carried encounter (refraction, reflection, stop), and `print(solution)` gives one line for each:
- `unsupported_relative_amplitude`: the incident amplitude on rays the encounter does not carry (misses, rear or grazing incidence, total reflection, outside the fan, a stop's screen or beyond a face's edge);
- `dark_samples`: output samples no ray reaches;
- `newton_residual`: the inverse map's residual;
- `azimuthal_orders`: the orders carried on the meridian (empty for the lattice leg);
- `caustics`: the caustic index κ of the carried rays;
- `folded_relative_amplitude`: the share of the carried amplitude on rays of another caustic index, beyond a fold of the map, which carry no field;
- `undersampled_relative_amplitude` (lattice leg): the share on rays that leave the surface steeper than the output lattice samples. The carried field turns at n_t k0 s_t across the plane, so the spacing must stay below λ0 / (2 n_t sin θ_t), whatever the input needed;
- `band_removed_relative_amplitude` (lattice leg): the share in propagating plane waves beyond sin = 0.995 of the transmitted medium, which the re-expansion removes.

A leg is refused, with the reason, when the folded, undersampled or
band-removed share passes 1 %, and in the meridian cases listed
[above](#the-meridian-systems-of-revolution). An option no spectrum has is
refused on every path. Unsupported amplitude is reported, never refused: it is
light that leaves the path (a stop, an edge, a fan), whose share depends on
the system.

## Reading results

`propagation.Propagator(system, *, planes=None, fans=None, realization="auto", lattice=None)`
takes nothing else; an unknown keyword raises. Called as
`propagator(light, carrier=None)` it returns a `Solution`.

- **Light.** A beam from `fields` (`PlaneBeam`, `FocusedBeam`) brings its own
  carrier. A sampled field (`ElectricField`, `TransverseElectricField`) or a
  `fields.Spectrum` takes `carrier=` (a `fields.Carrier`) at the call. A
  carrier is needed only when an element carries something. Light given in
  its medium is the light arriving in the path's first region; for a system
  that names no medium give `OpticalSystem(..., medium=air)`.
- **Solution.** `spectra` holds one `fields.Spectrum` per segment of the path,
  the incident light first; `spectrum` is the last, the light leaving the
  system. `elements` holds the elements as the light met them, `reports` one
  `TransportReport` per carried encounter, and `legs` the prepared legs.
- **E and H.** `field(target, ...)` returns an `ElectromagneticField`: at each
  point the sum of the spectra of the segments that reach it (in front of a
  mirror the incident and reflected light; zero where the path brings no
  light). `electric(target, ...)` gives E alone (no magnetic modes are
  formed), and `coverage(target, ...)` the target's grid, the segments of each
  point, the computed points and the sampling used.

```python
import numpy as np
import vecdiff as vd

air, glass = vd.media.air, vd.media.Dielectric(1.5, "glass")
cap = vd.surfaces.SphericalCap(-40.0, vd.geometry.Frame([0, 0, 10.0]))
lens = vd.system.OpticalSystem(
    vd.system.Refraction(vd.surfaces.Plane.z(0.0), air, glass),
    vd.system.Refraction(cap, glass, air),
)
beam = vd.fields.PlaneBeam(
    vd.fields.profiles.Gaussian(8.0),
    vd.fields.polarization.x,
    wavelength0=1.0,
    region=air,
    frame=vd.geometry.Frame([0, 0, -5.0]),
)
solution = vd.propagation.Propagator(lens)(beam)
focus = solution.field(z=85.0)  # sampled where the rays arrive
meridian = solution.electric(y=0, window=(-20, 20, 60, 110), shape=(126, 101))
S = vd.observables.poynting(focus)  # W/m²
I = focus.irradiance()  # S . n on the plane
print(focus.grid.shape, meridian.grid.shape, float(I.max()))
print(solution.coverage(z=85.0).sampling)
```

Targets are `z=f`, `y=a`, `x=b` (global coordinate planes, with charts (y,z),
(x,z) and (x,y)), any placed `GraphSurface` (sampled with `.sample(u=..., v=...)`
or automatically), or a sampled `PointGrid`, `CartesianGrid` or `SurfaceGrid`.
E and H stay global Cartesian components. A point lies in a segment when it is
on the side of the encounter before it that the light leaves into and on the
side of the encounter after it that the light arrives from, sided on the
actual surfaces (a closed surface by its implicit form); boundary samples
need `side="minus"` (the side the light arrives from there) or `side="plus"`,
one-sided reconstructed fields, not the raw Fresnel trace.
Uncomputed samples are NaN in E and H and marked by `.valid`; they cannot be
used as incident FFT data. Evanescent near fields are not supplied by
evaluating nearer a surface.

### Spectra

Every Spectrum has `region`, `wavelength0`, `grid` (its chart) and `plane`
(that chart's Frame), `direction` (+1 forward along the normal, −1 back),
`evaluate(grid)` for E and H, `electric(grid)`, `propagate(distance)` (the
field on its chart moved along the normal) and `relabeled(region)`. The kinds
are `PlaneSpectrum` (an FFT lattice), `LatticeSpectrum` (a lattice leg's
output), `HankelSpectrum` (a meridian leg's output) and `BeamSpectrum` (a beam
referred to a chart). Every sum, on a Spectrum or a Solution, takes the same
options: `method` ("auto", "nufft", "direct"), `tolerance`, `chunk_size` and
`threads`; anything else raises.

### Automatic target sampling

`z=f`, `y=a`, `x=b`, and unsampled placed surfaces are sampled automatically.
The default window is the light's footprint on the target: the rays of the
segments that reach it (where their light is at least a hundredth of its
brightest intensity) are carried straight to its plane, and their extent is
padded by four diffraction widths λ0/(n NA) of their cone (at least 25 λ0/n,
at most a quarter of the footprint's width beyond that), sampled in those
segments' medium. A focal plane thus needs no window. Where
only the incident light reaches the target, the window is the projection of
the finite input chart and optical path, including one input-chart width
beyond the last surface for longitudinal views. Neither is an energy
enclosure or a focus search. Use `window=(umin, umax, vmin, vmax)` for
detector crops; the target coordinate is never changed to maximize intensity.

For planes, spacing is at most 0.4 λ0/n_max, resolving the retained
propagating complex-field band. Curved charts use a pilot estimate of chart
stretch; highly varying curvature needs explicit sampling and convergence
checks. `shape=(Nv, Nu)` or a sampled surface overrides automatic resolution.
Automatic requests above two million points ask for an explicit window or
shape rather than silently coarsening. Sampling choices are not physical
accuracy estimates; `solution.coverage(target).sampling` records them.

### Coverage

`field(..., coverage="complete")` is the default: every point takes the
spectra of its segments, also in the band of heights a curved surface spans
where its spectrum is the continuation of the carried field beyond its plane
(the method's field there, to its order; a focus inside a ball lens lies in
that band). `coverage="reduced"` excludes those bands, measured along each
encounter's normal over the rays it carries (over the incident chart for
`alternatives.WavePropagator`), for the segments on either side of it.
There is no interpolation across the excluded bands. Equal-index bookkeeping
interfaces do not create gaps. Planar interfaces have zero-width height bands;
both modes agree there, with the chosen one-sided boundary convention.
`solution.coverage(target, coverage=...)` returns a `CoverageReport`: the
mode, the computed mask, `segments` (which segments reach each point),
`regions` (the first of them) and `extended` (points computed in a band that
reduced coverage leaves out). The `propagation/coverage.py` module owns
classification, masks and modal evaluation.

**Complete means spatial coverage of the chosen forward model.** In the curved
height bands this is an extension of its retained propagating modes; it does
not restore evanescent fields, guarantee agreement with Fresnel surface traces,
or certify near-interface Maxwell accuracy. In a continued-face lens model,
points outside the nominal rim follow the continued graphs; they are not
silently relabeled as the exterior of a finite lens.

## Between surfaces: plane waves

Between encounters the field is an exact superposition of propagating plane
waves of its region. `fields.angular_spectrum(field, region)` constructs a
`PlaneSpectrum` from a sampled field; `PlaneTransport.spectrum(spectrum,
distance)` moves it along the normal and keeps it modal for the next
operation; `spectrum.propagate(distance)` evaluates E and H on the translated
lattice (`magnetic=False`, also for `propagation.propagate`, returns E alone
without forming H). ASM propagation uses the exact homogeneous dispersion and
can retain evanescent modes, subject to the chosen branch and distance;
`model="fresnel"` uses the paraxial longitudinal phase, an approximation.

## Exact references

`references` holds the exact solutions the method is checked against; the
method never calls them. Each is one physical problem that takes `incident=`
any propagating field (plane waves, FFT spectra or beams) and its regions as
`media.HomogeneousRegion`s; see also
[validation](VALIDATION.md#exact-references-layered-sphere-and-cylinder).
- **`MieSphere`**: a homogeneous sphere.
- **`LayeredSphere`**: concentric layers.
- **`InfiniteCylinder`**: a circular cylinder at normal and oblique incidence.

`evaluate(grid, side=, term=)` gives the total field or one Debye term:
- `term=1` is the first entry, the quantity the method models;
- `term=p` is the waves that met p − 1 reflections;
- `term=0` is the external reflection and diffraction outside;
- `("after", P)` is the rest.

All stay exact at size parameters near 10⁴: log-scaled Bessel functions and
Wronskian-reduced coefficients replace the ill-conditioned direct
Hankel-basis solve. [`exact_references.py`](../general/examples/exact_references.py)
checks their identities.

## Checks of the method

Reflection legs (`general/tests/test_reflection.py`): plane waves against each
wave's boundary-value problem, to 1e-13 (R + T = 1, total reflection, ideal
mirror); focused beams against their plane-wave sums, with a focal line
(κ = 1) and a point focus (κ = 2) on the way back; a glass sphere against
Debye term 0 outside and, with a refraction first, term 2 inside. The errors
fall as 1/size (measured when the reflection legs were added):

| Case | 10 λ | 20 λ | 40 λ | 80 λ |
|---|---|---|---|---|
| Sphere, external reflection, lattice leg (term 0) | 25 % | 13 % | 6.1 % | 3.2 % |
| Same, meridian leg | 29 % | 14 % | 7.2 % | 3.6 % |
| Sphere, refraction and one internal reflection (term 2) | 36 % | 19 % | 9.7 % | 4.8 % |

Through the propagator (`general/tests/test_paths.py`), against exact
solutions, with errors falling as the system grows (measured when the paths
were added):

| Case | sizes | measured |
|---|---|---|
| Stop vs the equivalent hard pupil, flat face, meridian / lattice | a = 10, 20, 40 λ | 0.27, 0.049, 0.019 % / 2.1, 1.2, 0.66 % |
| Same, lens surface | a = 10, 20, 40 λ | 0.30, 0.16, 0.10 % / 0.64, 0.23, 0.065 % |
| Plane mirror, collimated beam (both) | | 1e-11 |
| Plane mirror, converging beam | foci 20, 40, 80 λ | 1.7, 0.88, 0.44 % |
| Fold mirror at 45°, lattice | waists 4, 8, 16 λ | 4.6, 2.5, 1.3 % |
| Plate on a back mirror (retro), both | 1, 2, 4 | 0.22, 0.11, 0.052 % |
| Plate ghost t12 r21 r21 t21, both | 1, 2, 4 | 1.1, 0.54, 0.27 % |
| Closed `Sphere`, Debye term 0, both | R = 10–80 λ | 29, 14, 7.2, 3.6 % |
| Closed `Sphere` in, far side, out: exterior term 2, lattice | R = 10–80 λ | 12.5, 4.8, 2.1, 0.90 % |
| Cassegrain, power deficit at focus (ideal mirrors) | scales 1, 2, 4 | 4.1, 1.1, 0.27 % |

The tests assert that each error falls with size and stays below a bound at
the largest size, not the values themselves.

### Regression against the handoff

Raw complex-field L2 errors in percent: on the 5 λ0 focal disk (161 radii, Eq. 8.2) for the spheres, on the 3.5 λ0 disk (140² grid) for the cylinder; n = 1 → 1.5 unless stated; R = 1000 λ0 unless stated. The handoff's numbers come from the archived prototype with its 80° fans. vecdiff uses the archive's fans: tapers 50°→56° for the aplanatic cap (identical to 80° for this sphere), 72°→80° on closed spheres, and 34°→38° for the meniscus's second surface. The opt-in module `general/tests/test_eikonal_regression.py` (`VECDIFF_SLOW=1`, marker `slow`) pins the rows marked †. The numbers are carried from the measurements recorded when the method was introduced (then through `PropagationPipeline`); the names below are the current ones.

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

The caustic correction (the flux factor continued through a focus inside a
leg, commit f145321) left the slow sphere and two-surface rows unchanged:
they were re-run after it and passed with the same pins.

## Other methods

These are not the method. They are kept to check it and for fields without
a carrier; `propagation.Propagator` never calls them.

### Wave transport: `alternatives.WavePropagator`

`alternatives.WavePropagator(system, transmission="spectral" | "local",
max_sine=0.85, transport_tolerance=1e-10)` carries a sampled field or a
`PlaneSpectrum` exactly to and from each surface (`SurfaceTransport`) with a
Fresnel treatment between, needs no carrier, and returns the same `Solution`
(its `reports` are the treatment's, its `legs` the incident and transmitted
surface transports). A beam is refused: sample it on a `CartesianGrid`. Its
path holds refractions of unbounded faces only: edges, stops and reflections
are refused. Charts must be ordered, nonintersecting graphs with the input
lattice's orientation and transverse origin. Wave transport does not include
returning branches or a finite-body rim solver. See the
[runnable example](../general/examples/wave_transport.py) and the
[validation records](VALIDATION.md#wave-transport).

```python
import numpy as np
import vecdiff as vd

wavelength = 532e-9
axis = (np.arange(48) - 24) * wavelength / 2
plane = vd.sampling.CartesianGrid(axis, axis)
x, y = plane.points[..., 0], plane.points[..., 1]
envelope = np.exp(-(x * x + y * y) / (3 * wavelength) ** 2)
incident = vd.fields.TransverseElectricField(
    plane, wavelength, np.stack((envelope, 0.3j * envelope), axis=-1)
)
cap = vd.surfaces.SphericalCap(40 * wavelength)
system = vd.system.OpticalSystem(
    vd.system.Refraction(cap, vd.media.air, vd.media.Dielectric(1.5, "glass"))
)
solution = vd.alternatives.WavePropagator(system, transmission="spectral")(incident)
print(solution.field(z=20 * wavelength).irradiance().max())
```

### Surface transfer: `SurfaceTransport`

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

### Fresnel treatments: `SpectralFresnel` and `LocalFresnel`

`SpectralFresnel.transmit` applies ordinary tangent-plane Fresnel
transmission separately to incident angular components before summing them
coherently. It interpolates the directional response, while Q retains each
mode's exact phase. It does not infer a single direction from the summed field.
The result contains the E/H trace and a report of retained modes, omitted modal
L2 and separation ranks. Advanced settings are supplied as a treatment
object, for example `transmission=SpectralFresnel(nodes=18, coefficients="unity")`.

For a controlled comparison, `SpectralFresnel(coefficients="unity")`
sets the electric amplitudes to `t_s = t_p = 1` at each interaction. The default
is `coefficients="fresnel"`. Both choices retain Snell refraction, polarization
rotation and vector E/H transport. The unity choice isolates Fresnel weighting;
it is neither an energy-conserving dielectric boundary model nor a separate
scalar Helmholtz solver. [Notebook 05](../general/notebooks/05_single_lens_lithography.ipynb)
runs this vector unity-transmission control and Fresnel transmission through
the same wave transport and high-NA immersion interface. An additional
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
Both treatments operate within the current forward propagating-band model.

### The exact planar boundary: `scatter_plane`

`alternatives.scatter_plane` solves a single planar dielectric boundary
(`system.PlanarInterface(frame, minus, plus)`, whose normal points from
`minus` into `plus`) mode by mode, returning reflected and transmitted
expansions. It retains transmitted evanescent fields under total internal
reflection; incident evanescent waves and exact grazing/critical limits
require another treatment.
