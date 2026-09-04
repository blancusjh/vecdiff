# Physical architecture

A field is an electromagnetic **state**. A medium, surface, domain, or layer
stack describes its **configuration**. A propagation calculation applies a
physical law to that state and configuration. Sampling and Fourier transforms
are numerical representations, not alternative physical objects.

## State and configuration

`ElectricField` carries global Cartesian components, domain, sampling, vacuum
wavelength, and medium. `Ez=None` is unspecified information.
`TransverseElectricField` is a subclass with exactly that missing component,
not TE polarization. `complete(direction=...)` selects a propagation branch
and returns a full electric field without mutating the original.

`ElectricSpectrum` represents
`E(r) = sum_j a_j exp(i k_j · r)`; each `a_j` includes its quadrature weight.
Every wavevector satisfies dispersion and every amplitude satisfies `k·a=0`.
`Z0 H_j = (k_j/k0) × a_j`. Evanescent k are complex and use the bilinear, not
Hermitian, dispersion/transversality relations.

`PlaneDomain` and `Frame` describe placement. `CartesianGrid`, `PointSampling`,
and `SurfaceSampling` describe discretization. A `Surface` provides positions,
tangents, normals, and area density. A `DielectricInterface` combines a surface
with its two media and an orientation. A `LayerStack` describes parallel media
and physical thicknesses. None owns FFT sizes or solver tolerances.

## Laws and numerical realizations

| Module | Responsibility |
| --- | --- |
| `interfaces/fresnel.py` | Per-k s/p basis, Snell mapping, vector Fresnel amplitudes; fixed-tangential-basis admittances for layers |
| `propagation/propagation.py` | Sampled-field spectrum, missing-component completion, homogeneous propagation |
| `propagation/interface_transform.py` | Infinite-plane map or per-k curved-surface physical-optics trace |
| `propagation/surface_radiation.py` | Maxwell radiation from prescribed equivalent currents, by Green or spectral evaluation |
| `propagation/layered_propagation.py` | Stable coherent multiple-interface composition and region-specific total fields |
| `propagation/multiple_scattering.py` | Complex-amplitude feedback equation with convergence controls |
| `fourier/` | Transforms only; no field, interface, or literature-model ownership |
| `observables/` | Measurements and residuals; never hidden normalization inside propagation |

The curved Fresnel map retains each incident k until after its local basis and
coefficients are applied. It does not infer a ray direction from the phase of
a superposed field. Linearity is tested explicitly.

The layer recursion sums repeated reflection/refraction with complex phases.
Its returned `LayeredElectricField` is a spectral electric-field representation
with no compatibility alias.
Only decaying internal propagation factors are formed; forward fields are
anchored at each layer's left boundary and backward fields at its right.
Exactly grazing/critical layers are rejected explicitly because this basis
requires a separate limiting formulation. Noncritical evanescent layers work.

General closed-body resonances require a physical spectral round-trip map and
validation that remain pending. The generic feedback iterator does not supply
that map. No auxiliary-source boundary solver is included.

## Strict reference separation and breaking API

`references/mie.py`, `references/richards_wolf.py`, and
`references/stratton_chu.py` are standalone reference implementations/adapters.
Only the Richards–Wolf reference consumes a shared field abstraction. Tests and
benchmarks may import them; the production package cannot. The wheel contains
only `vecdiff` packages.

There is no legacy package, compatibility facade, or alias for an old field
name. Old examples and notebooks were retired or rewritten, not kept runnable
through an adapter. The [migration record](migration.md) explains the choices.
