# Operational propagation

The primary general-library path is `propagate_interfaces(source, system, grid)`:

\[
R_{\xi_2}Q^{-1}_{n_i,\Sigma}TQ_{n_0,\Sigma}R_\xi .
\]

It repeats the interface operation in each incident/transmitted medium.
`propagate_high_frequency` exposes the same implementation. Notebook 05 uses this
path. No equivalent-current, Green, Weyl or ray-tube propagation occurs in it.

## Operators

- `PlanarASM` is **R**. It multiplies each plane coefficient by
  `exp(i kz distance)`, with the exact homogeneous dispersion. Its `spectrum`
  method preserves coefficients between interfaces; calling the operator returns
  E/H on the translated Cartesian plane by IFFT.
- `SurfaceTransport.forward` is **Q**. For a graph `z=h(u,v)`, it evaluates
  `sum A(ku,kv) exp(i(ku u + kv v + kz h(u,v)))`. An analytic `EikonalField`
  can instead be evaluated directly on the first surface.
- `fresnel_transmission` is **T**. The local normal and eikonal direction determine
  Snell refraction and the vector Fresnel matrices. It returns incident,
  reflected and transmitted E/H traces and the incident transverse-projection
  error. Reflections are retained as traces, not recursively propagated.
  When both indices are identical, T is exactly the identity on E **and H**;
  no local polarization projection is applied to a transparent interface.
- `SurfaceTransport.inverse` is **Q inverse**. It solves for outgoing plane
  coefficients whose physical surface-tangential electric trace best matches
  the target in L2(dS), including the chart metric and surface area.
  Longitudinal E and H follow homogeneous Maxwell constraints. It returns a
  `PlaneSpectrum` and an `InverseReport`; solver nonconvergence raises an error.
  Constant-height charts use a direct FFT and inverse propagation phase.
  Variable-height charts use an iterative least-squares solve.

The inverse fits the tangential electric trace only. Earlier experimental
options (joint E/H fitting, an integrated power closure and a spatial flux
closure inside Q^-1) were removed on 2026-09-23: they enforced power budgets
without improving the field or the boundary conditions. Their studies are kept
outside the source tree in `output/deprecated_2026-09-23/`.

For graph slope `g = grad(h)`, covariant chart components are `e = E_xy + g E_z`.
Their physical squared norm is `e* G^-1 e`, where `G = I + g g^T`.
The least-squares operator applies `sqrt(J) G^-1/2`, with `J = sqrt(1 + |g|²)`;
its adjoint includes the same symmetric weight. This is a geometric integration
weight in the residual, not a gain multiplying the propagated field.

On a variable-height surface, multiplication by height in spatial coordinates
does not commute with longitudinal dispersion in Fourier coordinates. Therefore
**changing h to -h does not invert Q**. It does for constant height, within the
retained band. The regression suite checks both cases against direct modal sums.

Q factors the phase into a carrier and a Chebyshev/Bessel series. Each term uses
an IFFT and pointwise multiplication. A Bessel-tail bound controls truncation;
the conjugate transpose of exactly that approximation is used in LSMR.
There is no dense surface-to-surface matrix. Cost is
`O(iterations × rank × N log N)` and storage is `O(rank × N)`. Rank grows with
sag times longitudinal spectral bandwidth: this is not a promise of constant
cost for arbitrary macroscopic sag or numerical aperture.

## Sampling and physical scope

Pass a padded `CartesianGrid` at the first vertex. Consecutive chart frames must
have the same orientation and transverse origin; surfaces must not intersect.
The optical geometry may be rotated as a whole. The present implementation
supports forward propagating modes below a declared `max_sine` (default 0.85);
evanescent coupling, TIR, grazing transmission and folded paths need another
physical model and are not silently approximated here.

Each `EikonalField` or analytic plane-wave branch is transported independently;
outputs are coherently summed. A `PlaneSpectrum` declares one locally eikonal
branch. Between interfaces, spectral derivatives estimate its phase gradient as
`Im(sum(conj(E) grad(E))) / |E|²`, with direction normalized to the eikonal
sphere. A multiray interference field near a caustic has no unique eikonal and
must not be interpreted as one such branch. Homogeneous diffraction through a
focus after the last interface is supported.

Surface edges (`apertures`) do not cut the field; only declared stops
(`stops`) do. Light passing outside an element continues in the surrounding
medium and is added at the element's exit; see
[surface edges and pupils](PUPILS_AND_EDGES.md). Before this change every edge
acted as an absorbing stop; studies dated before 22 September 2026 describe
that model. The numerical surface chart is extended outside the edge by
retaining the rim height at each azimuth. This extension is not a physical lens
rim. Resolve hard edges and stops, or supply an explicit smooth pupil.
The FFT cell is periodic: increase sample count at fixed width, then width at
fixed spacing. Also test `max_sine` independently of both.

`solution.evaluate(grid)` uses FFT ASM on a matching translated lattice and a
plane-wave Fourier sum elsewhere. FINUFFT accelerates arbitrary points when
installed (`pip install '.[nufft]'`); otherwise the direct Fourier sum is used.
Observations must lie beyond the final surface envelope and reference plane.

## Accuracy and comparison

Small inverse residuals certify fitting of tangential E in the chosen angular
band. They do **not** certify the full dielectric boundary problem. Local
Fresnel/WKB magnetic traces differ from the reconstructed homogeneous H at finite
wavelength. Check those differences and signed flux without normalization.
`InverseReport.relative_residual` is the physical tangential E residual;
`magnetic_relative_residual` reports the physical tangential H mismatch
independently. `solver` is `fft` or `lsmr`; a direct solve has zero iterations.
Neither residual includes the normal D/B conditions or a global scattering solve.

Notebook 05 uses the same Gaussian and explicit outer 8-wavelength taper for
both the spherical and stigmatic lenses. Its executed 160² to 192² study gives:

| Check | Sphere | Oval |
|---|---:|---:|
| Complex focal-probe grid change | 6.0e-5 | 3.43e-5 |
| Map-probe change / maximum reference E | 4.67e-4 | 5.51e-4 |
| 128λ to 160λ cell-width change | 2.06e-4 | 5.64e-5 |
| Tangential E inverse residual | 1.89e-4 | 4.90e-5 |
| Reconstructed H versus local Fresnel H | 1.95% | 1.16% |
| Reconstructed surface flux difference | 0.0189% | 0.0050% |

The H discrepancy is a model limitation, not an integration error hidden by
refinement. Unit tests separately verify plane limits, local boundary conditions,
flux, curved equal-index controls and grid convergence.

Prescribed-current tools live in `vecdiff_general.comparisons`.
`solution.evaluate_green_comparison()` explicitly evaluates final Fresnel traces
with that separate model. It is never called by the operational path.

Background: [Kim et al., stepwise angular spectrum on curved surfaces (2014)](https://opg.optica.org/fulltext.cfm?uri=oe-22-10-12659)
discusses variable-height ASM. This implementation uses a separated phase
expansion and a numerical inverse, rather than that paper's staircase algorithm.

## Macroscopic experiment

`validation/operational_propagation.py` checks a 532 nm beam through a lens with
2 mm curvature radius, 0.2 mm clear-aperture radius and 0.1 mm central thickness.
A 384² solve took 1.78 s and a 512² solve 3.70 s on the development machine
(single BLAS thread). The exit Q required 11–13 terms and four inverse iterations.
The complex-field grid change was 5.0e-7; reducing `max_sine` from 0.85 to 0.2
changed the field by 2.7e-7. The reconstructed H discrepancy was 0.0844% and
surface flux discrepancy 9.8e-9. These are one smooth, modest-NA example, not a
speed or accuracy guarantee for arbitrary lenses. [Raw measurements](operational_macro_results.json).

