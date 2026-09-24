# Local energy-flux factors: investigation of 2026-09-21

Follow-up: the spatial flux correction through every interface was later removed from the library (2026-09-23); its study is archived in `output/deprecated_2026-09-23/`.

## Result

A spatial amplitude factor followed by Maxwell spectral reconstruction improves
the tested 2-wavelength lens's local power distribution, tangential boundary
jumps and independent BEM field comparison. This improvement persists across
the tested sample counts and padding change. It is a useful result beyond the
earlier unsuccessful global-factor experiment.

The improvement is not universal. Normal D and B jumps get slightly worse in
that lens, and all four boundary jumps worsen for the tested 20-wavelength
lenses. The implementation therefore lives in
`general/src/vecdiff_general/propagation/experiments/flux.py`, is explicitly
opt-in, and is not called by the primary propagation API. No notebook results
have been silently replaced with these trial fields.

## Derivation and implementation

Let Q synthesize a transmitted forward Maxwell spectrum b on the interface.
At each point, form the desired signed normal flux from the local Fresnel
incident and reflected traces:

\[
s_*(x)=\tfrac12\Re[(E_i+E_r)\times(H_i+H_r)^*]\cdot\nu.
\]

This includes interference terms. For the lossless local plane-wave interface
model it equals the sum of incident and signed reflected normal powers, and
also equals the prescribed transmitted Fresnel power. Directions and normals
enter through the local reflected and transmitted fields and their magnetic
fields. No stigmatic-surface assumption is made.

Given the reconstructed field Qb_j, compute its actual normal power density
s_j using E×H*, then use the local multiplier

\[
a_j(x)=\exp\!\left[\operatorname{clip}\left(
\frac{\eta}{2}\log\frac{s_*(x)}{s_j(x)},-\delta,\delta\right)\right].
\]

All reported runs use eta=0.5, delta=0.2, six iterations. For eta=1 without
clipping, a_j² s_j=s_* before reconstruction. The smaller step damps the
nonlocal feedback produced by reconstruction. Points are controlled only when
both target and current flux exceed 1e-6 times the peak target flux; other
points retain multiplier one. Negative target flux is outside this experiment's
supported forward model. The code reports the fraction of illuminated samples
that cannot be controlled and the fraction whose steps were limited; neither
condition is silently interpreted as convergence.

The update is

\[
\widetilde b_{j+1}=Q^{-1}\{a_j E(Qb_j)\},\qquad
b_{j+1}=\sqrt{\frac{\int s_*\,dS}{\int s(Q\widetilde b_{j+1})\,dS}}
\widetilde b_{j+1}.
\]

Q-inverse fits tangential electric data in the existing forward angular band;
longitudinal E and all H are recomputed from the Maxwell spectrum. The final
constant multiplier commutes with homogeneous propagation and enforces total
chart power. The spatial multiplication is never returned directly as a
purported Maxwell field. Leakage into target-dark parts of the finite chart is
included in the error and total power, even though the local factor does not
directly control those samples.

Because Q-inverse is nonlocal, this is an alternating correction and
projection, not an exact local normalization after reconstruction. There is
no convergence theorem or automatic selection of a supposedly accurate iterate.
All intermediate results are retained. Reported comparisons use the same fixed
six iterations, not a separately optimized iteration count for each case.

## Geometry, controls and scope

Wavelength is 1 length unit, incident electric amplitude is 1 V/m, indices
are 1 and 1.5, and illumination is an axial x-polarized plane wave.

- Plano-convex: aperture radius a=D/2, thickness 0.6a, spherical exit radius
  -4a for D=2 and 20. D=200 uses the notebook's weak lens with radius -2500.
- Freeform: the same planar entrance and thickness, with exit sag
  h(x,y)=-(x²+0.6y²)/(4D). This is an astigmatic, non-axisymmetric surface;
  the D=200 freeform is not geometrically the same as the weak spherical lens.
- Sphere: diameter 0.6, with the existing restricted forward chart of radius
  0.45 times the sphere radius. The full sphere supplies the Mie reference.

Each case uses N=64,128,192 samples per axis at chart width 3a, plus N=192 at
width 4.5a. Both sampling and padding matter. These runs establish a repeated
trend, not a converged large-lens solution. In particular, the 200-wavelength
plano-convex boundary residual grows with sample count; its previously small
coarse-grid power mismatch was not evidence of field accuracy.

Only the final interface is corrected. Earlier fields and local reflected
traces are held fixed. This isolates the correction's effect; it is not yet a
validation of corrections fed through multiple interfaces.

Independent reference comparisons use 50 common exterior points on two planes,
0.4 and 1 wavelength beyond the exit vertex, with a 5×5 transverse grid on
[-0.3a,0.3a]². As in notebook07, the comparison adds incident minus equal-index
pupil fields to approximate bypass illumination around the lens. That comparison
construction differs from the absorbing-pupil primary API. Reference gaps
therefore include model differences, not just numerical errors.

## Measurements at N=128 and chart width 3a

All following percentages mean the stated dimensionless diagnostic multiplied
by 100. Flux RMS is the relative area-weighted L2 error in normal power density
over the whole chart. Eτ, Hτ, Dν and Bν are maximum interface jumps on r<0.95a,
normalized by 1 V/m, 1/Z0 A/m, epsilon0 C/m² and mu0/Z0 T respectively.
The comparison side is the prescribed local incident+reflected field; these
are transmission reconstruction checks, not global scattering boundary checks.

### Plano-convex lenses

| D/λ0 | Method | Flux RMS % | Eτ % | Hτ % | Dν % | Bν % |
|---:|---|---:|---:|---:|---:|---:|
| 2 | Ours | 26.05 | 38.18 | 36.13 | 27.24 | 22.22 |
| 2 | Ours + local factor | 15.84 | 28.96 | 26.06 | 28.54 | 24.37 |
| 20 | Ours | 9.563 | 16.08 | 16.04 | 16.62 | 14.85 |
| 20 | Ours + local factor | 5.776 | 18.45 | 20.02 | 20.53 | 18.22 |
| 200 | Ours | 0.04263 | 4.32e-8 | 0.5471 | 2.752 | 2.730 |
| 200 | Ours + local factor | 0.0006656 | 0.09922 | 0.5455 | 2.749 | 2.727 |

### Astigmatic freeform lenses

| D/λ0 | Method | Flux RMS % | Eτ % | Hτ % | Dν % | Bν % |
|---:|---|---:|---:|---:|---:|---:|
| 2 | Ours | 25.26 | 37.07 | 34.79 | 27.42 | 23.50 |
| 2 | Ours + local factor | 15.35 | 28.05 | 24.45 | 28.85 | 25.94 |
| 20 | Ours | 8.818 | 16.18 | 15.05 | 16.55 | 13.57 |
| 20 | Ours + local factor | 5.766 | 26.57 | 23.00 | 20.27 | 15.90 |
| 200 | Ours | 0.3266 | 1.14e-7 | 2.353 | 9.001 | 5.385 |
| 200 | Ours + local factor | 0.005661 | 1.013 | 1.850 | 8.904 | 5.379 |

### Independent BEM comparison: D=2 plano-convex lens

Gaps are relative vector sample norms on the 50 probes defined above.

| Method | Electric-field gap % | Magnetic-field gap % |
|---|---:|---:|
| Ours | 25.93 | 26.23 |
| Ours + local factor | 22.22 | 22.29 |
| BEM reference | 0 by definition | 0 by definition |

BEM was refined through edge sizes 0.30, 0.22 and 0.16 wavelengths, with 858,
1632 and 2988 unknowns. The electric-field change on these same probes fell
from 1.85% to 0.463%; the last magnetic-field change was 0.469%. Those changes
are convergence evidence, not a rigorous reference error bound.

The correction improves the BEM gap at all four tested chart configurations.
With the wider 4.5a chart at N=192, the electric gap changes from 29.86% to
24.64%. Absolute values still depend on padding. Do not call the corrected
forward method an accurate full-wave solution on the basis of this improvement.

### Independent Mie comparison: D=0.6 sphere

| Method | Electric-field gap % | Magnetic-field gap % |
|---|---:|---:|
| Ours | 38.72 | 39.28 |
| Ours + local factor | 37.54 | 38.07 |
| Mie reference | 0 by definition | 0 by definition |

The change is modest and the model gap remains large. A local amplitude
constraint does not supply the omitted full-body scattering physics.

## Maxwell checks and execution time

Across the 28 cases, the largest corrected homogeneous Maxwell residual at
finite-difference step λ0/160 was 7.69e-8 using fixed unit-field normalization.
The checks include both divergences, both curl equations and both vector wave
equations. They are evaluated away from the interface. Halving the difference
step from λ0/80 gives the expected decrease in derivative truncation error;
the exact angular-spectrum dispersion and transversality are also enforced by
the spectrum type. Passing these checks does not establish boundary accuracy.

Integrated target power is matched to at most 4.45e-16 relative error by
construction. No illuminated samples had nonpositive-current-flux exclusions
in these runs. These facts are diagnostics, not independent accuracy evidence.

At N=128, added correction time was approximately 0.42 s for the D=2 spherical
exit lens, 1.02 s for D=20, and 0.36 s for the weak D=200 case. Timings include
six inverse solves and reconstruction, exclude independent diagnostics, and
are single runs on this machine. They are additional work beyond the original
propagation. The BEM reference build/solve/evaluation times were about 1.7,
4.7 and 14.5 s for the three meshes. Complete timing records are in the output.

## Reproduce and inspect

```sh
.venv/bin/python general/examples/local_flux_investigation.py
.venv/bin/pytest -q general/tests/test_flux_experiment.py general/tests/test_operational_propagation.py
```

The experiment writes `output/local_flux_investigation/results.json` (all
196 iterates), `summary.md` (all geometries and refinements, with adjacent
method rows), and seven PNG figures showing exit-surface flux and the first
local amplitude factor on identical chart coordinates. These are surface
diagnostics, not focal-plane intensity maps. Flux plots use W/m² directly.

The seven added tests cover oblique planar Fresnel transmission including
reflected power, exact-wave fixed points, recovery of a manufactured amplitude
error on planar and curved surfaces at two physical scales, homogeneous Maxwell
residuals, unsupported signed target flux, and the inability of a flux-only
constraint to determine phase. The focused run including existing operational
tests passes all 23 tests. Changed Python files pass Ruff.

## Decision

Keep this correction available for experiments. It demonstrably improves some
field comparisons and extends to non-stigmatic surfaces. It has not passed
the gate for a default correction: local flux and all four boundary conditions
do not improve together across cases, larger configurations are not converged,
and repeated-interface feedback is untested. The next refinement needs to
constrain electric and magnetic boundary traces together with flux, and assess
any amplitude factor after spectral reconstruction rather than only before it.
