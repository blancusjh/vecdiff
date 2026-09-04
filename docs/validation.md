# Physical validation

Validation distinguishes identities of the local boundary law, numerical
convergence of propagation, and the accuracy of the physical approximation.

## Exact plane interface

`tests/test_maxwell.py` checks s, p, and circular polarization for equal-index,
air-to-glass, and glass-to-air interfaces from normal incidence to 89.9 degrees.
It includes both critical and Brewster angles. The tests reconstruct incident,
reflected, and transmitted fields on a tilted, displaced physical plane, then
check all source-free dielectric boundary conditions:

\[
\mathbf n\times(\mathbf E_2-\mathbf E_1)=0,\qquad
\mathbf n\times(\mathbf H_2-\mathbf H_1)=0,
\]

\[
\mathbf n\cdot(\epsilon_2\mathbf E_2-\epsilon_1\mathbf E_1)=0,\qquad
\mathbf n\cdot(\mathbf B_2-\mathbf B_1)=0.
\]

They also check dispersion, `k dot E = 0`, lossless normal Poynting-flux
balance, Brewster cancellation, and evanescent decay under total internal
reflection.

## Curved open interface

The local Fresnel boundary data satisfy the four conditions to floating-point
accuracy by construction. This is a necessary check. It is not the final
accuracy claim.

`benchmarks/validate_physics.py` reconstructs the reflected and transmitted
fields from the finite cap and evaluates the same four jumps again at physical
points on the cap. It reports those residuals separately from surface and
angular quadrature changes. Only propagating output waves enter this benchmark;
the JSON records that choice because edge-generated evanescent fields can be
relevant near a finite aperture.

The direct dyadic Green evaluator is cross-checked against a separately
preserved Stratton-Chu implementation. Numerical curls independently verify
both frequency-domain Maxwell curl equations away from the source surface.

## Dielectric spheres and Mie theory

`references/mie.py` wraps `miepython` 3.3.0 and is kept outside the core. First,
its field on both sides of a sphere is checked against all four boundary
conditions with shrinking radial offsets and against an expanded multipole
order. This qualifies the reference for the requested indices and sizes.

The native comparison deliberately applies only one local Fresnel encounter on
the illuminated hemisphere, sets its shadow-side interior trace to zero, and
radiates the resulting guessed trace on the complete sphere. It omits:

- repeated internal reflection and refraction;
- the second exit encounter;
- coherent resonant feedback;
- creeping and diffracted shadow fields.

Its error against Mie therefore measures the failure of that one-encounter
extension for a closed resonant object. It is not evidence against the exact
planar Fresnel map. A production sphere method needs a multiple-scattering
iteration or a Maxwell boundary-integral solver; either belongs in a future
solver package with convergence and resonance controls.

## Reproduction

```bash
python -m pip install -e '.[nufft,validation]'
python -m pytest -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.validate_physics
```

The generated JSON contains package versions, discretizations, raw residuals,
field changes under refinement, and timings. No fitted scale, phase alignment,
or reference-derived normalization is applied to native fields.

## Recorded results

The committed run used vacuum wavelength 1, `n1=1`, `n2=1.5`, cap aperture
`0.6 R`, and reconstructed-boundary observations through `0.3 R`. Values are
relative RMS residuals. `Δquad` is the relative change of the concatenated
reconstructed fields under simultaneous surface and angular refinement.

| `R/λ` | tangential E | tangential H | normal D | normal B | `Δquad` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.199 | 0.325 | 0.159 | 0.275 | 6.0e-13 |
| 1 | 0.0982 | 0.305 | 0.163 | 0.154 | 1.1e-11 |
| 2 | 0.104 | 0.139 | 0.0746 | 0.114 | 7.0e-10 |
| 5 | 0.130 | 0.169 | 0.0910 | 0.145 | 1.8e-8 |
| 10 | 0.0648 | 0.115 | 0.0602 | 0.0814 | 4.1e-7 |
| 50 | 0.0313 | 0.0492 | 0.0260 | 0.0396 | 3.5e-6 |

The residual generally decreases at macroscopic scale but is not monotone for
this hard-edged, fixed-shape cap. Refinement changes are much smaller than the
residuals, so they do not explain the measured jumps.

The planar sweep contains 36 cases. Its largest boundary residual was
`1.16e-15`; its largest relative flux error was `1.51e-15`.

The closed-sphere diagnostic is numerically stable under quadrature refinement
(field changes from `2.1e-8` down to `3.7e-12`) while its one-encounter field
errors versus Mie remain 30–95%, depending on radius, region, and E/H. Their
nonmonotonic size dependence and the recorded Mie scattering/backscatter
efficiencies are consistent with a resonant closed object. These numbers are
reported as evidence that repeated interactions are required, not as an error
estimate for an open interface.
