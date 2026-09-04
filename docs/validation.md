# Scientific validation and acceptance limits

The suite separates four questions: does the algebra satisfy Maxwell's laws,
does a discretization converge, does the physical approximation apply, and
does an independent reference agree? One successful check cannot replace the
others. Finite tests cannot prove correctness in every geometry or resonance.

**Scope of the main-method claim:** the repository's main method is the per-k
spectral interface transformation. Auxiliary-source boundary matching is a
different method. Its sphere errors and runtimes below must not be reported as
the spectral method's performance. The [implementation roadmap](future_implementations.md)
prioritizes refractive DUV, telescope, and microscope workflows within the
lossless material scope, before real-metal and EUV extensions.

## Conventions and boundary checks

Time dependence is `exp(-i omega t)`, vacuum wavelength is explicit, and returned
H is `Z0 H_SI`. Media are homogeneous, isotropic, nonmagnetic, and lossless.
The source-free dielectric jumps are tangential E, tangential H, normal D, and
normal B. Tests reconstruct both output fields before checking them; they also
check dispersion, transversality, and normal Poynting-flux balance.

Boundary residuals are RMS values. For refinement studies the fixed scales are
incident electric amplitude and normalized magnetic amplitude. Normal D is
additionally divided by the larger relative permittivity. The closed-sphere
benchmark uses quadrature weights and the same held-out points at every order.

## Planes and coherent layers

Plane tests cover normal, oblique, Brewster, critical, near-grazing, circular,
equal-index, and total-internal-reflection regimes, including a tilted and
translated interface. The recorded 36-case plane sweep had maximum boundary
and relative flux errors of `1.16e-15` and `1.51e-15`.

Layer tests check all four conditions at every boundary in three- and
four-region stacks, both s/p polarizations, both index directions, and rotated
frames. Independent Airy complex amplitudes agree within `6e-16` in the cavity
example. Explicit coherent round trips agree with the all-orders slab field
within `2.4e-13`. Frustrated-TIR flux errors are below `6e-16`; a 1000-wavelength
evanescent gap checks overflow safety. Exact critical layers are unsupported
and rejected, not silently regularized. Generic feedback raises on failure.

## Curved open surfaces: an approximation, even after numerical convergence

Local Fresnel boundary traces satisfy the imposed conditions to roundoff by
construction. The Green propagator is independently compared with Stratton–Chu,
and E/H curl identities are tested away from sources. These checks do **not**
establish that the prescribed currents solve the curved dielectric boundary.

The cap benchmark evaluates reconstructed, propagating-only spectra at physical
surface points. This is a continuation diagnostic, not a singular on-surface
Green integral. Its error includes local physical optics, hard-aperture effects,
and omitted evanescent content. It must not be labeled pure curvature error.

With aperture `0.6 R`, observations through `0.3 R`, wavelength 1, and indices
1/1.5, the recorded fine-grid results are:

| R / wavelength | Et jump | Ht jump | Dn jump | Bn jump | Quadrature field change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | .199 | .325 | .159 | .275 | 6.0e-13 |
| 1 | .0982 | .305 | .163 | .154 | 1.1e-11 |
| 2 | .104 | .139 | .0746 | .114 | 7.0e-10 |
| 5 | .130 | .169 | .0910 | .145 | 1.8e-8 |
| 10 | .0648 | .115 | .0602 | .0814 | 4.1e-7 |
| 50 | .0313 | .0492 | .0260 | .0396 | 3.5e-6 |

These nonzero jumps remain after quadrature convergence. Macroscopic scale
does not automatically confer exact boundary accuracy.

## Closed objects and resonances

The original one-encounter sphere diagnostic remains a labeled failure case
in `benchmarks.validate_physics`: it misses exit interactions and feedback,
with 30–95% errors against Mie. It is not a production solver or compatibility
implementation.

The new `solve_closed_interface` instead matches the complete boundary using
auxiliary Maxwell dipoles. The complex solve includes interactions implicitly.
No Mie data, fitted amplitude, or fitted phase enter it. Mie 3.3.0 is an optional
external reference. All four held-out boundary jumps, exterior/interior E/H,
closed-surface flux, source order, rank, condition, and offset are recorded.

At index 1.5, wavelength 1, offset fraction .5, and order 16 (512 source
locations, 1536 fitting points), the measured results are:

| Sphere diameter / wavelength | Largest bulk E/H error vs Mie | Largest held-out boundary jump | Closed flux error |
| ---: | ---: | ---: | ---: |
| .5 | 2.22e-8 | 2.36e-5 | 2.71e-10 |
| 1 | 8.98e-9 | 1.78e-5 | 4.19e-10 |
| 2 | 8.04e-8 | 5.28e-5 | 1.73e-8 |

Orders 8/12/16 are retained in `benchmarks/results/closed_sphere.json` so the
convergence trend is visible. The matrices are ill-conditioned; singular-value
truncation is explicit (`rcond=1e-12`). A separate four-wavelength-diameter
stress case is deliberately retained in `closed_sphere_stress.json`: the tested
order/offset are insufficient and produce substantial Mie errors. Do not infer
universal accuracy from the three smaller cases.

The larger sphere was then tested at order 24 with offset fraction .25
(1152 source locations, 3456 fitting points). Bulk E/H errors dropped from
13–19% to **0.30–0.44%**, with a maximum held-out jump of `5.09e-3` and a closed
flux error of `3.17e-3`. `closed_sphere_refinement.json` retains this result.
Both source count and placement changed, so this is an improvement under two
numerical changes, not a one-parameter convergence proof. The dense solve took
155 seconds; no claim of arbitrary-precision or universal high-Q accuracy is made.

The index-2 sphere example compares a nine-point wavelength scan with Mie and
refines its worst point from order 12 to 16. Maximum sampled bulk error was
`9.18e-5` at order 12. The dense curve is explicitly the reference, not additional
native solves. A sparse wavelength scan cannot certify narrow high-Q peaks.

## Reproduction and scientific use

```bash
python -m pip install -e '.[test,notebooks,nufft,validation]' -c requirements-validation.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m benchmarks.validate_physics
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m benchmarks.closed_sphere
python scripts/notebooks.py --check --execute
```

The pinned numerical environment targets Python 3.12. The general installation
supports Python 3.10+; CI also checks other supported versions without those
environment-specific pins. Notebook source synchronization is distinct from
Jupyter execution. The development host blocked kernel networking, so local
Jupyter execution was **not** certified locally. All six notebooks subsequently
executed successfully in the [CI notebook job](https://github.com/blancusjh/vecdiff/actions/runs/33922646859),
which uploads their executed outputs. The first unpinned Python 3.12/3.13 test
jobs exposed a SciPy 1.18 rank-scalar JSON serialization issue; diagnostics now
convert the rank explicitly to a Python integer. Numerical assertions passed.
All seven example calculation functions ran locally and their assertions passed.

Before using a new configuration: refine sampling and domain size; check all
four reconstructed boundary conditions and flux; vary source placement for
closed-boundary matching; refine wavelength spacing near peaks; verify against
an independent solution where available. Preserve parameters, package versions,
raw residuals, and convergence history. Absorbing/magnetic/anisotropic media,
general multi-body scattering, singular surface quadrature, and universal
high-Q accuracy are not implemented or claimed.
