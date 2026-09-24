# Q transport and complete-curl boundary audit

## Decision

The baseline audit below is supplemented by the implemented
[curvature-order study](CURVATURE_ORDERS.md). That study independently generates
orders 0–3 from the supplied equations, reproduces the approximately 811-fold
local lens-exit improvement, and tests propagated cavity and lens-exit fields.
The external coefficient package and FEM snapshots remain unavailable; they are
not needed by the new implementation. The generated orders 1–3 now also match
exact single-interface cylinder and sphere solutions at the asymptotic rate, and
a finite-lens run with entrance-stop diffraction confirms the small (below 0.1%)
focal effect against an external full-wave FEM solution; see sections 7 and 8
of that study. The following supplied-result discussion
describes the provenance of the baseline audit.

The supplied higher-order results justify an experiment, but do not establish an
improvement to this repository's transmitted focal fields. The experimental
branch is `research/maxwell-boundary-audit`. The primary composition remains
`R Q⁻¹ T Q R` with the existing Fresnel interaction.

The pasted report contains two distinct findings:

- Its lens-exit magnetic jump falls from 2.166e-3 to 2.671e-6, about 811 times,
  under its local continuation and complete-curl audit.
- Its full-lens complex-field discrepancy changes from 7.1339% to 7.1402%.
  That is not a focal-field improvement.

Those full-lens comparison figures are **supplied measurements**, not reproduced
measurements of this repository. The named external `taylor_interface.py`, coefficient arrays, and high-order
axisymmetric FEM snapshots are absent from the accessible workspace. Our FEM
uses lowest-order three-dimensional tetrahedral edge elements. The two solvers
must not be conflated. The report also uses aperture-current radiation for its
candidate comparison; those numbers cannot be attributed to our Q-based
evaluation path without running that path.

## Measurements of the current implementation

Run the [audit script](../general/examples/maxwell_boundary_audit.py) to produce
the [full tables and images](../output/maxwell_boundary_audit/REPORT.md) and
[machine-readable results](../output/maxwell_boundary_audit/results.json).
No gain, phase alignment, fitted correction, aperture filling or change to T is
used. Wavelength is 532 nm.

### Homogeneous propagation

For each spherical, parabolic and non-axisymmetric polynomial graph, evaluate a
known coherent Maxwell plane-wave packet independently at the surface. Compare
Q with these values and invert the **analytic** trace. Test the reconstructed
field at independent three-dimensional points, not just at the fitting nodes.

Across 18 cases with R/λ₀ = 2, 5, 10, 100, 1,000 and 100,000:

| Quantity | Largest relative error |
|---|---:|
| Q electric field on the surface | 2.67e-12 |
| Q⁻¹ electric field on the tangent plane | 1.46e-11 |
| Reconstructed electric field at independent 3D points | 1.37e-11 |

For the spherical graph at R = 10λ₀:

| Reconstruction | Tangent-plane complex-E error |
|---|---:|
| Ours: tangential least-squares Q⁻¹ | 7.12e-12 |
| Scaled adjoint used as inverse | 0.1675 |
| Flat FFT of curved samples followed by negative sag | 0.4141 |

These tests support retaining the current inverse. They do not identify a
missing homogeneous amplitude factor. The direct oracle, adjoint-identity
check and independent-point reconstruction avoid relying on a round trip alone.

The largest observation patch is 42.56 mm wide. Its packet's angular bandwidth
shrinks as R increases; the grid resolves its envelope. This is not a solution
of a centimetre lens at fixed numerical aperture. These tests cover forward
propagating modes, not unstable recovery of evanescent content. They also do
not reproduce the supplied report's tilted-carrier or dipole-Weyl experiments.

### Interface conditions after reconstruction

The prescribed Fresnel traces satisfy algebraic local matching to roundoff.
That does not show that Q⁻¹ can represent both E and H simultaneously in its
selected outgoing Fourier space.

Both reflected and transmitted traces are reconstructed, and their fields are
evaluated on independent Gauss nodes on the same physical graph. Reflected
propagation uses time reversal in the real medium: `(E*, -H*)` is a forward
field, so the existing homogeneous inverse can analyze it. This adds no
dielectric boundary solve or material-interaction correction.

For R = 10λ₀, a 48² inverse grid, axial incidence and n₁ = 1 → n₂ = 1.5,
the 36² independent surface quadrature gives:

| Geometry | Method | E tangent | Z₀H tangent | D normal | Z₀H normal | Normal flux |
|---|---|---:|---:|---:|---:|---:|
| Sphere | Ours: electric fit | 0.0420 | 0.0671 | 0.0749 | 0.0348 | 0.0308 |
| Sphere | Ours: joint E/H fit | 0.0439 | 0.0509 | 0.0580 | 0.0384 | 0.0267 |
| Paraboloid | Ours: electric fit | 0.0330 | 0.0584 | 0.0659 | 0.0281 | 0.0256 |
| Paraboloid | Ours: joint E/H fit | 0.0346 | 0.0438 | 0.0523 | 0.0311 | 0.0227 |
| Freeform | Ours: electric fit | 0.0746 | 0.1302 | 0.1430 | 0.0525 | 0.0662 |
| Freeform | Ours: joint E/H fit | 0.0786 | 0.1072 | 0.0980 | 0.0585 | 0.0439 |

All entries are relative norms, not percentages. E and impedance-scaled H jumps
are normalized by the incident electric surface L2 norm. Normal D uses the jump
in n²E and an additional n₁² denominator. Flux is the L2 jump in the total normal
Poynting vector divided by its incident L2 norm. Incident/reflected interference
is retained. Quadrature uses physical dS. The full report includes prescribed
Fresnel rows, reverse index contrast, grid and quadrature refinement.
For these fixed 48² reconstructions, doubling the independent quadrature to
72² changes each reported normalized residual by at most 2.42e-4. This is an
integration check; it does not establish convergence of the finite-patch model.

Joint fitting decreases some magnetic, displacement and flux residuals but
increases electric and normal-magnetic residuals. This does not establish a
better full optical solution. The finite patch, outgoing angular restriction
and implicit periodic plane representation remain part of this experiment;
these numbers are not isolated errors in Fresnel coefficients or curvature
terms. There is no accepted change to the default inverse based on this table.

### A sampled fit can conceal unresolved phase

At R = 100,000λ₀, the coarse 32² spherical chart fits its tangential-E boundary
data to approximately 4.8e-11 at the collocation nodes. The same reconstructed
fields give a mismatch of order one at independent surface points. The
transmitted local transverse wavevector exceeds that grid's Nyquist limit by
more than 1,100 times.

This is an explicit aliasing failure control. Even the independent quadrature
cannot give a converged continuum norm for this unresolved case. Its role is
to disprove accuracy inferred from collocation matching. It does **not** show
that the resolved macroscopic method has an order-one physical error.

For genuinely macroscopic, fixed-angle propagation, a resolved carrier/envelope
or local-chart representation must precede any accuracy claim. Merely scaling
geometry while retaining a small FFT grid is insufficient. There is no evidence
here that this particular failure caused the earlier notebook's small-lens
power deficit; those are different discretizations and physical problems.

## Complete curl of a retained asymptotic field

The [derivative adapter](../general/src/vecdiff_general/propagation/experiments/asymptotic.py)
accepts supplied amplitude coefficients and their first and second **Cartesian**
derivatives. It does not generate interface coefficients. With

\[
E=e^{ik_0 S}B_N,\qquad B_N=\sum_{m=0}^N A_m/(ik_0)^m,
\]

it differentiates the actual retained field:

\[
\partial_i E=e^{ik_0S}(\partial_iB_N+ik_0 S_i B_N),
\]

\[
\partial_i\partial_jE=e^{ik_0S}
[\partial_i\partial_jB_N+ik_0(S_i\partial_jB_N+
S_j\partial_iB_N+S_{ij}B_N)-k_0^2S_iS_jB_N].
\]

The [Maxwell diagnostics](../general/src/vecdiff_general/observables/maxwell.py)
then compute `H = curl(E)/(i k₀ Z₀)`. In particular, the term
`curl(A_N)/(i k₀)^(N+1)` is included. The audit does not separately truncate H
and thereby remove the first unmatched term.

The same derivatives supply divergence, vector Helmholtz and the full
`curl curl E − n² k₀² E` residual. A longitudinal Helmholtz solution is included
as a negative control: Helmholtz alone passes, while divergence and Maxwell
curl-curl fail. Additional tests compare derivatives with direct plane waves
and Cartesian finite differences, retain the final curl term for orders 0–3,
and detect a wrong magnetic trace despite exact tangential-E matching.

Normal derivatives must come from the actual three-dimensional continuation.
Differentiating samples `E(x,y,h(x,y))` supplies surface derivatives and cannot
replace them. The new [independent coefficient generator](../general/src/vecdiff_general/propagation/experiments/taylor_interface.py)
constructs these derivatives directly from the supplied recurrence.

## Criteria for accepting the higher-order experiment

1. Reproduce the original coefficients and their derivative continuation,
   with derivative depth independent of retained asymptotic order.
2. Compare all four boundary conditions and bulk equations for each order,
   including the few-wavelength and oblique cases where the supplied report
   says higher orders make the residual worse.
3. Pass those traces through this repository's Q⁻¹ and evaluate the same jumps
   on independent surface quadrature. Local improvement must survive
   reconstruction to count as an improvement of our composed method.
4. Compare absolute complex fields and Poynting flux on fixed intermediate,
   meridional and detector coordinates, including boundary, grid, spectral-band
   and reference convergence. Do not align amplitude, phase or focal position.
5. Audit the reference's one-sided material traces, including rims, and retain
   its physical scope separately from single-pass or finite-pupil models.

The [frequency-explicit Maxwell residual analysis of Chaumont-Frelet and Vega](https://arxiv.org/abs/2009.09204)
also makes clear that residual reliability depends on stability and resolution.
Small local jumps alone do not certify a complete radiating solution.

The present evidence supports further investigation of the interface
continuation and spectral representation. It does not yet demonstrate an
improvement in focal intensity or solve the main method's power discrepancy.

## Reproduce

```sh
.venv/bin/python general/examples/maxwell_boundary_audit.py
.venv/bin/python -m pytest general/tests/test_maxwell_audit.py \
  general/tests/test_operational_propagation.py \
  general/tests/test_representation_contracts.py -q
```

Generated results record source SHA-256 hashes. No external report values are
inserted into the generated measurement tables. The notebook outputs and
production propagation implementation are not modified by this experiment.
