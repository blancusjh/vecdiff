# Curvature orders and independently propagated Maxwell fields

## Scope and measured conclusion

The experimental branch is `research/maxwell-boundary-audit`. The implementation
generates the Fresnel term and the first three inverse-wavenumber corrections
directly from the supplied eikonal, transport and divergence equations. It does
not require the unavailable external `taylor_interface.py` or its arrays.

The [executed numerical report](../output/curvature_validation/REPORT.md)
contains the original experiment's measured fields, residuals, failure cases
and refinement data. [Notebook 11](../general/docs/notebooks/11_curved_interfaces.ipynb)
presents the validated results: exact-solution convergence, the convergent
angular range, the single-pass validity criterion, and four thin curved lenses
at orders 0 and 3 against BEM with a Stratton–Chu check. Order 3 enters real
lenses through `experiments.orders.OrderInteraction`, a per-plane-wave
interaction for the production chain (plane and spherical surfaces).
The results establish a local boundary improvement and a propagated-field
improvement in a shallow periodic cavity. The production-Q lens-exit experiment
does **not** show a substantial increase in focal intensity. The earlier main
method's full-lens power deficit remains unresolved.

A later review (section 7) checked the generated coefficients against exact
Maxwell solutions rather than against their own residuals. Through order 3 they
match the exact single-interface fields of a dielectric cylinder (TE and TM) and
sphere, with errors falling as (kR)^−(N+1). Section 8 adds a finite
plano-convex lens whose exit is lit by the diffracted field of the entrance
stop. There, too, orders 1–3 change the focal field by less than 0.1%, as the
1/(kR) theory predicts, and an external full-wave FEM solution of the same lens
confirms that the missing light outside the aperture, not curvature order, is
the leading error. The main-method deficit is about a factor of two and cannot
come from these terms.

All comparisons retain complex vectors and physical amplitudes. No fitted gain,
phase alignment, empirical energy factor or reference-dependent coefficient
enters the candidate. The existing default Fresnel interaction remains the
leading term; higher orders are an opt-in experimental API.

## 1. Local graph coordinates and units

Use the phasor convention exp(−iωt), nonmagnetic lossless media, k₀ = 2π/λ₀,
and H̄ = Z₀H = curl(E)/(ik₀). Let L be a physical length scale and define

\[
u=x/L,\quad v=y/L,\quad s=z/L-h(u,v).
\]

The physical interface is z = Lh(u,v). Polynomial coefficients are normalized
Taylor coefficients, so the stored coefficient of uᵃvᵇsᶜ already includes the
inverse factorial. All electric vectors use fixed Cartesian components.
Dimensionless Cartesian derivatives are

\[
D_x=\partial_u-h_u\partial_s,\quad
D_y=\partial_v-h_v\partial_s,\quad D_z=\partial_s.
\]

Physical derivatives introduce 1/L. Differentiating surface samples alone would
omit the normal derivative terms and would give the wrong three-dimensional curl.
The recurrence constructs those normal derivatives explicitly.

For a plane wave in medium 1, with real unit direction d and transverse amplitude
a, the dimensionless incident phase is

\[
S_i=n_1[d_xu+d_yv+d_z(h+s)].
\]

Its restriction at s = 0 is shared by all three branches. The numerical phase is
exp(ik₀LS); the mth dimensionless amplitude coefficient is divided by (ik₀L)ᵐ.
Thus physical Aₘ equals the stored coefficient divided by Lᵐ.

## 2. Eikonal normal continuation

Let g = (hᵤ,hᵥ), p = (Sᵤ,Sᵥ), and G = 1 + g·g. Substitution into
|∇S|² = n² gives

\[
G S_s^2-2(g\cdot p)S_s+p\cdot p-n^2=0,
\]

and hence

\[
S_s=\frac{g\cdot p\;\pm\sqrt{(g\cdot p)^2+G(n^2-p\cdot p)}}{G}.
\]

For incidence into the positive graph normal, reflection selects the minus root
in medium 1; transmission selects the plus root in medium 2. Expanding the right
side in s determines each successive normal Taylor coefficient from the previous
ones. The square root is an ordinary analytic branch away from its turning point.

The implemented generator supports real propagating incident directions and
noncritical propagating transmitted branches. It rejects grazing, total internal
reflection and invalid square-root branches. It is not a uniform grazing
approximation or an implementation of general evanescent incidence.

## 3. Amplitude continuation and material interaction

For every retained m, w = ∇S, the original equations are

\[
w\cdot A_m=-\nabla\cdot A_{m-1},\qquad
2w\cdot\nabla A_m+(\Delta S)A_m=-\Delta A_{m-1},
\]

with A₋₁ = 0. In the graph coordinates the transport equation becomes

\[
\partial_s A_m=
-\frac{w_x\partial_u A_m+w_y\partial_v A_m+
\tfrac12(\Delta S)A_m+\tfrac12\Delta A_{m-1}}
{w_z-w_xh_u-w_yh_v}.
\]

This supplies normal derivatives after the surface amplitude is found. At each
order, the three reflected and three transmitted Cartesian amplitudes are six
unknowns. They satisfy:

1. Two tangential electric matching equations.
2. Two tangential magnetic coefficient matching equations, with
   H̄ₘ = w × Aₘ + curl(Aₘ₋₁).
3. The reflected and transmitted divergence recurrences.

The tangent test directions are (1,0,hᵤ) and (0,1,hᵥ). They span the tangent
plane; orthonormalization is unnecessary for imposing zero tangential jumps.
The resulting 6×6 Taylor system is solved coefficient by coefficient using its
constant matrix. Order zero gives ordinary Fresnel reflection and transmission.
Each subsequent order uses derivatives of the preceding coefficients, not a
fitted multiplicative field correction.

The default Taylor degree is 2N + 4; Aₘ is retained through degree
2(N−m) + 2. This leaves two Cartesian derivatives of A_N for the bulk Maxwell
audit. Increasing derivative depth leaves the retained surface coefficients
unchanged in the test suite; it does not insert an additional optical order.

Normalized polynomial products are evaluated by a batched Fortran convolution
kernel, with an independent NumPy fallback. The optical algebra and geometry
composition remain in the Python interface.

## 4. Complete curl and independent checks

For the actual retained electric field,

\[
E^{[N]}=e^{ik_0 S}\sum_{m=0}^{N}\frac{A_m}{(ik_0)^m},
\]

the magnetic field includes

\[
\bar H^{[N]}=e^{ik_0S}\left[
\sum_{m=0}^{N}\frac{w\times A_m+\nabla\times A_{m-1}}{(ik_0)^m}
+\frac{\nabla\times A_N}{(ik_0)^{N+1}}\right].
\]

The final term is included both in local traces and full derivative jets.
Dropping it would check the coefficient identities already imposed during
construction. The tests compare the complete curl with a five-point finite
difference of the actual three-dimensional local electric continuation.

The audit records tangent E/H and normal D/B jumps with physical dS weights.
E and Z₀H use the incident electric surface norm; the D jump additionally divides
by n₁². It also records divergence/(k|E|), Helmholtz/(k²|E|) and the full
curl-curl Maxwell residual/(k²|E|), where k = n k₀. Small locally imposed electric
jumps alone are not a field-accuracy certificate.

Tests cover flat/tilted-plane Fresnel limits, an index-matched curved interface,
rotation covariance for mixed curvature, derivative depth, full-curl consistency,
order/size scaling and invalid branch rejection. The general recurrence contains
no spherical scattering coefficients.

## 5. Homogeneous reconstruction and repeated encounters

For a homogeneous medium, the graph trace of a spectrum is

\[
(Q_h a)(\rho)=\int a(\kappa)
e^{i\kappa\cdot\rho+iq(\kappa)h(\rho)}\frac{d^2\kappa}{(2\pi)^2}.
\]

The operational reconstruction solves the tangential electric least-squares
problem for this same map. Its adjoint is used inside the inverse algorithm;
it is not substituted for Q⁻¹. A geometric normalization cannot turn an arbitrary
finite-patch trace adjoint into its inverse.

`TaylorInterfaceOperator` applies the local expansion separately to each populated
incident spectral direction, sums coherently, and passes the resulting trace to
the production `SurfaceTransport.inverse`. The order-zero propagated field is
tested against the existing primary one-interface path. The lens example uses
this integration, not an aperture-current radiation substitute.

The periodic benchmark uses a direct Fourier trace matrix for the corresponding
homogeneous inverse on a shallow graph. It fits electric data only; magnetic and
normal conditions are measured afterward at independently shifted points.
Candidate modes are propagating; the independent Fourier–Maxwell reference also
includes evanescent channels. This distinction is recorded with the results.

For encounters, total asymptotic order and reflection count are independent:

\[
b_m=\sum_{j=0}^{m}T_j a_{m-j},\qquad
a_{p+1,m}=\sum_{j=0}^{m}R_j\mathcal P a_{p,m-j}.
\]

No products above the retained total order are kept. The layer sums finite
round-trip terms, retaining each coherent phase. An exact resonant flat slab and
an independent polynomial-product calculation test the encounter implementation.
No assertion of universal infinite-series convergence is made.

## 6. What the lens test establishes

The local negative-sphere exit has R = 30λ₀ and aperture radius 12λ₀. The direct
incident wave is Ex = 0.8 V/m in index 1.5 glass. This is the known illumination
of an ideal single-pass exit test, not the full field arriving from a finite lens.
Production Q⁻¹ uses an absorbing pupil, a Cartesian grid and max_sine = 0.92.

All orders share the same material geometry, incident field, inverse tolerances,
detector coordinates and absolute scale. The retained terms barely change the
focal field. Grid and window sensitivity exceed that small order effect, so
there is no justified claim of a converged full-lens intensity correction.
The cavity's strong improvement must not be transferred to this different model.

## 7. Exact-solution validation of orders 1–3

Small boundary and bulk residuals show that the recurrence is self-consistent.
They do not by themselves show that the coefficients equal the asymptotic
expansion of a true solution. An exact reference supplies that.

[`references/debye.py`](../general/src/vecdiff_general/propagation/references/debye.py)
builds the first Debye term of a plane wave meeting one cylinder or sphere. The
transmitted field uses only the one-way Hankel function that carries it away
from the interface. The reflected field likewise never re-enters. There are no
internal returns. These are exact Maxwell fields, verified to satisfy
tangential E and H continuity to about 1e-15. Near the illuminated vertex their
1/(kR) expansion must equal the local recurrence at every order.

[`curvature_exact_references.py`](../general/examples/curvature_exact_references.py)
compares complex E and Z₀H at five surface points up to 0.3R from the vertex.
Incidence angles reach 21°. Relative L2 error through order N, index 1 → 1.5,
sphere with x polarization:

| kR | Reflected N=0 | N=1 | N=2 | N=3 | Transmitted N=0 | N=3 |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 1.93e-2 | 3.36e-3 | 9.09e-4 | 3.81e-4 | 2.10e-3 | 7.27e-5 |
| 50 | 9.66e-3 | 8.25e-4 | 1.09e-4 | 2.22e-5 | 1.05e-3 | 4.99e-6 |
| 100 | 4.83e-3 | 2.05e-4 | 1.34e-5 | 1.35e-6 | 5.22e-4 | 3.23e-7 |
| 200 | 2.41e-3 | 5.13e-5 | 1.67e-6 | 8.36e-8 | 2.61e-4 | 2.04e-8 |
| 400 | 1.21e-3 | 1.28e-5 | 2.08e-7 | 5.21e-9 | 1.31e-4 | 1.28e-9 |

Each doubling of kR divides the order-N error by 2^(N+1), the asymptotic rate.
The TE and TM cylinder give the same rates; see
`output/curvature_validation/exact_references.json`. A concave vertex, with the
centre on the incident side, gives identical magnitudes. That is expected: A_m
scales as R^−m, so reversing curvature conjugates the retained series.
`tests/test_exact_single_interface.py` enforces the rates.

**Second implementation.** The external research bundle contains the original
coefficient code. It solves the matching conditions in closed form with an
inverse admittance operator and continues the phase by an explicit recurrence;
this repository solves a 6×6 Taylor system. At a sphere vertex, for both
curvature signs, both index directions, TE and TM and incidence cosines 1, 0.95
and 0.85, the two agree through order 3 to 3.3e-15 relative to max(|A_m|, |A₀|).

**High-to-low index.** For n₁ > n₂ the exact single-interface field contains an
additional wave that the local series cannot contain. Its error is the same at
every order and decays with kR. Along the surface its tangential wavenumber is
0.662 n₁k₀, against sin θc = 0.667, for n₁/n₂ = 1.5. It is the critical-angle
wave launched from the ring where incidence is critical, a non-local effect of
the closed body. It is not a coefficient error. The recurrence code path is
index-symmetric and chart-covariant to 2e-16. The n₁ > n₂ lens exit is
therefore validated through the same algebra, but not isolated by an exact
single-interface reference.

## 8. Finite lens with entrance-stop diffraction

[`polar_lens.py`](../general/src/vecdiff_general/propagation/experiments/polar_lens.py)
was left unfinished by the original experiment. As written it could not run at
its defaults: an internal band of sine 0.4 plus a 23.6° rim tilt meets the exit
at 47°, beyond the 41.8° critical angle, and the recurrence correctly refuses.
Its 12 radial and 24 azimuthal quadrature nodes were also several times too few.
It had no tests.

It has been rebuilt on one symmetry. A sphere is invariant under rotation about
its centre, so the local field of one plane wave at one point depends only on the
incidence angle and the s/p split of the amplitude. `SphericalExitTable`
evaluates the general graph recurrence on 96 Chebyshev nodes in incidence angle
and rotates the result to every (mode, exit point) pair. Against direct
recurrence calls at random directions, points and complex polarizations it
agrees to 1e-14 at every order. No sphere-specific formula enters.

`PolarTaylorLens` places a hard stop of radius a on the planar entrance, with
tangential E = tE₀ on the disk. Its exact angular spectrum is sampled by
Gauss–Legendre nodes in |κ| and a trapezoid in azimuth. Both sizes follow from
the trace bandwidth, and a test checks the on-axis sum against the closed form
tE₀(1 − J₀(aκmax)).

**The band is set by convergence of the series, not by the critical angle.**
Glass-to-air terms grow quickly as incidence approaches the critical angle.
For R = 30λ₀ the ratio of the last retained term to the one before is:

| Incidence | 0° | 20° | 30° | 33° | 36° | 38.8° |
|---|---:|---:|---:|---:|---:|---:|
| Transmitted angle | 0° | 30.9° | 48.6° | 54.8° | 61.8° | 70.0° |
| Largest term ratio, orders 1–3 | 0.006 | 0.09 | 0.29 | 0.52 | 1.20 | 7.3 |

Beyond about 34° the order-3 term exceeds order 2: the asymptotic series has
passed its smallest term, and adding it makes the field worse. The lens
therefore scans the recurrence directly (`convergent_incidence`) and limits
incidence to where every term is at most half the one before. The floor for
that ratio is the nominal size of the previous term, because the first
correction vanishes by symmetry at normal incidence. Here the limit is 32.8°
incidence, internal band κ ≤ 0.160 nk₀, keeping 96.5% of the disk-spectrum
power. Larger bands are rejected with an explicit error. An earlier draft of this
section used a 70° transmission cap. That admitted divergent order-3 rim terms,
visible as an order-3 peak jump of 0.005%.

`curvature_finite_lens.py`, λ₀ = 532 nm, E₀ = 1 V/m, 43 × 76 modes, production
Q⁻¹ with max_sine = 0.92, 512² grid:

| N | Peak \|E\|² on axis (V²/m²) | Surface power (W) | Reconstructed power (W) | E residual | H residual |
|---|---:|---:|---:|---:|---:|
| 0 | 61.1810 | 1.47831e-13 | 1.47526e-13 | 0.0547 | 0.0715 |
| 1 | 61.1798 | 1.47855e-13 | 1.47531e-13 | 0.0547 | 0.0718 |
| 2 | 61.1787 | 1.47846e-13 | 1.47529e-13 | 0.0547 | 0.0718 |
| 3 | 61.1788 | 1.47845e-13 | 1.47529e-13 | 0.0547 | 0.0718 |

Orders 2 and 3 now agree to 1e-7 in peak, as a converging series should.

| Check | Relative change |
|---|---:|
| Orders 0 → 3, complex focal-plane field | 7.7e-4 |
| Orders 0 → 3, complex axial field | 1.1e-4 |
| Orders 0 → 3, power through the 6λ₀ detector | 7e-7 |
| Modes × 1.5, axial field | 6e-15 |
| Grid 192² → 256² → 384² → 512², axial field | 9.4e-3, 2.6e-3, 3.0e-3 |

The internal quadrature is converged. The production Cartesian Q⁻¹ settles to
about 0.3% only from 256² samples; 192² (0.5λ₀ spacing at max_sine 0.92) is
near its sampling limit and moves the focus by about 1λ₀.

### Boundary conditions before and after propagation

[`curvature_lens_boundary_audit.py`](../general/examples/curvature_lens_boundary_audit.py)
checks all Maxwell interface conditions on the exit face with physical dS
weights. The glass side is always incident plus reflected. The air side is
either the local transmitted field, or the field that production Q⁻¹
reconstructs from it and evaluates back on the same surface. Relative jumps,
whole aperture, 256² grid:

| Air-side field | N | Tangential E | Tangential Z₀H | Normal D | Normal B | Normal flux |
|---|---|---:|---:|---:|---:|---:|
| Local interface | 0 | 4e-15 | 2.2e-3 | 5e-16 | 1e-15 | 3.5e-4 |
| Local interface | 3 | 4e-15 | 1.3e-5 | 5e-16 | 1e-15 | 3.5e-6 |
| After Q⁻¹ | 0 | 4.1e-2 | 6.6e-2 | 2.8e-2 | 2.9e-2 | 3.9e-2 |
| After Q⁻¹ | 3 | 4.1e-2 | 6.6e-2 | 2.8e-2 | 2.9e-2 | 3.9e-2 |

The interface field satisfies every condition, and order 3 reduces the magnetic
jump 170-fold. After Q⁻¹ the surface jumps are 3–7%, independent of order,
21–26% in the last grid spacing before the rim, and unchanged at 512². These
surface residuals are **not** downstream field errors: most of them are
near-field content concentrated at the rim that a propagating-only spectrum
cannot carry and that never reaches the focus. The next subsection measures
the downstream effect directly; it is about 0.2%.

### Where order 3 helps: size sweep

[`curvature_size_sweep.py`](../general/examples/curvature_size_sweep.py) repeats
the exit-face audit for R/λ₀ = 3 to 100 at fixed shape (a = 0.4R, thickness
R/6) and a fixed internal band, sine 0.1, so incidence stays below 29.3° at every
size. The Q⁻¹ grid keeps about 0.4λ₀ spacing. The series-convergence limit is
recorded rather than enforced.

| R/λ₀ | kR in glass | Convergent incidence | Local tangential Z₀H jump, N = 0 / 1 / 2 / 3 | Gain N0→N3 | Propagated tangential E / Z₀H, any N | Axial field change N0→N3 |
|---:|---:|---:|---|---:|---|---:|
| 3 | 28 | 10.2° | 2.2e-2 / 1.7e-2 / 2.2e-2 / 5.3e-2 | 0.4× (worse) | 0.28 / 0.32 | 1.8e-3 |
| 5 | 47 | 18.9° | 1.2e-2 / 5.6e-3 / 4.5e-3 / 6.5e-3 | 1.8× (best N = 2) | 0.17 / 0.20 | 8.0e-4 |
| 10 | 94 | 26.0° | 4.7e-3 / 1.0e-3 / 3.9e-4 / 2.7e-4 | 17× | 0.066 / 0.091 | 3.0e-4 |
| 20 | 188 | 30.7° | 2.7e-3 / 2.9e-4 / 5.1e-5 / 1.6e-5 | 170× | 0.049 / 0.072 | 1.6e-4 |
| 30 | 283 | 32.8° | 1.9e-3 / 1.4e-4 / 1.7e-5 / 3.6e-6 | 530× | 0.041 / 0.067 | 1.1e-4 |
| 60 | 565 | 35.4° | 1.0e-3 / 3.7e-5 / 2.3e-6 / 2.4e-7 | 4200× | 0.028 / 0.049 | 5.9e-5 |
| 100 | 942 | 36.8° | 6.4e-4 / 1.4e-5 / 5.0e-7 / 3.3e-8 | 19000× | 0.021 / 0.040 | 3.6e-5 |

Order 3 improves the interface field only once the series converges over the
whole band. Below R ≈ 5λ₀ the band exceeds the convergent incidence and higher
orders make the interface worse; at 5λ₀ order 2 is the best truncation. From
10λ₀ the gain grows roughly as (kR)³, as the asymptotic theory predicts: the
jump scales as (kR)^−1 at order 0 and (kR)^−4 at order 3. The
normal flux jump follows the same pattern. The propagated surface residual
falls with size from 28% to 2% and does not depend on N; as shown next, it
overstates the downstream effect of propagation, which at 30λ₀ is about 0.2%.

### Where the propagation error comes from

The transport has two steps. The forward map Q, from a plane-wave spectrum to
the curved surface, is the angular spectrum evaluated with the sag inside the
integral; it is exact. The inverse Q⁻¹ is a least-squares fit of a surface
trace by a finite band of outgoing plane waves. The propagation jumps above are
exactly the residual of that fit, QQ⁻¹T − T; everything after it is ordinary,
exact ASM. Q⁻¹ recovers compatible data, the trace of a genuine outgoing
field, to about 1e-11 (see the boundary audit).

[`q_inverse_representability.py`](../general/examples/q_inverse_representability.py)
feeds the same order-0 exit trace to the same Q⁻¹, 256² grid. Interior means
more than 4λ₀ from the rim:

| Trace | max_sine | Tangential E, whole / interior | Tangential Z₀H, whole / interior |
|---|---:|---|---|
| Hard absorbing pupil (production) | 0.92 | 3.5% / 1.0% | 5.6% / 2.5% |
| cos² rim taper, 2λ₀ | 0.92 | 1.1% / 0.59% | 4.9% / 1.4% |
| cos² rim taper, 4λ₀ | 0.92 | 0.14% / 0.11% | 3.0% / 0.26% |
| Hard absorbing pupil | 0.99 | 3.3% / 1.3% | 8.2% / 5.9% |

Softening the rim reduces the electric residual 25-fold; widening the band does
not help. The residual is not in the ASM and not a defect of the Q⁻¹ algorithm.
On a curved chart the sag couples position and spectrum, so Q⁻¹ is a
least-squares inverse rather than a Fourier transform, and evanescent waves are
excluded because their weight e^(−|q|h) varies exponentially across the sag. A
trace that jumps at the rim needs evanescent content, so the best propagating
fit rings from the rim inward. The
taper is only a diagnostic. It changes the target, and the magnetic residual in
the taper zone stays high because a tapered E and H no longer form a Maxwell
pair.

**Downstream effect.** [`q_inverse_downstream.py`](../general/examples/q_inverse_downstream.py)
propagates the same hard-pupil trace three independent ways and compares the
fields at 30–60λ₀ beyond the rim, on axis and off axis. A joint E/H fit differs
from the electric fit by 0.17%, and a Stratton–Chu radiation integral of the same
trace, with no band limit and no fit, by 0.23%. The surface residual is
therefore mostly near-field rim content; the propagation step changes the
downstream field by about 0.2%. (A dipole-oracle variant was discarded: dipoles
radiate strongly along the surface at grazing angles, outside any propagating
band, so it does not test lens-like traces.) The large remaining errors come
from what the absorbing stop removes, the light that passes outside the lens.
See [surface edges and pupils](PUPILS_AND_EDGES.md).

### Comparison with an independent full-wave FEM solution

An external research bundle (reference only, not part of this repository)
contains a degree-10 axisymmetric vector FEM solution of this same lens: n = 1.5,
R = 30λ₀, aperture 12λ₀, centre thickness 5λ₀, cylindrical sidewall, unit
x-polarized illumination. It includes internal returns, sidewall scattering and
the light that passes outside the lens. Its own audit reports a 0.78%
normal-D jump concentrated at the rim, so it is a strong reference, not an
oracle. [`curvature_fem_comparison.py`](../general/examples/curvature_fem_comparison.py)
compares complex on-axis E over 48–60λ₀ with no alignment. The single-pass models
have absorbing stops, so the exact on-axis field of the complementary opaque
disk, (z/r) e^{ik₀r} with r = √(z² + a²), is added separately:

| Model | Error | + outside light | Peak \|E\|² (z) |
|---|---:|---:|---:|
| FEM, full lens | — | — | 73.29 (52.38λ₀) |
| Plane wave 0.8 at the exit, bundle candidate | 6.8% | 5.4% | 64.54 (51.52λ₀) |
| This finite lens, N = 0, 512² | 10.2% | 4.7% | 77.24 with outside light (51.42λ₀) |
| This finite lens, N = 3, 512² | 10.2% | 4.7% | 77.24 with outside light (51.42λ₀) |

Three conclusions follow. The light passing outside the aperture is a leading
missing term; adding it halves the single-pass error. With it included, the
entrance-diffraction model is the closest of these models to the full-wave field.
Orders 1–3 do not change the comparison in the fourth digit. The remaining
4.7%, a peak about 5% high and a focus about 1λ₀ early, belong to internal
returns, sidewall scattering and the discarded 3.5% of edge-diffracted power.

**What this shows.** The first correction is of relative size 1/(kR) times
geometric factors. The measured focal change of about 8e-4 sits in that range.
The notebook-07 deficit is roughly a factor of two for lenses a few wavelengths
across. There kR is small, the asymptotic series is not convergent, and the
earlier diagnosis attributes most of the loss to the missing field outside the
entrance support. The FEM comparison here confirms that mechanism on a larger
lens. Curvature orders do not address it.

## 9. Where higher orders are worth using

Evidence from sections 7 and 8 and the periodic cavity:

- **Lower bound: convergence.** The series must converge over the incidence
  angles present. For glass to air this needs about R ≥ 10λ₀ (kR ≥ 100 in
  glass). At 3λ₀ order 3 makes the interface worse; at 5λ₀ order 2 is the best
  truncation. At R = 30λ₀ convergence stops near 33° incidence.
- **Upper bound: relevance.** The order-0 interface error is roughly 0.6/(kR):
  0.5% at 10λ₀, 0.06% at 100λ₀, about 3e-6 for a 1 cm lens at 532 nm
  (extrapolated). Once it is below the accuracy target, higher orders add
  nothing measurable. Power through a focal detector changes by below 1e-6.
- **Useful window.** Microlenses of about 10–100λ₀ with accuracy targets below
  0.1%; steep local incidence still inside the convergent range, where order-0
  error reaches percent level; and multi-pass or resonant systems. In the
  index-3 periodic cavity with 32 round trips, order 0 levels off at 3.2e-4 and
  order 3 reaches 2.7e-8, because local errors accumulate coherently over
  encounters. This is the strongest evidence for the higher orders.
- **Order 1 carries most of the gain**, 5–47 times in the size sweep. Orders 2
  and 3 are worth adding from about 20λ₀.
- **Propagated, not only local.** With every other error removed (section 10),
  order 3 makes the field transmitted into a 10λ₀ sphere 25 times more
  accurate after propagation (1.7e-4 to 6.9e-6 against exact Debye), provided
  the corrections are truncated term by term and smoothly (see there).
- **Precondition.** The gain is visible only when the rest of the model is
  more accurate than the order-0 interface error, below about 1e-3 for R of
  10–100λ₀. Propagation itself changes the downstream field by about 0.2% at
  30λ₀; missing physics (light outside the lens, sidewall, returns) is larger.

## 10. Sphere under a Gaussian beam: internal returns against exact Debye terms

**Question.** Does a better interface field give a better propagated field?
In the lens comparisons of sections 8 and notebook 11 the answer was hidden
by larger errors (rim, sidewall, BEM discretization). Here every other error is
removed: a smooth, band-limited Gaussian beam meets a glass sphere (n = 1.5),
and every propagated field is compared with its exact counterpart.

**The iteration (operational Debye series).** `experiments.returns.ElementReturns`
applies T and R at every encounter. What is transmitted leaves; what is
reflected stays inside and is carried (ASM, then Q) to the other surface:

    a_1 = T_front s,   exit_0 = R_front s
    a_{p+1} = R_j P a_p,   exit_p = T_j P a_p     (j = back for odd p, front for even p)

so the interior field is sum_p (R_f P R_b P)^k (...) = (I - M)^{-1}(...), a
Neumann series in the round trip M. Backward travel uses the same Q, Q^-1,
ASM and interaction on the same surface described in a reversed frame (mirror
lattice). For a sphere, pass p is exactly Debye term p:
`references.debye.SphereDebyeSeries` gives each term (interior T21 R11^(p-1) j,
exterior T21 R11^(p-1) T12 h1 / 2, term 0 = (R22 - 1)/2) for any plane-wave
sum. The terms sum to Lorenz–Mie to 6e-14 inside and 5e-11 outside.

For an axially symmetric, x-polarized beam (TM amplitude A(α) cos β, TE
amplitude −B(α) sin β) only the m = ±1 multipoles occur, and
`references.axial_beam.AxialBeamSphere` evaluates every Debye term and the
whole Mie field with beam-shape coefficients p_n, q_n (projections of the
angular spectrum on the far-field patterns of M_o1n and N_e1n). One sum over n
per point replaces one Mie solution per plane wave, which makes dense maps of
the whole plane cheap: 3·10⁵ points in about 5 minutes. It agrees with the
per-plane-wave sum to 1e-9 (the lattice beam's band edge is the only
difference) and with the plane-wave Mie code as the waist grows.

**Beam and geometry.** The incident field is the same set of lattice plane
waves on both sides (dropped below 1e-12 of the peak). For a Gaussian the
footprint times the angular spread is fixed, so the steepest local incidence
is smallest for w0 ≈ sqrt(R λ / π); the local series must converge over it.

**Result, R = 10λ₀, w0 = 1.8λ₀** (notebook 11, section 9; data `curved_interfaces_study.sphere_beam("R10")`). Relative
L2 error against the exact Debye term on the axis and transverse lines:

| Pass p | Share of the interior field | Interior error, order 0 | Interior error, order 3 | Exit error, order 0 | Exit error, order 3 |
|---:|---:|---:|---:|---:|---:|
| 0 (external reflection) | – | – | – | 8.8e-4 | 2.2e-4 |
| 1 (entered once) | 0.98 | 1.7e-4 | **6.9e-6** | 1.0e-3 | 9.8e-4 |
| 2 (one internal reflection) | 0.27 | 4.4e-3 | 9.5e-3 | 1.8e-2 | 1.9e-2 |
| 3 | 0.049 | 4.2e-2 | 4.9e-2 | 0.13 | 0.13 |

**Result, R = 20λ₀, w0 = 2.5λ₀** (`sphere_beam("R20")`, three
passes; the untruncated remainder beyond pass 3 is 1% of the interior field, so
totals are not compared here):

| Pass p | Share of the interior field | Interior error, order 0 | Interior error, order 3 | Exit error, order 0 | Exit error, order 3 |
|---:|---:|---:|---:|---:|---:|
| 0 (external reflection) | – | – | – | 2.1e-4 | **3.2e-6** |
| 1 (entered once) | 0.97 | 3.9e-5 | **7.1e-6** | 1.1e-4 | **5.9e-5** |
| 2 (one internal reflection) | 0.27 | 3.3e-4 | **1.8e-4** | 1.0e-3 | 1.0e-3 |
| 3 | 0.052 | 3.8e-3 | 4.3e-3 | 3.4e-2 | 3.4e-2 |

At 20λ₀ order 3 improves the external reflection 66×, the entered field 5.5×,
the field leaving the sphere 1.9× and the first internal return 1.9×. The
Q^-1 residuals (6e-6 to 8e-4 for passes 0 to 2) show the charts still hold the
light there; from pass 3 on they reach 4e-3 to 0.13.

**What this shows.**

- *Yes: a better interface field gives a better propagated field* once the rest
  is accurate. At 10λ₀ pass 1 inside the sphere is 25× more accurate at order 3
  and the external reflection 4×; at 20λ₀ the reflection 66×, pass 1 5.5×, the
  exit and the first internal return about 2×.
- *The first order-3 implementation hid this.* A hard switch to order 0 at the
  convergent limit makes the retained field discontinuous in the direction of
  each plane wave. Summed over a beam, the corrections of the waves inside the
  limit no longer cancel where the beam is dark, and radiate: pass-1 error
  3.3e-3, twenty times worse than order 0, although the surface field inside
  the beam was 30× better. Three changes fix it (`OrderInteraction`
  defaults): each correction term keeps its own convergent limit (graded
  optimal truncation: 76°, 60°, 49° for orders 1, 2, 3 at this entrance),
  each is blended out smoothly over the last 20% of its range, and
  corrections are applied only where the incident envelope (intensity
  smoothed over λ/2) exceeds 1e-3 of its maximum, fully above 0.1. The
  regression test `test_beam_corrections_do_not_radiate_from_dark_chart` keeps it.
- *Later passes are limited by representation, not by order.* A sphere turns
  the internal light by about 2θt per reflection; after two reflections it
  meets the hemisphere charts at steep angles, where the plane-wave band and
  graph charts fail (Q^-1 residuals 3e-2 to 0.9). Pass 2 also suffers a
  mechanism of order 0 itself: individual beam waves meet the back surface
  beyond the critical angle (rays never do), and the square-root kink of
  flat Fresnel there does not cancel between waves. Spreading the kink
  (`critical=` option, width ~ (kR)^(-2/3)) removes the spurious ring (fit
  residual 3e-2 to 3e-4) but does not yet improve pass 2 at order 3, whose
  convergent band is taken from the transmitted series. These are open.
- *Too small a sphere for its beam.* At R = 5λ₀, w0 = 1.5λ₀ (a run made during
  this investigation, not kept in the notebook) the lit surface
  reaches beyond the order-3 limit and order 3 is worse than order 0 even on the
  surface (pass 1: 3.7e-3 against 9.8e-4). This is the lower bound of section 9.

The procedure therefore works as intended for a lens-like element (pass 1 and
the first returns), and the resonant sum is written in operational form; a
closed sphere with many returns needs charts that follow the light around the
surface, which the graph-chart transport does not provide.

## 11. Reproduction

From the repository root, with Python ≥3.10 and C/Fortran compilers:

```sh
python -m venv .venv
.venv/bin/python -m pip install './general[test,notebooks,nufft]'
.venv/bin/python -m pytest general/tests/test_taylor_interface.py general/tests/test_curvature_encounters.py general/tests/test_maxwell_audit.py general/tests/test_operational_propagation.py general/tests/test_representation_contracts.py general/tests/test_exact_single_interface.py general/tests/test_polar_lens.py -q
.venv/bin/python general/examples/curvature_local.py
.venv/bin/python general/examples/curvature_layer.py TE 6
.venv/bin/python general/examples/curvature_layer.py TM 6
.venv/bin/python general/examples/curvature_layer.py TE 7
.venv/bin/python general/examples/curvature_layer.py TM 7
.venv/bin/python general/examples/curvature_lens_exit.py 30 0.4 192
.venv/bin/python general/examples/curvature_lens_exit.py 30 0.4 256
.venv/bin/python general/examples/curvature_report.py
.venv/bin/python general/examples/curved_interfaces_notebook.py
.venv/bin/python general/examples/curvature_exact_references.py
.venv/bin/python general/examples/curvature_finite_lens.py 192 1
.venv/bin/python general/examples/curvature_finite_lens.py 192 1.5
.venv/bin/python general/examples/curvature_finite_lens.py 256 1
.venv/bin/python general/examples/curvature_finite_lens.py 384 1
.venv/bin/python general/examples/curvature_finite_lens.py 512 1
.venv/bin/python general/examples/curvature_lens_boundary_audit.py 256
.venv/bin/python general/examples/curvature_lens_boundary_audit.py 512
.venv/bin/python general/examples/curvature_size_sweep.py 3 5 10 20 30 60 100
.venv/bin/python general/examples/q_inverse_representability.py 256
.venv/bin/python general/examples/curvature_fem_comparison.py PATH/TO/curved_interface_research_bundle_2026-09-22
.venv/bin/python general/examples/curved_interfaces_study.py R10 R20 picture maps
.venv/bin/python scripts/notebooks.py --pattern 11_curved --timeout 0
```

The notebook builder executes the presentation cells against the freshly saved
data and embeds the actual PNG plots. It does not claim to rerun simulations
when displaying existing arrays. All simulations are the preceding explicit
commands. Timings in the report are measured on the current machine and may
include concurrent workload; they are not solver-speed rankings.

Raw JSON, complex NPZ arrays, PNGs and source hashes live in
`output/curvature_validation/`. The accompanying ZIP includes the local package,
native sources, tests, scripts, this derivation, report and saved results. It does
not include the unavailable external FEM snapshots or claim to reproduce their
boundary audit. The independent reference in this experiment is the included
shallow periodic Fourier–Maxwell solver.
