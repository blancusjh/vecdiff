# Polar propagation and energy-flux review

Review date: 2026-09-20. This review adds a reproducible experiment, not a new
propagation backend or an amplitude correction to the primary solver.

Follow-up: [local spatial-factor investigation](LOCAL_FLUX_INVESTIGATION.md)
tests a direction-dependent flux target followed by Maxwell reconstruction,
including independent BEM/Mie comparisons and non-stigmatic freeform surfaces.

## Reproduce

```sh
.venv/bin/python general/examples/polar_flux_audit.py
```

Results are written to `output/polar_flux_audit/results.json`.

## Reference priorities and the 200-wavelength case

The notebook deliberately sets `run_bem=False` and `run_fem=False` for its large
lenses. Neither solver was attempted at that size. This was a resource-budget
choice for our implementations, not a fundamental limit of either formulation.

Our present boundary element method (BEM) uses a full 3D triangular RWG PMCHWT
discretization and dense matrices. There are approximately three unknowns per
triangle, with 16 N² bytes for a single complex128 matrix. The default cap is
5,000 unknowns. Our finite element method (FEM) uses first-order 3D edge elements,
a surrounding perfectly matched layer, and sparse direct factorization; its
default cap is 40,000 edge unknowns. Neither implementation has the acceleration
or symmetry reduction needed for a useful large-lens reference on this machine.

An illustrative estimate for the notebook's weak D=200 λ0 lens has aperture
radius 100 λ0, curvature radius 2,500 λ0, thickness 60 λ0 and total boundary area
99,286 λ0². Equilateral triangles of edge 0.2 λ0 would give about 5.73 million
triangles, 17.2 million unknowns and **4.73 decimal petabytes for one dense
matrix**, before factorization. This assumed edge length is an illustration of
cost, not an accuracy prescription or a generated mesh.

BEM deserves priority for the homogeneous dielectric lenses considered here.
Retain FEM as a secondary reference, with resolution and convergence status
visible; the coarse results do not establish that FEM itself is unsuitable.

For rotationally symmetric lenses, a promising next reference is a vector
body-of-revolution boundary integral solver. Lai and O'Neil's
[penetrable-axisymmetric-object solver](https://arxiv.org/abs/1810.07067)
uses a Müller second-kind integral equation, Fourier separation, accelerated
modal kernels and adaptive quadrature, including geometries with edges. This
provides a relevant formulation for reducing a surface problem to generating
curve problems. It does not certify our present solver at D=200 λ0.

## Existing polar Fourier implementation

The pasted modal Fourier/Hankel formulas agree with `fourier/polar.py`:

\[
\widehat f_m(\kappa)=2\pi(-i)^m\int f_m(r)J_m(\kappa r)r\,dr,
\qquad
f_m(r)=\frac{i^m}{2\pi}\int\widehat f_m(\kappa)J_m(\kappa r)\kappa\,d\kappa.
\]

The actual public keyword is `mode="polar"`. For example:

```python
fourier.transform(values, mode="polar", coordinates=(r, phi),
                  frequencies=(kappa, spectral_phi), weights=dr_weights)
```

The audit transforms exp(-r²) using 96 Gauss radial nodes on [0,6], 16 angles
and 80 radial spectral samples on [0,6]. Against π exp(-κ²/4), the relative
sample norm error is 2.87e-15. This validates this transform example, not a
complete polar propagation method. The radial implementation is direct Bessel
quadrature, O(Nr Nκ) per mode; it is not automatically faster than a Cartesian
FFT. Its finite forward and inverse quadratures are not exact matrix inverses.

Axisymmetric geometry does not make every Cartesian vector component radial.
The audit refracts an axial x-polarized plane wave at a spherical cap of radius
8 λ0, sampled up to radius 2 λ0, from n=1 to n=1.5. Ex is m=0, Ey vanishes to
roundoff, Ez and Hz contain m=±1, Hx contains m=±2 and Hy contains m=0,±2.
Retaining only J0 loses the longitudinal field. In a cylindrical vector basis,
the same axial linear polarization belongs to m=±1 vector sectors; the basis
rotation accounts for these different Cartesian orders.

## What should change in the operational architecture

The main PlanarASM, PlaneSpectrum and SurfaceTransport are currently Cartesian.
Selecting the mathematical polar transform alone does not change these objects.
A polar backend should preserve R Q-inverse T Q R as the composition:

1. R multiplies each modal spectrum by exp(i kz Δz), with
   kz = sqrt((n k0)² - κ²) on the chosen radiation branch.
2. For a rotationally symmetric surface height h(r), scalar Cartesian component
   synthesis in Q becomes

   \[
   (Q_m\widehat f_m)(r)=\frac{i^m}{2\pi}\int
   \widehat f_m(\kappa)J_m(\kappa r)e^{ik_z(\kappa)h(r)}\kappa\,d\kappa.
   \]

   Vector spectra must enforce k·E=0 and obtain H from k×E, retaining all
   required component orders. Analytic incident fields can still be evaluated
   directly on the surface.
3. T remains the local vector Fresnel operator. Angular mode coupling must be
   retained when the geometry or illumination breaks rotational symmetry.
4. Q-inverse solves the discretized curved synthesis problem with physical
   quadrature weights. Replacing h(r) by -h(r) is not generally its inverse:
   the height depends on the output coordinate inside a nonlocal integral.

Cache reusable Bessel matrices and use few angular modes when justified by the
field. Use r dr and κ dκ measures and the appropriate spectral flux form, not
an unweighted Euclidean coefficient norm. Benchmark speed and radial, spectral,
angular and aperture convergence before replacing notebook05's backend.

## Flux factors and the stigmatic notes

Reviewed local source:
`/Users/blancus/Downloads/NOTES/Refraction In Stigmatic Systems/refraccion_sistema_estigmatico_corregido.pdf`.

The general time-averaged flux is

\[
\langle\mathbf S\rangle=\tfrac12\Re(\mathbf E\times\mathbf H^*).
\]

The screenshot's n |E|² u/(2 Z0) applies to a single transverse plane wave
or a local geometrical-optics wave in a lossless nonmagnetic medium. It should
not replace E×H* for a general interfering angular spectrum.

The notes distinguish two operations:

- Section 3.1, equations 28–31: Fresnel reflection and transmission satisfy
  Rq + Tq = 1, with Tq = ni cos(θi) |tq|² / (n0 cos(θ0)).
- Sections 4 and 6, especially 6.1, equations 44–53: the factor
  A = |z0| ℓi / (|zi| ℓ0) transports amplitudes between reference spheres by
  their ray-tube areas. Section 12 additionally derives the cosine factors
  for the particular plane-to-sphere perspective mapping.

For corresponding ray-tube patches, carrying a power fraction τ from a to b,

\[
\left|\frac{E_b}{E_a}\right|
=\sqrt{\frac{\tau n_a|\hat u_a\cdot\hat\nu_a|\,dS_a}
{n_b|\hat u_b\cdot\hat\nu_b|\,dS_b}}.
\]

Here ν is a surface unit normal. The Jacobian is essential when amplitudes are
remapped between ray coordinates. It is not an additional universal factor
for evaluating the same ASM wave field on another surface: that propagation
already includes wave spreading. At one interface dS is shared and Fresnel
already includes the local flux split. Applying both factors indiscriminately
can count a geometric or transmission effect twice.

One can algebraically force a positive local transmitted flux using
c(x)² = [Sin(x)+Sr(x)]·ν / [Sapprox(x)·ν], with signed reflected flux and only
where this ratio is well defined and nonnegative. This does not determine
phase, polarization, reflected amplitudes or the four Maxwell jumps.
Multiplying both E and H by a nonconstant c also introduces terms such as
grad(c)×E into the curl equations and grad(c)·E into divergence. Reprojecting
onto Maxwell angular spectra then changes the prescribed local flux again.

## Experiment: a constant power correction

For plano-convex lenses D/λ0=2,20,200, the audit uses a 64² Cartesian grid of
width three aperture radii. The 200 case uses the notebook's weak-lens
geometry; these are not geometrically similar lenses across all three sizes.
Each exit field is reconstructed by Q(Q-inverse(T(...))). The experiment
multiplies both its E and H by a single c=sqrt(Ptarget/Preconstructed), where
Ptarget is the transmitted local Fresnel power integrated on the same chart.
Stages are tested separately; the correction is not fed into the next stage.

Boundary residuals are maxima over r<0.95 aperture radius; they compare the
reconstructed transmitted field against the stored local incident+reflected
traces. E, H, D, B jumps use the fixed reference scales Eref=1 V/m, Href=1/Z0,
Dref=epsilon0, Bref=mu0/Z0, respectively. `boundary_max` is the maximum of
these four dimensionless maxima. Power uses the entire sampled chart and
E×H*, not the local plane-wave formula.

| D/λ0 | Variant | Relative power mismatch | Largest scaled boundary jump |
|---:|---|---:|---:|
| 2 | Operational reconstruction | 7.6713e-2 | 3.8011e-1 |
| 2 | Global power rescaling | 2.2e-16 | 4.1803e-1 |
| 20 | Operational reconstruction | 7.7647e-3 | 1.9988e-1 |
| 20 | Global power rescaling | 0 | 1.9968e-1 |
| 200 | Operational reconstruction | 3.1575e-7 | 1.0515e-2 |
| 200 | Global power rescaling | 2.2e-16 | 1.0515e-2 |

The prescribed local Fresnel traces satisfy their jumps to roundoff, as expected
algebraically. They use the locally transverse projected incident field and
ray-derived magnetic field; they are not an independent full-wave boundary
solution. Reconstruction into a finite forward spectrum need not preserve
those prescribed traces. This is where the audit sees discrepancies.

The experiment rejects total-power agreement as a sufficient accuracy test.
It does not rule out a correctly derived Jacobian in a new ray-coordinate
operator, nor does this fixed-grid audit establish convergence. No production
normalizer has been added. The immediate architectural direction is vector
polar transport with consistent measures, alongside a symmetry-aware boundary
integral reference and independent boundary, PDE, flux and refinement checks.
