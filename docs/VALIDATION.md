# Validation

The checks separate numerical convergence, correctness within a declared
physical model, and agreement with independent Maxwell solutions. A local
identity is not a substitute for testing the propagated field.

## Conventions and normalization

Positions and vacuum wavelength are metres; E is V/m and H is A/m, with time
dependence `exp(-i omega t)`. Public magnetic data is H, not Z0 H. Media are
lossless, isotropic, nondispersive and nonmagnetic. Surface normals and material
sides are explicit.

Boundary diagnostics report tangential E/H and normal D/B separately.
Normalization uses a declared incident amplitude; a dark interference point never
changes the scale. Power is the signed integral of `Re(E × H*) / 2` over an
oriented surface. Electric amplitude squared is not labelled SI intensity.

## What the tests cover

| Area | Independent or physical reference |
|---|---|
| Fourier transforms and grids | Centered FFT2/IFFT2 against direct sums, Parseval, odd/even sizes; Hankel and nonuniform sums against analytic Gaussian pairs and FINUFFT; typed Cartesian/polar grids |
| Homogeneous propagation and layers | Known FFT-bin modes, tilted planes, longitudinal completion, evanescence; Airy formula for coherent slabs and frustrated TIR |
| Plane interfaces | All four Maxwell jump conditions, signed normal flux, Brewster cancellation, TIR phase and decay |
| Operational path R Q⁻¹ T Q R | Plane-interface limits, direct modal sums for Q, inverse residuals, grid and window convergence |
| Edges and stops | Index-matched element invisible to 1e-12; inside and outside light recombine to the incident wave; stops cut outside light |
| GOTS surfaces | Constant optical path and independent Snell refraction for both index orderings |
| Ideal imaging | Passed normal power, Maxwell transversality, low-NA scalar agreement, inverse pupil compensation |
| Full-wave references | Mie convergence, BEM/FEM interface traces, equal-index media, flux, resource limits, Müller/PMCHWT agreement and conditioning |
| Stigmatic library | Constant optical path, reference-sphere flux, Debye direct/Hankel/NUFFT agreement, Fresnel pupil Parseval power |
| Cross-library | Maxwell residuals of the dedicated Debye focus; general Green radiation against the saved legacy stigmatic fixture |

## Full-wave references

The operational `propagate_interfaces()` never imports a reference solver or
falls back to Green integration. The comparison API is
`vecdiff_general.propagation.references`:

| Method | Formulation | Core / solve | Numerical controls |
| --- | --- | --- | --- |
| `DielectricBEM` | PMCHWT (default) or Müller (`formulation="muller"`) boundary integral equations, RWG currents on triangles | Own Fortran / dense LAPACK | Surface mesh; regular, adjacent, self and field quadrature |
| `DielectricFEM` | Vector curl-curl scattered field, first-family Nédélec tetrahedra, Cartesian PML | Own Fortran / sparse SuperLU | Volume mesh, padding, PML width and strength |
| `MieSphere` | Lorenz–Mie multipoles | Python / SciPy | Multipole order |

PMCHWT is the Poggio–Miller–Chang–Harrington–Wu–Tsai formulation and RWG the
Rao–Wilton–Glisson basis. BEM and FEM solve the full vector transmission problem
for one homogeneous lossless inclusion in a homogeneous exterior, including
internal reflections, for analytic `PlaneWaveExpansion` illumination. They are
**first implementations requiring convergence studies**, not certified ground
truth: dense BEM is limited in size and FEM uses first-order elements.

```python
import numpy as np
import vecdiff_general as optics
from vecdiff_general.propagation.references import (
    DielectricBEM,
    DielectricFEM,
    MieSphere,
    SurfaceMesh,
    tetrahedral_mesh,
)

wavelength = 532e-9
air = optics.HomogeneousRegion("air", optics.Dielectric(1.0))
glass = optics.HomogeneousRegion("glass", optics.Dielectric(1.5))
source = optics.PlaneWaveExpansion(
    [[0, 0, 2 * np.pi / wavelength]], [[1, 0, 0]], [0, 0, 0], wavelength, air
)
body = optics.SphereVolume([0, 0, 0], 0.3 * wavelength)
bem = DielectricBEM(source, SurfaceMesh.sphere(body, subdivisions=2), glass)
fem = DielectricFEM(
    source,
    tetrahedral_mesh(
        body,
        edge_size=0.2 * wavelength,
        physical_padding=0.4 * wavelength,
        pml_width=0.4 * wavelength,
    ),
    glass,
)
mie = MieSphere(source, body, glass)
probes = optics.PointGrid([[0, 0, 0.6 * wavelength]])
fields = [solution.evaluate(probes) for solution in (bem, fem, mie)]
```

Gmsh (`reference-mesh` extra) generates spheres and coaxial spherical/planar
lenses with their cylindrical rim; BEM accepts any closed oriented triangle mesh.
Solved BEM currents are cached when `cache=` or `VECDIFF_BEM_CACHE` names a
directory (the notebooks use `.cache/bem`).

### BEM

With lengths in vacuum wavelengths, `k0 = 2π`, `k = k0 n`, unknowns `j = Z0 J`
and `m = M` in RWG bases `f_a`, and `g = exp(i k r)/(4πr)`:

\[
 L_{ab}^{(n)}=i k_0\iint g\left[f_a\cdot f_b-
 \frac{(\nabla_s\cdot f_a)(\nabla_s\cdot f_b)}{k^2}\right]dS_xdS_y,
 \qquad
 K_{ab}^{(n)}=\iint f_a\cdot(\nabla_xg\times f_b)dS_xdS_y,
\]

\[
\begin{pmatrix} L_e+L_i&-(K_e+K_i)\\
 K_e+K_i&n_e^2 L_e+n_i^2 L_i\end{pmatrix}
\binom{j}{m}=-\binom{\langle f,E_{inc}\rangle}{\langle f,Z_0H_{inc}\rangle}.
\]

Stratton–Chu alone is a representation given boundary data; PMCHWT supplies the
equations for that data. Self panels use target-centred Duffy integration,
adjacent panels refinable product quadrature, distant pairs a 3×3 rule. Assembly
and field evaluation run in parallel with OpenMP.

#### Müller formulation

`DielectricBEM(..., formulation="muller")` solves for the same `j`, `m` in the
same RWG basis, so `evaluate`, caching and `max_unknowns` are shared. The
equations come from the second-kind Müller formulation instead. It combines the
rotated traces of both sides, weighted by `n²` for E and 1 for H, so that the
hypersingular parts of `n² T` cancel:

\[
\begin{aligned}
 j-n\times(K_e-K_i)j-n\times(n_e^2T_e-n_i^2T_i)m&=n\times Z_0H_{inc},\\
 m-\tfrac1c\,n\times(n_e^2K_e-n_i^2K_i)m+\tfrac1c\,n\times(n_e^2T_e-n_i^2T_i)j
 &=-\tfrac{n_e^2}{c}\,n\times E_{inc},
\end{aligned}
\qquad c=\tfrac{n_e^2+n_i^2}{2},
\]

with `T = i k0 (S + ∇∇·S/k²)` and `K = ∇×S` (principal value) as in PMCHWT.
The system is identity plus compact on smooth boundaries and has no interior
resonances. The identity is only well conditioned with a stable pairing, so
the equations are tested with rotated Buffa–Christiansen functions `n × b` on
the barycentric refinement (Buffa and Christiansen, *Math. Comp.* 76 (2007)
1743–1769; coefficients as in Andriulli et al., *IEEE TAP* 56 (2008)
2398–2412). This is the mixed RWG/BC Müller scheme of Yan, Jin and Nie,
*IEEE TAP* 59 (2011) 1299–1310. The integral terms become `⟨b, K f⟩` and,
after integration by parts with the div-conforming `b`, the PMCHWT `L`
kernels. Both kernels are formed at identical points so the `∇∇` parts cancel
exactly. Test sub-triangles use the same rules as PMCHWT: Duffy inside the
source panel, the product rule when touching it, 3×3 when distant.

The measurements below use a sphere of radius 0.3 λ with `n = 1.5`
(`ka = 1.88`) on icosphere levels 0–3, lit by an `x`-polarized plane wave, on
10 OpenMP threads. GMRES is unrestarted with relative residual 1e-6. Field
errors are relative to Mie over two interior and two exterior probes.

| Unknowns | Gram cond `⟨n×b, f⟩` | Cond PMCHWT | Cond Müller | GMRES PMCHWT | GMRES Müller | E error vs Mie, PMCHWT | E error vs Mie, Müller | E difference Müller − PMCHWT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 60 | 2.15 | 6.1 | 5.0 | 35 | 12 | 0.24 | 0.22 | 0.037 |
| 240 | 2.86 | 11 | 6.0 | 69 | 13 | 0.067 | 0.067 | 0.0078 |
| 960 | 3.13 | 44 | 6.7 | 122 | 15 | 0.016 | 0.017 | 0.0030 |
| 3840 | 3.34 | 186 | 7.0 | 212 | 16 | 0.0035 | 0.0044 | 0.0013 |

Testing with rotated RWG functions would give the Gram `⟨n×f, f⟩`, which is
numerically singular (cond ≈ 10¹⁷ at levels 1–3), so it is not used.

At `ka = 5` (radius 0.796 λ) with 240, 960 and 3840 unknowns:

- Müller: cond 20, 17, 18; GMRES iterations 28, 27, 28.
- PMCHWT: cond 24, 11, 25; GMRES iterations 82, 138, 244.
- E errors vs Mie: 0.27, 0.074, 0.019 (Müller) and 0.21, 0.059, 0.014 (PMCHWT).

On the gmsh plano-convex and biconvex lenses (430–890 unknowns) the two
formulations agree to 2–3 %. Müller's condition number there (15–28) stays
below PMCHWT's (28–39).

At equal mesh Müller is slightly less accurate than PMCHWT, not more. Its
value is an independently derived set of equations with a bounded spectrum.
Assembly loops over 6× as many (sub-triangle, panel) pairs and cannot use
PMCHWT's symmetry, so it takes about 12× the PMCHWT time. Measured times are
1.6 s at 960 unknowns, 19–26 s at 3840, and 33 s plus a 1.5 s solve at 4980
(a gmsh `ka = 5` sphere near the default limit, 0.4 GB dense, E error vs Mie
0.014).

### FEM

The scattered field solves

\[
\int (\nabla\times v)\cdot\mu_{PML}^{-1}(\nabla\times E_s)
-k_0^2v\cdot\epsilon_{PML}E_s\,dV
=k_0^2\int_{body}(n_i^2-n_e^2)v\cdot E_{inc}\,dV,
\]

with edge bases `λ_a ∇λ_b − λ_b ∇λ_a`, stretching `s = 1 + i α depth²` and
tensor `diag(s_y s_z / s_x, s_x s_z / s_y, s_x s_y / s_z)`. Lowest-order
scattered H is constant per cell, so pointwise curl-H residuals are not a
convergence test for FEM; use the weak residual with field refinement.

### Interpreting the comparisons

Boundary checks use the actual mesh faces. A weak-solve residual near machine
precision does not imply small boundary or observation errors. Our method keeps
a forward branch with no internal returns. Eligible finite lenses use an
approximate two-medium rim step; other lenses keep the separated inside/outside
path. See [propagation](PROPAGATION.md#surface-edges-and-stops).
The sphere is followed through a central chart behind a declared stop. Its
differences from Mie therefore include modelling differences, not only numerical
error. Notebook 06 reports both.

### Provenance

The formulations were studied from these upstream sources; no upstream code was
copied, translated or linked.

| Upstream | Inspected commit | Relevant source |
| --- | --- | --- |
| [SCUFF-EM](https://github.com/HomerReid/scuff-em) | `9c6d0cb7695463af803dee8d04cdae939740cdcc` | BEM assembly, panel interactions, Taylor–Duffy, field evaluation |
| [DOLFINx](https://github.com/FEniCS/dolfinx) | `66c9677b695f8f9447939f8a710e35fe72970b4e` | `demo_pml.py`: weak form and PML tensors |
| [Basix](https://github.com/FEniCS/basix) | `8c5a57f2226fdaf7a62c1c1343e8ff757f46f555` | First-family Nédélec elements |

SCUFF-EM is GPLv2-or-later, DOLFINx LGPL and Basix MIT; any future code reuse
needs a license review.

## Legacy stigmatic fixture

`validation/reference_data/legacy_stigmatic.*` stores geometry, Fresnel
coefficients and Green radiation from the original stigmatic code (revision
`7a1a235`). The general Green path reproduces it to 3.4e-12 relative E, and the
dedicated Debye focus differs from the general Green result by 8.11e-5, within
its 1e-4 budget. To regenerate the fixture with a legacy interpreter:

```sh
python -B validation/export_legacy_stigmatic.py --source /path/to/stigmatic-7a1a235
```

## Reproduce

```sh
.venv/bin/python -m pytest -q
(cd general/notebooks && ../../.venv/bin/jupyter execute --inplace --timeout=-1 *.ipynb)
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

## Limits of this evidence

Global closed-surface energy budgets, reciprocity, material dispersion and fully
coupled curved scattering in our method remain open. Finite detector windows are
not normalized to the incident power.
