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
| Full-wave references | Mie convergence, BEM/FEM interface traces, equal-index media, flux, resource limits |
| Stigmatic library | Constant optical path, reference-sphere flux, Debye direct/Hankel/NUFFT agreement, Fresnel pupil Parseval power |
| Cross-library | Maxwell residuals of the dedicated Debye focus; general Green radiation against the saved legacy stigmatic fixture |

## Full-wave references

The operational `propagate_interfaces()` never imports a reference solver or
falls back to Green integration. The comparison API is
`vecdiff_general.propagation.references`:

| Method | Formulation | Core / solve | Numerical controls |
| --- | --- | --- | --- |
| `DielectricBEM` | PMCHWT boundary integral equations, RWG currents on triangles | Own Fortran / dense LAPACK | Surface mesh; regular, adjacent, self and field quadrature |
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
