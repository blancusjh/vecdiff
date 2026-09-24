# Full-wave propagation references

The primary API remains the operational `R Q^-1 T Q R` composition.
`propagate_interfaces()` and its `evaluate()` do not import a reference solver
or fall back to Green integration. The independent comparison API lives in
`vecdiff_general.propagation.references`.

## Implemented methods

| Method | Formulation | Core / solve | Numerical controls |
| --- | --- | --- | --- |
| `DielectricBEM` | PMCHWT dielectric boundary integral equations, RWG surface currents on triangles | Own Fortran / SciPy dense LAPACK | Surface mesh, regular/adjacent/self quadrature, field quadrature |
| `DielectricFEM` | Vector curl-curl scattered-field equation, first-family Nédélec tetrahedral edge elements, Cartesian PML | Own Fortran / SciPy sparse SuperLU | Volume mesh, physical padding, PML width and strength |
| `MieSphere` | Lorenz–Mie spherical multipoles | Python / SciPy special functions | Multipole order, observation and flux sampling |

**BEM** means boundary element method. **PMCHWT** is the
Poggio–Miller–Chang–Harrington–Wu–Tsai dielectric surface formulation.
**RWG** means Rao–Wilton–Glisson, the pair-of-triangles current basis.
**FEM** means finite element method. **PML** means perfectly matched layer.
Mie is an analytical spherical reference, not another general geometry solver.
The older method of fundamental solutions (MFS) is no longer used in notebook 07.

Both new methods solve the full vector Maxwell transmission problem for a
single homogeneous, lossless, nonmagnetic inclusion in a homogeneous exterior.
They include reflection and multiple scattering within the body, and evaluate
both electric and magnetic fields. They accept analytic `PlaneWaveExpansion`
illumination, including oblique propagation and coherent superpositions.
Absorbing materials, magnetic contrast and multiple dielectric regions are
not implemented. No SCUFF-EM, DOLFINx, Basix or FEniCS runtime is imported.

These are **first implementations requiring convergence studies**, not
accuracy-certified ground truth. Dense BEM is intentionally limited in size;
FEM uses first-order elements and a sparse direct solve. Neither implements
fast multipole acceleration or high-order discretization. A macroscopic dense
reference run is not silently replaced by another propagation model.

## Build and use

A C compiler, modern Fortran compiler (tested with gfortran), Python and Ninja
are needed to build the NumPy F2PY extension. Meson-Python builds and installs
it as part of the Python package; there is no compilation at import time in a
normal wheel installation. Meson editable installs rebuild changed native
sources. Developer installation from the repository root:

```sh
.venv/bin/python -m pip install meson-python ninja 'numpy>=2'
.venv/bin/python -m pip install --no-build-isolation -e './general[reference-mesh]'
```

`reference-mesh` installs Gmsh for optional geometry/mesh generation only.
Supplied `SurfaceMesh` / `TetrahedralMesh` arrays require no Gmsh import. The
sphere surface mesher also uses only NumPy/SciPy. All coordinates and mesh
sizes exposed by Python are in metres; native kernels use vacuum wavelengths.
Output E is V/m and H is A/m with the `exp(-i omega t)` convention.

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
mesh = SurfaceMesh.sphere(body, subdivisions=2)
bem = DielectricBEM(source, mesh, glass)
volume_mesh = tetrahedral_mesh(
    body,
    edge_size=0.2 * wavelength,
    physical_padding=0.4 * wavelength,
    pml_width=0.4 * wavelength,
)
fem = DielectricFEM(source, volume_mesh, glass)
mie = MieSphere(source, body, glass)
probes = optics.PointGrid([[0, 0, 0.6 * wavelength]])
fields = [solution.evaluate(probes) for solution in (bem, fem, mie)]
```

Automatic Gmsh geometry supports spheres and coaxial convex spherical/planar
lenses, including the cylindrical rim. BEM accepts other closed oriented
triangular meshes. FEM accepts supplied conforming tetrahedra with inclusion
labels and a Cartesian outer PML box. Generated mesh surfaces approximate the
curved geometry with planar facets; mesh refinement includes geometric error.

## BEM equations and quadrature

Let lengths be divided by the vacuum wavelength, `k0 = 2π`, and `k = k0*n`.
The unknowns are `j = Z0 J` and `m = M`, expressed in RWG bases `f_a`.
For `g = exp(i k r)/(4πr)`, the weak operators are

\[
 L_{ab}^{(n)}=i k_0\iint g\left[f_a\cdot f_b-
 \frac{(\nabla_s\cdot f_a)(\nabla_s\cdot f_b)}{k^2}\right]dS_xdS_y,
 \qquad
 K_{ab}^{(n)}=\iint f_a\cdot(\nabla_xg\times f_b)dS_xdS_y.
\]

The assembled transmission system is

\[
\begin{pmatrix} L_e+L_i&-(K_e+K_i)\\
 K_e+K_i&n_e^2 L_e+n_i^2 L_i\end{pmatrix}
\binom{j}{m}=-\binom{\langle f,E_{inc}\rangle}{\langle f,Z_0H_{inc}\rangle}.
\]

The surface currents determine interior and outgoing exterior fields. This is
a dielectric boundary-integral solution related to the Stratton–Chu surface
representation. **Stratton–Chu alone is a representation given boundary data**;
PMCHWT supplies the coupled equations determining the unknown boundary data.
It is therefore a concrete way to implement the general surface-integral
reference requested, rather than prescribing our Fresnel approximation as
input to an integral.

Self-panel singularities use target-centred Duffy integration. Shared-edge and
shared-vertex panels use separately refinable product quadrature; this is not
SCUFF-EM's complete Taylor–Duffy machinery. No softening length is introduced.
Exact jump terms and a principal-value self-panel calculation give one-sided
fields at face interiors. Edge/vertex traces are rejected because the normal
is not unique. Off-surface near-boundary evaluation still requires increasing
`field_quadrature`. Inspect quadrature convergence separately from mesh error.

## FEM equations and PML

The scattered electric field solves

\[
\int (\nabla\times v)\cdot\mu_{PML}^{-1}(\nabla\times E_s)
-k_0^2v\cdot\epsilon_{PML}E_s\,dV
=k_0^2\int_{body}(n_i^2-n_e^2)v\cdot E_{inc}\,dV.
\]

Tetrahedral edge bases are `λ_a ∇λ_b − λ_b ∇λ_a`, with a consistent global
edge orientation. Four-point tetrahedral quadrature integrates degree-two
polynomials exactly; spatially varying PML coefficients require refinement.
For each coordinate, `s = 1 + i α depth²`, where depth rises from zero to one
across the PML. The diagonal transformed tensor is
`diag(s_y s_z / s_x, s_x s_z / s_y, s_x s_y / s_z)`; epsilon includes `n²`.
The outer scattered tangential E is zero. Observations in the PML or outside
the physical box are rejected. Increase padding and PML width, and vary
strength independently of mesh refinement.

H is reconstructed from `curl E / (i k0 Z0)` plus the analytical incident
field. Lowest-order scattered H is constant within each cell. Consequently,
strong pointwise curl-H and vector Helmholtz residuals are not a consistent
accuracy test for this discretization. Use the weak residual together with
field convergence, interface jumps and energy flux; do not compare a near-zero
linear-system residual with another method's continuum PDE residual.

## Validation and notebook interpretation

`general/tests/test_native_references.py` checks Mie field convergence,
interface traces, equal-index media, wavelength scaling, quadrature, Maxwell
residuals away from interfaces, flux and resource limits. Notebook 07 shows
geometry-specific tables, refinement records and field plots with our method
first. It does not label coarse numerical solutions as ground truth.

BEM and FEM boundary checks use the **actual mesh faces**, not displaced points
on an ideal analytical sphere. Their reported residuals include pointwise
normal and tangential jumps that can converge much more slowly than selected
far-field values. A weak solve residual near machine precision does not imply
small boundary or observation errors.

The primary method has absorbing pupils and keeps a forward branch. A closed
body under plane-wave illumination also has bypass illumination and rim
scattering. The notebook explicitly compares `incident + ours(lens) −
ours(equal-index pupil)` as a forward approximation, and makes the unmodified
solution available as `ForwardComparison.raw`. This correction is confined to
comparison helpers. For spheres the operational chart uses a declared central
aperture, avoiding equatorial graph singularities and unsupported TIR paths.
Differences from Mie therefore include modeling differences; they are not pure
numerical errors in the operational operators.

## Upstream implementation provenance

See [native reference provenance](NATIVE_REFERENCE_PROVENANCE.md) for exact
upstream commits and inspected files. Implementations are our own Fortran
routines derived from the formulations; no upstream solver code was copied
verbatim or linked. Any future code translation must preserve the applicable
upstream license and notices.
