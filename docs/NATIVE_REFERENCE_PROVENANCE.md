# Native reference implementation provenance

The reference repositories were cloned under `~/External/Repositories` on
2026-09-20. They are study material, not runtime or build dependencies. The
project's numerical core is Fortran; Python owns validated inputs, geometry,
sparse/dense solves through SciPy, units and diagnostics. No upstream solver
source was copied or mechanically translated in this implementation.

| Upstream | Inspected commit | Role and relevant source |
| --- | --- | --- |
| [SCUFF-EM](https://github.com/HomerReid/scuff-em) | `9c6d0cb7695463af803dee8d04cdae939740cdcc` | `libs/libscuff/AssembleBEMMatrix2018.cc`, `PanelPanelInteractions.cc`, `TaylorDuffy.cc`, `GetFields.cc`: electric/magnetic current blocks, RWG integration and field representation |
| [DOLFINx](https://github.com/FEniCS/dolfinx) | `66c9677b695f8f9447939f8a710e35fe72970b4e` | `python/demo/demo_pml.py`: scattered-field weak form and transformed permittivity/permeability tensors |
| [Basix](https://github.com/FEniCS/basix) | `8c5a57f2226fdaf7a62c1c1343e8ff757f46f555` | `cpp/basix/e-nedelec.cpp`, `test/test_nedelec.py`: first-family edge elements and orientation conventions |

SCUFF-EM source carries GPLv2-or-later notices. DOLFINx carries LGPL notices;
Basix's inspected Nédélec source carries an MIT notice. Reading a formulation
is distinct from copying an implementation. A future translation or source
reuse needs a specific license review and retained notices; this file does
not relicense any upstream source.

## Deliberate implementation differences

- BEM uses vacuum-wavelength scaling, `j=Z0 J`, `m=M`, and our documented
  `exp(-i omega t)` matrix signs. Independent E **and** H comparisons against
  Mie check the sign convention. Upstream block signs cannot be pasted without
  reconciling unknown and equation conventions.
- Self-panel Duffy quadrature is implemented directly. Adjacent-panel product
  quadrature is refinable but is less sophisticated than SCUFF-EM's singular
  integration. We do not claim its accuracy or acceleration.
- Assembly and field evaluation run in parallel (OpenMP, GCC's libgomp linked
  explicitly because the C wrapper may be built by Apple clang). Assembly
  accumulates with atomic updates, so results depend on the thread count only
  through summation order. Panel pairs farther apart than twice their summed
  sizes use a 3×3 product rule (`far_quadrature`, `far_ratio`), and field
  points farther than three panel sizes a 4×4 rule (`field_far_quadrature`,
  `field_far_ratio`). On a 6λ lens at λ/3 (4806 unknowns) this changes the
  fields by 7.6e-5 relative L2, far below the λ/3-to-λ/4 discretization change
  (about 1%), and assembly went from about 200 s serial to 15 s; the dense LU
  is now the main cost. Pass `far_ratio=inf, field_far_ratio=inf` for the
  uniform rules.
- DOLFINx's cited PML example is two-dimensional. Our solver is a three-dimensional
  tetrahedral vector implementation of the same coordinate-transform tensor
  construction. There is no scalar substitute for the vector problem.
- Only the lowest-order analytical Nédélec bases are needed here. Basix's
  general polynomial tabulation and code are not embedded.
- Gmsh is an optional mesher; it does not assemble or solve Maxwell equations.
  Its binary/package is installed separately, never vendored into this repo.
- NumPy F2PY supplies the C/Python binding layer. Meson compiles our `.f90`
  files, and SciPy supplies established LAPACK/SuperLU linear solves.

## Primary documentation

- [SCUFF-EM implementation notes](https://homerreid.github.io/scuff-em-documentation/forDevelopers/Implementation/)
- [DOLFINx electromagnetic PML demo](https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_pml.html)
- [NumPy F2PY with Meson](https://numpy.org/doc/stable/f2py/buildtools/meson.html)

Validation is against physical invariants and the independent local Mie
implementation. No claim of numerical agreement with a running SCUFF-EM or
DOLFINx executable is made; those packages were not built for these tests.
