# General vector propagation

This is the independently installable `vecdiff_general` distribution. It has no runtime dependency on the other Logos Optics package.

```sh
python -m pip install '.[test,notebooks,nufft,reference-mesh]'
python -m pytest -q
```

Run these commands from this package directory. Building requires C and Fortran
compilers; Meson-Python builds the native reference kernels. Matplotlib,
notebooks, FINUFFT and Gmsh mesh generation are optional extras; the base runtime
dependencies are NumPy and SciPy. See [native references](../docs/REFERENCE_PROPAGATION.md)
for editable installation and convergence requirements.

The source follows `geometry/`, `surfaces/`, `sampling/`, `media/`, `fields/`, `interfaces/`, `fourier/`, `propagation/`, `observables/`, and `visualization/`. Consult the [architecture](../docs/ARCHITECTURE.md) for responsibilities and units, and [validation](../docs/VALIDATION.md) for supported physical claims.

## Physical experiments

- [Planar dielectric optics](docs/notebooks/01_planar_dielectric_optics.ipynb)
- [Diffraction by apertures](docs/notebooks/04_diffraction_by_apertures.ipynb)
- [Refraction and focusing by a lens](docs/notebooks/05_refraction_and_focusing_by_a_lens.ipynb)
- [Polarization of vector beams](docs/notebooks/06_polarization_of_vector_beams.ipynb)
- [Our method versus BEM, FEM and Mie](docs/notebooks/07_full_wave_reference_validation.ipynb)
- [Fresnel compensation](docs/notebooks/08_compensating_fresnel_transmission.ipynb)
- [Vector imaging and lithography](docs/notebooks/10_single_lens_lithography.ipynb)
- [Curved interfaces: local orders, lenses, spheres and internal returns](docs/notebooks/11_curved_interfaces.ipynb)

See the [notebook guide](../docs/NOTEBOOKS.md) for the models, figures and execution instructions.

## Operational interface propagation

The primary general path now composes planar ASM R, curved transport Q, local Fresnel T and a solved Q inverse. Green methods are isolated in `vecdiff_general.comparisons`. See [the operational method and measured checks](../docs/OPERATIONAL_PROPAGATION.md). Notebook 05 is executed with this path. Earlier current-based validation below/above applies only to the separate comparison model.

The opt-in [curvature expansion](../docs/CURVATURE_ORDERS.md) generates corrections
through order three and includes complete-curl boundary diagnostics, physical
encounter composition, an independent periodic Maxwell reference, and a lens-exit
test through the production Q inverse. Its Fortran kernel accelerates Taylor
polynomial products. The measured cavity improvement does not establish that
the full-lens focal-power discrepancy is resolved.
