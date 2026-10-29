# General vector propagation

This is the independently installable `vecdiff_general` distribution. It has no runtime dependency on the other vecdiff package.

```sh
python -m pip install '.[test,notebooks,nufft,reference-mesh]'
python -m pytest -q
```

Run these commands from this package directory. Building requires C and Fortran
compilers; Meson-Python builds the native reference kernels. Matplotlib,
notebooks, FINUFFT and Gmsh mesh generation are optional extras; the base runtime
dependencies are NumPy and SciPy. See [validation](../docs/VALIDATION.md#full-wave-references)
for the native reference solvers.

The source follows `geometry/`, `surfaces/`, `sampling/`, `media/`, `fields/`, `interfaces/`, `fourier/`, `propagation/`, `observables/`, and `visualization/`. Consult the [architecture](../docs/ARCHITECTURE.md) for responsibilities and units, [propagation](../docs/PROPAGATION.md) for the operational method, and [validation](../docs/VALIDATION.md) for supported physical claims.

The physical notebooks are in [`notebooks/`](notebooks/); see the [notebook guide](../docs/NOTEBOOKS.md).
