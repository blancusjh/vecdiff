# vecdiff

This is the independently installable `vecdiff` distribution: vector optics
by eikonal–angular-spectrum transport. It is pure Python; its runtime
dependencies are NumPy and SciPy, and it has no runtime dependency on the
other packages of the repository.

```sh
python -m pip install '.[test,notebooks,nufft]'
python -m pytest -q
```

Run these commands from this package directory. Matplotlib, the notebook
tools and FINUFFT are optional extras (`notebooks`, `nufft`). The independent
full-wave solvers that validate it are a separate package in `../validation`;
see [validation](../docs/VALIDATION.md#full-wave-references).

The source follows the library's ontology: `geometry/`, `media/`,
`surfaces/` (space and matter), `system.py` (the optical system), `fields/`
(light), `transport/` and `propagation/` (the method), `observables/`
(measurement), `references/` (exact solutions to check against),
`alternatives/` (other methods), with `fourier/`, `sampling/`, `backend.py`,
`visualization/`, `io/` and `catalog.py` beside them. Start with the
[README](../README.md) quick start, then the [method](../docs/PROPAGATION.md);
the [architecture](../docs/ARCHITECTURE.md) gives responsibilities and units,
and [validation](../docs/VALIDATION.md) the supported physical claims.

Runnable scripts are in [`examples/`](examples/) and executed notebooks in
[`notebooks/`](notebooks/); see the [notebook guide](../docs/NOTEBOOKS.md).
