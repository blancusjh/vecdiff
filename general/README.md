# General vector propagation

This is the independently installable `vecdiff` distribution. It has no runtime dependency on the other vecdiff package.

```sh
python -m pip install '.[test,notebooks,nufft]'
python -m pytest -q
```

Run these commands from this package directory. The package is pure Python;
Matplotlib, notebooks and FINUFFT are optional extras, and the base runtime
dependencies are NumPy and SciPy. The independent full-wave solvers that
validate it are a separate package in `../validation`; see
[validation](../docs/VALIDATION.md#full-wave-references).

The source follows `geometry/`, `surfaces/`, `media/`, `system.py`, `fields/`, `transport/`, `propagation/`, `observables/`, `references/`, `alternatives/`, with `fourier/`, `sampling/` and `visualization/` beside them. Consult the [architecture](../docs/ARCHITECTURE.md) for responsibilities and units, [propagation](../docs/PROPAGATION.md) for the operational method, and [validation](../docs/VALIDATION.md) for supported physical claims.

The physical notebooks are in [`notebooks/`](notebooks/); see the [notebook guide](../docs/NOTEBOOKS.md).
