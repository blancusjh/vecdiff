# Dedicated stigmatic vector optics

This is the independently installable `vecdiff_stigmatic` distribution. It has no runtime dependency on the other Logos Optics package.

```sh
python -m pip install '.[test,notebooks,nufft]'
python -m pytest -q
```

Run these commands from this package directory. Matplotlib, notebooks and FINUFFT are optional extras; the base dependencies are NumPy and SciPy.

The source follows `geometry/`, `surfaces/`, `sampling/`, `media/`, `fields/`, `interfaces/`, `fourier/`, `propagation/`, `observables/`, and `visualization/`. Consult the [architecture](../docs/ARCHITECTURE.md) for responsibilities and units, and [validation](../docs/VALIDATION.md) for supported physical claims.
