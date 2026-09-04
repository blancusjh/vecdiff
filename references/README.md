# Reference implementations

This package is outside `vecdiff` and never imported by the production core.

- `mie.py` wraps an independent Lorenz-Mie near-field implementation.
- `richards_wolf.py` implements ideal aplanatic focusing.
- `stratton_chu.py` and `kirchhoff.py` expose preserved radiation references.
- `stigmatic.py` exposes the historical Cartesian-oval comparison.
- `legacy/` preserves the complete version 0.2 package, tests, examples, and
  documents so old numerical claims remain reproducible during migration.

Some radiation references accept currents prescribed by the native method.
That checks propagation from those currents, not whether the currents solve a
dielectric boundary problem. Each benchmark records this dependence.
