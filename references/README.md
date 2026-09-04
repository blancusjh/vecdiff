# Independent reference theories

These checkout-only modules are outside the installed `vecdiff` package. The
core never imports them; architecture tests enforce that dependency direction.

| Module | Purpose and independence |
| --- | --- |
| `mie.py` | Adapter to pinned independent Lorenz–Mie E/H fields; no native boundary or propagation routine is called |
| `richards_wolf.py` | Ideal aplanatic objective reference; uses ElectricSpectrum only as a shared Maxwell state representation |
| `stratton_chu.py` | Standalone E-only radiation integral with an independent Hessian contraction; can compare prescribed-current propagation |

Agreement between two prescribed-current integrals validates propagation from
those currents, not the currents' correctness for a dielectric boundary.
Richards–Wolf focusing is not a general dielectric lens model.

There are no historical wrappers or bundled legacy modules. Old models and
their outputs remain recoverable from Git history, outside the maintained API.

Primary sources: [Richards and Wolf (1959)](https://doi.org/10.1098/rspa.1959.0200),
[Mie field implementation](https://miepython.readthedocs.io/en/latest/15_2D_fields.html),
and [multilayer derivations and conventions](https://arxiv.org/abs/1603.02720).
