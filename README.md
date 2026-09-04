# vecdiff

`vecdiff` treats refraction and reflection as linear spectral transformations.
Every populated incident plane wave is transformed independently at an
oriented dielectric interface using its own local s/p basis and Fresnel
coefficients. The resulting electric and magnetic boundary data are propagated
as Maxwell fields.

The package is organized by physical role:

| Package | Meaning |
| --- | --- |
| `fields/` | Electric fields and spectra |
| `media/` | Constitutive media |
| `geometry/` | Domains, frames, and placements |
| `surfaces/` | Parametric physical surfaces |
| `interfaces/` | Oriented dielectric boundaries and `fresnel.py` |
| `propagation/` | Homogeneous and interface-induced field propagation |
| `sampling/` | Spatial and surface quadrature nodes |
| `fourier/` | Cartesian, polar, Hankel, and NUFFT algorithms |
| `observables/` | Maxwell residuals, flux, and field observables |

Only the public API wrapper `vecdiff/__init__.py` lives at the package root.
Historical methods and literature-specific implementations live in the
repository-level `references/` package. The core never imports them.

## Physical scope

The infinite-plane spectral interface map is the exact Maxwell solution for
lossless, isotropic, nonmagnetic media. It supports oblique incidence,
Brewster incidence, and the complex transmitted wave under total internal
reflection.

For a curved interface, `interface_transform` applies the local tangent-plane
Fresnel law to **each incident k**, accumulates the boundary phasors, and forms
equivalent electric and magnetic surface currents. Radiation from those
currents satisfies Maxwell's equations in each homogeneous region. The local
boundary trace is a physical-optics approximation unless the surface is a
plane; the benchmark evaluates the boundary conditions again after spectral
reconstruction.

A closed resonant body such as a dielectric sphere requires repeated internal
reflection/refraction or a full boundary solve. Applying the open-interface
transform once to its illuminated hemisphere omits resonant feedback, creeping
waves, and the exit encounter. `references/mie.py` provides an independent
Lorenz-Mie comparison that makes this domain limit measurable.

## Minimal example

```python
import numpy as np
from vecdiff import Medium, Plane, DielectricInterface, plane_wave
from vecdiff import interface_transform

n1, n2 = Medium(1.0), Medium(1.5)
theta = np.deg2rad(35)
incident = plane_wave(
    direction=(np.sin(theta), 0, np.cos(theta)),
    polarization=(0, 1, 0),
    wavelength=0.532,
    medium=n1,
)

result = interface_transform(
    incident,
    DielectricInterface(Plane(), n1, n2),
)
E_transmitted, H_transmitted = result.transmitted.evaluate([[0, 0, 1]])
```

`H` is represented as `Z0 * H_SI`; lengths and vacuum wavelength may use any
common unit. The phasor convention is `exp(-i omega t)`.

## Installation and validation

```bash
python -m pip install -e '.[nufft,validation]'
python -m pytest -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m benchmarks.validate_physics
```

The benchmark records raw Maxwell boundary residuals, quadrature convergence,
and independent Mie comparisons in `benchmarks/results/physics.json`. See
[`docs/validation.md`](docs/validation.md) for what each check establishes.

Version 0.3 intentionally replaces the mixed 0.2 API. The former package,
tests, examples, and notebooks remain in `references/legacy/` for reproducible
comparison and do not participate in the new public API.
