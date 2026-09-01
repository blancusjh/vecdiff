# vecdiff

General vectorial diffraction propagation: a composable spectral
interface-operator engine for **arbitrary smooth curved surfaces**
(`vecdiff.wave`), anchored by an essentially exact Hankel/Debye chain for the
stigmatic Cartesian oval and an independent Franz/Stratton–Chu Maxwell
reference solver that referees it. The project contains the Python package,
runnable examples, and tests for working with sampled electromagnetic fields
in Cartesian, polar, and circular representations.

Two engines, one physics, at two fidelities:

| | engine | scope | fidelity |
|---|---|---|---|
| classic chain | Hankel transfer `G -> G' -> focal plane` | the stigmatic Cartesian oval | exact within its class (pinned to 2e-9 against Franz/Stratton–Chu) |
| `vecdiff.wave` | spectral interface operators, composable | any smooth surface — sphere, conic, asphere, freeform — and *systems* of them | leading order in `1/kR`, improving with size |

The seam between them is `vecdiff.wave.stigmatic`: it feeds the general
operator the one surface the exact chain owns, so the exact solver referees
the general one (`vecdiff.wave.referee`, exercised in
`tests/test_unification.py`).

<p align="center">
  <img src="docs/assets/quiver_harmonic_readme.gif" alt="Harmonic instantaneous focal field, linear x incidence" width="420">
  <img src="docs/assets/quiver_harmonic_circular.gif" alt="Harmonic instantaneous focal field, circular incidence" width="420">
</p>

<p align="center">
  <em>Instantaneous focal field <code>Re[E e<sup>-i&omega;t</sup>]</code> over the optical cycle
  (cross-maximizing edge pupil): linear-x incidence (left) makes the field
  <em>breathe</em>; circular incidence (right) makes it <em>rotate</em>.</em>
</p>

## What Is Included

- **The general operator engine** (`vecdiff.wave`): a curved dielectric
  interface as a non-diagonal operator on the vector angular spectrum, built
  from the local tangent-plane Fresnel model and a surface return integral —
  the azimuthal Bessel kernel for surfaces of revolution, a general type-3
  NUFFT of the surface currents for freeforms. Operators compose
  (`InterfaceOperator`, `FreeSpace`, `System`), so a multi-surface system is
  an ordered product; aplanatic pupils, vector aerial imaging of masks, and
  two packaged ray-traced projection objectives (EUV and hyper-NA DUV
  immersion) ride on the same spectrum.
- **The stigmatic bridge** (`vecdiff.wave.stigmatic`): the Cartesian oval seen
  through the wave `Surface` interface, the matching point-source
  illumination, and a `referee` that compares the general operator against the
  exact chain on the surface they share.
- Field containers for Cartesian, circular, and polar transverse components,
  tagged with the reference surface their samples live on.
- Ray geometry of the stigmatic Cartesian oval: path lengths, meridional and
  incidence angles, the local Fresnel frame, and the grazing-incidence aperture
  limit, all from the transverse radius on the surface.
- Fourier-grid helpers for Cartesian propagation workflows.
- Hankel-transform utilities for radially symmetric propagation problems.
- Fresnel and diopter operators for vectorial field transmission, carrying the
  geometric amplitude factor that conserves the Poynting flux between the two
  reference spheres.
- An independent Franz / Stratton-Chu reference solver used to arbitrate
  modelling conventions against an exact Maxwell field.
- Polarization diagnostics and visualization helpers.
- Time-harmonic field animation (the quiver above shows the instantaneous
  real field `Re[E e^{-i omega t}]` of a focused, cross-maximizing pupil over
  the optical cycle; regenerate it with `python examples/harmonic_field_animation.py`).
- Example scripts that generate reproducible figures and comparison outputs.

## Installation

Create and activate a Python environment, then install the package from the
repository root:

```bash
python -m pip install -e .
```

The package requires Python 3.10 or newer.  The general surface transform of
`vecdiff.wave` (freeform surfaces, near-linear cost on large apertures) also
wants `finufft`:

```bash
python -m pip install -e ".[wave]"
```

Without it the axisymmetric spectral path still works through the Bessel
kernel and direct sums.

## Quick Check

Run the test suite from the repository root:

```bash
pytest
```

## Examples

Examples live in `examples/` and are intended to be run from the repository
root:

```bash
python examples/reference_vs_model.py          # the model against an exact Maxwell field
python examples/cartesian_simple.py            # single-diopter propagation basics
python examples/aperture_scalar_vs_vectorial.py  # scalar (t- = 0) vs vectorial focus
python examples/maximize_cross_polarization.py   # edge pupil maximizing the cross field
python examples/two_diopter_imaging.py           # orientation-dependent vectorial imaging
python examples/resolution_inversion.py          # scalar resolves, vectorial fuses
python examples/wave_error_scaling.py            # error law of the general operator vs exact Maxwell
python examples/wave_radial_diopter.py           # Quabis tight spot through a real diopter
python examples/wave_nanojet.py                  # photonic nanojet via the closed-body word
python examples/wave_light_needle.py             # longitudinal light needle from one surface
python examples/wave_vortex_diopter.py           # spin-orbit vortex switch at an interface
python examples/wave_freeform_astigmat.py        # freeform astigmatism via the NUFFT path
```

Generated artifacts are written under `examples/output/`. See
[`examples/README.md`](examples/README.md) for the current example list and
output layout.

## The transfer chain

An incident field is prescribed on the reference sphere `G` centred on the
object point, transferred to the sphere `G'` centred on the image point, and
reduced to the focal plane:

```
E_G  --[ transfer operator ]-->  E_G'  --[ Debye reduction ]-->  E(focal plane)
```

The transfer operator is diagonal in the local Fresnel frame, with eigenvalues

```
radial        A(Q) t_p cos(alpha_i)/cos(alpha_0)
azimuthal     A(Q) t_s
longitudinal  A(Q) t_p sin(alpha_i)/cos(alpha_0)
```

and `A(Q) = |z0| l_i / (|zi| l_0)` the factor that conserves the mean Poynting
flux between the two spheres.  The focal-plane step integrates over the
sine-mapped pupil coordinate `u = |zi| sin(alpha_i)` with the `1/cos(alpha_i)`
Jacobian that substitution carries.

These conventions were settled by measurement rather than argument.  At an
image-side numerical aperture of 0.91, against the Franz / Stratton-Chu
integral -- an exact Maxwell field -- the chain above reproduces the reference
to a relative RMS of 2e-9 in absolute amplitude and phase.  Run
`examples/reference_vs_model.py` to reproduce the comparison, including the
variants that get it wrong.

Fields carry their geometry: `Grid.reference` records whether the samples sit
on a reference sphere or on a tangent plane, and the propagators dispatch on it
rather than assuming.

## The general operator engine

`vecdiff.wave` treats an interface as an operator on the vector angular
spectrum, defined for any smooth surface through its normal and measure alone.
Because each operator maps spectra to spectra, interfaces *compose*: a system
of many surfaces is their ordered product interleaved with free propagation.

```python
import numpy as np
import vecdiff.wave as vw

grid = vw.Grid.from_spacing(0.25, 256)
pw   = vw.plane_wave_spectrum(grid, wavelength=1.0, n=1.0, polarization="x")

kappa = vw.stigmatic_conic_constant(1.0, 1.5)           # plane-wave stigmat
front = vw.InterfaceOperator(vw.Conic(radius=+6.0, conic=kappa),
                             n1=1.0, n2=1.5, aperture=5.5)
back  = vw.InterfaceOperator(vw.Plane(), n1=1.5, n2=1.0, aperture=5.5)

system = vw.System([front, vw.FreeSpace(8.0), back])    # a word in the algebra
out    = system(pw)                                     # spectrum -> spectrum
```

A genuinely non-axisymmetric surface uses the same operator through the
general NUFFT surface transform (`vw.Freeform2D`, `vw.surface_transform`);
`vw.load("euv")` / `vw.load("duv")` ship two ray-traced projection objectives
for vector PSFs and aerial images (`vw.Pupil`, `vw.ImagingSystem`, `vw.Mask`).

And the two engines meet on the stigmatic oval:

```python
from vecdiff import CartesianSurface
import vecdiff.wave as vw

oval = CartesianSurface(n0=1.0, ni=1.5, z0=-30.0, zi=20.0)
report = vw.referee(oval)          # exact chain vs general operator
print(report["profile_rms"])       # ~2e-2 at NA_i ~ 0.6
```

### The amplitude measure, measured

The return integral radiates the refracted surface field with an amplitude
measure selected by `measure=`:

- `"franz"` (default): the Kirchhoff obliquity pair `(n.k_t + n.d)/2` times
  the chart radiation factor `k/kz` — the stationary-phase content of the
  exact Franz radiation of the surface currents.  It reduces to the bare
  transform in the planar limit and, measured against the Franz/Stratton–Chu
  Maxwell reference on the stigmatic oval, holds the absolute focal amplitude
  to −3…−8% with **no trend in NA or in the size parameter**
  (`examples/wave_error_scaling.py`).
- `"flat"`: the bare surface transform of the original construction.  Its
  phase (focal profile) is percent-level too, but its absolute amplitude runs
  +9% at NA 0.32 to +40% at NA 0.87 — an error controlled by aperture, not by
  size.  Kept for comparison and for reflection, where the Franz convention
  is not yet worked out.

Validation status is pinned by tests: the planar Fresnel limit to ~0.5%
(`test_wave_rigor`), the exact-chain and Stratton–Chu referees
(`test_unification`, `test_stratton_chu_referee`), Richards–Wolf 1959 to
2–4×10⁻⁴ profile RMS and Quabis 2000 spot areas to 1–6%
(`test_literature`).

## Package Layout

```text
vecdiff/          Python package (the exact stigmatic chain)
vecdiff/reference/  Independent Maxwell reference solver
vecdiff/wave/     The general spectral interface-operator engine
vecdiff/wave/stigmatic.py  The bridge and referee between the two engines
examples/         Runnable scripts and generated-output conventions
tests/            Unit tests (test_unification.py referees the two engines)
docs/assets/      README and documentation media
docs/roadmap/     Planned work and open design debt
```
