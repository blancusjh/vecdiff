# vecdiff

Radial Hankel and Fresnel field-propagation utilities for vector diffraction
calculations. The project contains a small Python package, runnable examples, and
tests for working with sampled electromagnetic fields in Cartesian, polar, and
circular representations.

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

The package requires Python 3.10 or newer.

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

## Package Layout

```text
vecdiff/          Python package
vecdiff/reference/  Independent Maxwell reference solver
examples/         Runnable scripts and generated-output conventions
tests/            Unit tests
docs/assets/      README and documentation media
docs/roadmap/     Planned work and open design debt
```
