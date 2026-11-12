# vecdiff

vecdiff computes the vector electromagnetic field of light through an
optical system by eikonal–angular-spectrum transport. Across each surface
the light meets, the field is carried along the rays of its carrier (the ray
family the light declares) from a reference plane in front of the surface to
the same plane on the side it leaves into, with the Fresnel dyadic of that
surface and the flux factor of the ray tube; between surfaces it is an exact
superposition of plane waves. The transfer is the leading-order solution of
Maxwell's equations in the size of the system relative to the wavelength:
nothing is added by hand, it is exact for a plane wave on a plane interface,
and on curved surfaces its error falls as 1/(R/λ).

The library is meant for **macroscopic** systems, tens to thousands of
wavelengths across: lenses, mirrors, stops, ghosts. It does not model
absorbing or dispersive media within a calculation (real indices only),
coatings, partial coherence, evanescent near fields of a surface, or rim
scattering of a finite body, and it computes fields, not analysis outputs
such as PSF, MTF, Strehl ratio, Zernike or OPD maps, or a focus search.

<p align="center">
  <img src="docs/assets/focal_polarization_linear.gif" alt="Focal-plane polarization over one optical cycle, linear-x input" width="420">
  <img src="docs/assets/focal_polarization_circular.gif" alt="Focal-plane polarization over one optical cycle, circular input" width="420">
</p>

<p align="center">
  <em>Instantaneous focal field <code>Re[E e<sup>-i&omega;t</sup>]</code> of the Cartesian-oval
  lens of notebook 03 over two optical cycles: arrows show the transverse field over its
  time-averaged intensity (scale I<sup>0.3</sup>, so the secondary maxima stay visible).
  Linear-x input (left) oscillates; circular input (right) rotates.</em>
</p>

## Quick start

A collimated Gaussian beam through a plano-convex lens, flat side first, and
E, H and the Poynting vector at its focus:

```python
import numpy as np
import vecdiff as vd

# The system: what the light meets, in order. Lengths in metres here; any
# unit works if the wavelength is given in it.
air, glass = vd.media.air, vd.media.Dielectric(1.5, "glass")
curved = vd.surfaces.SphericalCap(-25e-3, vd.geometry.Frame([0, 0, 3e-3]))
lens = vd.system.OpticalSystem(
    vd.system.Stop(4e-3, z=-0.5e-3),  # an opaque screen with a 4 mm hole
    vd.system.Refraction(vd.surfaces.Plane.z(0.0), air, glass),
    vd.system.Refraction(curved, glass, air),
)

# The light: 2 mm waist, polarized along x, 532 nm, its plane 1 mm before the lens.
beam = vd.fields.PlaneBeam(
    vd.fields.profiles.Gaussian(2e-3),
    vd.fields.polarization.x,
    wavelength0=532e-9,
    region=air,
    frame=vd.geometry.Frame([0, 0, -1e-3]),
)

solution = vd.propagation.Propagator(lens)(beam)
print(solution)  # per element: what it did not carry, and why

# E and H on the plane of best focus, over a 40 um square.
focus = solution.field(
    z=52.7e-3, window=(-20e-6, 20e-6, -20e-6, 20e-6), shape=(81, 81)
)
S = vd.observables.poynting(focus)  # time-averaged, W/m²
centre = (40, 40)
print("E (V/m):", np.round(focus.electric.values[centre], 2))
print("H (A/m):", np.round(focus.magnetic[centre], 4))
print("S (W/m²):", np.round(S[centre], 3))
print(f"power through the square: {vd.observables.power(focus):.3e} W")
```

It runs in about 8 s and 1.6 GB on a laptop CPU and prints (measured now):

```text
Solution through 3 element(s), 3 carried; leaving as HankelSpectrum in air 2
  stop 1 (radius 0.004, in air): unsupported 1.8e-02, folded 0.0e+00, ...
  surface 2 (Plane, air -> glass): unsupported 7.2e-04, folded 0.0e+00, ...
  surface 3 (SphericalCap, glass -> air 2): unsupported 8.3e-04, folded 0.0e+00, ...
E (V/m): [-18.64+337.96j   0.    +0.j     0.    +0.j  ]
H (A/m): [ 0.    -0.j     -0.0495+0.8971j  0.    +0.j    ]
S (W/m²): [  0.      0.    152.053]
power through the square: 7.525e-09 W
```

- **The report.** The stop holds back 1.8 % of the incident amplitude, the
  Gaussian's tail beyond 4 mm (exp(−(4/2)²) = 0.018). `air 2` is the air
  behind the lens, a region distinct from the air in front.
- **The focus.** Paraxially it lies f = R/(n − 1) = 50 mm behind the curved
  vertex, at z = 53 mm; spherical aberration of the flat-first lens brings the
  axial maximum to 52.7 mm (the axis scan in
  [`plano_convex_lens.py`](general/examples/plano_convex_lens.py) finds it).
  A scalar diffraction integral over the traced wavefront gives |E| = 339 V/m
  there with t = 0.96, against 338 V/m above; an unaberrated lens would give
  π w² t / (λ f) = 454 V/m.
- **Units.** E is in V/m (the beam's profile is 1 V/m at its centre), H in
  A/m, S in W/m². Powers are in watts because the lengths are in metres.

## Install

Python 3.10 or newer. The library is pure Python; NumPy and SciPy are its
only dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e './general[test,notebooks,nufft]'
```

`nufft` installs FINUFFT, which the transport uses for its plane-wave sums
when it is present (strongly recommended); `notebooks` installs Matplotlib and
the Jupyter execution tools; `cuda12` or `cuda13` installs CuPy for a CUDA
GPU ([GPU backend](#gpu-backend)). `requirements-dev.lock` records the
qualified development environment; the root `pyproject.toml` configures code
style.

The independent full-wave solvers that validate the library (BEM, FEM, MFS,
meshing) are a separate package, `vecdiff_validation` in `validation/`. It
builds a Fortran extension and needs a C and a Fortran compiler (tested with
gfortran); the library never imports it. To run the whole test suite:

```sh
.venv/bin/python -m pip install meson-python ninja 'numpy>=2'
.venv/bin/python -m pip install --no-build-isolation './validation[test,mesh]'
.venv/bin/python -m pip install -e './stigmatic[test,notebooks,nufft]' ruff
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

## The library, in the order of its ontology

Public objects are reached through the namespace of what they are
(`vd.system.Refraction`, `vd.propagation.Propagator`); the package top level
re-exports nothing.

| Namespace | What it is |
|---|---|
| `geometry`, `media`, `surfaces`, `volumes` | Space and matter: frames; materials (`Dielectric`, `air`, `vacuum`) and the regions they fill; surfaces (caps, aspheres, ovals, freeforms, closed bodies); volumes for the references |
| `system` | The optical system: the ordered path of `Refraction`, `Reflection` and `Stop` the light meets |
| `fields` | Light: beams (`PlaneBeam`, `FocusedBeam`) from a profile and a polarization, sampled fields, the carrier, and `Spectrum`, the plane-wave field of a region |
| `transport` | The method across one surface: the ray map and its caustic index, the dyadics, the leg on a lattice or on a meridian, its report |
| `propagation` | The method through the whole system: `Propagator(system)(light)` returns a `Solution`, with E and H on any target; plane waves between surfaces |
| `observables` | What is measured on a field: Poynting vector, irradiance, power, energy, Stokes parameters, spin, helicity, momenta, forces |
| `references` | Exact solutions to check against: `MieSphere`, `LayeredSphere`, `InfiniteCylinder`, with their Debye terms |
| `alternatives` | Other methods, for checks: wave transport (`WavePropagator`, `SurfaceTransport`, `SpectralFresnel`, `LocalFresnel`) and the exact planar boundary |
| `fourier`, `sampling`, `backend` | Numerical infrastructure: transforms, grids, the CPU/CUDA switch |
| `visualization`, `io`, `catalog` | Tools: plots, JSON prescriptions, catalog lenses |

`visualization` and `references` are imported explicitly, so importing the
package loads neither plotting nor the reference solvers.

## Documentation and examples

- [The method](docs/PROPAGATION.md): the system as a path, the transport leg,
  its two realizations, sequences, reports, results, exact references and the
  regression record; other methods last.
- [Architecture](docs/ARCHITECTURE.md): responsibilities, units and source map.
- [Validation](docs/VALIDATION.md): how the method and the other methods are
  checked, and the measured records.
- [Fourier transforms and grids](docs/FOURIER_AND_GRIDS.md): Cartesian, polar and
  nonuniform transforms with physical coordinates.
- [Notebooks](docs/NOTEBOOKS.md): eight executed optical studies.
- [Changes](docs/CHANGES.md): what changed for a reader of the earlier API.

Runnable scripts in [`general/examples/`](general/examples/), each under
20 s on a laptop; each prints a few numbers with what to expect and why:

| Script | Shows |
|---|---|
| [`plano_convex_lens.py`](general/examples/plano_convex_lens.py) | The quick start with an axis scan for the focus and automatic focal-plane sampling |
| [`paraboloidal_mirror.py`](general/examples/paraboloidal_mirror.py) | An ideal mirror: the light comes back, its axis turned over |
| [`plate_ghost.py`](general/examples/plate_ghost.py) | A ghost path built from the same two surfaces as the direct light |
| [`beams_and_pupils.py`](general/examples/beams_and_pupils.py) | `fields`: profiles, polarizations, zoned pupils, against Rayleigh, Airy and symmetry |
| [`one_transport_leg.py`](general/examples/one_transport_leg.py) | `transport`: the ray map, a lattice leg, its report and a caustic inside the leg |
| [`exact_references.py`](general/examples/exact_references.py) | `references`: Mie, layered sphere, cylinder, Debye terms |
| [`observables_of_light.py`](general/examples/observables_of_light.py) | `observables`: power, irradiance, Stokes, spin, momenta, radiation force |
| [`wave_transport.py`](general/examples/wave_transport.py) | `alternatives.WavePropagator`, the other method, on a sampled field |

## Physical contracts

- Lengths are in one unit, that of the vacuum wavelength `wavelength0`. E and H are phasors in V/m and A/m, so S is in W/m²; quantities that integrate over area or involve the frequency (power, momentum, spin, force) are SI when lengths are metres. Time dependence is `exp(-i omega t)`.
- `ElectricField(grid, wavelength0, values)` owns electric samples, with `Ex`, `Ey`, `Ez` views and `norm2()` (|E|², electric amplitude squared, not an irradiance). `ElectromagneticField` holds E and H on one grid; its `poynting()`, `irradiance()` and `power()` preserve coherent interference.
- `Grid` owns physical coordinates, placement, and any integration measure. `CartesianGrid` (alias `PlaneGrid`) components use a declared right-handed frame; E values are global Cartesian vectors.
- A beam (`fields.PlaneBeam`, `fields.FocusedBeam`) is a source-free sum of propagating plane waves in one medium. It takes a scalar `profile` from `fields.profiles` (`Uniform`, `Gaussian`, `Annulus`, `Zones`, and their products) and a separate `polarization` from `fields.polarization` (`x`, `y`, `radial`, `azimuthal`, or any Cartesian or polar pair), evaluates anywhere through a quadrature chosen for the evaluation region, and declares its rays as a `fields.Carrier`.
- `TransverseElectricField` prescribes two local plane components; the longitudinal one follows from the medium. A complete electric vector is validated, never silently projected.
- A medium (`media.Dielectric`) is what fills space; a region (`media.HomogeneousRegion`) is a part of space with its medium. The system derives its regions from its path; the references take regions.
- Momentum observables use the dual-symmetric densities of Berry and Bliokh–Bekshaev–Nori: `observables.spin_density` and `helicity_density` read E/H samples; `canonical_momentum`, `spin_momentum` and `orbital_angular_momentum` read an `ElectricFieldJet` (`observables.plane_wave_jet(expansion, grid)`), with ε μ P = p_o + ∇×S/2. `observables.force` and `torque` integrate the Maxwell stress tensor over a closed surface quadrature.

## GPU backend

One global switch moves arrays and computation to a CUDA GPU. Make it at the
top of a script, before building any vecdiff object:

<!-- fragment -->
```python
from vecdiff import backend

backend.set_backend("CUDA")  # "CPU" is the default; needs CuPy and a CUDA device
```

Everything built and computed afterwards lives on the GPU, in double precision;
results match the CPU to roundoff. NumPy arrays and lists passed in are moved to
the device on entry, results stay there until `backend.to_numpy(array)`, and
plotting converts on its own. Objects built before a switch keep their arrays on
the old device and must be rebuilt. Geometry, meshing, the reference solvers in
`references` and FINUFFT compute on the CPU; their inputs and results
are moved for them. `backend.bd` is the current array module (NumPy or CuPy) for
creating arrays on the same device. `VECDIFF_BACKEND=CUDA pytest general/tests`
runs the test suite on the GPU.

The gain depends on the operation. Measured when the backend was added, on a
laptop RTX 4060 (8 GB) against 16 CPU threads: focal maps of a `FocusedBeam`
about 14× faster, angular-spectrum propagation of a 2048² field 10×, an
order-1 Hankel transform 37× and the curved-surface transfer `SurfaceTransport`
6–9×, all matching the CPU to roundoff. Device memory bounds the lattice:
angular-spectrum propagation holds about 3 GB at 2048². FINUFFT stays on the
CPU: in double precision it outpaces cuFINUFFT on consumer cards.

## Results

Figures from the executed [notebooks](docs/NOTEBOOKS.md).

<p align="center">
  <img src="docs/assets/sphere_focal_components.png" alt="Ex, Ey, Ez focal intensities: Debye, full Mie, local and spectral propagation" width="860">
</p>

**Macroscopic sphere, R/λ = 100 (notebook 07).** Focal |E<sub>x</sub>|², |E<sub>y</sub>|², |E<sub>z</sub>|²
from local and spectral Fresnel wave transport (`alternatives`) through the same glass sphere (n = 1.5, illumination
radius 35λ), beside the first transmitted Debye term (p = 1) and the full Mie field, which includes
additional scattering (scale I<sup>0.3</sup>, so the rings are visible).

<p align="center">
  <img src="docs/assets/lens_focal_polarization.png" alt="Focal-plane polarization, spherical and stigmatic exit faces" width="860">
</p>

**Lens focusing (notebook 03).** A plane–convex glass lens (n = 1.5, exit radius 120λ, λ = 532 nm)
through wave transport (`alternatives.WavePropagator`) with spectral Fresnel transmission. Focal-plane transverse intensity and
polarization: the spherical exit face (left) blurs the focus by spherical aberration; the Cartesian-oval
exit face (right) gives a sharp Airy pattern with dark rings.

<p align="center">
  <img src="docs/assets/lens_meridional_fields.png" alt="Meridional total and longitudinal electric intensity through focus" width="860">
</p>

**Through focus (notebook 03).** Total and longitudinal (|E<sub>z</sub>|²) electric intensity in the
meridional plane for both exit faces (scale I<sup>0.3</sup>); the dashed line marks the nominal image plane.

<p align="center">
  <img src="docs/assets/diopter_fresnel_compensation.png" alt="Uncompensated and compensated stigmatic diopter focal planes" width="860">
</p>

**Fresnel compensation in a stigmatic diopter (notebook 04).** Ray-tube/Debye focus at NA ≈ 0.84 with
and without compensating the Fresnel transmission. The compensated first minimum (gold dashes) lies on
the Airy first-zero circle (black); the uncompensated one (cyan) is wider.

<p align="center">
  <img src="docs/assets/aperture_diffraction.png" alt="Diffraction by a slit, two slits and a circular hole at three distances" width="760">
</p>

**Diffraction by apertures (notebook 02).** Vector angular-spectrum propagation behind a single slit,
a double slit and a circular hole at three distances, on a common I<sup>0.3</sup> intensity scale, with
the transverse polarization.

<p align="center">
  <img src="docs/assets/al1225_scaled_models.png" alt="Thorlabs AL1225 scaled models: meridional and focal intensity" width="700">
</p>

**Thorlabs AL1225 (notebook 06).** Spectral wave-transport fields of the catalog asphere, geometrically
scaled to R/λ = 3, 10, 30 and 100: meridional and focal-plane |E|².

## License

Original project code and documentation are licensed under the [Apache License 2.0](LICENSE).
Third-party code, publications, datasets, fonts, and other assets retain their
respective licenses and copyright notices; this license does not replace them.
