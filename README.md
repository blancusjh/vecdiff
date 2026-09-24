# Logos Optics

Two independent vector optical propagation libraries, organized around **space, matter, and fields**:

- **`vecdiff_general`** — homogeneous ASM/Fresnel propagation, dielectric interfaces, coherent layers, curved-surface radiation, local spectra, and multi-surface high-frequency transport.
- **`vecdiff_stigmatic`** — Cartesian-oval design, entrance/exit reference spheres, geometric vector transfer, Debye focusing, and a separate paraxial Fourier-pupil model.

The packages have separate source trees, distributions, and tests. Neither imports the other. Their corresponding foundational types are intentionally implemented locally; external validation compares physical results. The original repository and its branches remain intact.

The common [Fourier API and typed grids](docs/FOURIER_AND_GRIDS.md) provide Cartesian and polar transforms with explicit physical coordinates.

[Full-wave references](docs/REFERENCE_PROPAGATION.md) and
[notebook 07](general/docs/notebooks/07_full_wave_reference_validation.ipynb)
compare the operational method with our Fortran dielectric BEM and vector FEM
solvers, plus analytical Mie spheres. Tables group parameters within each
geometry and retain boundary checks, convergence, field plots and timings.

[Surface edges and pupils](docs/PUPILS_AND_EDGES.md): a lens edge no longer cuts the field; only explicitly declared stops do. Keeping the light that passes outside a lens reduces the on-axis error against full-wave references by up to half.

The [general architecture review](docs/GENERAL_ARCHITECTURE_REVIEW.md) records the latest separation of geometry and volumes, reproduced gaps, and the next proposed steps. Stigmatic architecture changes are deferred.

Start with the **[executed notebook guide](docs/NOTEBOOKS.md)** or the **[implemented architecture](docs/ARCHITECTURE.md)**. The seven notebooks cover planar optics, aperture diffraction, explicit lens focusing, vector beams, ideal projection, Fresnel compensation, polarization-selective holography, and scalar/vector lithography including an ideal DUV reduction relay. Field intensity maps use hot or inferno. Grayscale polarization glyphs appear only on figures intended to show the polarization of a complete vector field; component, scalar-reference and circuit-comparison panels omit them. Projection and DUV are combined in the imaging notebook; holography is part of the vector-beam notebook. The [polarization guide](docs/POLARIZATION.md) describes automatic figure sizing, intensity-aware ellipse placement, and animation.

## Install and verify

Python 3.10 or newer is required. Building the general package also needs a C
compiler and a Fortran compiler (tested with gfortran). Run from this directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install meson-python ninja 'numpy>=2'
.venv/bin/python -m pip install --no-build-isolation -e './general[test,notebooks,nufft,reference-mesh]'
.venv/bin/python -m pip install -e './stigmatic[test,notebooks,nufft]' ruff
.venv/bin/python -m pytest -q
.venv/bin/python scripts/notebooks.py
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Both libraries require NumPy and SciPy. Matplotlib, Jupyter execution, and FINUFFT are optional extras. Gmsh (`reference-mesh`) generates meshes for the reference notebook. Neither package requires the other. Either can be installed alone with `python -m pip install ./general` or `python -m pip install ./stigmatic`.

```sh
.venv/bin/python scripts/notebooks.py --check
.venv/bin/python scripts/verify_independent_wheels.py
.venv/bin/python general/examples/planar_interface.py
.venv/bin/python stigmatic/examples/focal_field.py --plot
```

The notebook runner uses the calling interpreter and a project-local kernel specification. Executed outputs are retained in the notebooks. `requirements-dev.lock` records the qualified development environment; the libraries' dependency declarations are in their individual `pyproject.toml` files. The root `pyproject.toml` configures code style.

## Organization

Both packages follow corresponding ontology folders rather than a flat collection of numerical routines:

```text
src/vecdiff_<general or stigmatic>/
    geometry/       mathematical geometry (general); existing ray geometry (stigmatic)
    surfaces/       geometrical shape equations
    sampling/       physical grids and integration measures
    media/          constitutive materials and their spatial assignment
    fields/         electric samples and explicit solution representations
    interfaces/     adjacent media, Fresnel laws, optical prescriptions
    fourier/        Cartesian FFT, radial Hankel, harmonics, NUFFT synthesis
    propagation/    physical evolution, scattering, radiation and focusing
    observables/    energy, Poynting flux, power and detector formation
    visualization/  component maps, polarization ellipses and animation
```

Fourier routines accept numerical arrays and lattice parameters only. Optical
angular-spectrum construction and propagation live in `propagation/`. Interface
transmission and reflection are separate operators, each returning the requested
field on its existing surface. See [the operator example](docs/INTERFACE_OPERATORS.md).

The general package keeps spatial volumes in `volumes.py`, separate from mathematical geometry and material assignment.

The general package also has `io/prescription.py` for explicit SI JSON optical prescriptions. Each distribution has `tests/` and `examples/`; the physical notebooks are in `general/docs/notebooks/`. External cross-library qualification lives in `validation/`.

## Physical contracts

- Lengths and vacuum wavelength are metres. Public E and H are SI phasors in V/m and A/m; time dependence is `exp(-i omega t)`.
- `ElectricField(grid, wavelength0, values)` owns electric samples, with convenient `Ex`, `Ey`, `Ez` views. It has no medium or duplicate sampling state.
- `Grid` owns physical coordinates, placement, and any integration measure. `PlaneGrid` components use a declared right-handed frame; complete E values are global Cartesian vectors.
- `TransverseElectricField` prescribes two local plane components. Longitudinal completion needs the material dispersion and a branch supplied separately. A complete electric vector is validated, never silently projected.
- Material context belongs to regions and systems. `MaterialDistribution` represents piecewise spatial `n(r)`; a geometrical body does not imply that a solver includes its rim scattering.
- A solution computes consistent H from its spatial/modal representation. `ElectromagneticField` holds evaluated E/H; its `poynting()` and `power()` preserve coherent interference.
- `norm2()` is electric amplitude squared. It is not labeled intensity in W/m². Energy density takes explicit nondispersive material context.

## A general propagation

```python
import numpy as np
import vecdiff_general as optics

wavelength = 532e-9
region = optics.HomogeneousRegion("air", optics.Dielectric(1.0))
axis = (np.arange(64) - 32) * wavelength / 2
grid = optics.PlaneGrid(axis, axis)

radius_squared = np.sum(grid.points[..., :2] ** 2, axis=-1)
envelope = np.exp(-radius_squared / (3 * wavelength) ** 2)
values = np.stack((envelope, 0.3j * envelope), axis=-1)
electric = optics.TransverseElectricField(grid, wavelength, values)

spectrum = optics.angular_spectrum(electric, region, evanescent="discard")
propagated = spectrum.propagate(10 * wavelength, model="asm")
power_in_watts = propagated.power()
```

An interface normal points from its `minus` region into its `plus` region. Exact plane scattering retains reflected and transmitted expansions, including TIR transmission; one-sided evaluation includes all branches present on that side. Coherent layers include repeated reflections and evanescent gaps. Curved local-interface scattering and high-frequency transport have explicitly different accuracy contracts.

## A dedicated stigmatic focus

```python
import numpy as np
import vecdiff_stigmatic as optics

wavelength = 532e-9
system = optics.StigmaticSystem(
    object_z=-10e-3,
    image_z=6e-3,
    object_material=optics.Dielectric(1.0),
    image_material=optics.Dielectric(1.5),
)
rays = system.rays(aperture=2e-3)
source = optics.cartesian_source(rays, wavelength, polarization=(1.0, 0.0))
transferred = optics.transfer(rays, source)

axis = np.linspace(-2, 2, 41) * wavelength
grid = optics.PlaneGrid(axis, axis, optics.Frame(origin=[0, 0, system.image_z]))
focus = optics.DebyeFocus(transferred)
focal_field = focus.evaluate(grid, backend="hankel")
ray_power_budget = transferred.power_budget()
```

Direct, Hankel, and optional NUFFT Debye evaluation preserve all complex Cartesian components and SI H. Intermediate reference-sphere and physical-interface fields remain inspectable. The separately named `fresnel_focus` models an ideal paraxial thin lens; it is not substituted for the Cartesian-oval model.

## Qualification and limits

Read [validation](docs/VALIDATION.md) for checks and measured errors, and [implementation status](docs/IMPLEMENTATION_STATUS.md) for supported regimes and remaining extensions. The notebooks state their physical models and distinguish numerical convergence from model accuracy.

The general Green calculation reproduces the saved stigmatic physical-optics reference without phase or amplitude fitting. That preservation check does not turn either physical-optics model into an exact closed-body dielectric solver. The [original audit](docs/ARCHITECTURE_REVIEW.md) and [earlier blueprint](docs/SEPARATE_ARCHITECTURES_AND_PHYSICAL_CHECKS.md) remain historical design evidence; the implemented architecture document describes the current tree.
