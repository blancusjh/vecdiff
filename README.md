# Logos Optics

Two independent vector optical propagation libraries, organized around **space, matter, and fields**:

- **`vecdiff_general`** — homogeneous ASM/Fresnel propagation, surface transfer, local/spectral Fresnel interaction, and field/system/target propagation.
- **`vecdiff_stigmatic`** — Cartesian-oval design, entrance/exit reference spheres, geometric vector transfer, Debye focusing, and a separate paraxial Fourier-pupil model.

The packages have separate source trees, distributions, and tests. Neither imports the other. Their corresponding foundational types are intentionally implemented locally; external validation compares physical results. The original repository and its branches remain intact.

Documentation:

- [Notebooks](docs/NOTEBOOKS.md): seven executed optical studies using the current methods.
- [Architecture](docs/ARCHITECTURE.md): responsibilities, units and source map.
- [Propagation](docs/PROPAGATION.md): homogeneous propagation and the R Q⁻¹ T Q R path.
- [Fourier API and grids](docs/FOURIER_AND_GRIDS.md): Cartesian, polar and nonuniform transforms with physical coordinates.
- [Validation](docs/VALIDATION.md): tests, full-wave references (BEM, FEM, Mie) and their limits.

## Install and verify

Python 3.10 or newer is required. Building the general package also needs a C
compiler and a Fortran compiler (tested with gfortran). Run from this directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install meson-python ninja 'numpy>=2'
.venv/bin/python -m pip install --no-build-isolation -e './general[test,notebooks,nufft,reference-mesh]'
.venv/bin/python -m pip install -e './stigmatic[test,notebooks,nufft]' ruff
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Both libraries require NumPy and SciPy. Matplotlib, Jupyter execution, FINUFFT
and Gmsh (`reference-mesh`) are optional extras. Neither package requires the
other; either installs alone with `python -m pip install ./general` or
`python -m pip install ./stigmatic`. `requirements-dev.lock` records the
qualified development environment; the root `pyproject.toml` configures code
style. See [notebooks](docs/NOTEBOOKS.md) to execute the notebooks.

## Organization

The general package separates physical data from algorithms:

```text
vecdiff_general/
    geometry/       coordinate frames
    surfaces/       surface shapes and normals
    sampling/       grids and integration measures
    media/          material properties and spatial assignment
    fields/         field samples, spectra and source representations
    system.py       surface/media adjacency and optical prescriptions
    fourier/        numerical transforms
    propagation/
        homogeneous.py          propagation in a homogeneous medium (ASM/Fresnel)
        surface_transfer.py     surface trace Q and outgoing reconstruction
        surface_interaction.py  transmission and reflection at a boundary
        pipeline.py             composition along an optical path
        references/             Mie, BEM and independent reference solutions
    observables/    energy, power, polarization and detector formation
    visualization/  plots and animations
```

Angular-spectrum **construction** belongs to `fields/`; evolution in a
homogeneous medium belongs to `propagation/homogeneous.py`. Surface geometry
and adjacent materials are specified independently of the interaction model.
A pipeline can contain zero, one or several optical boundaries.

The stigmatic package retains its independent Cartesian-oval and Debye model.
Each distribution has its own tests. External comparisons live in `validation/`.

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

An interface normal points from its `minus` region into its `plus` region. `scatter_plane` retains reflected and transmitted expansions, including evanescent transmission under TIR. `PropagationPipeline(system, transmission="local" | "spectral")` accepts an existing incident field and a target (`z=f`, `y=a`, `x=b`, or a placed surface), returning an `ElectricField`; see the [pipeline example](general/examples/propagation_pipeline.py).

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

Read [validation](docs/VALIDATION.md) for the reference checks and measured
errors. The curved-surface pipeline is a forward approximation. Its current
transport uses a finite, periodic FFT chart and a declared propagating angular
band. Finite material edges, stops and returning branches are outside this
pipeline's supported model. Exact planar scattering and complete reference
transfer have different, explicitly documented spectral coverage.
