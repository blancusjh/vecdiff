# Notebooks

The eight main notebooks are in [`general/notebooks/`](../general/notebooks/).
Notebook 08 shows the method; 01 and 02 show plane waves at a plane boundary
and in free space, which the method uses between and at surfaces; 03, 05, 06
and 07 use the other method, wave transport (`alternatives`), and 04 is a
ray-tube study outside both. All except 06 were executed in place on
2026-11-11 against the current library; the runtimes below were measured
then, on an Apple M4 laptop (10 cores, 16 GB) shared with other work.

| Notebook | What it shows | Method | Runtime |
|---|---|---|---|
| [08 · Eikonal–angular-spectrum transport](../general/notebooks/08_eikonal_angular_spectrum_transport.ipynb) | `propagation.Propagator` on the aplanatic sphere (R = 100λ, with the 1/R error law from 20 to 250λ), two aplanatic surfaces in sequence (NA 1.6) and a cylinder's line focus (R = 30λ, lattice leg). Component intensities, phase, signed E_z, polarization ellipses and error maps against `MieSphere`, `LayeredSphere` and `InfiniteCylinder` (first entry). | the method | 15 s |
| [01 · Planar dielectric optics](../general/notebooks/01_planar_dielectric_optics.ipynb) | Exact planar Fresnel scattering (`alternatives.scatter_plane`), Brewster angle, TIR, evanescent penetration and polarization. | exact plane boundary | 8 s |
| [02 · Diffraction by apertures](../general/notebooks/02_diffraction_by_apertures.ipynb) | Homogeneous angular-spectrum propagation of prescribed slit and circular-screen fields, with a window check. | plane waves | 83 s |
| [03 · Refraction and focusing by a lens](../general/notebooks/03_refraction_and_focusing_by_a_lens.ipynb) | Spherical versus Cartesian-oval exit faces through `alternatives.WavePropagator`: meridional and focal fields, Poynting flux, polarization and a focal animation. | wave transport | 12 s |
| [04 · Compensating Fresnel transmission](../general/notebooks/04_compensating_fresnel_transmission.ipynb) | The ray-tube/Debye diopter study: Fresnel compensation, equal incident power, Airy comparison and quadrature checks. | ray tube and Debye integral | 17 s |
| [05 · High-NA lithography by wave transport](../general/notebooks/05_single_lens_lithography.ipynb) | Dense-mask imaging at NA ≈ 1.10 through `alternatives.WavePropagator`: transverse-intensity unity reference (`ts = tp = 1`) versus total electric intensity with Fresnel transmission, plus polarization-prepared illumination; occupied NA, throughput, registered gap depths and timings. | wave transport | 71 s |
| [06 · Thorlabs AL1225](../general/notebooks/06_full_wave_reference_validation.ipynb) | The catalog prescription (`catalog.thorlabs_al1225`) and stored spectral wave-transport fields for scaled AL1225 models at R/λ = 3, 10, 30, 100, with timings and independent-reference availability. Full-size propagation remains pending. | wave transport (stored) | not re-executed |
| [07 · Macroscopic interface operators](../general/notebooks/07_macroscopic_interface_operators.ipynb) | Local and spectral wave transport on the same R/λ = 100 sphere: electric fields, Debye and full-Mie differences, complete operator timings. | wave transport | 50 s |

## Run

Install the development environment described in the [README](../README.md),
with `general[notebooks,nufft]`; notebook 06 also needs the `vecdiff_validation`
package (`validation[mesh]`). Select the project Python interpreter as the
Jupyter kernel, then restart the kernel and run all cells. Notebooks 05 and
07 load saved independent reference arrays from `validation/reference_data/`.

Notebook 06 loads `validation/reference_data/al1225_size_study.npz`, which is
not in the repository; regenerate it outside the notebook with

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/python validation/al1225_benchmark.py
```

The stored run took 397 s (focal) and 543 s (all fields) for R/λ = 100 on a
1280² lattice alone. Re-running R/λ = 3, 10 and 30 now took 0.9, 0.9 and
10.6 s (stored: 1.2, 1.1 and 11.9 s) in 0.66 GB; the R/λ = 100 case, about
eleven times the R/λ = 30 lattice, was not re-run within this machine's
memory budget, so 06 keeps the outputs of its last execution.

Notebook 05 solves only the three dense-mask cases, with two incoherent
illumination directions per case. Calibration experiments and refinement
reruns are excluded from the notebook. Two short scripts reproduce the
method and the other method outside the notebooks:

```sh
.venv/bin/python general/examples/plano_convex_lens.py
.venv/bin/python general/examples/wave_transport.py
```

## Numbers that changed on re-execution

The printed numbers of 01, 02, 04 and 07 are unchanged (07's two timings
changed by a few percent). In 03 and 08 only timings and one report value
changed:

- **08, cylinder:** the reported unsupported amplitude rose from 4.58e-02 to
  7.91e-02, with the field unchanged (error 0.8089 % before and after). Since
  commit f882600 rays that the fan cuts count as unsupported on the lattice,
  as on the meridian; the archive's fan in this cell cuts the transverse sine
  beyond 1.9 u_max and incidence beyond 85°. Reproduced with the code at
  ac10128 (4.58e-02) and at f882600 (7.91e-02).
- **08, timings:** the Mie reference at R = 100 λ0 took 1.8 s instead of
  82.6 s (radial functions by recurrence, commit 66c98ae); the lattice leg of
  the cylinder 2.2 s instead of 15.6 s.
- **05** had no stored outputs; it now has them.

## Models

06 contains the physical **Thorlabs AL1225** prescription and an
executed meridional geometry plot. `catalog.thorlabs_al1225()` constructs its
finite optical system in metres: 12.5 mm diameter, 4 mm center thickness,
an aspheric entrance (conic plus A4–A16) and a plane exit, both faces ending
at their 6.25 mm edges. N-BK7 dispersion is evaluated at the requested
wavelength; at 780 nm the physical rim gives R/λ = 8012.82. `scale != 1`
denotes a geometrically scaled model, not the manufactured part. The source
coefficients and orientation are documented in
[`catalog.py`](../general/src/vecdiff/catalog.py). The full-size field,
runtime and independent accuracy are **pending**; the geometry plot is not a
propagation result. The stored scaled-model fields use explicitly continued
faces (wave transport refuses edges); the BEM comparison retains h/λ = 0.20
and the 5,000-unknown capacity limit, and missing BEM means unknown
independent accuracy, not zero error. Meridional observation spacing grows
with the model to remain at most 0.5λ, and focal views span ±8λ. Full focal
time includes preparation, both interfaces and all 6,561 E/H observation
points; all-fields time adds the axial scan, complete meridian and interface
reconstruction, with `coverage="complete"` (plus-side limits on interfaces).

03 prescribes open graph surfaces with explicit illumination. Its geometry keeps
the entire FFT chart valid without a finite-rim treatment. The
illumination support is not a material edge or a stop.

05 uses an immersion interface with a spherical central region and an explicit
continuous-slope outer continuation. Prescribed converging illumination comes
from a virtual mask; upstream preparation optics are outside the model. All
three cases run wave transport at 1 W incident power.
The unity reference displays transverse electric intensity; physical Fresnel
images display total electric intensity. The unity case still computes the
full vector field. Its transverse intensity has the scalar fringe shape in the
symmetric two-order model, without implying a general scalar equivalence. Changing incident polarization preserves angular power
and tests high-NA polarization contrast. It is not a general
inverse-Fresnel compensator. Measured outgoing bandwidth, throughput and
registered resolution accompany the images. These images do not establish
full-wave accuracy for this geometry.

04 is a geometrical ray-tube study, separate from both transports.

07 computes both wave-transport treatments; its focal references are frozen
independent data, while the candidates are freshly computed. Pure-Q checks
are in the tests, and `validation/propagation_branches.py` checks independent
E/H references outside the notebook.

The previous sphere and seven synthetic-body comparisons, including the
arbitrary two-freeform-face case, are preserved with their original outputs in
[the archive](../general/notebooks/archive/06_synthetic_reference_validation.ipynb),
which predates the current API and is not re-executed. Its models are
described in [validation](VALIDATION.md#full-wave-references).

The notebooks distinguish discretization changes, pure-transfer errors,
reference-solver errors and physical model differences. See
[the method](PROPAGATION.md) and [validation](VALIDATION.md) for their contracts.
