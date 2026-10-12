# Notebooks

The seven main notebooks are in [`general/notebooks/`](../general/notebooks/).
They contain executed figures and numerical checks. They use the current
package; none loads retired solvers from Git history.

| Notebook | Study and current implementation |
|---|---|
| [01 · Planar dielectric optics](../general/notebooks/01_planar_dielectric_optics.ipynb) | Exact planar Fresnel scattering, Brewster angle, TIR and polarization. |
| [02 · Diffraction by apertures](../general/notebooks/02_diffraction_by_apertures.ipynb) | Homogeneous ASM for prescribed slit and circular-screen fields, with a window check. |
| [03 · Refraction and focusing by a lens](../general/notebooks/03_refraction_and_focusing_by_a_lens.ipynb) | Spherical versus Cartesian-oval exit faces through `PropagationPipeline`: meridional/focal fields, polarization and animation. Sampling/refinement checks are outside the presentation notebook. |
| [04 · Compensating Fresnel transmission](../general/notebooks/04_compensating_fresnel_transmission.ipynb) | The retained ray-tube/Debye diopter study: Fresnel compensation, equal incident power, Airy comparison and quadrature checks. |
| [05 · High-NA lithography through the propagation pipeline](../general/notebooks/05_single_lens_lithography.ipynb) | Dense-mask imaging at NA ≈ 1.10 through `PropagationPipeline`. Transverse-intensity unity reference (`ts=tp=1`) versus total electric intensity with Fresnel transmission, plus polarization-prepared illumination. Plain intensity images and profiles lead the comparison; separate polarization edge views and a longitudinal section retain physical scales. Includes occupied NA, throughput, registered gap depths and timings. |
| [06 · Thorlabs AL1225](../general/notebooks/06_full_wave_reference_validation.ipynb) | Catalog prescription and newly measured spectral pipeline fields for scaled AL1225 models at R/λ = 3, 10, 30, 100. Fields use `hot`, followed by complete timings and independent-reference availability. Full-size propagation remains pending. |
| [07 · Macroscopic interface operators](../general/notebooks/07_macroscopic_interface_operators.ipynb) | Local and spectral propagation on the same R/λ=100 sphere, with electric fields first, then Debye/full-Mie differences and complete operator timings. |

## Run

Install the development environment described in the [README](../README.md),
including `general[notebooks,nufft,reference-mesh]`. Select the project Python
interpreter as the Jupyter kernel, then restart the kernel and run all cells.
Notebook 06 loads the saved AL1225 measurements in `validation/reference_data/`;
recomputing them needs the compiled BEM extension and Gmsh. Notebook 07 also
uses saved independent reference arrays from that directory.

Notebook 05 solves only the three dense-mask cases, with two incoherent
illumination directions per case. Calibration experiments and refinement
reruns are excluded from the notebook.
Notebook 06 presents the AL1225 prescription and saved benchmark measurements.
Regenerate the measurements outside the presentation notebook with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/python validation/al1225_benchmark.py
```

The measured scaled models have R/λ = 3, 10, 30, 100; the physical part has
R/λ = 8012.82. These are distinct. The spectral pipeline uses explicit
continued faces, and the finite-body BEM comparison retains h/λ = 0.20 and
the 5,000-unknown capacity limit. Missing BEM means unknown independent
accuracy, not zero error. Meridional observation spacing now grows with the
model to remain at most 0.5λ, and focal views span ±8λ. Full focal time
includes preparation, both interfaces and all 6,561 E/H observation points;
all-fields time adds the axial scan, complete meridian and interface
reconstruction. Timings describe recorded solver runs, not notebook reloads.

The previous sphere and seven synthetic-body comparisons, including the
arbitrary two-freeform-face case, are preserved with their original outputs in
[the archive](../general/notebooks/archive/06_synthetic_reference_validation.ipynb).

A short non-notebook example is available as
[`general/examples/propagation_pipeline.py`](../general/examples/propagation_pipeline.py):

```sh
.venv/bin/python general/examples/propagation_pipeline.py
```

## Models and migration

06 contains the physical **Thorlabs AL1225** prescription and an
executed meridional geometry plot. `optics.thorlabs_al1225()` constructs its
finite optical system in metres: 12.5 mm diameter, 4 mm center thickness,
an aspheric entrance (conic plus A4–A16) and a plane exit. N-BK7 dispersion
is evaluated at the requested wavelength; at 780 nm the physical rim gives
R/λ = 8012.82. `scale != 1` denotes a geometrically scaled model, not the
manufactured part. The source coefficients and orientation are documented in
[`catalog.py`](../general/src/vecdiff_general/catalog.py).
The full-size field, runtime and independent accuracy are **pending**;
the geometry plot is not a propagation result. This finite prescription is
not silently converted into the continued open faces used by the pipeline.

03 prescribes open graph surfaces with explicit illumination. Its geometry keeps
the entire FFT chart valid without the retired finite-rim treatment. The
illumination support is not a material edge or a stop.
05 uses an immersion interface with a spherical central region and an explicit
continuous-slope outer continuation. Prescribed converging illumination comes
from a virtual mask; upstream preparation optics are outside the model. All
three cases execute the public propagation pipeline at 1 W incident power.
The unity reference displays transverse electric intensity; physical Fresnel
images display total electric intensity. The unity pipeline still computes the
full vector field. Its transverse intensity has the scalar fringe shape in the
symmetric two-order model, without implying a general scalar equivalence. Changing incident polarization preserves angular power
and tests high-NA polarization contrast. It is not a general
inverse-Fresnel compensator. Measured outgoing bandwidth, throughput and
registered resolution accompany the images. These images do not establish
full-wave accuracy for this geometry.
04 remains a geometrical ray-tube study, explicitly separate from the FFT
pipeline. The archived synthetic study uses Mie only for the sphere and fixed-mesh BEM for general
geometries, whose remaining discretization uncertainty is stated explicitly.
It compares these closed-body references, including rims and reflections,
with the full forward pipeline on explicitly continued graph faces. The
continued material background and omitted reflected/rim fields are physical
model differences; the sphere continuation also alters its outer 5% of radius.
Meridional coverage excludes points unavailable to the forward plane spectra.
Interface tables distinguish prescribed transmitted traces from reconstructed
traces and report the BEM facet offsets. These comparisons do not establish
full-wave pipeline accuracy for finite bodies. 07 replaces the phase-space experiment with the supported
operators; its focal references are frozen independent data, while the candidate
is freshly computed.

The notebooks distinguish discretization changes, pure-transfer errors,
reference-solver errors and physical model differences. See
[Propagation](PROPAGATION.md) and [Validation](VALIDATION.md) for their contracts.

## Operator API migration

03 and 07 use the field/system/target organization, with local or spectral
transmission selected on the propagator. 07 computes both candidates through
that API; pure-Q checks are in tests and `validation/propagation_branches.py`
checks independent E/H references outside the notebook. 05 retains explicit
spectral unity/Fresnel controls and E/H for power. The AL1225 comparison
helper constructs an OpticalSystem internally. The stored results in 05 and
the synthetic archive predate the API refactor; the active AL1225 results
were measured using the current pipeline.

Notebook 06 now regenerates the AL1225 meridian with `coverage="complete"`
through the public propagation operator. It evaluates all points in the
continued-face model and takes plus-side limits on interfaces. `coverage="reduced"`
remains available independently of local/spectral transmission. The extra
coverage is not a finite-rim or evanescent-field correction, and independent
near-interface accuracy is still pending.
