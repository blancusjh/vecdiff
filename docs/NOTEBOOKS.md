# Physical experiments with the API

Nine notebooks investigate optical phenomena with explicit geometry, readable calculations and retained figures. Field intensity maps use hot or inferno. Grayscale polarization overlays are reserved for figures that study the polarization of a complete vector field. Individual components, scalar references and circuit intensity comparisons have no glyphs. Canvas composition is written directly in each notebook.

| Notebook | Physical question |
|---|---|
| [01 · Planar dielectric optics](../general/docs/notebooks/01_planar_dielectric_optics.ipynb) | Reflection/refraction, Brewster and critical angles, evanescent fields, Fabry–Pérot layers, frustrated TIR, slab confinement and TIR retardance. Consolidates the former three planar notebooks. |
| [04 · Aperture diffraction](../general/docs/notebooks/04_diffraction_by_apertures.ipynb) | Slits, circular apertures, diffraction fringes and the Airy limit. |
| [05 · Lens refraction and focusing](../general/docs/notebooks/05_refraction_and_focusing_by_a_lens.ipynb) | Explicit spherical and matched GOTS surfaces, aberration, longitudinal fields and focal flux. |
| [06 · Vector beams](../general/docs/notebooks/06_polarization_of_vector_beams.ipynb) | Radial/azimuthal focusing, secondary rings, phase animation and polarization-selective holography. |
| [07 · Full-wave reference validation](../general/docs/notebooks/07_full_wave_reference_validation.ipynb) | Our operational method versus Fortran BEM, vector FEM and Mie; separate geometry sections, grouped parameters, defined residuals, field maps, convergence and timings. Millimetre-scale rows explicitly mark unrun full-wave references. |
| [08 · Fresnel compensation](../general/docs/notebooks/08_compensating_fresnel_transmission.ipynb) | Derive inverse illumination analytically; compare with a constant-amplitude pupil at equal input power; measure complex-field error, throughput and pupil convergence. |
| [09 · Local flux factors and post-focus fields](../general/docs/notebooks/09_local_flux_and_postfocus_fields.ipynb) | Diagnose repeated downstream structure by comparing flux correction, sample spacing and FFT-window width; six lens cases, shared meridional/focal maps and BEM refinement. |
| [10 · Vector imaging and lithography](../general/docs/notebooks/10_single_lens_lithography.ipynb) | Visible single-lens projection and ideal 193 nm reduction, a circuit mask, orientation-dependent scalar/vector contrast, partial coherence and convergence. |
| [11 · Curved interfaces: local orders, lenses, spheres and internal returns](../general/docs/notebooks/11_curved_interfaces.ipynb) | Order 0 (Fresnel) versus order 3 of the local curvature expansion: exact single-interface fields, the convergent angular range, four curved 6λ₀ lenses against BEM with a Stratton–Chu check, exit-face boundary conditions, a glass sphere under a Gaussian beam with every internal return against the exact Debye terms and Lorenz–Mie, and repeated encounters in a cavity. |

The former projection notebook was consolidated into 10; 07 now contains full-wave reference validation. The former projection/DUV and holography experiments belong in 10 and 06. Number 09 now contains a new local-flux and numerical-window investigation. The original repository's notebooks and separate stigmatic implementation remain untouched.

## Models and interpretation

The planar examples retain coherent reflected and transmitted fields, including evanescent transmission. The waveguide section solves the analytic symmetric-slab TE₀ dispersion equation with SciPy, then evaluates the mode with the library; it does not imply that a general waveguide eigenmode solver exists.

The aperture notebook prescribes a thin-screen amplitude transmission with a finite edge transition. It compares the propagated field with the ideal sharp-edge scalar Airy limit and checks sensitivity to the periodic FFT window. It does not solve scattering from a conducting screen.

The lens notebook uses repeated R Q^-1 T Q R with outgoing ASM evaluation. It checks grid refinement, padding, inverse residuals, magnetic reconstruction and flux; see [operational propagation](OPERATIONAL_PROPAGATION.md).

The second lens experiment replaces the spherical exit face with its matched-curvature GOTS design in the collimated-input limit. Diameter, central thickness, materials, illumination, plotting axes and detector coordinates are shared. Corresponding field maps use shared color scales. A wide aperture and broad incident beam expose spherical aberration: the shared aperture radius and beam waist are 48 wavelengths, with 20 wavelengths of central thickness. The dedicated stigmatic library stays independent. See [the surface API](GOTS_SURFACE.md).

Figures render at 180 DPI. Continuous field maps use interpolated cell shading only after their physical evaluation grids resolve the displayed structure. In the lens notebook, the meridional fields are evaluated on a 161 × 321 grid (0.225 wavelengths transversely and 0.641 wavelengths axially), while focal planes use 129 × 129 points at 0.094 wavelengths per coordinate. Aperture transmission maps retain discrete cell shading so the prescribed mask is not visually softened.

## Ideal imaging applications

The application sections use the [ideal aplanatic imaging operators](IDEAL_IMAGING.md). Their numerical aperture, signed magnification, Jones pupil and normal-power normalization are explicit. These models preserve Maxwell transversality in each homogeneous image region. They do not predict the aberrations or interface losses of an unspecified objective.

Fresnel compensation changes the prepared illumination, not the material's coefficients. At fixed incident power, recovery of the target shape has a throughput cost. The paired-axis example cancels differential transmission while leaving common attenuation.

The holographic phase mask is a prescribed thin Jones operator. Its design lattice differs from the displayed focal grid: the latter is evaluated from the vector spectrum at 257 × 257 points with 39 nm spacing, using FINUFFT. Enlarging a raster is not used as a substitute for field evaluation.

The lithography notebook shows aerial fields. Its circuit mask is copied from the original stigmatic branch, with the source revision, crop and physical scale recorded in the notebook and [asset provenance](../general/docs/assets/README.md). Matched close-ups and line profiles resolve vertical and horizontal trace groups without glyphs covering the traces. A separate diagonal-input view combines the x/y responses coherently for each source direction, then adds source directions incoherently; its glyphs describe polarization over total vector-field intensity. The DUV experiment is an ideal two-objective 4f relay, not a manufactured multi-surface DUV prescription. Finite-thickness mask scattering, coatings, absorption and resist chemistry require additional physical models. The DUV source quadrature is refined from 8 to 16 to 32 directions, followed by a 64-direction reference check, independently of spatial sampling. Relative intensity changes are 4.528%, 1.926% and 0.769%; the displayed 32-direction ensemble retains a measurable source-quadrature error. Incoherent-source maps add Stokes values; their glyphs show the polarized component, with opacity equal to the projected degree of polarization.

## Run and inspect

```sh
python -m pip install -e './general[notebooks,nufft]'
python scripts/notebooks.py
python scripts/notebooks.py --check
python scripts/notebooks.py --pattern 08_compensating
.venv/bin/python -m ipykernel install --user --name logos-optics --display-name "Logos Optics (project interpreter)"
python scripts/notebook_gallery.py
```

For interactive notebook 05, select **Logos Optics (project interpreter)** and restart the kernel before running all cells after library changes. The first cell checks the imported package location and propagation API so a stale kernel fails with a clear message.

The collection retains executed outputs, including the TIR, radial-focus and stigmatic-focus animations. The [execution report](../validation/results/notebooks.json) records current cell and figure counts. Source fingerprints detect edits after execution; rerun after library changes as well. Notebook 07 records its measured execution times and explicitly identifies unconverged or budget-skipped reference cases. Its macroscopic weak-lens checks are not a full-wave validation of fixed-NA macroscopic lenses; see [reference propagation](REFERENCE_PROPAGATION.md).

Notebook 07 uses the same meridional and focal-plane field views as notebook 05:
electric norm squared in V²/m² and axial Poynting flux in W/m², without incident-field
normalization. Each comparison uses a common detector plane. Each field panel
has its own linear physical colorbar so weaker fields remain visible; compare
the colorbar numbers and line profiles for amplitudes, not colors across panels.
The coarse FEM reconstruction appears separately as an unresolved mesh
diagnostic. Tables inherit the notebook theme rather than forcing white backgrounds.
Its propagation backend is **Cartesian FFT ASM**, labeled in the plots; the
standalone Fourier `mode="polar"` is not wired into the operational Q inverse.
The revised small-body runs use FFT width 24 times the clear radius and 256²/384²
samples; weak macroscopic cases use width 12 times the clear radius. A separate
fixed-spacing window study compares widths 3a, 12a, 18a and 24a on the same probes.
Field maps render the sampled values without Gouraud smoothing. For reference
comparisons the shared detector is selected from Mie or BEM when available.
Comparison maps use the **hot** palette without percentile clipping. BEM samples
are cached with their numerical inputs and implementation fingerprint so plotting
changes can reuse the same field values.
All **Ours** rows now use the unmodified operational field. The former
`incident + raw - equal-index control` addition is retained only in the explicitly
labeled standalone historical sphere audit; its illustration has been removed
from notebook 07. It is not part of the propagation method.
Our method is run with the third curvature order (`OrderInteraction(3)`) in every
full-wave comparison. Lens edges keep the light passing outside a lens; the
sphere is followed behind a declared 0.45R stop, so its comparison with the
whole-sphere references includes that illumination difference. Solved BEM
currents are cached in `output/bem_cache` and reloaded on later runs. The earlier
power-audit, flux-closure and finite-beam sections were removed with the
deprecated fitting options (archived in `output/deprecated_2026-09-23/`).

Notebook 09 revisits the repeated downstream spots in the earlier notebook 07 D=4λ0
plano-convex image. It distinguishes the physical FFT-window width from sampling
interval and displayed pixel density, retains corrected and uncorrected fields,
and reports the remaining BEM mesh uncertainty. Read its
[field analysis](POSTFOCUS_FIELD_STUDY.md) before interpreting the original
small-window maps as isolated-lens predictions.

The export command builds [a local figure gallery](../output/applications/index.html) directly from retained outputs, without rerunning propagation. The output directory is ignored by Git and can be regenerated.

The experiments cite their inspiration: Diffractio's calibrated Jones-mask example and Ansys's polarization and lithography examples. Those references guide the experiments; they are not validation against either product.

## Figure quality and compensation controls

Automatic glyph placement resolves bright cores and polarization changes before spending the display budget on weak structure. Refinement preserves nested Cartesian positions or exact polar rings. Intensity-based sizing is enabled by default; `scale_by_intensity=False` disables sizing independently of `adaptive_density`. All glyphs are grayscale.

Notebook 08 derives the electric Fresnel amplitudes, the normal-power multipliers, the inverse pupil, equal-input-power normalization and the resulting focal-field identity. The control is constant-amplitude circular illumination on the same annulus. An ideal lossless high-index dielectric (n = 2.4) transmitting into air at NA = 0.995 makes the polarization aberration substantial: the uncompensated complex-vector shape error is 20.995%. The normalized inverse recovers the ideal focal shape to numerical precision, while throughput decreases from 68.915% to 55.323%. These values describe the stated ideal material model, without identifying a specific optical glass. A common absolute intensity scale, normalized profiles and complex-vector shape errors separate power loss from distortion. Pupil quadrature is refined independently of that algebraic identity.

The focal comparison in 08 uses the same automatic density and intensity sizing as other maps, with a polar layout. Its shared absolute intensity scale is linear, with zero mapped to black.
