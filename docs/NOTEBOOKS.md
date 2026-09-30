# Notebooks

Seven executed notebooks in [`general/notebooks/`](../general/notebooks/) study
optical phenomena with the general library. Each states its physical model,
checks its numerical convergence and keeps its figures. Field maps are linear,
with physical colorbars; polarization glyphs appear only on figures about the
polarization of a complete vector field.

| Notebook | Physical question |
|---|---|
| [01 · Planar dielectric optics](../general/notebooks/01_planar_dielectric_optics.ipynb) | Reflection and refraction, Brewster and critical angles, evanescent fields, Fabry–Pérot layers, frustrated TIR, slab confinement and TIR retardance. |
| [02 · Diffraction by apertures](../general/notebooks/02_diffraction_by_apertures.ipynb) | Slits, circular apertures, fringes and the Airy limit, with an FFT-window check. |
| [03 · Refraction and focusing by a lens](../general/notebooks/03_refraction_and_focusing_by_a_lens.ipynb) | A spherical and a matched stigmatic (GOTS) lens through R Q⁻¹ T Q R: rays, meridional and focal fields, longitudinal fields, polarization and convergence. |
| [04 · Compensating Fresnel transmission](../general/notebooks/04_compensating_fresnel_transmission.ipynb) | A stigmatic diopter (Cartesian oval, z₀ = −10 mm → zᵢ = 6 mm, NA 0.84096): local Fresnel transmission and energy conservation give p/s transfer weights; dividing the entrance field by them recovers the transverse Airy focus at equal input power. Airy and first-minimum contours, throughput and quadrature checks. |
| [05 · Vector imaging and lithography](../general/notebooks/05_single_lens_lithography.ipynb) | A schematic circuit reticle through a visible single lens and an ideal 193 nm 4× relay with partial coherence; scalar versus vector contrast on vertical and horizontal trace groups. |
| [06 · Our method against full-wave solvers](../general/notebooks/06_full_wave_reference_validation.ipynb) | Small bodies (1–6λ₀): BEM checked against Mie on spheres, then our field against BEM over whole meridional sections and three-lobe focal planes of plano-convex and biconvex lenses, and when rim coupling is needed for thick lenses. |
| [07 · Macroscopic interface operators](../general/notebooks/07_macroscopic_interface_operators.ipynb) | Q, T and the explicit Q⁻¹ against exact Maxwell fields at macroscopic size: a single spherical interface up to ka = 10⁵, a glass ball lens (meridional section and focal plane) against exact Debye p = 1 and full Mie, and the same lens with ray-transport Q⁻¹ against the exact field. |

## Models and interpretation

- **02** prescribes a thin-screen transmission with a finite edge transition; it
  does not solve scattering from a conducting screen.
- **03** uses a 128λ₀ cell; a fixed-spacing study up to 768λ₀ changes its
  fields by less than 6·10⁻⁴ of the peak.
- **04** propagates through the real diopter surface: local Fresnel transmission
  on the oval, energy conservation along ray tubes, and a Bessel (Debye) synthesis
  of the image-space spectrum. No aplanatic objective is assumed.
- **05** uses the [ideal imaging operators](PROPAGATION.md#ideal-aplanatic-imaging).
  They conserve normal power and Maxwell transversality in each homogeneous
  region but do not predict the aberrations of an unspecified objective. The
  lithography source quadrature is refined from 8 to 64 directions.
- **06** compares our forward method with BEM, which also includes reflections
  inside the lens and sidewall scattering; at 1–4λ₀ the differences are mainly
  this model difference. See the
  [element Fresnel number criterion](PROPAGATION.md#validity-element-fresnel-number).

## Run

Install the project (see the README), then register the project kernel once:

```sh
.venv/bin/python -m ipykernel install --user --name logos-optics --display-name "Logos Optics (project interpreter)"
cd general/notebooks
../../.venv/bin/jupyter execute --inplace --timeout=-1 03_refraction_and_focusing_by_a_lens.ipynb
```

Rerun a notebook after library changes; executed outputs are stored in the
notebook. Notebooks 06 and 07 cache expensive data (BEM currents and reference fields)
in the git-ignored `.cache/` at the repository root.
The caches are keyed by numerical inputs and solver sources, so stale entries
are never reused; delete `.cache/` to reclaim space. Notebook 06's references
come from `general/notebooks/lens_references.py`.
