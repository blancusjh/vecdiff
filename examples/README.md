# Examples and Outputs

All runnable examples save generated artifacts under:

```text
examples/output/<example_name>/
```

Run examples from the repository root with `python examples/<script>.py`.
Each script prints the path of the files it saves.

## Basics

| Example | Purpose | Main outputs |
| --- | --- | --- |
| `cartesian_simple.py` | Cartesian-polarized circular pupil propagated through one diopter. | `input_field_components.png`, `propagated_field_components.png`, `input_polarization.png`, `propagated_polarization.png` |
| `circular_simple.py` | Circular-polarized circular pupil propagated through one diopter. | `input_field_components.png`, `propagated_field_components.png`, `input_polarization.png`, `propagated_polarization.png` |
## Scalar vs vectorial focusing

| Example | Purpose | Main outputs |
| --- | --- | --- |
| `aperture_scalar_vs_vectorial.py` | Wide-aperture focal field: scalar (t- = 0) vs vectorial maps, radial cuts, cross-polarized clover, focal-plane polarization map. | `focal_plane_maps.png`, `radial_cuts.png`, `focal_plane_polarization.png` |
| `maximize_cross_polarization.py` | Edge-weighted pupil on a high-index diopter maximizing the cross-polarized fraction; cross-channel shape for linear vs circular input. | `cross_polarization_maps.png`, `cross_channel_shape.png`, `focal_plane_polarization.png` |

## Imaging

| Example | Purpose | Main outputs |
| --- | --- | --- |
| `two_diopter_imaging.py` | Two-line mask imaged by conjugate diopters: scalar vs vectorial contrast depends on the mask orientation relative to the incident polarization. | `image_gallery.png`, `profiles.png`, `image_plane_polarization.png` |
| `resolution_inversion.py` | Two canonical point objects resolved by the scalar model but fused by the vectorial one (high-index and ordinary-glass systems, super-resolving edge pupil, zoom-FFT renders). The polarization of each vectorial image is analyzed in three separate figures per system: component analysis (incident vs. image), polarization-ellipse maps (incident field vs. focal plane), and the ellipse-angle maps. | `inversion_{high_index,glass}.png`, `components_{high_index,glass}.png`, `polarization_maps_{high_index,glass}.png`, `ellipse_angles_{high_index,glass}.png` |
| `lithography.py` | Synthetic lithography diagnostic mask through two diopters, scalar vs vectorial. | `lithography_pattern_check.png` |

## Animation

| Example | Purpose | Main outputs |
| --- | --- | --- |
| `harmonic_field_animation.py` | Time-harmonic animation (`vecdiff.animate_harmonic_field`) of the instantaneous real field `Re[E e^{-i omega t}]` for a cross-maximizing edge-weighted pupil. Cartesian (linear x) case *breathes*; circular case *rotates*. Both carry the system parameters and refresh the two README hero GIFs. | `harmonic_field_cartesian.gif`, `harmonic_field_circular.gif` (and `docs/assets/quiver_harmonic_readme.gif`, `docs/assets/quiver_harmonic_circular.gif`) |

`_common.py` and `_output.py` are shared helpers, not runnable examples.

## Output Layout

The intended structure is:

```text
examples/output/
  cartesian_simple/
  circular_simple/
  aperture_scalar_vs_vectorial/
  maximize_cross_polarization/
  two_diopter_imaging/
  resolution_inversion/
  lithography/
  harmonic_field_animation/
```

Older local runs may still have legacy generated files in `examples/output/`,
`examples/vecdiff_results/`, top-level `output/`, or
`output_circular_pupil_resolution/`. Those are historical artifacts, not the
current convention.
