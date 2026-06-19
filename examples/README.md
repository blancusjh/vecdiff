# Examples and Outputs

All runnable examples save generated artifacts under:

```text
examples/output/<example_name>/
```

Run examples from the repository root with `python examples/<script>.py`.
Each script prints the path of the files it saves.

| Example | Purpose | Main outputs |
| --- | --- | --- |
| `cartesian_simple.py` | Cartesian-polarized circular pupil propagated through one diopter. | `input_field_components.png`, `propagated_field_components.png`, `input_polarization.png`, `propagated_polarization.png` |
| `circular_simple.py` | Circular-polarized circular pupil propagated through one diopter. | `input_field_components.png`, `propagated_field_components.png`, `input_polarization.png`, `propagated_polarization.png` |
| `measure_of_spot_size.py` | Computes Rayleigh, FWHM, and first-minimum spot-size diagnostics. | `spot_size_metrics.png` plus printed metric values |
| `arbitrary_cartesian_fft_diopter.py` | FFT propagation of an arbitrary Cartesian field through a diopter. | `incident_field_components.png`, `propagated_field_components.png`, `propagated_polarization.png` |
| `aperture_angle_method_comparison.py` | Hankel-vs-FFT comparison over aperture angle and incident polarization cases. | Per-case comparison PNGs, `summary_errors.png`, `summary_errors.csv` |
| `lithography.py` | Synthetic lithography diagnostic mask through two diopters, scalar vs vectorial. | `lithography_pattern_check.png` |
| `resolution_two_features.py` | Minimal two-feature lithography scalar/vectorial comparison. | `scalar_vectorial_comparison.png` |

## Output Layout

The intended structure is:

```text
examples/output/
  cartesian_simple/
  circular_simple/
  measure_of_spot_size/
  arbitrary_cartesian_fft_diopter/
  aperture_angle_method_comparison/
  lithography/
  resolution_two_features/
```

Older local runs may still have legacy generated files in `examples/output/`,
`examples/vecdiff_results/`, top-level `output/`, or
`output_circular_pupil_resolution/`. Those are historical artifacts, not the
current convention.
