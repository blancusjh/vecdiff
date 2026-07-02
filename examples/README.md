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
| `lithography.py` | Synthetic lithography diagnostic mask through two diopters, scalar vs vectorial. | `lithography_pattern_check.png` |

`_common.py` and `_output.py` are shared helpers, not runnable examples.

Earlier exploratory scripts (aperture sweeps, spot-size metrics, scalar vs
vectorial comparisons, method comparisons) were removed; that line of work now
lives, consolidated and documented, in
[`investigation/focal_plane_aperture_study/`](../investigation/focal_plane_aperture_study/CONCLUSIONS.md).

## Output Layout

The intended structure is:

```text
examples/output/
  cartesian_simple/
  circular_simple/
  lithography/
```

Older local runs may still have legacy generated files in `examples/output/`,
`examples/vecdiff_results/`, top-level `output/`, or
`output_circular_pupil_resolution/`. Those are historical artifacts, not the
current convention.
