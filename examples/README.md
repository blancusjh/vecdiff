# Curated scientific examples

Run from the checkout root after installing `.[examples,validation]`. Each application module listed below
exposes `run()` for notebooks/tests and a guarded command-line entry
point. Importing an example performs no calculation or plotting.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m examples.cavity_resonance
```

Each command writes a labeled `figure.png` and `results.json` containing
parameters, assumptions, checks, and package versions under
`examples/output/<name>/`. Use `--output <directory>` to choose another location.
Lengths are expressed relative to vacuum wavelength unless a label states
otherwise; H uses the documented Z0 normalization.

| Module | Why it belongs | Acceptance check |
| --- | --- | --- |
| `field_propagation` | Distinguish unspecified Ez from zero and propagate a sampled electric state | Reversibility and integrated normal flux on a propagating-only lattice |
| `plane_interface` | Demonstrate per-k Fresnel reflection, Brewster incidence, and TIR | Reconstructed four-condition Maxwell residuals and R+T |
| `cavity_resonance` | Show why multiple encounters must sum complex amplitudes | Independent Airy amplitude, explicit feedback, standing wave |
| `frustrated_tir` | Demonstrate an effect a single interface cannot capture | Evanescent-gap coupling, thin/thick limits, flux |
| `curved_interface` | Distinguish integral convergence from physical boundary accuracy | Two source quadratures and separately labeled reconstructed jumps |
| `vector_focus` | Compare polarization and vortex fields at equal pupil power | Independent Richards–Wolf model, angular convergence, explicit model limits |
| `interface_assembly` | Construct repeated per-k encounters from physical interfaces | Independent planar-layer agreement; explicit curved feedback and boundary diagnostics |

General resonant-body behavior remains pending. The planar cavity and layer
examples do not establish it. The [notebooks](../docs/notebooks/README.md)
provide interpretation and suggested convergence experiments. The
[migration record](../docs/migration.md) explains consolidations and retirements.

`macroscopic_focus.py`: 10 mm stigmatic conic at 193.368 nm; resolved vector maps, local-spectrum kernel bounds, quadrature and NUFFT comparisons. Requires the `nufft` extra.

`image_formation.py` supplies the explicitly periodic vector convolution and
normalized incoherent-source sum used in the application notebooks. It accepts
a supplied transfer function and does not infer one from an optical prescription.
`notebook_tools.py` contains plotting conventions and measured-width helpers.
The notebooks themselves contain the setup, physics calls, analysis and checks.
