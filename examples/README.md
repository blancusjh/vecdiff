# Curated scientific examples

Run from the checkout root after installing `.[examples,validation]`. Each
module exposes `run()` for notebooks/tests and a guarded command-line entry
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
| `sphere_resonance` | Test a self-consistent closed field against Mie | Held-out boundary conditions, bulk E/H, flux, worst-point source refinement |

The sphere example is the slowest (roughly a minute on this development host).
Its native wavelength scan is intentionally identified as sparse, not a
certificate of peak resolution. The [notebooks](../docs/notebooks/README.md)
provide interpretation and suggested convergence experiments. The
[migration record](../docs/migration.md) explains consolidations and retirements.
