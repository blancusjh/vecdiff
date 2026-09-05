# Benchmarks

Run the physical validation sweep with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.validate_physics
```

`results/physics.json` is the committed raw result. It contains plane-interface
Maxwell residuals, reconstructed open-cap residuals for radii `0.5λ` through
`50λ`, and the deliberately incomplete one-encounter sphere diagnostic against
Mie. See `docs/validation.md` before interpreting the sphere errors.

`python -m benchmarks.macroscopic` records independent source/polar/table
refinements, reconstructed boundary jumps, propagating power, time, and process
memory for R=50,100,200 wavelengths in `results/macroscopic.json`.
`python -m benchmarks.assembly_convergence` independently varies the spectral
bandwidth, period, and surface quadrature of the coherent two-cap assembly.
Both studies retain unresolved errors. `python -m benchmarks.plot_macroscopic`
regenerates the README plot from the committed measurements.
