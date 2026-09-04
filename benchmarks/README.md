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
