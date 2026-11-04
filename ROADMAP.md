# Roadmap

Planned work, ordered by priority. These items describe future work, not current
capabilities or committed release dates. See [validation](docs/VALIDATION.md)
for the evidence and limits of the current implementation.

## Next

- [ ] Expand independent validation for nonspherical lenses with mesh-refinement
  studies and spatially resolved complex E/H error measurements.
- [ ] Measure sensitivity to FFT chart size, sampling and angular-band limits,
  and document practical convergence guidance.
- [ ] Extend repeatable performance measurements to include memory use and
  repeated runs across optical sizes.

## Later

- [ ] Investigate reference solvers that can handle larger nonspherical models
  beyond the current dense BEM capacity.
- [ ] Evaluate the feasibility of the physical-size AL1225 lens after establishing
  resource requirements and validation at smaller scales.
- [ ] Investigate finite edges, stops and returning branches as extensions of the
  forward model, with independent validation before claiming support.
