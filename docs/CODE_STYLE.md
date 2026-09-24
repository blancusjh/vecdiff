# Readability and physical traceability

Code is written for a reader following the optics. The package structure follows
the physical ontology, and each file has one clear responsibility.

- Use four-space indentation, spaces around operators, and an 88-character
  formatting target. Separate logical steps with blank lines.
- Separate input preparation, validation guards, geometrical construction,
  coefficients, operator construction, and application with blank lines. Put
  visible space before and after loops and guard blocks when adjacent to other
  statements. Do not pack a function into a page to minimize its line count.
- Write one operation per statement. Avoid semicolon-separated statements,
  compressed conditionals, and positional tuples whose physical meaning changes.
- Prefer names such as `incident_direction`, `normal_wavenumber`, and
  `geometric_amplitude`. Conventional mathematical names are appropriate in a
  small, documented derivation, not throughout an entire propagation pipeline.
- Keep geometry construction, material assignment, field representation,
  boundary laws, propagation, observation, and plotting in their own modules.
- Document units, array shapes, coordinate bases, phase convention, and the
  approximation at public numerical boundaries.
- Split numerical work into the corresponding physical steps. Explain a sign,
  Jacobian, or stability choice where it enters the formula.
- Use immutable public inputs and avoid material or sampling state duplicated
  inside electric-field samples.
- Build notebooks around physical phenomena: introduce the geometry and
  illumination, calculate through the public API, visualize fields and
  observables, and explain what changes when physical parameters change.
  Keep detailed regression checks in tests; state model approximations in
  the example where they affect interpretation.

Run both checks before accepting a change:

```sh
.venv/bin/python -m ruff check general stigmatic validation scripts
.venv/bin/python -m ruff format --check general stigmatic validation scripts
```

Formatting is a minimum requirement. The numerical algorithms also need a human
readability review; automated formatting cannot repair confused responsibilities
or unexplained physics.

A mathematical transform must not acquire an optical dependency for convenience.
An operation should be a function or an operator; do not introduce a class merely
to bundle intermediate arrays. Transmission and reflection are independently
requestable operations. Numerical tests do not establish that these conceptual
boundaries or the presentation are appropriate; review them separately.
