# Clean break and example retirement record

Version 0.3 has no backward-compatibility promise or implementation. No old
field aliases, import forwarding, deprecated facade, or historical package is
shipped. The previous code, figures, tests, and notebooks remain in Git history
(the pre-refactor main revision is `b12a26220972824c424af82a99158a6199ffdae0`).

The former 21 example programs and ten substantive notebooks were reviewed by
purpose. Preserving every filename or old numerical claim would defeat a clean
scientific API. Supported ideas were rewritten or consolidated; unsupported
claims were explicitly retired instead of being made to run through adapters.

| Former examples / notebook topics | New disposition |
| --- | --- |
| Cartesian, polar, and circular basic examples; harmonic animation | `field_propagation`, `vector_focus`, notebooks 01/05. Coordinate and polarization demonstrations consolidated; cosmetic animation retired |
| Longitudinal component; polarization aberrations; radial/vortex/light-needle examples | New equal-pupil-power vector-focus comparison and notebook 05. No unsupported “needle” performance claim |
| Aperture limits; scalar-vs-vector aperture; cross-polarization maximization; deformation-vs-cross | Curved-interface limits and vector-focus comparisons, notebooks 04/05. Optimization and scalar-equivalence claims retired without an explicit objective/detector model |
| Spectral-interface variants; freeform astigmatism | One native per-k interface implementation and documented freeform geometry in notebook 04; no competing legacy chains |
| Reference-vs-model; scalar reference; wave-error scaling | Independent Stratton–Chu test plus reproducible plane/cap/Mie benchmarks. Historical model-specific scalar/stigmatic wrappers removed |
| Nanojet | Closed-sphere field and resonance validation against Mie; no blanket nanojet accuracy claim |
| Two-diopter imaging; resolution inversion; resolved features; two-point resolution; lithography pattern | Retired application-specific imaging claims. A coherent imaging/detector/object model and converged multi-surface propagation are not yet provided |
| Baseline spot metrics; central-spot exploration | Explicit component norms and equal-input comparisons in notebooks 01/05; no automatic interpretation as power or resolution |
| Quabis replication, vecdiff comparison, and diopter notebooks | Replaced by a clearly labeled ideal Richards–Wolf reference notebook; not presented as an experimental replication or exact dielectric objective |

New workflows add what the former suite did not establish: coherent cavity
round trips, stable frustrated TIR, held-out closed-boundary checks, independent
Mie E/H comparisons, and visible failure/convergence cases. No `ScalarField`
class is introduced because no maintained scalar physical workflow requires it.
