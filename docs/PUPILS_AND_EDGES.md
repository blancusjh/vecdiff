# Surface edges and pupils

## The rule

A surface **edge** (`OpticalSystem.apertures`) is where a material face ends.
It never cuts the field. A **pupil** is an opaque stop declared explicitly
(`OpticalSystem.stops`); it cuts every field reaching it outside its radius.
Where no stop is declared, nothing is cut.

```python
lens = o.OpticalSystem(interfaces, apertures=(a, a))            # bare lens
mounted = o.OpticalSystem(interfaces, apertures=(a, a), stops=(a, None))
```

Stops are stored in JSON prescriptions as `stop_m`. Earlier versions treated
every edge as a pupil. The same result is now obtained by declaring a stop at
the edge.

## What "not cutting" means at a finite lens

The part of the incident wave that misses the glass keeps travelling in the
surrounding medium. It cannot be propagated with the glass index, because it
never enters the glass. `propagate_high_frequency` therefore handles each
**element**, a run of interfaces between visits to the source medium
(identified by material), as follows:

1. At the element's entrance edge, the field inside the edge enters the
   element. The field outside it is inverted in the surrounding medium on the
   entrance chart, which is flat beyond the rim.
2. That outside light is carried across the element by exact ASM and added
   at the element's exit. Later elements and the observation region receive it.
3. Inside the element, light reaching a later chart beyond that surface's edge
   crosses the chart's flat extension with normal-incidence Fresnel, a simple
   model of leaving through the sidewall. It is exact for an index-matched
   element, which is then invisible.
4. A stop cuts everything on its chart outside its radius: light inside the
   element and light passing outside it.

`interface_spectra` keeps each interface's own Q⁻¹ result, and
`exterior_spectra` keeps the light that passed outside each element.

Limits: sidewall reflection and light entering through a sidewall are not
modeled. A system that never returns to its source medium is open; beyond a
finite edge there is no defined medium for the field, so such an edge must
carry a stop no larger than itself, or the call is rejected. An edge inside an
element larger than its entrance edge is also rejected, because light passing
outside would meet that face. `PolarTaylorLens` follows the same rule, with
exact per-mode Fresnel on the flat extension.

Tests (`general/tests/test_edges_and_stops.py`): an index-matched element is
invisible to 1e-12; inside and outside light recombine to the incident wave to
1e-12; a stop at the edge equals the same stop on an unbounded surface; an exit
stop also blocks light passing outside; open systems and growing edges are
rejected; prescriptions keep their stops.

## Why this matters

The transport step is not the large error. Q, from a spectrum to a curved
surface, is exact. Q⁻¹ is a least-squares inverse, because on a curved chart
the sag couples position and spectrum; evanescent waves are excluded to keep it
stable. A trace with a hard rim leaves a 3–7% surface residual, but downstream
three independent propagators of the same trace (electric fit, joint E/H fit,
Stratton–Chu) agree to about 0.2% ([details](CURVATURE_ORDERS.md), section 8).
Treating edges as pupils removed the light passing outside the lens, which was
the leading error against full-wave references.

## Validation

[`pupil_validation.py`](../general/examples/pupil_validation.py) and
[`curvature_fem_comparison.py`](../general/examples/curvature_fem_comparison.py).
On-axis complex electric field, no gain, phase or focal alignment. BEM is the
native dielectric BEM; mesh divisors 12 and 16 differ by 1.7% (plano-convex)
and 0.9% (biconvex). FEM is an external degree-10 solution of the R = 30λ₀,
a = 12λ₀, 5λ₀ lens, used as a reference only. Both references solve a bare
lens in free space, with no mount.

Relative error of the on-axis complex electric field, with and without a stop
at the lens edge:

| Case | Axial field error, stop at edge | Axial field error, no stop |
|---|---:|---:|
| Plano-convex, D = 4λ₀, production, vs BEM | 49% | 39% |
| Biconvex, D = 4λ₀, production, vs BEM | 41% | 20% |
| R = 30λ₀, production 256², vs FEM | 9.5% | 5.0% |
| R = 30λ₀, `PolarTaylorLens` 512², vs FEM | 10.2% | 5.7% |

Peak on-axis intensity \|E\|² in V²/m² for a 1 V/m incident wave:

| Case | Peak \|E\|², reference | Peak \|E\|², stop at edge | Peak \|E\|², no stop |
|---|---:|---:|---:|
| Plano-convex, D = 4λ₀, production | 14.31 (BEM) | 6.38 | 9.89 |
| Biconvex, D = 4λ₀, production | 13.81 (BEM) | 8.72 | 13.80 |
| R = 30λ₀, production 256² | 73.29 (FEM) | 63.2 | 80.1 |
| R = 30λ₀, `PolarTaylorLens` 512² | 73.29 (FEM) | 61.1 | 82.2 |

Axial position of that peak, beyond the entrance, for R = 30λ₀: FEM 52.38λ₀;
stop at edge 50.9λ₀ (production) and 51.3λ₀ (`PolarTaylorLens`); no stop
52.6λ₀ and 52.5λ₀. Integrated power through the focal plane was not compared,
because the saved references give only on-axis fields.

- Not cutting at the edge improves every case, and for R = 30λ₀ it moves the
  focus to where FEM places it.
- For R = 30λ₀ the peak now overshoots FEM by 9–12%. The flat-extension sidewall model
  lets through internal light that a real glass sidewall reflects almost
  totally, and internal returns are absent.
- The small plano-convex lens stays about 39% off. `PolarTaylorLens`, with
  exact per-mode Fresnel instead of estimated ray directions, gives the same
  38–40%. At a = 2λ₀ the sidewall is as large as the entrance face; a
  single-pass model is at its limit there.
- Curvature orders 1–3 change none of these numbers beyond the fourth digit.
- For a lens in a mount, declare the mount as a stop; the former column then
  applies.

## Validity of the single-pass model: element thickness

Our method separates, at each element's entrance edge, the wave entering the
glass from the wave passing outside. Within about √(λ₀T) of the rim, for centre
thickness T, light crosses between the two, and that separation is not exact.
The single-pass model therefore needs a large element Fresnel number a²/(λ₀T).
Measured with 6λ₀ lenses against BEM, which itself agrees with Mie to 5–7% at
this size ([notebook 11](../general/docs/notebooks/11_curved_interfaces.ipynb),
section 3):

| Lens (D = 6λ₀, n = 1.5) | Element Fresnel number a²/(λ₀T) | Axial field error of ours vs BEM |
|---|---:|---:|
| Biconvex R = ±20λ₀, T = 0.6λ₀ | 15.0 | 4.1% |
| Biconvex R = ±20λ₀, T = 0.9λ₀ | 10.0 | 8.1% |
| Biconvex R = ±20λ₀, T = 1.2λ₀ | 7.5 | 18% |
| Biconvex R = ±20λ₀, T = 1.8λ₀ | 5.0 | 53% |
| Biconvex R = ±20λ₀, T = 2.4λ₀ | 3.8 | 55% |
| Four thin lenses, T = 0.7λ₀ | 12.9 | 3.3–3.6% |

Our chain conserves power in every case, so the thick-lens gap is not lost
light: a thick glass cylinder a few wavelengths wide concentrates light behind
it far more than a single pass predicts. The R = 30λ₀ lens of the FEM comparison
has a²/(λ₀T) = 29 and a 5% error. The notebook-07 lenses (a²/(λ₀T) ≈ 3–4) sit
in the unreliable range.

## Reproduce

```sh
.venv/bin/python -m pytest general/tests/test_edges_and_stops.py -q
.venv/bin/python general/examples/q_inverse_downstream.py 256
.venv/bin/python general/examples/pupil_validation.py PATH/TO/curved_interface_research_bundle_2026-09-22
.venv/bin/python general/examples/curvature_finite_lens.py 512 1 free
.venv/bin/python general/examples/curvature_fem_comparison.py PATH/TO/curved_interface_research_bundle_2026-09-22
```

Without the bundle path, `pupil_validation.py` runs only the BEM cases. The
bundle is an external reference, not part of this repository.
