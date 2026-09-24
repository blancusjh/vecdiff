# Post-focus structure: local flux factor versus FFT-window size

The supplied notebook07 screenshot is revisited in
[notebook09](../general/docs/notebooks/09_local_flux_and_postfocus_fields.ipynb).
Its reusable calculation is
[postfocus_field_study.py](../general/examples/postfocus_field_study.py).

## Diagnosis for the supplied D=4λ0 plano-convex example

The repeated bright spots in |E|² and patchy downstream |Ez|² are strongly
affected by **physical FFT-window width**. Doubling the sample count at the
original width leaves them almost unchanged. Widening the window at the same
sample spacing removes the prominent later axial peaks over the observation
interval. Six local flux-correction steps do not remove the pattern at the
original width, and one of the largest later peaks becomes brighter.

This is deterministic interference from the finite periodic representation,
not evidence of physical turbulence. The FFT lattice has kx=2πj/L, so
E(x+L,y,z)=E(x,y,z). In this example, translating probes by one cell changes
the field by only 7.7e-15 relative. Direct and NUFFT evaluation agree to
7.3e-13. Changing the evaluation backend or displayed pixel density preserves
that same field and its unwanted replicas.

The diagnosis is supported by controlled window changes, not simply visual
smoothness. For background on angular-spectrum sampling restrictions, see
[Matsushima and Shimobaba (2009)](https://doi.org/10.1364/OE.17.019662).
No propagation filter, Gaussian smoothing or image denoising is applied here.

## Quantitative comparison

The lens has aperture radius a=2λ0, thickness 1.2λ0, exit radius -8λ0, index
1.5 and incident electric amplitude 1 V/m. λ0=532 nm. The original chart is
3a=6λ0 wide; the widest displayed chart is 18a=36λ0. Both use spacing
Δx=0.125λ0. The common detector is the sampled BEM axial maximum at
z=4.1205λ0 (2.192 µm). The downstream region starts at z=6.1205λ0.

E and Ez gaps are relative complex-field sample norms over that meridional
region, against the refined but still unconverged BEM solution.

| Window L/λ0 | N per axis | Method | Downstream E gap | Downstream Ez gap |
|---:|---:|---|---:|---:|---:|
| 6 | 48 | Ours | 58.94% | 139.85% |
| 6 | 48 | Ours + factor | 58.17% | 137.45% |
| 6 | 96 | Ours | 58.75% | 140.04% |
| 6 | 96 | Ours + factor | 57.98% | 137.73% |
| 36 | 288 | Ours | 22.45% | 24.76% |
| 36 | 288 | Ours + factor | 21.12% | 23.39% |

At the original width, the later axial intensity peak near z=12.7λ0 increases
from 8.15 to 8.51 V²/m² with the factor. Widening the window removes this
distinct secondary maximum. The factor's improvement after padding is modest;
it does not explain removal of the repeated pattern.

The raw transmitted field, before the benchmark's bypass correction, also
has late peaks at the narrow width and loses those peaks at the wider width.
The effect is not solely a consequence of the comparison convention.

## Qualitative assessment

### Meridional section

At the original width, the field has additional axial maxima, repeated
off-axis patches and extensive downstream longitudinal structure. The local
factor shifts their amplitudes and positions slightly but retains the pattern.

At the wider width, the central maximum is followed by a much smoother
downstream envelope. Ez retains its two principal longitudinal lobes and a
decaying distribution instead of the original patchwork. Those lobes are
physical vector structure; they should not be suppressed to make the field
look scalar.

Wider-window fields and BEM have more similar spatial organization, but their
amplitudes and complex fields still differ. Window refinement does not supply
returning waves, rim scattering or other omitted full-scattering physics.

### Common detector plane

The total electric field retains a central spot. Ez has two lobes aligned
with the incident linear polarization. The factor does not cause a major
change in spot topology in this example. Increasing the window suppresses
the repeated stripe-like structure surrounding the longitudinal lobes.

All methods use identical detector coordinates and one color scale per field
quantity. There is no separate peak alignment or amplitude normalization.
Figures use actual 321×161 meridional and 161×161 detector-plane samples.

### Other scenarios

- **Biconvex, D=4λ0:** narrow-window secondary bright regions remain with the
  factor. The downstream E gap against BEM changes from 32.39% to 33.30%,
  and Ez from 90.79% to 95.12%. Widening removes the prominent repetitions;
  the corresponding wide-window E gaps are 15.85% and 15.69%. The factor
  barely changes the spatial organization there. The last BEM E refinement
  change is about 4.9%, so the small difference between the factors is not
  precision evidence.
- **Astigmatic freeform, D=4λ0:** narrow-window repeated patches survive the
  factor; the wider window has one dominant focus and a smoother tail. The
  detector retains its central spot and longitudinal lobes. This is a
  numerical window comparison without an independent full-wave reference.
- **Plano-convex, D=2λ0:** narrow-window lateral copies of the focal spot are
  visible near the detector edges. Widening removes them and the large late
  axial oscillations. The factor worsens the wide-window downstream E gap
  against BEM from 8.89% to 9.63%, and Ez from 37.65% to 43.35%. The last
  reference refinement change is about 0.32%. The earlier benefit on a few
  near-exit probes therefore does not extend to this downstream region.
- **Plano-convex, D=20λ0:** narrow-window axial ripples and crossed Ez fringes
  remain with the factor. The wider window yields a smoother beam envelope
  and principal longitudinal lobes, with diffraction structure still present
  near the lens. No independent full-wave reference is claimed.
- **Weak plano-convex, D=200λ0:** the factor makes almost no visible change to
  the narrow-window patchwork around the detector's Ez lobes. Wider windows
  produce much cleaner lobes and beam envelopes. The 9a→12a window change
  still affects Ez appreciably, so this is not a convergence certificate.

### Observation-grid control

[postfocus_observation_check.py](../general/examples/postfocus_observation_check.py)
increases the z count from 321 to 10,241 while holding spectra fixed, on the
axis and an off-axis line that probes Ez. For D=200λ0 it finds seven late axial
maxima in the narrow-window field instead of five at the map's sampling, and
zero in the wider-window field at both densities. The original on-axis main
peak misses less than 0.85% at narrow width and about 0.021% at wider width.
The tested off-axis Ez maximum differs by about 1.03% at narrow width and
roundoff at wider width. Counts of weak ripples require sampling care; the
causal window diagnosis survives this independent check. These two lines do
not certify the entire 2D observation grid.

## Reference and model limitations

All these periodic spectra can satisfy the homogeneous Maxwell equations and
conserve power over their numerical period. Those checks do not establish that
the field represents one isolated lens. A scalar amplitude update followed by
projection onto the same FFT lattice preserves that lattice's repetition period.

For the D=4 plano-convex case, BEM is refined through 810, 1530 and 2988
unknowns. Electric-field changes on common meridional probes are 17.1% and
10.4%. This is not a converged quantitative reference. Refinement changes peak
intensity appreciably; the earlier coarse BEM image is not exact ground truth.

As in notebook07, the comparison uses E_inc + E_lens - E_equal-index-pupil
to approximate bypass illumination around a closed body. The primary API
instead has absorbing pupils. Gaps include that model difference and omitted
multiple scattering, not only numerical errors.

The correction is applied at the last interface with preceding stages fixed.
Its target includes the prescribed incident-plus-reflected normal flux, and
its returned fields are Maxwell spectra. It does not change the spectral
lattice or repetition period.

The widest charts are numerical controls, not universally safe window sizes.
Required padding depends on propagation distance, angular content and the
observation region. A smooth map is not a convergence certificate.

## Cases and reproduction

The study includes plano-convex D/λ0=2,4,20,200, biconvex D/λ0=4 and an
astigmatic freeform D/λ0=4. BEM is run for the two small plano-convex cases and
the biconvex lens. Other cases explicitly use the widest-window operational
field as a numerical comparator, not ground truth. Each has geometry,
meridional maps, common detector maps, line profiles, grouped tables and timing.

```sh
.venv/bin/python general/examples/postfocus_field_study.py
.venv/bin/python scripts/notebooks.py --pattern 09_local_flux
.venv/bin/python scripts/notebooks.py --check --pattern 09_local_flux
```

`output/postfocus_field_study/` contains complex fields, reference refinement
checks, numerical controls, timings and PNG figures. Source fingerprints
invalidate the numerical cache after study or library changes. The notebook
wraps long figure titles and uses tables that inherit the notebook theme.
