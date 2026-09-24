# Original main-path energy deficit: unresolved

The original notebook 07 field deficit has **not been fixed**. This investigation
uses the original plane-wave illumination, wavelength, indices and body shapes.
Fresnel `T` was not changed. None of the experimental corrections below has been
promoted into the library or substituted for notebook results.

## What the implementation actually does

`propagation/high_frequency.py` multiplies both E and H by a disk mask before
**every** Fresnel operation. Outside each disk, the target supplied to
`Q.inverse()` is zero. Thus the implemented chain contains an extra aperture
operator at every interface. The `OpticalSystem` documentation calls these radii
surface support and explicitly says that a radius does not specify an absorbing
stop. The propagation implementation nevertheless treats them as stops.

This is an inconsistency in our implementation and physical problem definition.
It is not a request from the user, and cannot be explained away by changing the
illumination used for the comparison.

For the sphere, `comparison_geometry()` additionally assigns **both** entrance
and exit supports the radius `0.45 * sphere.radius`. The entrance disk therefore
admits only 20.25% of the geometrical cross section illuminated in Mie and BEM.
Its propagated field is clipped a second time at the exit. The displayed grey
sphere does not reveal this reduction in illuminated support.

`Q.inverse()` also projects these clipped targets onto a restricted space of
outgoing Maxwell fields. The existing optional power factor repairs that
projection's integrated flux budget; it neither reconstructs omitted boundary
data nor restores missing coherent field contributions.

## Original-case causal experiments

The experiments are isolated under `output/main_path_investigation/scripts/`.
Run them from the repository root with `.venv/bin/python`. They deliberately
modify operators in an isolated Python process and are not production methods.

The following numbers are the maximum sampled on-axis |E|², in V²/m², over
the **same axial interval and sample coordinates for every variant** in a case.
They are not normalized fields or independently aligned plots. The grid has
256² samples, width 24 times the original support radius. Lens BEM uses the
existing divisor-12 mesh and remains subject to mesh error. Sphere uses Mie.

### Plano-convex lens, D = 4 wavelengths

| Calculation | Maximum sampled on-axis E² |
|---|---:|
| Original main method | 6.378 |
| Suppress entrance clipping only | 5.347 |
| Original path plus an entrance-complement ASM field | 11.291 |
| Replace exterior exit trace with the incident field | 11.256 |
| BEM | 14.231 |

### Biconvex lens, D = 4 wavelengths

| Calculation | Maximum sampled on-axis E² |
|---|---:|
| Original main method | 8.720 |
| Suppress entrance clipping only | 7.968 |
| Original path plus an entrance-complement ASM field | 15.033 |
| Replace exterior exit trace with the incident field | 12.949 |
| BEM | 13.838 |

The exterior-field trials change interference and the axial profile substantially.
They are **not validated repairs**: the main path still omits material-rim
coupling, and no conservation or boundary-condition certificate follows from a
closer peak. These trials must not become a fitted background correction.

### Sphere, D = 2 wavelengths

The original path has maximum sampled on-axis |E|² = 0.191 over the interval
starting 0.08 wavelengths behind the rear vertex; Mie reaches 19.429 there.
Adding the entrance-complement field only reaches 1.205 and does not recover
the near-surface enhancement. This is different from the notebook detector at
z = 0.798 µm, where Mie is about 7: the observation locations must not be mixed.

An additional coverage control holds the exit support at 0.45R and increases
the entrance support. With a fixed 12-wavelength-wide, 256² grid, expanding
the entrance from 0.45R to 0.98R increases the near-exit axial maximum from
0.208 to 3.328. It remains far below Mie. Enlarging both supports can trigger
the existing TIR rejection. Simply selecting a bigger disk is not a complete
or validated full-sphere implementation.

## Does excluding evanescent modes explain the deficit?

An isolated extension of Q includes complex longitudinal wavenumbers and the
corresponding complex adjoint, while keeping the original masks and T. This
is a diagnostic prototype, not an accepted library feature. The narrow interval
0.99k < k_perp < 1.01k is excluded to avoid the grazing singularity of the
transverse-Cartesian representation. Fields are
evaluated by direct modal sums because the current FINUFFT path requires real
wavevectors. Adjoint checks on completed solves have relative errors below
2e-14. This is not a convergence certificate for the physical result.

At 192² samples, width 12 times the original support radius:

| Case | Q spectrum | Maximum sampled on-axis E² | Output power [W] |
|---|---|---:|---:|
| Plano-convex D=4 | Original, sin(theta) <= 0.85 | 6.315 | 3.728e-15 |
| Plano-convex D=4 | Propagating modes to 0.99 | 5.968 | 3.840e-15 |
| Plano-convex D=4 | Propagating + evanescent, k_perp <= 2k | 5.911 | 3.828e-15 |
| Biconvex D=4 | Original, sin(theta) <= 0.85 | 8.621 | 3.804e-15 |
| Biconvex D=4 | Propagating modes to 0.99 | 7.921 | 3.890e-15 |
| Biconvex D=4 | Propagating + evanescent, k_perp <= 2k | 7.811 | 3.890e-15 |
| Sphere D=2 | Original, sin(theta) <= 0.85 | 0.200 | 3.555e-17 |
| Sphere D=2 | Propagating modes to 0.99 | 0.250 | 3.885e-17 |
| Sphere D=2 | Propagating + evanescent, k_perp <= 2k | 0.255 | 3.792e-17 |

Evanescent modes reduce the tangential trace-fitting residual, but do **not**
recover the missing power or field enhancement in these original cases. Higher
cutoffs also become expensive and poorly conditioned. Tightening the solve or
adding these modes is not an established cure for this deficit.

## Conclusion

The statement that the problem is simply a defective Q inverse was too strong.
The current main method models a clipped forward path, whereas the supplied
reference illumination excites a complete dielectric body. Our implementation
introduced that mismatch. The scalar power correction does not solve it.

A valid architectural repair must distinguish a numerical surface chart from
a physical absorbing pupil and account for the field outside each transmitted
path. For a closed body this also requires its remaining material boundaries.
Those fields have amplitudes and phases; conservation alone cannot determine
them. Prescribing zero outside a chart, filling it with incident field, or
rescaling the output are different physical assumptions, not interchangeable
Q inverses. In wavelength-scale cases, the single local-eikonal and forward-only
approximations must also be tested rather than assumed accurate.

The original meridional and focal maps and the new causal profile figure were
visually inspected. The main fields remain underpowered. **There is no accepted
fix in this investigation and notebook 07 has not been relabelled as fixed.**
