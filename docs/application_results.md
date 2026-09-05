# Results from the optical application notebooks

The six [executed notebooks](notebooks/README.md) contain the actual setup,
calculation, embedded figures and numerical checks. This page is a guide to their
results and the limits of those results. Electric-field norms are distinguished
from transported power throughout:

$$
\mathcal E(\mathbf r)=\|\mathbf E(\mathbf r)\|^2,
\qquad
\langle\mathbf S\rangle=\frac12\Re[\mathbf E\times\mathbf H_{SI}^{*}].
$$

## Main-method calculations

| Experiment | Measured result | What the check establishes |
| --- | --- | --- |
| 532 nm Gaussian beam, 1.2 µm waist | 0.5055% longitudinal electric-norm fraction at the waist; sequential forward propagation agrees with one combined step to approximately $2\times10^{-14}$ | A specified transverse field is completed and can be propagated again; window and pixel controls are included |
| Glass–air plane, $n_1=1.5$, $n_2=1$ | Brewster angle 33.690068°, critical angle 41.810315°; reconstructed plane-wave boundary jumps below $5\times10^{-15}$ | Exact planar Fresnel transformation, Snell refraction and energy balance |
| TIR at 55°, 532 nm | Amplitude penetration depth 118.59 nm; intensity penetration depth 59.29 nm | Nonzero evanescent transmission with the expected exponential decay |
| Stigmatic 24 mm diameter conic at 193.368 nm | NA 0.427353; local versus full-radiation complex E/H error approximately $1.11\times10^{-4}$ at held-out points | Accuracy of the local radiation expansion for the same Fresnel currents; **not** a global dielectric-boundary error |
| Circuit imaging with a 4 mm waist Gaussian illuminating that conic | Doubling the point-response window changes the complex image by approximately 0.002% | Controlled finite-window convolution in an explicitly local isoplanatic experiment |

The finite beam near the critical angle contains both propagating and
evanescent transmitted modes. A single ray angle cannot describe that complete
beam. The notebook evaluates the spectra on both sides and checks all four
Maxwell boundary conditions after their coherent superposition.

![Main method: partial reflection, Brewster-centred incidence, critical incidence and TIR. Every panel uses the same input-amplitude scale and hot colormap.](assets/planar_beam_fields.png)

The lithographic image is calculated from the **complex vector point response**,
not by blurring an intensity image. Each mutually incoherent source is propagated
separately before its component intensities are summed:

$$
\mathbf E_s=\mathbf h*(m e^{i\mathbf q_s\cdot\mathbf r}),
\qquad I=\sum_s w_s\|\mathbf E_s\|^2,\quad\sum_s w_s=1.
$$

The Gaussian illumination is a physical input choice, synthesized from Maxwell
plane waves. It differs from the uniformly illuminated spot study. The notebook
reports source-quadrature sensitivity as well: the coarse 49-point source differs
from the 213-point source by about 4.7% in aerial-image norm, so the coarse source
is insufficient for percent-level intensity work. Image-window convergence does
not establish source convergence or validate the local isoplanatic assumption.

![Main-method point response and its circuit-pattern image, with separate stated normalizations.](assets/lithographic_image.png)

## Independent reference calculations

The reference implementations remain outside `vecdiff/`. They can use its shared
field abstractions; the main implementation never imports them.

**Sphere resonance.** A fixed 2 µm diameter sphere with index 1.5 has a selected
scattering peak near 566.514 nm and a nearby valley at 573.500 nm. The Mie volume
mean of $\|E\|^2/|E_0|^2$ rises from about 1.22 to 2.15 between those two settings.
Extra multipoles change the sampled complex E/H by approximately $10^{-10}$,
and a shrinking-offset boundary check is shown in the notebook.

![Mie reference: off-resonant and resonant electric and magnetic meridional fields. The cyan circle is the physical sphere boundary; logarithmic scales reveal internal and exterior structure.](assets/sphere_resonance_fields.png)

The current single-encounter spectral sphere diagnostic has approximately **62%**
complex-electric-field error in both sampled interior and exterior regions at
the selected resonance. The notebook puts its field beside Mie and includes an
error map. **Closed-sphere resonant propagation remains pending.** These Mie
plots cannot be used as evidence that the main method already reproduces it.

**DUV patent-system pupil reference.** At 193.368 nm, NA 1.2, image index
1.59667693 and the stored 62 mm object field, the defined vector-pupil reference
produces x/y focal FWHM values of approximately 95.53/80.30 nm. The same transfer
is used for the vector PSF, meridional field and circuit image. A TE/TM
line-grating study uses analytic grating coefficients and reports source
refinement; the 421-to-1649-source comparison changes selected TM contrasts by
less than 0.005 in absolute contrast. Pixel and pupil-frequency spacing are
checked separately.

![DUV reference: wafer-scale circuit mask, coherent and partially coherent aerial images, and the three electric-component contributions.](assets/duv_circuit_reference.png)

This is a ray-traced wavefront plus an independent sine-condition pupil model,
with [data provenance](../examples/data/README.md). It assumes uniform entrance
illumination and omits coating losses and pupil transmission. **Main-method
propagation through the complete 48-encounter folded objective remains pending.**
Neither this reference nor the conic experiment certifies a production
lithography instrument, electromagnetic mask model, or resist process window.
