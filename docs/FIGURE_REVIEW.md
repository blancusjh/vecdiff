# Figure review · 17 September 2026

The six-notebook collection replaces three standalone application notebooks with focused sections in 06 and 10. The regenerated [figure gallery](../output/applications/index.html) contains the retained notebook outputs; [renderer diagnostics](../output/polarization/index.html) also cover prescribed patterns and animation frames.

## Changes checked visually

- Field backgrounds use hot or inferno. Glyph faces, rotation heads and outlines are grayscale.
- Automatic placement measures the bright core separately from the weak halo and responds to normalized-Stokes changes. Polar layouts preserve every radial station and reduce angular density in weak regions; Cartesian layouts use nested cells. Intensity sizing is on by default, with a fourth-root law and a 50% radius floor so weak diffraction orders remain legible.
- Dense linear arrows have a visible white interior and a continuous thin outline. No point markers are added.
- Notebook 08 uses a shared linear absolute intensity scale and an inferno palette with an exact black endpoint. The former gamma = 0.4 scale lifted weak rings too strongly. Physical nonzero minima are retained, including contributions from longitudinal E.
- The narrow annulus in 08 originally lost every glyph at its boundaries. Selecting interior native rings now fits complete glyphs on the illuminated band.
- Circuit overviews locate two matched close-ups. Vertical and horizontal traces show the different effects of x and y illumination; line profiles make the contrast difference measurable.
- Component and scalar-reference maps have no polarization glyphs. The circuit scalar/x-input/y-input comparisons also omit glyphs so image contrast remains readable. A separate coherent diagonal-input figure shows polarization over the complete vector field's total intensity.

## Physical content

Notebook 08 derives the inverse Jones pupil and its equal-input-power normalization. An ideal lossless n = 2.4 to air interface, NA = 0.995 and annular illumination produce a 20.995% uncompensated complex-vector shape error. The prepared illumination recovers the ideal field up to a constant amplitude (relative identity error 1.97e-15), while throughput changes from 68.915% to 55.323%. Refining the pupil from 385 to 577 nodes per axis changes the focal probes by 0.153%; the inverse identity alone is not a convergence test.

Notebook 10 uses the original branch circuit asset, with explicit crop and reticle scale. The 193 nm, NA = 0.85 reduction model compares scalar, x-polarized and y-polarized images. Increasing incoherent source directions from 8 to 16, 16 to 32 and 32 to 64 changes intensity by 4.528%, 1.926% and 0.769%, respectively. Displayed ensembles use 32 directions. The ideal relay and thin amplitude mask do not represent a manufactured DUV objective or finite-thickness mask scattering.

Validation: 208 tests pass, including 44 polarization presentation tests. Formatting and lint checks pass. Original repository status and the 41 dedicated stigmatic implementation files match their preservation baselines.

In notebook 06, the separate holography x/y channel panels have no glyphs. Only the complete coherent-field panel carries its polarization map over total |E|². A numerical assertion checks E = (E⁽ˣ⁾ + E⁽ʸ⁾)/√2 before rendering.
