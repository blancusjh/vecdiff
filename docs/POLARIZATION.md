# Polarization maps

The general library recovers the ellipse geometry, tangent rotation markers,
contrast halos and scalar angle maps from
`origin/claude/transfer-matrix-geometric-factor-664h9g` and the corresponding
Claude branch renderer. It adapts them to the explicit field/grid architecture.
The original branches and the separate stigmatic library are unchanged.

```python
import numpy as np
import matplotlib.pyplot as plt

from vecdiff_general.visualization import (
    plot_polarization,
    plot_polarization_scalar,
    plot_intensity,
    animate_polarization,
)

axis = plot_polarization(field)
axis.figure.savefig("polarization.pdf")

plot_polarization(field, density=1.25, crop=False)
plot_polarization(field, adaptive_density=False)  # uniform spatial density
plot_polarization(field, distribution="polar")  # exact concentric display rings
plot_polarization(field, scale_by_intensity=False)  # uniform glyph sizes
plot_polarization_scalar(field, "ellipticity")
figure, axes = plt.subplots(1, 2, figsize=(12, 5))
for axis, current in zip(axes, (incident, transmitted)):
    plot_intensity(current, axis=axis)
animation = animate_polarization(field, frames=48)
plot_polarization(field, phase=np.pi / 4)  # electric arrow from the glyph center
```

The [physical notebook](../general/docs/notebooks/06_polarization_of_vector_beams.ipynb)
shows radial and azimuthal focusing, longitudinal E, finite-pupil diffraction rings
and embedded animations. TIR retardance is in the consolidated planar notebook 01.
Notebook 05 compares the matched spherical and stigmatic lenses on their common
plane, then animates circular illumination through the same stigmatic lens.

Intensity backgrounds use **hot or inferno**. The default is inferno. Glyphs,
arrowheads and outlines are strictly grayscale. The default `color_by="uniform"`
uses white glyphs with neutral dark outlines; handedness is carried by the arrow
direction. Optional `color_by="handedness"` uses gray levels, not colored glyphs.

## Coordinates and physical meaning

The functions accept `ElectricField`, `TransverseElectricField`, or an evaluated
`ElectromagneticField` on a `CartesianGrid` or `PolarGrid`. Complete vectors are
projected onto the grid frame's u,v axes. Transverse inputs already use this local
basis. A polar grid changes node positions, not the vector-component convention.
Axes show local micrometres, so translated and tilted planes retain their meaning.

For `plot_polarization`, the background is projected electric amplitude squared, |Eu|² + |Ev|² in V²/m².
It is neither irradiance nor total |E|² when a normal component exists. Use
`poynting()` for energy flow and component maps to inspect normal E. Glyphs show
the local polarization state. Their maximum radius depends on spacing; by default
their radius also carries a compressed intensity cue with a visible floor.
It is a display scale, not a physical length or a proportional amplitude scale.

Positive S3 = 2 Im(Eu* Ev) rotates from u toward v under exp(-iωt). Tangent arrows
follow this convention. Static rotation heads have their tips on the ellipse at
parameter angle π/4, away from the major-axis endpoints. Their bases follow the
preceding arc, so thin ellipses receive smaller heads. States with
|minor/major| ≤ 0.04 have two
outward heads to show linear oscillation without implying a rotation sense.
Orientation ψ is masked at circular states, where the major axis is undefined.
The ψ scale spans ±90° (the endpoints describe the same axis). Ellipticity χ uses a fixed ±45° scale. Both use inferno; glyphs use grayscale.
Neither angle is interpolated across its branch cut.

## Intensity overlays and incoherent illumination

`plot_intensity(field, axis=axis)` renders total |E|², including the normal component, and superposes the field's transverse polarization. Supply a grid-shaped second argument and an explicit `label` for |Ez|², flux or a scalar-reference intensity. Set `polarization=False` whenever the panel's purpose is intensity comparison rather than polarization analysis. A dedicated polarization panel should use the same complete vector field for both its total-intensity background and its glyphs. A scalar model by itself has no polarization.

Canvas layout, panel titles and shared `Normalize`/`PowerNorm` objects belong in notebooks. The general library no longer supplies `compare_polarization` or `plot_components`. For controlled comparisons, establish identical subplot sizes before plotting; automatic glyph spacing uses the space available at that time.

Both `plot_polarization` and `plot_intensity` accept a list of mutually incoherent fields with nonnegative `weights`. The fields must share wavelength and physical grid coordinates. Weights are used as supplied, not silently normalized. Stokes values add statistically, rather than averaging phasors. Ellipse shape describes the polarized component and opacity is its projected degree of polarization, sqrt(S1²+S2²+S3²)/S0. Unpolarized states have no ellipse. An incoherent mixture has no single optical phase and cannot be phase-animated through this interface.

## Automatic settings

1. Project the field and exclude nodes below `min_intensity_fraction` of its peak
   projected norm (default 0.002, or 0.2%). The threshold is an intensity fraction.
2. Crop to the bounding support, with padding. Displaced and disconnected lobes
   remain in the view. `crop=False` preserves the full sampled window.
3. Choose figure proportions from the visible coordinate span. Existing axes are
   respected; their measured space determines glyph spacing. Narrow beams get
   short, wide figures, and colorbars follow the actual plot height.
4. Follow the grid chart with `distribution="auto"`: a Cartesian grid gives exact
   rows and columns at integer grid strides, anchored near the intensity centroid.
   A polar grid gives exact native rings about its geometrical origin, with
   periodic angular steps chosen from the sampled azimuths. The origin appears
   at most once. Surface area weights and a spacing floor in physical figure
   units set the density. A dark position stays empty; it is never shifted toward
   a bright neighbour. Intensity masking may leave gaps in rows or rings.
5. Set the finest spacing from the area above 25% of peak intensity, rather than
   the entire faint support. Figure size sets a 5.5-point minimum pitch at unit
   density. Bright cores and sufficiently bright changes in normalized Stokes
   state receive fine detail; other positions use nested steps of two or four.
   Cartesian refinement includes adjacent cells. Polar refinement acts on whole
   rings with periodic angular subdivisions, and retains sampled secondary
   radial maxima. Uniform states use fewer glyphs. `adaptive_density=False`
   disables this refinement without disabling intensity-based sizes.
6. Increase spacing if necessary to obey `max_glyphs`, rather than discarding the
   faintest lobes. Keep whole glyphs clear of the plot edges and inside the polar
   disk or annulus. White glyphs with thin neutral dark outlines maintain
   contrast over hot or inferno backgrounds. Legends are offset in points so their
   spacing does not depend on the physical aspect ratio.

`density` adjusts the bright-core target and physical figure pitch; `max_glyphs` is a ceiling (640 by default), not a requested
count. `figsize` overrides automatic dimensions when creating a figure. Placement
is chosen at plotting time; call the function again after materially resizing a
figure. `background=False`, `legend=False`, `colorbar=False`, and
`color_by="uniform"` support custom compositions. `intensity_gamma` affects only
background contrast (default gamma 0.3 reveals secondary maxima). Smooth intensity
rendering does not improve field sampling.

`distribution="cartesian"` or `distribution="polar"` explicitly chooses the display
layout. When this differs from the grid chart, the renderer linearly interpolates
the **complex Cartesian components**, then computes the local polarization.
It never interpolates Stokes angles and never extrapolates beyond the input
domain. Polar interpolation closes the azimuthal seam periodically. Display
interpolation is an approximation on the existing samples; it does not refine
propagation or alter the field. Exact display rings on a Cartesian source are
centered on the area-weighted intensity centroid. Native polar rings always use
the chart origin. Static views and animations evaluate the same display phasors.

## Intensity and glyph size

For projected intensity `I = |Eu|² + |Ev|²`, the default radius is

```text
radius = maximum_radius * [0.5 + 0.5 * (I / I_peak) ** 0.25]
```

This fourth-root compression distinguishes brighter states while keeping weak-ring
polarization readable. At 1%, 0.1%, and 0.01% of peak intensity, glyph radii are
65.8%, 58.9%, and 55.0% of their spacing-dependent maximum. The intensity threshold
still determines whether a glyph is drawn; the size floor never makes a dark
state visible. Size changes preserve the centers, ellipse axis ratio, handedness
and phasor phase.

| Display choice | Options | Weak-ring behavior |
| --- | --- | --- |
| Uniform sizes | `scale_by_intensity=False` | Clearest shape comparison, no size cue |
| Radius proportional to intensity | `size_power=1, min_size_fraction=0` | Secondary states quickly become invisible |
| Radius proportional to amplitude | `size_power=0.5, min_size_fraction=0` | Less severe shrinkage, still too small in faint rings |
| Fourth-root compression, default | `size_power=0.25, min_size_fraction=0.5` | Readable weak states with a gradual brightness cue |

These are radii, not areas. Both intensity and the maximum radius are normalized
per plot, so glyph size should not be used to compare absolute brightness across
different panels. The background retains its labeled quantitative scale.
The gallery's `intensity_size_comparison.png` holds the field, layout, axes and
density fixed across all four choices. `intensity_sizing.json` records the ratios.

Paired plots are composed in notebooks. Controlled comparisons supply a shared
normalization and physical coordinate window. Glyph placement is determined by
the field on that window, not by pixels in an enlarged raster.

`plot_ellipses` remains available. Its `relative_threshold` retains its previous
amplitude-fraction meaning. Its optional `stride` restricts eligible input nodes
for native layouts; physical spacing can thin these further. An alternate
interpolated layout uses physical spacing instead. The new API does not recreate
the old field resampling layer, phase-color controls or the ambiguous single-headed
orientation quiver.

## Phase and animation

`plot_polarization(field, phase=...)` draws the electric vector from the fixed
glyph center to Re(E exp(-i phase)), using the same semi-major-axis normalization
as the ellipse. The arrowhead aligns with the shaft, with its base symmetric
about that shaft. Short arrows receive smaller heads; they never extend behind
their origins. At a linear turning point the arrow has full length; at a zero
crossing it vanishes. No point markers are drawn. The ellipse trace is fainter
than the arrow so the fixed origin stays visually clear.
The shaft and head are one filled polygon with one continuous outline. Static
linear glyphs use the same construction with two heads; no separate shaft stroke
can cross into a head or leave a dark seam at its base.
The default `phase=None` uses the static, geometric markers described above.

`animate_polarization` advances this phase over a cycle, preserving local phasor
phase differences. Its optional `phase` sets the initial phase in radians. The
phase readout appears above the plot, where it cannot hide an outer glyph. The
movie shows monochromatic oscillation, not a pulse moving through the system.

## Verification and visual review

```sh
.venv/bin/pytest -q general/tests/test_polarization_views.py
.venv/bin/python validation/polarization_gallery.py
.venv/bin/python scripts/notebooks.py --pattern 06_polarization
```

The gallery writes PNG and PDF figures plus an HTML index under
`output/polarization/`. Twenty-two cases cover linear, circular, elliptic, radial,
azimuthal, annular, displaced faint lobes, narrow, dark and uniform fields,
Cartesian/polar grids, rotated frames, scalar maps, diffraction rings, phase
snapshots, small subplot panels, enlarged arrow details, controlled size-law
comparisons, exact rows/rings and both layouts inside an annular domain.
Retained physical notebook figures are included when available. The gallery also
exports the embedded movies as HTML and three frame snapshots from each movie.

Visual inspection led to three revisions: colorbars now follow the actual axes,
legend spacing is independent of aspect ratio, and narrow figures avoid excess
whitespace. Rotation heads were enlarged and edge glyphs are kept inside the
view. The automated checks verify rotation sign, phase/frame invariance, minimum
separation, count bounds, weak-lobe retention, intensity threshold semantics,
polar-origin deduplication, undefined orientation masking, animation phase and
legend clearance. Additional checks cover off-axis static heads, two-way linear
markers, secondary Airy maxima, fixed arrow origins throughout a full cycle,
head alignment and symmetry, absence of point markers, linear zero crossings,
agreement between phase snapshots and animation, symmetric placement around a
central null, and single-outline linear glyphs. Layout checks verify exact
Cartesian strides and polar radii/angles, whole-glyph annulus containment,
complex interpolation and its periodic seam, no extrapolation, monotonic size
compression without moved centers, and static/animation agreement for both layouts.

A direct spectral check on notebook 06's finite-pupil radial focus (768 × 768,
80 nm spacing) compared linear display interpolation at 120 polar positions with
evaluation of the same plane-wave expansion at those positions. The transverse
complex-field relative RMS difference was 0.175%. This qualifies display
interpolation for that example, not propagation convergence or other sampling
choices. The measurement is recorded in `validation/results/current_qualification.json`.

The quality revision adds checks that glyph strokes/fills remain grayscale, that bright structure retains the fine lattice while dim regions are thinned, and that narrow bright stripes receive multiple resolved glyphs. The full notebook collection is rerun after renderer changes. Historical interpolation measurements above refer to their stated grids and display positions, not a certification of every later layout.

Dense linear arrows retain a visible white shaft inside a thin gray outline. Native polar annuli start their display-ring selection on an interior grid ring, allowing complete glyphs to fit inside narrow illuminated bands without shifting them off their native coordinates. Both cases have regression tests.

Linear glyphs have a quiver-like shaft and clearly separated triangular heads. In static maps, two heads indicate an oscillation axis without inventing a phase direction; phase animations use one arrow from the fixed center. Each head is 24% of the static full length, with a width four times the shaft width. Ellipse strokes scale with drawn size so small ellipses do not become solid blobs.
