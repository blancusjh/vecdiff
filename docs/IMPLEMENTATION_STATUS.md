# Implementation status

The fresh project now contains two independently installable libraries, ontology-based folders, Fourier and propagation modules, executed notebooks, and external physical qualification. The original repository is preserved. This document supersedes the earlier flat, limited implementation status.

## Implemented and exercised

| Area | Delivered behavior | Evidence |
|---|---|---|
| Space and matter | Frames, typed Cartesian/Polar grids with explicit array order, graph/aspheric/oval geometry, explicit lens rim, piecewise spatial n(r), optical prescriptions | Geometry/material tests, JSON round trip |
| Electric ontology | Complete E and explicit transverse inputs; no field medium or duplicate sampling; consistent evaluated SI E/H | Plane and source tests; physical field visualizations |
| Fourier | Common transform/inverse with Cartesian/polar modes; radial and angular operations together in polar.py; optional NUFFT | Parseval, analytic Gaussian pairs, complex-vector backend comparisons |
| Homogeneous propagation | Vector ASM, explicit paraxial Fresnel, branches, evanescent decay and physical output grids | Dispersion, translation and energy tests; aperture diffraction notebook |
| Plane interfaces | Local Fresnel law, reflected/transmitted modes, total fields on either side | Four boundary conditions, Brewster, TIR, rotation/reversal and power |
| Coherent planar systems | Stable all-orders slab recursion with internal E/H | Airy resonances, standing-wave flux, thin/thick evanescent gap |
| Curved physical optics | Per-mode local transformation, equivalent currents, full dyadic Green radiation | Flat/equal-index control, quadrature refinement, reconstructed one-sided residuals |
| Local and forward spectra | Local observation-ball approximation and absolute bounds; propagating Weyl hemisphere | Direct Green/bound comparison; explicit half-space restriction |
| Macroscopic propagation | Separate-mode/eikonal ray transport through multiple faces, spreading and final diffraction | Analytic two-plane amplitudes, modal linearity, ray power and step refinement |
| Stigmatic path | Constant-OPL oval, entrance/exit reference grids, geometric Fresnel transfer, inspectable traces | OPL, Jacobians, power budget, retained legacy geometry/transfer |
| Focal propagation | Direct/Hankel/NUFFT Debye; separate ideal Fourier-pupil Fresnel | Complex E/H agreement, longitudinal parity and independent Bessel sign, Gaussian width and power |
| Observables and presentation | Energy density, signed Poynting power, Stokes/ellipses, animation, coherent/incoherent detector responses | Analytic polarization/energy/fringes and executable plotting notebooks |
| Teaching and verification | 6 physical notebooks, runnable examples, isolated wheels, source formatting/lint | Saved notebook outputs and reports under validation/results |
| Polarization presentation | Exact Cartesian rows or polar rings, whole-glyph domain containment, fourth-root intensity sizing, scalar angles, paired fields and continuous-outline electric arrows | 44 targeted tests, enlarged visual review, vector-beam and stigmatic-focus animations |

## Supported-domain limits

These are explicit model boundaries, not silent fallbacks:

- Initial materials are lossless, nondispersive, isotropic and nonmagnetic. No free current or charge is prescribed in the dielectric scattering models.
- Cartesian ASM uses a periodic plane cell. Evanescent removal and coefficient thresholding are explicit and report removed input RMS. Populated exact grazing completion is rejected.
- Single-interface local Fresnel scattering accepts propagating incident modes. Transmitted TIR evanescence is retained. Coherent layers separately handle internal evanescent waves and their coupling.
- Curved physical optics does not solve an arbitrary closed dielectric body. Finite support, omitted sides, and local-boundary approximation can leave nonzero reconstructed jump residuals.
- Propagating Weyl spectra omit evanescent radiation and apply beyond the source's global-z envelope. Local spectra have a declared observation ball and bounds relative to the supplied discrete currents.
- Forward curved assembly does not include returning reflections. High-frequency transport preserves each input phase but rejects intermediate caustics, transmitted TIR, grazing and vignetting. Numerical aperture support is not automatically an aperture diffraction model.
- Geometrical volume membership includes the specified rim; forward face propagation does not automatically solve scattering from it.
- Stigmatic ovals currently use real axial conjugates, unequal positive indices, a forward transmitted family, and the vertex-connected graph branch. Virtual conjugates and other branches are not claimed.
- Debye fields solve homogeneous Maxwell equations, while their connection to the refracting surface is approximate. The separate Fresnel pupil model is paraxial. Finite detector power is reported without artificial normalization.
- The generic feedback solver certifies its discrete linear equation only. Fully assembled coherent curved-body scattering remains a separate physical implementation and validation task.

## Preservation anchors

| Revision | Purpose |
|---|---|
| `7a1a235` | Dedicated stigmatic reference, including a saved converged physical-current Green calculation |
| `b12a262` | Combined historical revision retained as an additional comparison source |
| `5b8e6f3` | General/macroscopic source for preserving layers, radiation, local spectra and transport ideas |

No legacy branch has been deleted or merged. The saved fixture's JSON records revision, units, quadrature, illumination and the limited claim it supports.

## Further qualification and extensions

1. Map curved-boundary error against wavelength/curvature, illumination, numerical aperture and patch extent, adding independent full-wave references rather than relying on local identities.
2. Qualify coherent curved-interface coupling with a fixed representation, near-field interactions, and all four reconstructed boundary conditions before presenting it as a general Maxwell solver.
3. Extend the dedicated geometry only with explicit conjugate/branch semantics and new reference cases; retain its independent debugging path.
4. Extend material models and volume discretizations when required, with the appropriate dispersion-dependent energy expression and conservation laws.
5. Add higher-level optical/imaging prescriptions from actual source responses after their physical assumptions are specified. Keep FFT cell support, integration patches and physical apertures distinct.

The current implementation is inspectable and runnable across both workflows. These extensions describe additional physical regimes, not promised behavior hidden behind the existing APIs.

## General architecture refinement

`geometry/domains.py` has moved to `volumes.py`; `LensVolume` is now `VolumeBetweenSurfaces`. Mathematical geometry and material assignment remain separate. The dedicated stigmatic implementation is unchanged.

The [current general review](GENERAL_ARCHITECTURE_REVIEW.md) records reproduced findings and their status. FFT constructor invariants and aperture-node bounds are now checked. Surface chart retention, shared material boundaries and axial regularity remain open. The next stage is mathematical geometry, followed by typed surface grids and consistent material topology.

## General Fourier readability refinement

The general mathematical API now exposes `FFT2(g)`, `IFFT2(G)` and
`frequency_grid((x, y))`. Centering, spatial axes and normalization are explicit.
The continuous `transform`/`inverse` interface remains available for Cartesian
and polar coordinates. Modal normalization is applied in propagation, rather
than hidden in Fourier helper names. See [the conventions](FOURIER_AND_GRIDS.md)
for frequency units and migration details. The stigmatic implementation is unchanged.

## General polar and nonuniform refinement

`FT2`/`IFT2` and `HT`/`IHT` provide values-first mathematical operations with
explicit radial measures and separate angular/radial stages. `NUFFT(g, x, q)`
provides a signed unnormalized Fourier sum, using direct evaluation or FINUFFT.
The common continuous interface accepts nonuniform quadrature with explicit
weights. Signs, dimensions, normalization, trailing axes and inverse-accuracy
limits are documented in the Fourier guide and exercised in tests.

## FFT representation and aperture contracts

`PlaneSpectrum` rejects incompatible frequencies, bin order or phase origins.
General modal sums remain unrestricted by FFT layout. Sequential propagation
checks every supplied finite-aperture quadrature's nodes before scattering;
errors identify the conflicting interface. The check uses local surface
coordinates, preserves quadrature objects and weights, and permits explicitly
smaller integration patches. It does not certify full-aperture integration or
introduce an absorbing-stop model. Regression tests exercise rejection and retained physical behavior.

## Physical notebook replacement

The initial replacement removed 13 introductory/diagnostic notebooks. The current collection has six physical notebooks; its execution report records the retained figures and three complete animations. See the [catalog](NOTEBOOKS.md) and current execution report.

## GOTS surface in the general path

The existing Cartesian oval now exposes `from_conjugates(no, z0, ni, zi)`,
`gots`, `z(rho)`, `r(rho)`, `rho_from_r` and placed `parametric_points`.
Notebook 05 adds the revolved surface, stigmatic ray convergence and vector
image-plane fields. See the
[GOTS surface API](GOTS_SURFACE.md) for coordinates and branch limits.

## Matched lens example

The general oval accepts `object_z=-np.inf` for collimated input, with analytic
GOTS limits and a finite vertex-referenced optical path. Notebook 05 compares
matched spherical and stigmatic two-face lenses on common axes and color scales,
keeping diameter, central thickness, illumination and materials fixed.

The matched-lens demonstration now uses aperture radius and beam waist of
48 wavelengths and a central thickness of 20 wavelengths. The larger pupil
exposes spherical aberration while preserving the common geometry constraints
and plotting scales. Surface quadrature is refined to 72×192 nodes.

## Projection, holography and lithography applications

The planar experiments are consolidated into 01. The application collection now has two substantial notebooks: 08 derives and tests Fresnel compensation against constant-amplitude illumination; 10 combines visible projection and ideal DUV imaging. The polarization-selective hologram is a section of 06. Standalone 07, 09 and 11 have been removed.

`propagation/imaging.py` retains the ideal aplanatic operators documented in [IDEAL_IMAGING.md](IDEAL_IMAGING.md). Fourier remains mathematical. The dedicated stigmatic implementation is unchanged.

The renderer uses hot/inferno backgrounds with grayscale glyphs. Fine native rows or exact rings cover bright structure; nested subsets reduce dim-region clutter. Physical assertions and visual review support the examples, but do not establish accuracy for an unspecified manufactured objective.

The revised compensation experiment uses a high-NA annular pupil and explicitly derives its equal-power inverse illumination. Circuit imaging uses the original branch asset, matched horizontal/vertical trace close-ups and source-angle refinement through 64 directions. See the [figure review](FIGURE_REVIEW.md).

## Operational interface propagation

The primary general path now composes planar ASM R, curved transport Q, local Fresnel T and a solved Q inverse. Green methods are isolated in `vecdiff_general.comparisons`. See [the operational method and measured checks](OPERATIONAL_PROPAGATION.md). Notebook 05 is executed with this path. Earlier current-based validation below/above applies only to the separate comparison model.
