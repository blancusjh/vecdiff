# Brewster devices of revolution

Refractive elements whose surfaces are all met at the Brewster angle by a beam
along their axis. On such a surface p is the radial field and s the azimuthal
one, so each surface lets E_ρ through without reflection loss and attenuates
E_φ. This folder holds:

| File | What it does |
|---|---|
| `devices.py` | Closed forms and surface sequences: Brewster shells, shell stacks, the double-Brewster axicon |
| `meridional_operator.py` | The simple eikonal–angular-spectrum operator on one meridian (vertex-plane refraction operator plus order-1 Hankel propagation) for fields of azimuthal order 0 |
| `hyperbolic_cone.py` | `HyperbolicCone`, a cone with a hyperbolic tip as a vecdiff surface with an exact ray crossing, and the `Doughnut` profile |
| `cross_validate.py` | The operator against `vecdiff.propagation.Propagator` on one shell |
| `run_shell_stack.py` | Nested shells that suppress E_φ of a collimated doughnut beam |
| `run_brewster_axicon.py` | The double-Brewster axicon: a longitudinally polarized needle from circular light |
| `make_figures.py` | The figures in `figures/` |

Nothing in the vecdiff package was changed.

## Closed forms

At Brewster incidence θ_B = arctan n, incidence and refraction angles add to
90°. Every ray is turned by the same angle

    δ = θ_B − θ_t = 2θ_B − 90°,   sin δ = (n² − 1)/(n² + 1),

and the ratio of the s and p amplitude transmissions is

    t_s / t_p = cos δ = 2n/(1 + n²),   with p power transmission T_p = 1.

A sequence of Brewster surfaces therefore rotates the meridional ray field by a
sum of ±δ, multiplies E_φ/E_ρ by cos δ at each surface, and loses no radial
power. With N surfaces and circularly polarized input (equal radial and
azimuthal power), the azimuthal share of the output is c^N/(1 + c^N), where
c = cos² δ.

**Alternating shell.** Two parallel cones of slope n. Their rotations cancel and
the beam stays collimated. The shell shifts each ray radially by
d sin δ / cos θ_t, where d is the wall thickness measured normal to the wall
(0.67 d for n = 2). A spike (A, apex upstream) shifts rays inward and a funnel
(B, apex downstream) shifts them outward.

**Double-Brewster axicon.** An entrance cone of slope n, and an exit surface of
slope tan(3θ_B − 180°) that the refracted rays meet at the internal Brewster
angle. The two rotations add. The output is a converging conical wave of
half-angle 2δ:

    NA = sin 2δ = 4n(n² − 1)/(n² + 1)²,   T_p = 1 at both faces,   E_φ/E_ρ power × cos⁴ δ.

| Index n | Brewster angle θ_B (°) | Exit-face slope | Output half-angle 2δ (°) | Output NA | s/p power ratio per face, cos² δ |
|---|---|---|---|---|---|
| 1.515 (BK7) | 56.57 | −0.181 (shallow funnel) | 46.3 | 0.723 | 0.846 |
| √3 = 1.732 | 60.00 | 0 (flat face) | 60.0 | 0.866 | 0.750 |
| 2.000 | 63.43 | +0.182 (shallow cone) | 73.7 | 0.960 | 0.640 |
| 1 + √2 = 2.414 (diamond) | 67.50 | +0.414 | 90.0 (grazing) | 1.000 | 0.500 |

For n = √3 the element is an ordinary axicon with a flat back, lit tip first.
For comparison, a single refracting cone of index 2 deviates rays by at most
60°, and only at grazing exit, where its transmission goes to zero.

The light that enters at radius ρ leaves the exit face before crossing the axis
only if ρ > D tan δ/(1 + n tan δ), where D is the axial depth. The two faces
meet at ρ = D/(n − slope). The device is therefore lit by a ring. The tip is
never lit, so how the tip is rounded does not matter.

## Results

All runs: 633 nm, Hankel grid of 5000 nodes out to 700 µm (node spacing at most
0.14 µm), ray directions smoothed over 5 µm, timings on 2 cores.

### Cross-check against `vecdiff.propagation.Propagator` (`figures/fig_crosscheck.png`)

The case is one n = 2 shell with hyperbolic tips of a = 20 µm (vecdiff's
meridian realization needs this; see Limits), walls 10 µm apart on the axis,
and a doughnut input (ρ/w) e^(−ρ²/w²)(e_ρ + i e_φ) with w = 40 µm. Fields are
read 1 mm behind the shell.

| Method | Radial (TM) power transmitted (% of input) | Azimuthal (TE) power transmitted (% of input) | Overlap of E_ρ with the other method (%) | Overlap of E_φ with the other method (%) | Run time (s, 2 cores) |
|---|---|---|---|---|---|
| vecdiff `Propagator` (meridian) | 98.096 | 48.705 | 99.998 | 99.998 | 1.4 |
| `meridional_operator` | 98.100 | 48.589 | 99.998 | 99.998 | 4.5 |

The rounded tips are why TE/TM is 0.497 here rather than the sharp-cone 0.410.

### Shell stacks (`run_shell_stack.py`, `figures/fig_shell_stack.png`)

The input is the doughnut above, w = 40 µm. Walls are 10 µm apart on the axis
unless stated. "Mode match" is the overlap of the output E_ρ with the input E_ρ
propagated freely over the same length, i.e. with an ideal element
diag(1, c^(N/2)).

| Stack | Wall thickness normal to the wall (µm) | Tip rounding a (µm) | E_φ share of output power, simulated (%) | E_φ share, closed form c^N/(1+c^N) (%) | Radial power transmitted (%) | Mode match of E_ρ (%) | Run time (s, 2 cores) |
|---|---|---|---|---|---|---|---|
| ABABAB, n = 2 | 4.47 | 0 (sharp) | 0.472 | 0.470 | 99.89 | 98.7 | 8.6 |
| AAAAAA, n = 2 | 4.47 | 0 (sharp) | 0.452 | 0.470 | 96.02 | 63.9 | 7.9 |
| ABABAB, n = 2 | 13.42 | 0 (sharp) | 0.473 | 0.470 | 97.54 | 89.2 | 7.8 |
| (AB)×7, n = 1.515 | 5.51 | 0 (sharp) | 0.905 | 0.902 | 99.94 | 98.9 | 19.7 |
| ABABAB, n = 2 | 4.47 | 1 | 0.474 | 0.470 | 99.65 | 99.3 | 8.1 |
| ABABAB, n = 2 | 4.47 | 5 | 0.521 | 0.470 | 76.82 | 36.7 | 8.3 |
| ABABAB, n = 2 | 4.47 | 20 | 1.308 | 0.470 | 63.11 | 1.5 | 8.1 |

What the table shows:

- **Suppression matches the closed form.** For sharp tips it agrees within 0.4% relative, whatever the wall thickness and whether or not the shells alternate.
- **Alternating is what preserves the mode.** Without it the inward shifts add up and fill the dark core.
- **A rounded tip ruins a doughnut beam over millimetre gaps.** The tip acts as a microlens on the light that reaches it. With a 5 µm rounding only 36.7% of the mode survives. A ring beam that never reaches the tips avoids this.

### Double-Brewster axicon (`run_brewster_axicon.py`, `figures/fig_axicon_*.png`)

The axial depth is D = 300 µm. The input is a ring
e^(−((ρ−ρ0)/σ)²)(e_ρ + i e_φ) placed inside the radii that the device accepts.
That is ρ0 = 129.9 µm and σ = 14.4 µm for n = √3, and ρ0 = 127.5 µm and σ = 12.5 µm
for n = 2. In the "+ 4 shells" case, four nested n = 2 spike shells (walls
4.47 µm, 50 µm gaps) precede the axicon. For a ring far from the axis their
inward shifts (3 µm each) only move the ring. Each device is also run with
t_s replaced by t_p (the same geometry without diattenuation) and with an ideal
radially polarized ring.

| Device and input | Output NA (half-angle) | E_φ/E_ρ power at output (%) | Radial power transmitted (%) | FWHM of total intensity \|E\|² at the brightest plane (wavelengths) | FWHM of \|E_z\|² (wavelengths) | \|E_z\|² share of the intensity inside the FWHM (%) | On-axis needle length, FWHM (µm) |
|---|---|---|---|---|---|---|---|
| n = √3 axicon, circular input | 0.866 (60.0°) | 56.2 | 100.0 | 0.643 | 0.414 | 66.0 | 19.6 |
| n = √3, same geometry with t_s = t_p | 0.866 (60.0°) | 100.0 | 100.0 | 0.834 | 0.414 | 40.7 | 19.6 |
| n = √3, ideal radial input | 0.866 (60.0°) | 0.0 | 100.0 | 0.462 | 0.414 | 93.1 | 19.6 |
| n = 2 axicon, circular input | 0.960 (73.7°) | 40.9 | 100.1 | 0.449 | 0.373 | 87.9 | 15.5 |
| n = 2, same geometry with t_s = t_p | 0.960 (73.7°) | 99.8 | 100.1 | 0.605 | 0.373 | 61.7 | 15.5 |
| n = 2, ideal radial input | 0.960 (73.7°) | 0.0 | 100.1 | 0.383 | 0.373 | 98.5 | 15.5 |
| n = 2 axicon + 4 shells, circular input | 0.960 (73.7°) | 1.15 | 100.1 | 0.385 | 0.373 | 98.3 | 17.3 |
| n = 2 + 4 shells, t_s = t_p | 0.960 (73.7°) | 99.8 | 100.1 | 0.605 | 0.373 | 62.2 | 17.3 |
| n = 2 + 4 shells, ideal radial input | 0.960 (73.7°) | 0.0 | 100.1 | 0.383 | 0.373 | 98.5 | 17.3 |

The 0.1% excess in transmitted radial power is the numerical error of the
resampled ray map.

Ideal conical waves give closed-form widths to compare against. The TM part has
intensity sin²α J0² + cos²α J1² and the TE part J1², with x = k ρ sin α:

| Ideal conical wave | FWHM of total intensity, closed form (wavelengths) | FWHM, simulated (wavelengths) |
|---|---|---|
| TM only, NA 0.960 | 0.3834 | 0.3833 |
| TE = TM in power, NA 0.960 | 0.6069 | 0.6051 |
| TE/TM = 0.64¹⁰ = 1.15 %, NA 0.960 | 0.3849 | 0.3848 |
| TM only, NA 0.866 | 0.4621 | 0.4619 |

What the axicon results show:

- **A single n = 2 element narrows the needle on its own.** It passes the radial field without loss and leaves 40.9% E_φ/E_ρ. That brings the total-intensity width from 0.605 λ (no diattenuation) to 0.449 λ.
- **With four Brewster shells ahead of it, the needle matches ideal radial input.** Ten Brewster surfaces in all give 0.385 λ against 0.383 λ ideal, and 98.3% of the intensity inside the FWHM is longitudinal.
- **No surface loses radial power.** The only fabrication-sensitive element is the exit face, a cone of slope 0.182, or a plane for n = √3.

## Prior art and what was not found

| Work | How it works | Relation to this folder |
|---|---|---|
| F. P. Schäfer, US 4,755,027 (filed 1986); "On some properties of axicons", Appl. Phys. B 39, 1 (1986) | Transmission through an even number of opposed conical interfaces at Brewster incidence, with polarization-selective or antireflection coatings; four coated cones make a radial or tangential polarizer; ray and Fresnel analysis | The shell stack is this idea. The wave-optical analysis (radial shift, alternation, tip rounding) is added here |
| J. M. Guerra, Polaroid (1986), Appl. Opt. 29, 3741 (1990) | Two metal reflections in a bicone and one air–glass reflection at Brewster | Uses reflection; not purely refractive |
| Brewster axicons inside laser cavities (from 1972; TRW US 5,359,622; Kozawa & Sato, Opt. Lett. 30, 3063, 2005) | A conical Brewster window inside the resonator selects the radial mode over many passes; Kozawa–Sato add a 7-layer coating on a convex/concave cone pair | Mode selection inside a cavity; the devices here work outside one, in a single pass |
| Lithography illumination patents (US 6,392,800; US 7,511,886) | Conical frustum or axicon surfaces near Brewster, with coatings, in illumination optics | Coated, collimated, filtering |
| J. R. Knutson, US 2012/0320458, US 9,086,509 | Pairs of axicons that separate radial and tangential components | Beam splitting, not a needle |
| S. A. Degtyarev, S. N. Khonina et al., "Volcone", Photonics 9, 421 (2022); Khorin et al., Computer Optics 47, 742 (2023); Gubaev et al., Computer Optics 45, 828 (2021) | Refractive bi-conic axicon: Brewster refraction at an inner conical cavity, an outer cone collimates; circular input gives an azimuthally polarized vortex; a reviewer reports 6.6% conversion efficiency; ray tracing with Jones vectors and FDTD | Closest recent work; azimuthal and collimated, versus radial and conical here |
| H. Wang et al., Nat. Photon. 2, 501 (2008); G. M. Philip & N. K. Viswanathan, FiO 2013 (FW4F.4); L. Turquet et al., Opt. Express 26, 27572 (2018) | Longitudinal needles from a radially polarized beam focused by a high-NA lens with binary optics, or by a high-NA axicon (0.48 λ, 40 λ) | Need radially polarized input; here the element makes its own from circular light |

Not found in these searches: an axicon whose two faces are both at Brewster
incidence with their deviations adding; its closed forms
NA = 4n(n² − 1)/(n² + 1)² and c = cos² δ; and the n = √3 (flat back) and
n = 1 + √2 (NA 1) special cases. These searches were not exhaustive (no Scopus,
no patent full-text search after 2012).

## Limits

- **vecdiff's meridian realization does not carry these devices.**
  `MeridianTransport` needs the axial ray to meet no caustic on any leg. The
  tip of a cone breaks this, sharp or rounded: a rounded tip is a microlens
  that focuses within a few tip radii, and a sharp tip sends the near-axis rays
  across the axis inside the shell. vecdiff carries a single shell only when
  its tips are rounded to a ≥ (n − 1) × (axial wall separation); the
  cross-check above uses that case. The lattice realization handles caustics,
  but here it needs a spacing below λ/(2n sin θ) = 0.26 µm over ±250 µm, which
  exceeded 7.5 GB.
- **Steep graph surfaces.** For a graph surface vecdiff starts Newton from the
  height under the plane point. For the inner face of a slope-2 cone that start
  lies across the axis, and Newton converged to the far crossing: surface
  heights jumped from 69 µm to 564 µm between labels 54 µm and 56 µm.
  `HyperbolicCone` declares its implicit form and an exact crossing, which
  avoids this. A bracketing start in `InterfaceMap._intersect` would fix it for
  every steep graph surface; that change to the library is left for the
  maintainer to decide.
- **The operator.**
  - It is the vertex-plane method with phase-gradient rays smoothed over 5 µm.
  - It conserves radial power to 0.1%.
  - It carries no reflections, so ghosts between faces are not modelled.
  - The geometry is scaled down (beams of 40–170 µm, walls of a few µm). In a real device what carries over is the ratio of radial shift to beam size.

## Running

    python run_shell_stack.py [outdir]       # 1 min
    python run_brewster_axicon.py [outdir]   # 1 min
    python cross_validate.py                 # 10 s
    python make_figures.py [outdir]          # 2 min, after the two runs
