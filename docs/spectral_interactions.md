# Spectral interactions and macroscopic validation

## Constructed repeated encounters

`InterfaceAssembly` contains ordered physical interfaces with matching adjacent
media. `propagate_interfaces` constructs their forward and backward spectral
maps. For interface j, between regions j and j+1, the equations are

$$
\widetilde{\mathbf E}_{j+1}^{+}
=T_j^+\widetilde{\mathbf E}_{j}^{+}
+R_j^-\widetilde{\mathbf E}_{j+1}^{-},\qquad
\widetilde{\mathbf E}_{j}^{-}
=R_j^+\widetilde{\mathbf E}_{j}^{+}
+T_j^-\widetilde{\mathbf E}_{j+1}^{-}.
$$

Each map invokes the per-k Fresnel interface transformation. Spectra use global
coordinates, so exp(i k.Q) at each placed surface includes the propagation
phase between encounters. There is no additional phase multiplier to double
count that propagation. Unknowns are Ex/Ey amplitudes on one explicit transverse
spectral lattice per branch; Ez follows from k.E=0. A fixed-point iteration or
GMRES solves the resulting coherent equation. No reference coefficients or
auxiliary field representation enter the calculation.

Current scope is separated z-graph interfaces with a propagating, non-grazing
lattice in every medium. The period and bandwidth are numerical controls.
Back-facing modes, closed spheres, overlapping surface envelopes, and evanescent
coupling are rejected. `propagate_layers` remains available for parallel
evanescent gaps. Field evaluation checks the source-free spectral slab in the
requested region. Boundary diagnostics explicitly use spectral continuation
inside curved source envelopes and are not singular on-surface limits.

The planar assembly is compared with the independent stable layer recursion
for several index sequences, both encounter algorithms, and mixed complex
polarizations; all four boundary conditions are reconstructed at each plane.
The cavity example's maximum complex-field discrepancy is 1.98e-15 across
13 wavelengths. This validates the composition in its planar limit.

## Curved assemblies: remaining error is visible

The two-cap example uses indices 1/4/1, radii +80/-80, vertex gap 1, and aperture
radius 2, in reference-wavelength units. It displays the curved response and
reconstructed boundary jumps together. It is a model/convergence study, not a
validated physical resonator.

The independent control sweep at wavelength 1 is recorded in
`benchmarks/results/assembly_convergence.json`. At fixed period 7.2, increasing
the lattice from 6×6 to 10×10 changes the sampled downstream E/H field by 19.8%
and reduces the largest boundary jump from 29.0% to 7.31%. Doubling only surface
quadrature changes the field by 2.23e-9. Varying the period at fixed spacing
also changes the field. Thus spectral-window/bandwidth accuracy is not established,
even though feedback residuals are below 1e-10. A converged algebraic equation
does not repair an inadequate spectral or physical interface model.

General resonant-body accuracy, closed-sphere comparisons, and resolved high-Q
spectroscopy remain pending. The code must not describe this test as Mie validation.

## Macroscopic caps with the same spectral method

`SurfaceRadiation.evaluate_propagating` integrates the observation's azimuthal
phase analytically using the Jacobi–Anger expansion. The source-current harmonic
tail is checked; the vector projection and H reconstruction retain their added
harmonics. This is the same propagating hemisphere quadrature, verified against
explicit plane-wave summation at off-axis points and translated surfaces.
The polar transform processes nodes in blocks to bound temporary memory.

`python -m benchmarks.macroscopic` independently refines surface quadrature,
polar quadrature, and interpolation tables. Aperture radius is 0.6R, indices
are 1/1.5, and lambda=1. Boundary points extend to 0.3R; downstream E/H samples
are at z=R and x/R=-0.1,0,0.1. Combined-refinement results are:

| R / lambda | Et jump | Ht jump | Dn jump | Bn jump | Power imbalance | Downstream field change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 3.128% | 4.924% | 2.598% | 3.960% | 0.2006% | 0.0133% |
| 100 | 1.832% | 2.931% | 2.158% | 2.956% | 0.1012% | 0.0449% |
| 200 | 1.241% | 1.931% | 1.568% | 2.359% | 0.0510% | 0.0686% |

Power is integrated over the propagating reflected/transmitted spectra using
spectral Parseval normalization, divided by incident flux through the projected
aperture. Boundary jumps use the normalizations in `validation.md`. The final
column compares combined-refinement fields against the baseline, not against
an independent exact solution. Boundary continuation includes hard-aperture
and omitted-evanescent effects; it cannot isolate local-curvature error.

The combined R=200 run used 1600×32 surface nodes, 3200 polar nodes, a 6400-node
interpolation table, and current harmonics through order 2. It took 89.6 seconds
on the development host. The process high-water RSS reached 309 MiB across the
sequential sweep; this is not an isolated per-case measurement. No GPU or parallel
numerical threads were used. The raw file records environment versions and every
refinement. These measurements improve scale coverage but fail to establish
exact boundaries or complete DUV, telescope, or microscope performance.

## Reproducible presentation

All six application notebooks are committed with executed outputs; source synchronization
preserves them. The two original README GIFs are restored unchanged and labeled
with their historical model provenance. Current spectral results appear in
separate reproducible figures, including the unresolved boundary residuals.

For resonator theory and the distinction between coherent round trips and
intensity summation, see H. Kogelnik and T. Li, *Laser Beams and Resonators*,
Applied Optics **5**, 1550–1567 (1966),
[doi:10.1364/AO.5.001550](https://opg.optica.org/ao/abstract.cfm?URI=ao-5-10-1550).
That reference motivates validation; it is not a core implementation dependency.
