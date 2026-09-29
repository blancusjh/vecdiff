# Propagation

This guide covers the operational path of `vecdiff_general`: the interface
operators, surface edges and stops, the GOTS Cartesian oval and the ideal
imaging operators. For accuracy against full-wave references see
[validation](VALIDATION.md).

## The operational path

The primary path is `propagate_interfaces(source, system, grid)`:

\[
R_{\xi_2}Q^{-1}_{n_i,\Sigma}TQ_{n_0,\Sigma}R_\xi .
\]

The interface operation repeats in each incident/transmitted medium. A finite
two-face lens may also use the coupled-rim forward step described below between
its faces.
`propagate_high_frequency` exposes the same implementation. No
equivalent-current, Green, Weyl or ray-tube propagation occurs in it.

- `PlanarASM` is **R**. It multiplies each plane coefficient by
  `exp(i kz distance)`, with the exact homogeneous dispersion. Its `spectrum`
  method preserves coefficients between interfaces; calling the operator returns
  E/H on the translated Cartesian plane by IFFT.
- `SurfaceTransport.forward` is **Q**. For a graph `z=h(u,v)`, it evaluates
  `sum A(ku,kv) exp(i(ku u + kv v + kz h(u,v)))`. An analytic `EikonalField`
  can instead be evaluated directly on the first surface.
- `fresnel_transmission` is **T**. The local normal and eikonal direction
  determine Snell refraction and the vector Fresnel matrices. It returns
  incident, reflected and transmitted E/H traces. Reflections are retained as
  traces, not recursively propagated. When both indices are identical, T is
  exactly the identity on E and H.
- `SurfaceTransport.inverse` is **Q⁻¹**. It solves for outgoing plane
  coefficients whose surface-tangential electric trace best matches the target
  in L2(dS), including the chart metric. Longitudinal E and H follow homogeneous
  Maxwell constraints. It returns a `PlaneSpectrum` and an `InverseReport`;
  nonconvergence raises an error. Constant-height charts use a direct FFT;
  variable-height charts use an iterative least-squares solve.

For graph slope `g = grad(h)`, covariant chart components are `e = E_xy + g E_z`.
Their physical squared norm is `e* G^-1 e`, where `G = I + g g^T`. The
least-squares operator applies `sqrt(J) G^-1/2`, with `J = sqrt(1 + |g|²)`; this
is an integration weight in the residual, not a gain on the propagated field.

On a variable-height surface, multiplication by height in spatial coordinates
does not commute with longitudinal dispersion in Fourier coordinates, so
**changing h to -h does not invert Q**. It does for constant height.

Q factors the phase into a carrier and a Chebyshev/Bessel series. Each term uses
an IFFT and pointwise multiplication; a Bessel-tail bound controls truncation,
and LSMR uses the exact adjoint of that approximation. There is no dense
surface-to-surface matrix. Cost is `O(iterations × rank × N log N)`; rank grows
with sag times longitudinal spectral bandwidth.

### Sampling and scope

Pass a padded `CartesianGrid` at the first vertex. Chart frames must share
orientation and transverse origin; surfaces must not intersect. The path
supports forward propagating modes below a declared `max_sine` (default 0.85);
evanescent coupling, grazing transmission and folded paths are not silently
approximated. The coupled-rim path discards exit-face TIR samples because the
forward interface operator has no evanescent transmission branch.

Each `EikonalField` or analytic plane-wave branch is transported independently
and summed coherently. A `PlaneSpectrum` declares one locally eikonal branch;
a multiray interference field near a caustic has no unique eikonal.

The FFT cell is periodic. Increase the sample count at fixed width, then the
width at fixed spacing, and test `max_sine` separately. A cell only a few clear
radii wide lets periodic replicas add spurious downstream structure; the
full-wave comparisons of notebook 06 use 24a.

`solution.evaluate(grid)` uses FFT ASM on a matching translated lattice and a
plane-wave Fourier sum elsewhere (FINUFFT when installed). Observations must lie
beyond the final surface.

### What the residuals certify

Small inverse residuals certify the fit of tangential E in the chosen angular
band, not the full dielectric boundary problem. Local Fresnel magnetic traces
differ from the reconstructed homogeneous H at finite wavelength; notebook 03
reports that difference and the signed flux without normalization.
`InverseReport.relative_residual` is the tangential E residual and
`magnetic_relative_residual` the tangential H mismatch.

Prescribed-current tools live in `vecdiff_general.comparisons`;
`solution.evaluate_green_comparison()` uses that separate model and is never
called by the operational path.

Background: [Kim et al., stepwise angular spectrum on curved surfaces (2014)](https://opg.optica.org/fulltext.cfm?uri=oe-22-10-12659).
This implementation uses a separated phase expansion and a numerical inverse
instead of that paper's staircase algorithm.

## Electric-field operators at one interface

A transmission or reflection changes the electric samples on a fixed surface
and keeps the grid and vacuum wavelength. The direction field is supplied
explicitly, not inferred from an interfering total field.

```python
import numpy as np

from vecdiff_general import ElectricField, SurfaceGrid
from vecdiff_general.interfaces.directions import transmitted_direction
from vecdiff_general.interfaces.fresnel import transmit

normal = np.array([0.0, 0.0, 1.0])
incident_direction = np.array([0.6, 0.0, 0.8])
transmitted = transmitted_direction(
    incident_direction, normal, incident_index=1.0, transmitted_index=1.5
)
grid = SurfaceGrid([[0.0, 0.0, 0.0]], [normal], [1e-12])
field = ElectricField(grid, wavelength0=532e-9, values=[[0.8, 0.0, -0.6]])
transmitted_field = transmit(
    field,
    incident_direction,
    transmitted,
    normal,
    incident_index=1.0,
    transmitted_index=1.5,
)
assert transmitted_field.grid is field.grid
```

`transmission_coefficients` returns `(t_s, t_p)`, `reflection_coefficients`
returns `(r_s, r_p)`; `fresnel_reflection_matrix` and `reflect` compute only the
reflected branch. `scatter_plane` requests both. The normal points **into the
transmitted material**, so `cos_i = dot(u_i, normal)`. Under TIR the transmitted
direction is complex and is not normalized with a Hermitian norm.

## Surface edges and stops

A surface **edge** (`OpticalSystem.apertures`) is where a material face ends; it
never cuts the field. A **stop** (`OpticalSystem.stops`) is an opaque pupil that
cuts everything reaching it outside its radius.

```python
lens = o.OpticalSystem(interfaces, apertures=(a, a))  # bare lens
mounted = o.OpticalSystem(interfaces, apertures=(a, a), stops=(a, None))
```

Stops are stored in JSON prescriptions as `stop_m`. For each **element** (a run
of interfaces between visits to the source medium):

1. At the entrance edge, the field inside the edge enters the element; the field
   outside it is inverted in the surrounding medium on the entrance chart, flat
   beyond the rim.
2. That outside light is carried across the element by exact ASM and added at
   the element's exit.
3. For eligible small, thick two-face lenses, the interior and exterior fields
   exchange across the circular rim during forward propagation through the
   element. Each slice propagates in glass and air separately with ASM. This is
   a forward approximation to rim diffraction, not a vector sidewall solve.
   Other elements retain the separated path; inside light reaching a later
   chart beyond its edge crosses the flat extension with normal-incidence
   Fresnel.
4. A stop cuts light inside the element and light passing outside it.

`interface_spectra` keeps each interface's Q⁻¹ result and `exterior_spectra`
the light that passed outside each element. `coupled_elements` reports which
interface pairs used the rim step. `rim_coupling="auto"` (the default) applies
it only to two-face, equal-radius circular lenses without stops, with diameter
4–10λ₀, centre thickness at most 4λ₀ and element Fresnel number at most 8.
This empirical gate avoids
the observed peak overshoot for 1–2λ₀ bodies and leaves thin lenses on the
better-performing path. `"off"` retains the separated path; `"on"` requests the
rim step for every supported element. `rim_step` is the axial slice length in
vacuum wavelengths, default 0.05. The step omits sidewall Fresnel reflection,
returning waves and vector boundary conditions; its square-root index factors
approximate the normal-incidence flux scale. An open system (never returning to
its source medium) needs a stop no larger than its last edge; an interior edge
larger than the entrance edge is rejected.

Keeping the light that passes outside a lens is the largest single improvement
against full-wave references. On-axis complex field error, no gain, phase or
focal alignment:

| Case | Axial field error, stop at edge | Axial field error, no stop |
|---|---:|---:|
| Plano-convex, D = 4λ₀, vs BEM | 49% | 39% |
| Biconvex, D = 4λ₀, vs BEM | 41% | 20% |
| R = 30λ₀, a = 12λ₀, 256², vs FEM | 9.5% | 5.0% |

### Validity: element Fresnel number

Within about √(λ₀T) of the rim, for centre thickness T, light crosses between
the glass and the surrounding medium, and the entrance-edge separation is not
exact. The separated path therefore needs a large a²/(λ₀T). Measured with
6λ₀ lenses against BEM in [notebook 06](../general/notebooks/06_full_wave_reference_validation.ipynb)
(geometry 4):

| Lens (D = 6λ₀, n = 1.5) | a²/(λ₀T) | Separated axial E error | Auto coupled axial E error |
|---|---:|---:|---:|
| Biconvex R = ±20λ₀, T = 0.6λ₀ | 15.0 | 4.1% | 4.1% |
| Biconvex R = ±20λ₀, T = 0.9λ₀ | 10.0 | 8.2% | 8.2% |
| Biconvex R = ±20λ₀, T = 1.2λ₀ | 7.5 | 17.9% | 6.6% |
| Biconvex R = ±20λ₀, T = 1.8λ₀ | 5.0 | 53.1% | 10.3% |
| Biconvex R = ±20λ₀, T = 2.4λ₀ | 3.8 | 54.8% | 12.4% |
| Four thin lenses, T = 0.7λ₀ | 12.9 | 3.4–3.7% | unchanged |

These are complex vector E gaps over a common axial interval, not peak-only
scores. For the 4λ₀ plano-convex detector plane, the complex E and H gaps to a
λ₀/16 BEM mesh fall from about 33% to 11% on a common 41×41 point patch;
the BEM λ₀/12 to λ₀/16 change on that patch is 1.7%. The improved total-field
peak is 15.1 V²/m² versus BEM's 14.3, although the longitudinal component
overshoots. This remains an approximate forward model, and local power or
boundary continuity is not certified by those detector comparisons.
On that detector patch, halving the axial slice from 0.05λ₀ to 0.025λ₀
changes complex E by 0.37%; widening the FFT cell from 24a to 36a at fixed
spacing changes it by 0.63% (complex E_z by 2.5%). The residual E_z excess is
therefore a model limitation rather than an unresolved slice size.

## GOTS Cartesian oval

```python
import numpy as np
import vecdiff_general as optics

surface = optics.CartesianOval.from_conjugates(no=1.0, z0=-100e-6, ni=1.5, zi=60e-6)
G, O, T, S = surface.gots
rho = np.linspace(0, surface.rho_from_r(12e-6), 64)
phi = np.linspace(0, 2 * np.pi, 96, endpoint=False)
points = surface.parametric_points(rho[:, None], phi[None, :])
```

`gots_parameters(no, z0, ni, zi)` returns the four coefficients directly, from
the thesis Appendix D, D.3–D.8. `rho` is distance from the local vertex, not
aperture radius; `surface.z(rho)` and `surface.r(rho)` evaluate the explicit
parametrization and `parametric_points(rho, phi)` revolves it into the physical
frame. G, O, T and S have dimensions 1, 1/length, 1/length³ and 1/length².
`height(xy)` and `gradient(xy)` keep the cylindrical chart; `rho_from_r` inverts
the unsquared constant-optical-path equation.

Supported: real conjugates `z0 < 0 < zi`, positive unequal indices and a
single-valued vertex-connected graph. `z0=-np.inf` selects collimated
incidence, with `G=-(no/ni)**2`, `O=ni/(zi*(ni-no))`, `T=S=0`; to match a
sphere's signed vertex radius `R`, choose `zi = ni*R/(ni-no)`, giving `O=1/R`.
Virtual conjugates, infinite image distance and other oval branches raise
errors. Notebook 03 compares matched spherical and stigmatic lenses in this
collimated limit.

## Ideal aplanatic imaging

`vecdiff_general.propagation.imaging` maps fields between conjugate planes with
explicit ideal pupils (notebooks 04 and 05).

```python
from vecdiff_general.propagation.imaging import image_spectrum

image_modes = image_spectrum(
    object_modes,
    image_region,
    magnification=-0.5,
    numerical_aperture=0.8,
    image_frame=image_plane,
)
image = image_modes.propagate(0)
```

The input is a forward `PlaneSpectrum`. Signed magnification maps
`x_image = m * x_object`. For object transverse frequency q = λ₀ f, the aperture
accepts |q| ≤ |m| NA_image, and the amplitude factor

```text
gain = sqrt(n_object c_object / (n_image c_image)) / |m|
```

with the scaled transverse area conserves the normal Poynting power of the
passed modes; rejected modes lose their power. An optional `pupil` is a scalar,
an array on the object frequency grid, or a 2 × 2 Jones matrix in the **(p, s)**
mode bases. Objective aberrations, coatings and prescriptions are not inferred.

`focus_pupil(...)` maps a collimated Cartesian pupil to a vector focal spectrum
under the sine condition (ρ = f sin θ, amplitude sqrt(n_object / (n_image cos θ))).
`apply_pupil(spectrum, J)` applies a Jones operator per mode; an inverse pupil
may need extra incident power, which should be normalized explicitly. For an
interface response inside a power-normalized pupil use

```text
tau_s,p = t_s,p * sqrt(n_transmitted cos_transmitted / (n_incident cos_incident))
```

`scalar_image(...)` returns an array with the same aperture, Jacobian and exact
defocus, without polarization transport. The DUV example of notebook 05 is an
ideal 200 mm / 50 mm relay at 193 nm (m = −0.25, NA 0.85), not a traced
objective; a real system also needs a measured Jones pupil, rigorous mask fields
and a resist model.
