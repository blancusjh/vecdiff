# Macroscopic transport using a smooth phase

`propagate_high_frequency` now transports a forward dielectric path through
multiple placed curved interfaces. It preserves each phase component's optical
path, vector Fresnel transformations and geometric spreading, and constructs
ordinary final `SurfaceRadiation` for diffraction. It does not assume that an
image is a shifted on-axis response. The implementation is in `propagation/`;
constitutive interface laws remain in `interfaces/fresnel.py`.

## Relation to the gradient operator

For a field with one smooth phase,

$$E(r)=A(r)e^{ik_0L(r)},\qquad |\nabla L|=n,$$

$$-i\nabla E=e^{ik_0L}\big(k_0A\nabla L-i\nabla A\big).$$

The leading wavevector is $k_0\nabla L$ and the ray direction is $\nabla L/n$.
`EikonalElectricField` accepts the optical-path length, its analytic gradient,
and a transverse vector envelope. It verifies the eikonal magnitude and
transversality at the samples. It does not extract a reliable single phase
from arbitrary interfering fields. Neglecting the envelope derivative is an
explicit approximation; this input is not an exact Maxwell field.

Alternatively, passing `ElectricSpectrum` retains every incident phase component.
Each component is transported independently until complex fields are summed.
Linearity is tested. No phase-gradient estimate of the summed field replaces
these components. The present optimization is phase transport, not a Taylor
approximation of Fresnel coefficients about one central wavevector.

## Transport and numerical representation

At each encounter, the central ray carries the electric envelope and optical
path. Four displaced rays measure the area Jacobian. Between interfaces,

$$|A_2|=|A_1|\sqrt{
\frac{|\hat u\cdot\hat n_1|\,dS_1}{|\hat u\cdot\hat n_2|\,dS_2}}.$$

The geometric derivatives are checked by step refinement. The signed projected
ray-tube area is quadratic in propagation distance; an intermediate zero is
rejected instead of silently crossing a caustic without the required wave and
phase treatment. Reflection fields are returned at each face, but are not fed
back through the element. Surface quadrature, derivative step and final-radiation
kernel error must be checked separately.

For $N$ initial surface samples, $M$ phase components and $S$ interfaces,
transport costs $O(MNS)$. An explicitly supplied single phase makes $M=1$.
No wavelength-spaced aperture raster is constructed. This cost statement does
not guarantee that a fixed $N$ resolves every aberrated image or every aperture.

`backend='auto'` in Fourier synthesis uses direct sums for small products and
NUFFT for larger real-wavevector sums when installed. This removes plan-creation
overhead for thin observation patches without changing the optical model. A
3,321-point meridional trial took 1.50 s with direct sums versus 18.01 s with
repeated NUFFT plans. These are configuration-specific single-thread timings.

## Measured accuracy and performance

The [executed system notebook](notebooks/09_macroscopic_system_transport.ipynb)
and `benchmarks/results/high_frequency_transport.json` record parameters and
versions. Timings below use one CPU thread; no GPU or parallel transport is used.

A 625-mode Gaussian spectrum illuminates the first curved face. The reference
reconstructs its outgoing field at five points on the second face using the
full dyadic Green kernel, with independently refined surface quadrature.
The incident edge amplitude is below $10^{-5}$, suppressing truncation artifacts.

| $R/\lambda_0$ | 625-component transport: complex E/H error | One Gaussian eikonal: complex E/H error |
| ---: | ---: | ---: |
| 100 | 0.0883% | 25.90% |
| 200 | 0.0442% | 13.00% |
| 500 | 0.0177% | 5.203% |
| 1,000 | 0.00886% | 2.602% |

The 625-component transport isolates the inter-interface high-frequency
approximation. The single Gaussian eikonal also discards the incident field's
angular spread and its wave evolution. The latter is unsuitable at the smaller
scales in this table. Direct-reference quadrature changes stay below
$3.4\times10^{-9}$; geometric derivative-step changes are below $6\times10^{-12}$.

At 193.368 nm, a two-interface element with a 10-mm entrance curvature radius,
8-mm aperture diameter and 1-mm Gaussian waist gives:

| Final image calculation, 17 points | Samples per phase | Time |
| --- | ---: | ---: |
| One supplied Gaussian eikonal | 2,048 | 0.0347 s |
| 625 independently transported components | 2,048 | 30.793 s |

Their complex image fields differ by **0.0218%**, and the multiple-component
surface-quadrature change is $1.7\times10^{-11}$. This is approximately **888×**
faster for this workload. Both methods use high-frequency inter-interface
transport; this is not an exact full-Maxwell error or speedup claim. The final
radiation in this comparison uses the full Green kernel.

Transport alone through both faces at 8,192 samples takes approximately 0.09 s.
It stays at roughly that cost when the wavelength decreases from 0.1 mm to
193.368 nm. The notebook also recovers a finite object at $z=-20$ mm and image
at $z=40$ mm using a Cartesian-oval entrance and spherical exit. A displaced
object is propagated through unchanged surfaces.

## Current domain and remaining work

This is now implemented multi-interface macroscopic transport in its stated
high-frequency domain. It is not restricted to stigmatic geometry: displaced
sources follow the same algorithm. The recovered finite-conjugate example does
not require a separate stigmatic solver or compatibility API.

The path currently transmits through dielectric interfaces. Intermediate
caustics, grazing intersections, evanescent transmission and vignetting reject.
Upstream edge diffraction, multiple coherent feedback, metallic/coated mirrors,
and full folded DUV prescriptions are not covered by this implementation.
Global reconstructed boundary accuracy remains distinct from local Fresnel
identities and ray-tube flux conservation. The direct spectral/current route
remains available for verification and regimes where the approximation fails.

## Related literature

J. Kim, Y. Wang and X. Zhang, “Calculation of vectorial diffraction in optical
systems,” *JOSA A* **35**, 526–535 (2018),
[doi:10.1364/JOSAA.35.000526](https://doi.org/10.1364/JOSAA.35.000526), discusses
consistent vectorial tracing for estimating boundary fields in optical systems.
It provides methodological context; no reference implementation or runtime
ray-tracer dependency is imported into this path.
