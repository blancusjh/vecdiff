# A spatial amplitude factor derived from flux conservation

## Result and status

A ray-tube conservation law does give a **spatial amplitude function**, applicable
to general regular surfaces. It includes the Jacobian of the ray map, not just
the local incidence and refraction cosines. The factor follows from local power conservation along the mapped rays.

The tested law **does not solve the main method's intensity deficit**. Existing
curved ASM already includes homogeneous spreading. Its exact inverse must not
receive that spreading factor a second time. Replacing the inverse with the
complete geometrical ray map gives only small improvements in the studied
wavelength-scale lenses and leaves their large discrepancy with BEM/Mie.

The numerical comparisons use the same Fresnel coefficients and incident field
for the curved-ASM inverse and the geometrical ray-map inverse.

## Derivation for an arbitrary graph

Consider a nonmagnetic, lossless homogeneous medium and a regular, locally
transverse eikonal branch with real unit direction **u**. To leading order,

\[
\langle\mathbf S\rangle=\frac{n}{2Z_0}|\mathbf E|^2\mathbf u.
\]

Let the surface be \(\mathbf r_s(\mathbf x)=(\mathbf x,h(\mathbf x))\) and
the reference plane be \(z=z_p\). Straight rays in this medium meet the plane at

\[
\mathbf y(\mathbf x)=\mathbf x+(z_p-h)\mathbf q,
\qquad \mathbf q=\frac{\mathbf u_\perp}{u_z},
\qquad \ell=\frac{z_p-h}{u_z}.
\]

The mapped area is \(d^2y=J\,d^2x\), where, before a caustic,

\[
J=\det\left[I-\mathbf q\otimes\nabla h+(z_p-h)D\mathbf q\right]>0.
\]

The surface's oriented area element is \((-\nabla h,1)d^2x\). Equating power
through the two cross sections of the same ray tube gives

\[
|E_p(\mathbf y)|^2 u_z J
=|E_s(\mathbf x)|^2\left(u_z-\mathbf u_\perp\cdot\nabla h\right).
\]

Consequently,

\[
\boxed{F_{\Sigma\to P}(\mathbf x)=
\sqrt{\frac{u_z-\mathbf u_\perp\cdot\nabla h}{u_z J}}}
\]

and the leading ray-field relation is

\[
\mathbf E_p(\mathbf y(\mathbf x))\simeq
F_{\Sigma\to P}(\mathbf x)\mathbf E_s(\mathbf x)e^{ik\ell}.
\]

Polarization is constant along a straight ray in this homogeneous isotropic
medium. In a rotated chart it must also be expressed in the correct frame.
The inverse amplitude is the reciprocal **at corresponding mapped points**.
Multiplying fields at the original Cartesian coordinates by this function
does not implement that transport.

For general pairs of surfaces the same law reads
\(F^2=(\mathbf u\cdot\mathbf n_s)dA_s/
[(\mathbf u\cdot\mathbf n_p)dA_p]\), with consistently oriented sections.
This is the standard ray-tube amplitude transport law; see
[Demanet, *Waves and Imaging*, §§2.3–2.4](https://math.mit.edu/icg/resources/notes325.pdf).

### Relation to the stigmatic manuscript

For a spherical reference surface centred on the focus, radius R, and its
tangent plane, the perspective ray map has transverse scale \(1/\cos\alpha\).
The expression above reduces to \(F=\cos\alpha\) for sphere-to-plane transport,
and its reciprocal for plane-to-sphere transport. This reproduces the
perspective-mapping factors in §12 of
`refraccion_sistema_estigmatico_corregido.pdf` supplied by the user.

The result depends on the ray mapping: a sine-condition pupil mapping has a
different Jacobian. Importing its square-root-cosine factor into this chart
would describe a different map.

### Reflection and transmission

At one interface the transmitted and reflected rays start on the **same**
physical area. For each local s/p polarization,

\[
n_i\cos\theta_i(1-|r|^2)
=n_t\cos\theta_t |t|^2.
\]

Thus flux conservation there recovers the magnitude of the Fresnel coefficient
already used by T. It does not supply another independent gain. After T, use
the transmitted direction in the homogeneous transport Jacobian. A reflected
branch has its own map and transported power; it cannot be added to transmitted
power without representing that branch and its subsequent interactions.

## Where a numerical correction could enter

The boxed F relates a surface field to a field at different ray-mapped points.
It is a **transport amplitude**, not automatically a **correction amplitude**.
If an approximate transport already supplies amplitude \(F_{\rm existing}\),
the relevant correction in the eikonal limit would be
\(F_{\rm correction}=F/F_{\rm existing}\), with matching coordinates, phase,
polarization and ray budget. Adding F outright double counts existing transport.

Exact homogeneous ASM solves the wave equation and supplies the geometrical
spreading in its short-wavelength limit. It also supplies interference and
diffraction absent from a single-ray formula. A measured scalar flux ratio
alone does not determine those phases or additional incident/returning branches.

The graph inverse requires the direction derivatives Dq, hence neighboring
ray information (or second derivatives of a phase), in addition to grad h.
Computing these finite differences and 2×2 determinants costs O(N). The study's
ray inverse uses one nonuniform Fourier quadrature and transverse modal
completion. It does not use a Green kernel or call a reference solver.

## Implementation and checks

- `general/src/vecdiff_general/propagation/transport_flux.py`: ray coordinates,
  Jacobian, amplitude and caustic detection, with shape/unit-direction checks.
- `general/tests/test_transport_flux.py`: seven tests covering plane waves,
  the stigmatic cosine limit, focusing, a crossed caustic with positive endpoint
  determinant, local power conservation and invalid direction inputs.
- `general/examples/theoretical_flux_study.py`: factor measurements, independent
  Maxwell control, and the complete experimental ray-map inverse at **each**
  interface of the original R Q⁻¹ T Q R composition.
- `general/examples/theoretical_flux_report.py`: absolute-field plots and dark
  tables. Results are in `output/theoretical_flux_study/report.html`.

The implementation tests the projected ray-tube area throughout the segment,
not only its endpoint. A focus can be crossed with a positive endpoint area
determinant when both transverse directions reverse. Invalid/caustic samples
return NaN factors, not arbitrary capped gains. Local nonsingularity does not
establish global injectivity; the approximation also assumes a resolved single
eikonal branch. These conditions are particularly restrictive at a few wavelengths.

### Independent homogeneous Maxwell field

A Gaussian angular spectrum is completed into transverse Maxwell modes, then
evaluated on a smooth curved graph. Both 128² and 192² grids give:

| Operation | Relative complex E error | Power / original |
|---|---:|---:|
| Existing Q⁻¹ Q | 6.26×10⁻¹¹ | 1.000000000000004 |
| Complete geometrical ray map | 0.0132 | 1.0000048 |
| Extra F multiplied into the trace before Q⁻¹ | 0.00517 | 1.00799 |

Curved ASM also agrees with an independent direct modal evaluation to
4.65×10⁻¹² relative electric-field error. The inverse check is therefore paired
with an independent forward-field check, not used alone as evidence of accuracy.

The ray map nearly conserves power but does not reproduce the exact wave field.
This is a direct example of why power conservation alone is insufficient to
validate field accuracy. The last row is a deliberate negative control, not
a proposed modification.

### Factors measured on the main method

Final surface to vertex-plane transport; central 90% of each support, N=512:

| Geometry | Diameter / λ | Minimum F | Median F | Maximum F |
|---|---:|---:|---:|---:|
| Plano-convex | 4 | 0.9245 | 0.9952 | 1.0401 |
| Biconvex | 4 | 0.8863 | 1.0129 | 1.0407 |
| Sphere, central support | 2 | 0.9736 | 0.9918 | 1.00002 |
| Weak plano-convex | 200 | 0.9955 | 1.0001 | 1.0041 |

No caustic was crossed by these particular surface-to-vertex segments.
The 200λ case checks this local transport factor, not full-body BEM agreement.
The factor relates amplitudes at corresponding ray-mapped coordinates.

### Absolute detector intensities with the full ray inverse

Fixed common detector planes, incident amplitude 1 V/m, N=512. No plane-dependent
gain, peak alignment, or reference fitting is used. References are the existing
dense BEM/Mie data; the ray-map fields are newly evaluated on those coordinates.

| Geometry | Method | On-axis \(|E|^2\), V²/m² |
|---|---|---:|
| Plano-convex, 4λ | Ours | 6.205 |
| | Ours with ray-map Q⁻¹ | 6.297 |
| | BEM | 14.232 |
| Biconvex, 4λ | Ours | 8.274 |
| | Ours with ray-map Q⁻¹ | 8.516 |
| | BEM | 13.838 |
| Sphere, 2λ | Ours | 0.155 |
| | Ours with ray-map Q⁻¹ | 0.171 |
| | Mie | 7.072 |

Before and after the detector the large deficit also remains. The biconvex
on-axis intensity **decreases** before the detector even though the overall
plane intensity error improves slightly. Boundary-jump changes are mixed.
There is no consistent improvement sufficient to adopt this approximation.

Refinement of the full ray-map calculation from 512² to 768², at fixed FFT-cell
width, changes the complex field over the three planes by 0.884% (plano-convex),
0.810% (biconvex), and 0.771% (sphere). The tested refinement cannot account for
the much larger remaining deficits. All eight report figures were opened and
visually inspected; the very dark sphere row reflects its actual low amplitude,
not an omitted or constant field.

## What the evidence establishes

The proposed flux-conservation principle is valid and generalizes beyond
stigmatic surfaces. Its geometry produces a well-defined spatial factor, and
the implementation reproduces the independent conservation limits.

For the present cases, the factor does not recover the missing intensity.
Compatible fields have an accurate Q inverse;
incompatible Fresnel traces and the current absorbing supports remain separate
issues. The main method clips the field at each support, while BEM/Mie describe
the complete illuminated body. The sphere comparison additionally uses only a
central 0.45R chart in the main method. These facts do not excuse its deficit,
but they constrain any legitimate power target.

The field comparisons retain a substantial amplitude discrepancy despite the
small compatible-trace inverse error.

## Reproduce

From the repository root, using the unfitted main-method and reference fields
in `output/theoretical_flux_study/reference_fields/`:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python general/examples/theoretical_flux_study.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -c "import sys; sys.path.insert(0,'general/examples'); from theoretical_flux_study import lens_wave_checks; lens_wave_checks(768)"
.venv/bin/python general/examples/theoretical_flux_report.py
.venv/bin/pytest -q general/tests/test_transport_flux.py general/tests/test_operational_propagation.py
```
