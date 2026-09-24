# Ideal aplanatic imaging

`vecdiff_general.propagation.imaging` supplies explicit ideal pupil operators for the application notebooks. Geometry, matter, electric fields and mathematical Fourier transforms retain their existing responsibilities. These operators live in propagation because they map fields between optical planes.

## Conjugate image planes

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
defocused = image_modes.propagate(1e-6)
```

The input is a forward `PlaneSpectrum`. Signed transverse magnification maps local coordinates by `x_image = m * x_object`; a negative value inverts the object. Pixel spacing scales by |m|, while the physical frame locates and orients the image plane. An omitted frame retains the source frame; it does not calculate an image distance. Offset coordinate windows are mapped about the declared frame origin.

For object transverse frequency q = λ₀ f, the aperture accepts |q| ≤ |m| NA_image. The object and image direction cosines are

```text
c_object = sqrt(1 - |q|² / n_object²)
c_image  = sqrt(1 - |q|² / (m² n_image²))
```

The meridional and azimuthal components are transported between incident and emerging directions. Their amplitude factor is

```text
gain = sqrt(n_object c_object / (n_image c_image)) / |m|
```

Together with the scaled transverse area, this conserves the normal Poynting power of the modes passed by a unitary pupil. Rejected modes lose their power; the operator does not renormalize that loss. Both apertures must remain below grazing incidence, and the object pupil must fit inside the FFT frequency rectangle. Evanescent modes do not enter this ideal imaging model.

An optional `pupil` is a scalar amplitude, an array on the object frequency grid, or a 2 × 2 Jones matrix (constant or grid-dependent). Jones channels are ordered **(p, s)** in the transported mode bases. The caller must specify a realizable pupil. The API does not infer objective aberrations, coating losses or a lens prescription.

## Pupil illumination and focal planes

`focus_pupil(transverse_field, object_region, image_region, focal_length=..., numerical_aperture=...)` maps a prescribed collimated Cartesian pupil to a vector focal spectrum under the aplanatic sine condition. The pupil coordinate axes must be centered. Physical radius obeys ρ = f sin θ, hence aperture radius f NA / n_image.

Polarization rotates into the outgoing transverse plane. The amplitude factor sqrt(n_object / (n_image cos θ)) and Fourier measure n_image du dv / (λ₀ f) preserve normal power from the illuminated pupil. The focal frame is displaced by f along the input frame normal. Full E and H satisfy homogeneous Maxwell dispersion and transversality. Finite incident support can diffract; this is not an exact field solution at an objective's surfaces.

The result supports FFT evaluation and evaluation on another physical grid through `spectrum.evaluate(grid, backend="nufft")`. A smaller evaluation spacing resolves the same spectrum more finely; it does not add omitted pupil bandwidth or establish convergence of the original pupil quadrature.

## Compensation and scalar references

`apply_pupil(spectrum, J)` applies a Jones operator in each forward propagating mode's (p, s) basis. It keeps the grid and wavevectors; H follows the transformed E. An inverse Jones pupil may require extra incident power. Normalize that cost explicitly when comparing illumination strategies.

For a dielectric-interface response embedded in a power-normalized pupil, use

```text
tau_s,p = t_s,p * sqrt(n_transmitted cos_transmitted / (n_incident cos_incident))
```

Using raw electric transmission coefficients as these normalized pupil multipliers would count the flux Jacobian incorrectly. Notebook 08 also verifies reflected plus transmitted normal power for both polarizations.

`scalar_image(...)` returns an array using the same aperture, Jacobian and exact defocus dispersion, without vector polarization transport. Its output uses the same coordinate mapping as `image_spectrum`. The notebook normalizes scalar and vector inputs to equal normal power. A scalar result is not packaged as a Maxwell electric field. Scalar and x/y illumination comparison panels have no polarization glyphs. A separate diagonal-input panel shows polarization over the complete vector field's total intensity.

## Scope of the DUV example

Notebook 10 defines an ideal 200 mm / 50 mm two-objective relay at 193 nm with magnification −0.25 and image NA 0.85. The image operator represents the relay's ideal isoplanatic response, rather than tracing two manufactured lenses. Statistical source points and thin mask transmissions are explicit. An actual DUV system additionally needs a validated surface or measured Jones pupil, rigorous mask fields where relevant, absorbing material models and a dose/resist model.

## Verification

`general/tests/test_ideal_imaging.py` checks passed normal power, homogeneous propagation power, Maxwell transversality, even/odd FFT inversion, offset coordinate windows, low-NA scalar agreement, pupil attenuation and inverse compensation. The holography section of notebook 06 checks full-plane power independently of its cropped display grid. Notebook 10 compares source-angle quadrature and adds incoherent detector responses, never source phasors.
