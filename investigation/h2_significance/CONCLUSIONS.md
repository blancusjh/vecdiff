# H2 Significance Investigation (Exact vs Paraxial)

## What was tested
- Swept aperture and geometry with **both components active** (`EL` and `ER` not suppressed).
- Compared paraxial vs exact Fresnel coefficients.
- Metrics used:
  - `ER_energy(H2/H0)`
  - `ER_peak(H2/H0)`
  - `intensity_delta_from_H2` = weighted relative change in total intensity when removing only the `H2` term from `E'_R`.

## Main findings
- Increasing `a/R` and `R` (higher NA) consistently increases practical H2 impact.
- In the initial flat-profile sweep (`EL=ER=1` inside aperture), **paraxial** produced larger H2 metrics than exact.
- With **exact Fresnel**, H2 became clearly more relevant by keeping both components equal but using a **symmetric edge-weighted profile**:
  - `EL(r) = ER(r) = (r/a)^p` for `r<=a`, with `p` around `4..6`.
- Best tested practical exact case (balanced, not extreme):
  - `R = 2.0`, `a/R = 0.95`, `ni = 2.2`, `z0 = 2.0`, `zi = 8.0`, `p = 4`
  - Gives about `intensity_delta_from_H2 ≈ 0.048` (around 4.8% net intensity impact from H2 removal test).

## Recommended parameter set (exact default)
- `n0=1.0`, `ni=2.2`
- `z0=2.0`, `zi=8.0`
- `R=2.0`, `a=0.95*R`
- focus-refined mesh (`power ~2.2..2.5`)
- symmetric profile exponent `p=4`

## Notes
- `H2` appears in `E'_R` (not directly in `E'_L`), so the "both components" practical criterion is best measured through **total intensity** impact.
- Exact Fresnel implementation had two code issues fixed during this investigation:
  - `fresnel.py` now imports `numpy`.
  - `FresnelOvoid` constructor now accepts `n0/z0` correctly (with legacy `no/zo` compatibility).
  - `CartesianSurfaces.py` now passes correct parameter names to `GOTS_params`.
