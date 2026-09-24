# Fourier transforms and coordinate-aware grids

This document describes the general library. The dedicated stigmatic Fourier
implementation remains unchanged during this refinement.

A sample array's shape does not identify its coordinates. Its Grid declares a
sampling chart and maps each index to a physical position, including translation
and rotation of the observation plane.

| Grid | Numerical coordinates | Array order | Position at an index |
|---|---|---|---|
| `CartesianGrid` | `(u, v)` | `(v, u, ...)` | `origin + u[j]*frame.u + v[i]*frame.v` |
| `PolarGrid` | `(radius, azimuth)` | `(radius, azimuth, ...)` | `origin + r[i]*(cos(phi[j])*frame.u + sin(phi[j])*frame.v)` |

Trailing dimensions are vector components or other independent data. They do not
change the sample positions. At radius zero, all angular entries refer to the
same point and have zero radial area weight. A single-valued continuous function
must have consistent values there. Azimuth covers one uniform period without a
duplicate endpoint. `PolarGrid.dr_weights` may supply nonuniform radial quadrature.

`PlaneGrid` remains an alias for `CartesianGrid`. There is no duplicated grid
implementation. Existing FFT propagation explicitly requires a Cartesian grid;
introducing polar sampling does not silently resample data or change that solver.

## Direct centered transforms

```python
from vecdiff_general.fourier import FFT2, IFFT2, frequency_grid

G = FFT2(g)
g = IFFT2(G)
fx, fy = frequency_grid((x, y))
```

`FFT2` and `IFFT2` transform axes `(0, 1)` of arrays ordered `(y, x, ...)`.
Trailing vector components or batches are neither shifted nor transformed.
The forward transform is unnormalized; the inverse divides by `ny*nx`, as in
NumPy. Both spatial and frequency arrays are centered, including for odd sizes.
The spatial reference is index `(ny//2, nx//2)`.

`frequency_grid` accepts either numerical 1D axes or matching 2D coordinate
meshes made with `np.meshgrid(x, y, indexing="xy")`. It checks uniform spacing,
increasing axes and rectilinear mesh structure. It returns **cycles per coordinate
unit**, in the same centered order as `FFT2`. Multiply by `2*pi` for angular
frequencies. No optical Grid object is passed into the mathematical package.

Propagation explicitly uses `FFT2(E)/N` for plane-wave amplitudes and `N*IFFT2`
for their synthesis. That normalization belongs to the modal sum. The reference
point is the central physical grid point, preserving phase on translated and
rotated planes without referring evanescent propagation to a remote origin.

The general API removes `fft_modes` and `inverse_fft_modes`; these are not aliases
for the new functions because both normalization and ordering have changed.
The old `(shape, spacing)` frequency helper is replaced by `(x, y)` coordinates.

## A common mathematical interface

Cartesian and polar modes use

```text
F(q) = integral f(x) exp(-i q dot x) d²x
f(x) = integral F(q) exp(+i q dot x) d²q / (2*pi)²
```

The Fourier module consumes numerical arrays and coordinate axes. It imports no
optical fields, materials, grids, or propagation algorithms.

```python
import numpy as np
from vecdiff_general import CartesianGrid, fourier

u = np.linspace(-6, 6, 128, endpoint=False)
v = np.linspace(-5, 5, 96, endpoint=False)
grid = CartesianGrid(u, v)
values = np.exp(-np.sum(grid.points[..., :2] ** 2, axis=-1))

spectrum = fourier.transform(
    values,
    mode=grid.coordinate_system,
    coordinates=grid.coordinates,
)

reconstructed = fourier.inverse(
    spectrum,
    mode=grid.coordinate_system,
    coordinates=grid.coordinates,
)
```

Cartesian output uses centered FFT frequency order. Use
`fx, fy = fourier.frequency_grid(grid.coordinates)` and `qx, qy = 2*pi*fx, 2*pi*fy`
for its angular frequency coordinates. The transform values include `dx*dy`;
unlike `FFT2`, they approximate the continuous integral.
The coordinate-axis origins enter the transform phase. A frame supplies the
embedding in global space; Fourier coordinates here refer to that frame's chart.
The discrete Cartesian forward/inverse pair is exact to roundoff on its periodic
cell. Finite support and discretization still limit its continuous approximation.

```python
from vecdiff_general import PolarGrid

radius = np.linspace(0, 6, 501)
azimuth = 2 * np.pi * np.arange(32) / 32
grid = PolarGrid(radius, azimuth)
values = np.exp(-(radius[:, None] ** 2)) * np.ones((1, len(azimuth)))

frequency_radius = np.linspace(0, 12, 601)
frequency_azimuth = azimuth
frequency_coordinates = (frequency_radius, frequency_azimuth)

spectrum = fourier.transform(
    values,
    mode=grid.coordinate_system,
    coordinates=grid.coordinates,
    frequencies=frequency_coordinates,
    weights=grid.dr_weights,
)

reconstructed = fourier.inverse(
    spectrum,
    mode=grid.coordinate_system,
    coordinates=grid.coordinates,
    frequencies=frequency_coordinates,
)
```

Polar transformation decomposes angular harmonics and integrates each one with
its corresponding Bessel order. It handles non-axisymmetric scalar and vector
arrays. `weights` means **dr** for forward transformation and **dq** for inverse
transformation; radial Jacobians and angular measures are included internally.
Default radial weights are trapezoidal. Both angular axes use a uniform full
period; resolve the angular harmonics below their sampling limit.

The polar inverse is a quadrature approximation. Radial cutoffs, radial weights
and angular resolution must be converged. The tests use analytic Gaussian modes
of orders 0, +1 and -2, shifted angular origins, and Gauss radial quadrature to
verify phase, normalization and inverse reconstruction. They also compare
Cartesian and polar calculations at common frequency coordinates.

## Direct polar operations

```python
from vecdiff_general.fourier.polar import FT2, IFT2, HT, IHT

G = FT2(g, (r, phi), (q, theta), weights=dr)
g = IFT2(G, (r, phi), (q, theta), weights=dq)

G_radial = HT(g_radial, r, q, order=0, weights=dr)
g_radial = IHT(G_radial, q, r, order=0, weights=dq)
```

`FT2` applies an angular Fourier expansion, one Hankel integral per harmonic,
and the forward factor `2*pi`. `IFT2` reverses the Fourier sign and applies
`1/(2*pi)` after angular integration. Every stage is separate in the code.
`HT` and `IHT` contain no `2*pi` factor: their measures are `r*dr` and `q*dq`.
Optional weights specify `dr` or `dq`; the radial Jacobian is applied internally.
The first axis of Hankel data is radial; trailing components and batches are
independent, matching the coordinate-first convention of the other transforms.
Polar data uses `(radius, angle, ...)`. Angular frequency `q` is in radians per
coordinate unit. The angular axes retain their supplied origins and order.

The common `fourier.transform(..., mode="polar")` and `fourier.inverse` dispatch
to these functions. Neither applies a Cartesian 2D FFT to the polar array.
There is no universally inferred radial frequency grid: callers choose radial
frequency support and quadrature explicitly. The inverse integrals require
convergence with support and resolution; they are not exact matrix inverses.

The former general `hankel_transform(order, r, values, q)` becomes
`HT(g, r, q, order=order)`. Move an old last radial axis to the first axis when
migrating multidimensional input. Direct `polar.transform` / `polar.inverse`
calls become `FT2` / `IFT2`; the common top-level interface remains available.
Angular harmonic truncation and cylindrical synthesis remain available.

## Nonuniform operations

```python
from vecdiff_general.fourier import NUFFT

G = NUFFT(g, x, q)
G_direct = NUFFT(g, x, q, backend="direct")
```

This is the unnormalized sum `G[k] = sum_j g[j] * exp(-1j * q[k].x[j])`.
`x` has shape `(nodes, dimension)`, `g` has shape `(nodes, ...)`, and `q` has
shape `(..., dimension)`. Dimensions 1, 2 and 3 are supported. Trailing value
axes are independent components or batches. Angular frequencies and coordinates
are supplied explicitly; there is no implicit shift or `2*pi` conversion.

The default backend uses optional FINUFFT. The direct backend needs only NumPy
and also supports complex exponential arguments. Zero contributions are removed
before exponentiation, so an unoccupied growing term cannot overflow a result.
`sign=1` gives a positive-sign sum; **it does not invert a general nonuniform
matrix**. Consequently there is no misleading exact `INUFFT` operation.

For continuous integral quadrature, use explicit measures:

```python
from vecdiff_general import fourier

G = fourier.transform(
    g,
    mode="nonuniform",
    coordinates=x,
    frequencies=q,
    weights=dx,
)
g = fourier.inverse(
    G,
    mode="nonuniform",
    coordinates=x,
    frequencies=q,
    weights=dq,
)
```

Here `x` and `q` are node lists `(nodes, dimension)`. `dx` and `dq` mean full
spatial and angular-frequency integration weights, respectively, including any
coordinate Jacobian. Both calls accept `backend="direct"`, `tolerance`, and
`chunk_size`; these options apply only to nonuniform mode. The inverse applies
`1/(2*pi)**dimension`. Finite support and quadrature determine its accuracy.
The same operations are available as `nufft.transform` / `nufft.inverse`.

The former general `synthesize(q, amplitudes, x, backend=...)` becomes
`NUFFT(amplitudes, q, x, sign=1, backend=...)`. Its amplitudes already include
integration weights, so no additional normalization is applied. Field evaluation
and the forward surface-current transform now state their signs explicitly.

## Files

`fourier/cartesian.py` contains Cartesian transforms; `fourier/polar.py` contains
polar transforms, Hankel integrals and angular harmonics; `fourier/nufft.py`
contains nonuniform sums and weighted integral transforms. `fourier/__init__.py` exposes the common interface.
There is no separate Hankel module. `FFT2`/`IFFT2` and the coordinate-aware
`transform`/`inverse` share one implementation, with their normalizations stated
explicitly. Optical modal construction remains in `propagation/`.
