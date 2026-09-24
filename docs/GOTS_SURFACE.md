# GOTS Cartesian oval in the general library

```python
import numpy as np
import vecdiff_general as optics

surface = optics.CartesianOval.from_conjugates(no=1.0, z0=-100e-6, ni=1.5, zi=60e-6)
G, O, T, S = surface.gots
rho = np.linspace(0, surface.rho_from_r(12e-6), 64)
phi = np.linspace(0, 2 * np.pi, 96, endpoint=False)
points = surface.parametric_points(rho[:, None], phi[None, :])
```

`gots_parameters(no, z0, ni, zi)` also returns the four coefficients directly.
Their equations follow the supplied thesis, Appendix D, D.3–D.8, and the
historical `GOTS_parameters.py` and `CartesianSurfaces.py` at revision `7a1a235`.
Notebook 05 compares matched spherical and stigmatic lenses using the collimated-input limit.

`rho` is distance from the local vertex, not cylindrical aperture radius.
`surface.z(rho)` and `surface.r(rho)` evaluate the explicit GOTS parametrization.
`parametric_points(rho, phi)` revolves it and applies the physical frame.
Coordinates are metres; azimuth is radians. G, O, T and S have dimensions
1, inverse length, inverse length cubed and inverse length squared.

`height(xy)`, `gradient(xy)` and standard surface quadrature retain the existing
physical cylindrical chart. `rho_from_r` obtains the inverse parameter through
the unsquared constant-optical-path equation. This avoids quartic root ambiguity
and preserves the established graph calculation as an independent check of the
explicit parametrization. Shape coefficients do not assign material regions.

The supported branch has real conjugates `z0 < 0 < zi`, positive unequal
indices, and a single-valued vertex-connected graph. `z0=-np.inf` explicitly
selects collimated incidence along local +z; `zi` must remain finite. Invalid designs, nonfinite
or negative parameters, and parameters outside that graph raise errors. The
geometrical branch limit is not a certificate that all rays are forward or
away from grazing: propagation separately checks those physical conditions.
Virtual conjugates, infinite image distance, equal-index limits and other oval
branches are not implemented.

The tests compare explicit geometry with the existing graph, constant optical
path and independent Snell refraction, for both index orderings and multiple
length scales. Placement, coefficient units, axial regularity and invalid inputs
are also checked. The notebook checks local signed flux and complex-field
quadrature convergence, while explaining the physical-optics approximation.

## Collimated input and matched vertex curvature

For `z0=-np.inf`, the analytic GOTS limits are `G=-(no/ni)**2`,
`O=ni/(zi*(ni-no))`, and `T=S=0`. The vertex-referenced optical path is
`no*z + ni*sqrt(r**2 + (zi-z)**2) = ni*zi`. The library uses this finite
quantity rather than subtracting infinite paths. `design.object_point` raises
an error because there is no finite source point in this limit.

To match a spherical face's signed vertex radius `R`, choose
`zi = ni*R/(ni-no)`, giving `O=1/R`. The cylindrical sag is evaluated through
a rationalized analytic root and its gradient through the unsquared path.
Tests check the finite-conjugate limit, constant path, Snell refraction and
vertex curvature for both index orderings.

Notebook 05 retains the same entrance face, diameter, central thickness,
materials, incident beam and physical observation grids. It replaces only the
spherical exit face. The edge sag changes, as required by the aspheric shape.
Paired field maps use shared color scales and a common image plane.
