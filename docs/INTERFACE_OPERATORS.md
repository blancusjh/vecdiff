# Electric-field operators at an interface

A transmission or reflection acts on an electric field at a fixed physical
surface. It changes the electric samples, while retaining the grid and vacuum
wavelength. A direction field is supplied explicitly; it is not inferred from an
interfering total field.

```python
import numpy as np

from vecdiff_general import ElectricField, SurfaceGrid
from vecdiff_general.interfaces.directions import transmitted_direction
from vecdiff_general.interfaces.fresnel import (
    apply_fresnel,
    fresnel_transmission_matrix,
    transmit,
)

normal = np.array([0.0, 0.0, 1.0])
incident_direction = np.array([0.6, 0.0, 0.8])

transmitted = transmitted_direction(
    incident_direction, normal, incident_index=1.0, transmitted_index=1.5
)

transmission = fresnel_transmission_matrix(
    incident_direction,
    transmitted,
    normal,
    incident_index=1.0,
    transmitted_index=1.5,
)

incident_electric = np.array([0.8, 0.0, -0.6])
transmitted_electric = apply_fresnel(incident_electric, transmission)

# The equivalent field operation retains the physical surface grid.
grid = SurfaceGrid([[0.0, 0.0, 0.0]], [normal], [1e-12])
field = ElectricField(grid, wavelength0=532e-9, values=[incident_electric])

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

`transmission_coefficients` returns `(t_s, t_p)` only.
`reflection_coefficients` returns `(r_s, r_p)` only.
`fresnel_reflection_matrix` and `reflect` provide the reflected operation without
calculating a transmitted field or its amplitude coefficients. A complete
`scatter_plane` calculation deliberately requests both branches. The dedicated
stigmatic package exposes its own independent implementation of these operators.

The convention here is a normal **into the transmitted material**. Therefore
`cos_i = dot(u_i, normal)`; a normal into incident matter has the opposite sign.
At normal incidence, an arbitrary regular transverse basis gives the same
Cartesian operator. Under TIR, the transmitted direction is complex and must not
be normalized using a Hermitian norm. Existing boundary and flux tests cover
these cases; additional tests assert that the unrequested coefficients are never
called and that the field's physical support is preserved.
