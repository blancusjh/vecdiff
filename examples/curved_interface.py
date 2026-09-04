"""Per-k physical-optics transformation on an open spherical cap."""
import numpy as np
from vecdiff import (Medium, SphericalCap, DielectricInterface, plane_wave,
                     sample_surface, interface_transform)
from vecdiff.observables.electromagnetism import boundary_residuals

cap = SphericalCap(radius=20)
sampling = sample_surface(cap, (0, 10), (0, 2*np.pi), 160, 96)
n1, n2 = Medium(1), Medium(1.5)
incident = plane_wave(medium=n1)
result = interface_transform(incident, DielectricInterface(cap, n1, n2), sampling)
b = result.boundary
print("prescribed local trace:", boundary_residuals(
    b.incident_E+b.reflected_E, b.incident_H+b.reflected_H,
    b.transmitted_E, b.transmitted_H, sampling.normals, n1, n2,
    weights=sampling.weights,
))

points = np.array([[0, 0, 30], [.5, 0, 30]])
E, H = result.transmitted.evaluate(points)
print("transmitted E:", E)
