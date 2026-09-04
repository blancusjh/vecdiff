"""Exact per-k reflection and refraction at an infinite dielectric plane."""
import numpy as np
from vecdiff import Medium, Plane, DielectricInterface, plane_wave, interface_transform
from vecdiff.observables.electromagnetism import boundary_residuals

n1, n2 = Medium(1), Medium(1.5)
theta = np.deg2rad(35)
incident = plane_wave((np.sin(theta), 0, np.cos(theta)), (0, 1, 0), medium=n1)
result = interface_transform(incident, DielectricInterface(Plane(), n1, n2))
points = np.stack((np.linspace(-2, 2, 41), np.zeros(41), np.zeros(41)), axis=-1)
Ei, Hi = incident.evaluate(points)
Er, Hr = result.reflected.evaluate(points)
Et, Ht = result.transmitted.evaluate(points)
print(boundary_residuals(Ei+Er, Hi+Hr, Et, Ht, [0, 0, 1], n1, n2))
