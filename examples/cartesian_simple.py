import numpy as np
import matplotlib.pyplot as plt

from vecdiff import CartesianSurface, FieldCartesian, Grid
from vecdiff.view import plot_field


n0, ni = 1.0, 1.5
z0, zi = -10.0, 6.0
lam = 532e-6
R = 2.6

n_r, n_q, n_phi = 1000, 1000, 256
r = np.linspace(0.0, R, n_r)
q = (ni * R / (lam * zi)) * np.linspace(0.0, 1.0, n_q) ** 2
phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

pupil = r <= 0.95 * R
grid = Grid.from_polar(r, phi)
diopter = CartesianSurface(n0=n0, ni=ni, z0=z0, zi=zi)

E0 = FieldCartesian(x=1.0 * pupil, y=0.0 * pupil, grid=grid)
E = E0.propagate_through_diopter(zi, diopter, q)

print(f"theta = {np.rad2deg(np.arctan(R / zi)):.3f} deg")
plot_field(E0, half_size=R, title="Input Cartesian field")
plot_field(E, half_size=4.0 * 3.8317059702075125 / (0.95 * R), title="Propagated Cartesian field")
plt.show()
