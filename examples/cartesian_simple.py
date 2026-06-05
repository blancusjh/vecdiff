import numpy as np
import matplotlib.pyplot as plt

from vecdiff.CartesianSurfaces import CartesianSurface
from vecdiff.fields import Field
from vecdiff.grid import Grid
from vecdiff.propagation import propagate_to_focal_plane_through_diopter
from vecdiff.coordinate_transformation import polar_grid_to_cartesian_grid


# Optical system
# Unit convention:
# - R, z0, zi, r, x, y are in the same length unit (project convention: mm).
# - lam must use that same unit (532e-6 mm = 532 nm).
n0 = 1.0
ni = 1.5
z0 = -10.0
zi = 6.0
lam = 532e-6

# Input amplitudes and pupil
EX0_AMP = 1.0
EY0_AMP = 0.0
R = 2.6
N_R = 1000
N_Q = 1000
N_PHI = 256
N_XY = 512

r = np.linspace(0.0, R, N_R)  # radial source coordinate [same unit as R]
q_max = (ni / (lam * zi)) * R
q = q_max * np.linspace(0.0, 1.0, N_Q) ** 2.0
varphi = np.linspace(0.0, 2.0 * np.pi, N_PHI, endpoint=False)

a = 0.95 * R
P = (r <= a).astype(float)

# Cartesian input field
Ex0 = P * EX0_AMP
Ey0 = P * EY0_AMP

grid = Grid.from_polar(r, varphi)
E0 = Field.from_cartesian(Ex0, Ey0, grid, symmetric=True)
Sigma = CartesianSurface(n0=n0, ni=ni, z0=z0, zi=zi)
E_PRIME = propagate_to_focal_plane_through_diopter(E0, Sigma, q)

Ex = E_PRIME.x
Ey = E_PRIME.y

# 2D visualization on Cartesian mesh in physical coordinates (x, y)
q_view = 0.0045 * q_max
x_view = (R / q_max) * q_view  # convert q-window to physical x-window [same unit as R]
xs = np.linspace(-x_view, x_view, N_XY)
xx, yy = np.meshgrid(xs, xs)

# Convert physical coordinates to q-coordinates for interpolation:
# q = (q_max / R) * rho, with rho = sqrt(x^2 + y^2)
xx_q = (q_max / R) * xx
yy_q = (q_max / R) * yy

Ex_xy = polar_grid_to_cartesian_grid(Ex, q, varphi, xx_q, yy_q, fill_value=0.0)
Ey_xy = polar_grid_to_cartesian_grid(Ey, q, varphi, xx_q, yy_q, fill_value=0.0)

I_Ex = np.abs(Ex_xy) ** 2
I_Ey = np.abs(Ey_xy) ** 2
I_total = I_Ex + I_Ey

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
xs_lambda = xs / lam
extent = [
    float(xs_lambda[0]),
    float(xs_lambda[-1]),
    float(xs_lambda[0]),
    float(xs_lambda[-1]),
]

im0 = axs[0].imshow(I_Ex, extent=extent, origin="lower", aspect="equal", cmap="hot", vmin=0)
axs[0].set_title("|Ex|^2")
plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

im1 = axs[1].imshow(I_Ey, extent=extent, origin="lower", aspect="equal", cmap="hot", vmin=0)
axs[1].set_title("|Ey|^2")
plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

im2 = axs[2].imshow(I_total, extent=extent, origin="lower", aspect="equal", cmap="hot", vmin=0)
axs[2].set_title("|Ex|^2 + |Ey|^2")
plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

for ax in axs:
    ax.set_xlabel(r"$x / \lambda$")
    ax.set_ylabel(r"$y / \lambda$")

plt.tight_layout()
plt.show()
