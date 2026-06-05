import numpy as np
import matplotlib.pyplot as plt

from vecdiff.CartesianSurfaces import CartesianSurface
from vecdiff.fields import Field
from vecdiff.grid import Grid
from vecdiff.propagation import propagate_to_focal_plane_through_diopter
from vecdiff.coordinate_transformation import (
    circular_to_cartesian,
    polar_grid_to_cartesian_grid,
)


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
EL0_AMP = 0.0
ER0_AMP = 1.0
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

# Circular input field
EL0 = P * EL0_AMP
ER0 = P * ER0_AMP

grid = Grid.from_polar(r, varphi)
E0 = Field.from_circular(EL0, ER0, grid, symmetric=True)
Sigma = CartesianSurface(n0=n0, ni=ni, z0=z0, zi=zi)
E_PRIME = propagate_to_focal_plane_through_diopter(E0, Sigma, q)

EL = E_PRIME.L
ER = E_PRIME.R
Ex, Ey = circular_to_cartesian(EL, ER)

# 2D visualization on Cartesian mesh in physical coordinates (x, y)
q_view = 0.0045 * q_max
x_view = (R / q_max) * q_view  # convert q-window to physical x-window [same unit as R]
xs = np.linspace(-x_view, x_view, N_XY)
xx, yy = np.meshgrid(xs, xs)

# Convert physical coordinates to q-coordinates for interpolation:
# q = (q_max / R) * rho, with rho = sqrt(x^2 + y^2)
xx_q = (q_max / R) * xx
yy_q = (q_max / R) * yy

EL_xy = polar_grid_to_cartesian_grid(EL, q, varphi, xx_q, yy_q, fill_value=0.0)
ER_xy = polar_grid_to_cartesian_grid(ER, q, varphi, xx_q, yy_q, fill_value=0.0)
Ex_xy = polar_grid_to_cartesian_grid(Ex, q, varphi, xx_q, yy_q, fill_value=0.0)
Ey_xy = polar_grid_to_cartesian_grid(Ey, q, varphi, xx_q, yy_q, fill_value=0.0)

I_EL = np.abs(EL_xy) ** 2
I_ER = np.abs(ER_xy) ** 2
I_total = I_EL + I_ER

I_Ex = np.abs(Ex_xy) ** 2
I_Ey = np.abs(Ey_xy) ** 2

fig_lr, axs_lr = plt.subplots(1, 3, figsize=(12, 4))
xs_lambda = xs / lam
extent = [
    float(xs_lambda[0]),
    float(xs_lambda[-1]),
    float(xs_lambda[0]),
    float(xs_lambda[-1]),
]

im0 = axs_lr[0].imshow(I_EL, extent=extent, origin="lower", aspect="equal", cmap="hot", vmin=0)
axs_lr[0].set_title("Output |EL|^2")
plt.colorbar(im0, ax=axs_lr[0], fraction=0.046, pad=0.04)

im1 = axs_lr[1].imshow(I_ER, extent=extent, origin="lower", aspect="equal", cmap="hot", vmin=0)
axs_lr[1].set_title("Output |ER|^2")
plt.colorbar(im1, ax=axs_lr[1], fraction=0.046, pad=0.04)

im2 = axs_lr[2].imshow(I_total, extent=extent, origin="lower", aspect="equal", cmap="hot", vmin=0)
axs_lr[2].set_title("Output |EL|^2 + |ER|^2")
plt.colorbar(im2, ax=axs_lr[2], fraction=0.046, pad=0.04)

for ax in axs_lr:
    ax.set_xlabel(r"$x / \lambda$")
    ax.set_ylabel(r"$y / \lambda$")

fig_lr.suptitle("Output field in circular L/R representation")

fig_xy, axs_xy = plt.subplots(1, 3, figsize=(12, 4))

im3 = axs_xy[0].imshow(I_Ex, extent=extent, origin="lower", aspect="equal", cmap="hot", vmin=0)
axs_xy[0].set_title("Output |Ex|^2 from L/R")
plt.colorbar(im3, ax=axs_xy[0], fraction=0.046, pad=0.04)

im4 = axs_xy[1].imshow(I_Ey, extent=extent, origin="lower", aspect="equal", cmap="hot", vmin=0)
axs_xy[1].set_title("Output |Ey|^2 from L/R")
plt.colorbar(im4, ax=axs_xy[1], fraction=0.046, pad=0.04)

im5 = axs_xy[2].imshow(I_Ex + I_Ey, extent=extent, origin="lower", aspect="equal", cmap="hot", vmin=0)
axs_xy[2].set_title("Output |Ex|^2 + |Ey|^2")
plt.colorbar(im5, ax=axs_xy[2], fraction=0.046, pad=0.04)

for ax in axs_xy:
    ax.set_xlabel(r"$x / \lambda$")
    ax.set_ylabel(r"$y / \lambda$")

fig_xy.suptitle("Output field converted to Cartesian representation")

plt.tight_layout()
plt.show()
