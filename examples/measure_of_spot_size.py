import numpy as np
import matplotlib.pyplot as plt

from vecdiff import CartesianSurface, FieldCartesian, Grid
from vecdiff.view import radial_map


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

E0 = FieldCartesian(x=1.0 * pupil, y=1.0 * pupil, grid=grid)
E = E0.propagate_through_diopter(zi, diopter, q)

profile = np.abs(E.x[0]) ** 2 + np.abs(E.y[0]) ** 2
profile /= profile[0] + 1e-30

below_half = np.flatnonzero(profile <= 0.5)
fwhm = np.interp(0.5, profile[below_half[0] - 1:below_half[0] + 1][::-1], q[below_half[0] - 1:below_half[0] + 1][::-1])

turns = np.flatnonzero((np.diff(profile)[:-1] < 0.0) & (np.diff(profile)[1:] > 0.0))
first_min = q[turns[0] + 1] if turns.size else np.nan
rayleigh = 3.8317059702075125 / (0.95 * R)

for name, value in [("Rayleigh", rayleigh), ("FWHM", fwhm), ("first minimum", first_min)]:
    print(f"{name}: {value:.6g}")

half_size = 4.0 * rayleigh
image, extent = radial_map(profile, q, half_size, 600)

fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
im = ax.imshow(image, extent=extent, origin="lower", cmap="hot", vmin=0.0, vmax=1.0)
for radius, color, label in [(rayleigh, "cyan", "Rayleigh"), (fwhm, "lime", "FWHM"), (first_min, "magenta", "first minimum")]:
    if np.isfinite(radius):
        ax.add_patch(plt.Circle((0.0, 0.0), radius, color=color, fill=False, label=label))
ax.set_xlabel("qx")
ax.set_ylabel("qy")
ax.set_title("Spot size")
ax.legend(loc="upper right")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.show()
