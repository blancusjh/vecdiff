"""
Minimal two-feature scalar/vectorial lithography example.

Builds two line features, propagates them through two conjugate diopters,
and compares the scalar image with the vectorial image.
"""

import numpy as np
import matplotlib.pyplot as plt

from vecdiff import Grid, FieldCartesian, CartesianSurface
from _output import example_output_dir, print_saved


# ---------------------------------------------------------------------
# Optical parameters
# ---------------------------------------------------------------------

lam = 532e-6  # wavelength in mm

D1 = dict(n0=1.0, ni=2.4, z0=-6.0, zi=3.6)
D2 = dict(n0=D1["ni"], ni=1.0, z0=-D1["zi"], zi=3.6)

r_a = 7.0  # aperture radius in mm

alpha_max = np.arctan(r_a / abs(D1["z0"]))
Kc = D1["n0"] * (2.0 * np.pi / lam) * np.sin(alpha_max)

r_Airy = 3.8317059702075125 / Kc
d_Airy = 2.0 * r_Airy

theta = 0.25 * np.pi

sep = 0.90 * d_Airy
line_length = 2.7 * d_Airy
line_width = 0.35 * d_Airy


# ---------------------------------------------------------------------
# Numerical grid
# ---------------------------------------------------------------------

N = 769
L = 0.04  # mm

x = np.linspace(-L / 2.0, L / 2.0, N)
y = np.linspace(-L / 2.0, L / 2.0, N)
X, Y = np.meshgrid(x, y, indexing="xy")


# ---------------------------------------------------------------------
# Minimal utilities
# ---------------------------------------------------------------------

def kgrid_from_spacing(N, dx, dy):
    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dy))

    KX, KY = np.meshgrid(kx, ky, indexing="xy")

    return Grid.from_cartesian(KX, KY, domain="k")


def apply_circular_pupil(Ex, Ey, grid):
    kx = grid.X[0, :]
    ky = grid.Y[:, 0]
    KX, KY = np.meshgrid(kx, ky, indexing="xy")

    pupil = (KX**2 + KY**2) <= Kc**2

    return Ex * pupil, Ey * pupil, kx, ky


def normalize(I):
    I = np.asarray(I, dtype=float)
    return I / (np.max(I) + 1.0e-30)


def display(I):
    I = normalize(I)
    return np.clip(I, 0.0, 1.0) ** 0.82


# ---------------------------------------------------------------------
# Two-line mask
# ---------------------------------------------------------------------

ux = np.cos(theta)
uy = np.sin(theta)

vx = -uy
vy = ux

T = np.zeros_like(X, dtype=float)

for sign in (-1.0, 1.0):
    cx = sign * 0.5 * sep * ux
    cy = sign * 0.5 * sep * uy

    u = (X - cx) * ux + (Y - cy) * uy
    v = (X - cx) * vx + (Y - cy) * vy

    T[(np.abs(u) <= 0.5 * line_width) & (np.abs(v) <= 0.5 * line_length)] = 1.0


# Incident field: x-polarized mask
E0 = FieldCartesian(
    x=T.astype(complex),
    y=np.zeros_like(T, dtype=complex),
    grid=Grid.from_cartesian(X, Y),
    symmetric=False,
)


# ---------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------

def propagate_two_diopters(E0, vectorial=True):
    dx = float(np.mean(np.diff(E0.grid.X[0, :])))
    dy = float(np.mean(np.diff(E0.grid.Y[:, 0])))

    transmission = "vectorial" if vectorial else "identity"

    diopter1 = CartesianSurface(**D1)

    E1 = E0.propagate_through_diopter(
        z=diopter1.zi,
        ovoid=diopter1,
        method="fft",
        kgrid=kgrid_from_spacing(1024, dx, dy),
        transmission=transmission,
    )

    Ex1, Ey1, kx, ky = apply_circular_pupil(E1.x, E1.y, E1.grid)

    # Convert Fourier coordinates after the first diopter into focal-plane coordinates.
    scale = lam * D1["zi"] / (2.0 * np.pi * D1["ni"])

    xf = scale * kx
    yf = scale * ky
    XF, YF = np.meshgrid(xf, yf, indexing="xy")

    Ef = FieldCartesian(
        x=Ex1,
        y=Ey1,
        grid=Grid.from_cartesian(XF, YF),
        symmetric=False,
    )

    dxf = float(np.mean(np.diff(xf)))
    dyf = float(np.mean(np.diff(yf)))

    diopter2 = CartesianSurface(**D2)

    E2 = Ef.propagate_through_diopter(
        z=diopter2.zi,
        ovoid=diopter2,
        method="fft",
        kgrid=kgrid_from_spacing(1024, dxf, dyf),
        output="focal",
        wavelength=lam,
        transmission=transmission,
    )

    return E2


E_scalar = propagate_two_diopters(E0, vectorial=False)
E_vector = propagate_two_diopters(E0, vectorial=True)


# ---------------------------------------------------------------------
# Intensities
# ---------------------------------------------------------------------

# Diagnostic: image component parallel to the incident polarization.
I_scalar = normalize(np.abs(E_scalar.x)**2)
I_vector = normalize(np.abs(E_vector.x)**2)
I_diff = normalize(np.abs(I_vector - I_scalar))


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

xr = E_scalar.grid.X[0, :]
yr = E_scalar.grid.Y[:, 0]

extent_input = [x[0] / lam, x[-1] / lam, y[0] / lam, y[-1] / lam]
extent_output = [xr[0] / lam, xr[-1] / lam, yr[0] / lam, yr[-1] / lam]

fig, axes = plt.subplots(1, 4, figsize=(15, 4))

panels = [
    (T, extent_input, "Input mask"),
    (I_scalar, extent_output, "Scalar image"),
    (I_vector, extent_output, "Vectorial image"),
    (I_diff, extent_output, "|Vectorial - Scalar|"),
]

for ax, (A, extent, title) in zip(axes, panels):
    ax.imshow(
        display(A),
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )

    ax.set_title(title)
    ax.set_xlabel(r"$x/\lambda$")
    ax.set_ylabel(r"$y/\lambda$")
    ax.set_aspect("equal")

    # Simple central zoom.
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)

fig.suptitle(
    rf"$z_{{0,1}}={D1['z0']} \,\mathrm{{mm}}$, "
    rf"$z_{{i,1}}={D1['zi']} \,\mathrm{{mm}}$, "
    rf"$z_{{i,2}}={D2['zi']} \,\mathrm{{mm}}$, "
    rf"$r_a={r_a} \,\mathrm{{mm}}$, "
    rf"$\alpha_\max={np.degrees(alpha_max):.2f}^\circ$"
)

plt.tight_layout()
output_dir = example_output_dir(__file__)
path = output_dir / "scalar_vectorial_comparison.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
print_saved(path)
plt.show()


print("OK")
print(f"d_Airy = {d_Airy:.8e} mm")
print(f"sep = {sep:.8e} mm")
print(f"sep / d_Airy = {sep / d_Airy:.4f}")
print(f"alpha_max = {np.degrees(alpha_max):.4f} deg")
