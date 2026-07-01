import numpy as np
import matplotlib.pyplot as plt

from vecdiff import CartesianSurface, FieldCartesian, Grid
from vecdiff.polarization_visualization import plot_field_polarization
from vecdiff.view import plot_field
from _output import example_output_dir, print_saved


def field_in_focal_wavelengths(field, zi, ni):
    if field.grid.type == "cartesian":
        # FFT output is sampled in angular wavevector coordinates.
        scale = zi / (2.0 * np.pi * ni)
        grid = Grid.from_cartesian(
            scale * field.grid.X,
            scale * field.grid.Y,
            domain="focal/lambda",
            dual=field.grid.dual,
        )
    elif field.grid.type == "polar":
        # The Hankel convention is q = n*r_f/(lambda*f).
        scale = zi / ni
        grid = Grid.from_polar(scale * field.grid.r, field.grid.varphi)
        grid.domain = "focal/lambda"
    else:
        raise ValueError(f"Unsupported focal grid type: {field.grid.type!r}.")
    return FieldCartesian(field.x, field.y, grid=grid, symmetric=False)


n0, ni = 1.0, 1.5
z0, zi = -10.0, 6.0
lam = 532e-6
# Matches the uniform Cartesian Hankel case used for the thesis
# polarization maps and makes the cross-polarized Ey channel resolvable.
R = 2.6

n_r, n_q, n_phi = 1000, 1000, 256
r = np.linspace(0.0, R, n_r)
q = (ni * R / (lam * zi)) * np.linspace(0.0, 1.0, n_q) ** 2
phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

pupil = r <= 0.95 * R
grid = Grid.from_polar(r, phi)
diopter = CartesianSurface(n0=n0, ni=ni, z0=z0, zi=zi)

E0 = FieldCartesian(x=1.0 * pupil, y=0.0 * pupil, grid=grid)
E = E0.propagate_through_diopter(zi, diopter, q, method="hankel")
E_focal = field_in_focal_wavelengths(E, zi, ni)

input_half_size = R
propagated_half_size = 20.0
output_dir = example_output_dir(__file__)

fig, _ = plot_field(E0, half_size=input_half_size, title="Input Cartesian field")
path = output_dir / "input_field_components.png"
fig.savefig(path, dpi=220, bbox_inches="tight")
print_saved(path)

fig, axes = plot_field(E_focal, half_size=propagated_half_size, title="Propagated Cartesian field")
for ax in axes:
    ax.set_xlabel(r"$x/\lambda$")
    ax.set_ylabel(r"$y/\lambda$")
path = output_dir / "propagated_field_components.png"
fig.savefig(path, dpi=220, bbox_inches="tight")
print_saved(path)

ax, _ = plot_field_polarization(E0, half_size=input_half_size)
ax.set_title("Input Cartesian polarization")
path = output_dir / "input_polarization.png"
ax.figure.savefig(path, dpi=220, bbox_inches="tight")
print_saved(path)

ax, _ = plot_field_polarization(E_focal, half_size=propagated_half_size)
ax.set_title("Propagated Cartesian polarization")
ax.set_xlabel(r"$x/\lambda$")
ax.set_ylabel(r"$y/\lambda$")
path = output_dir / "propagated_polarization.png"
ax.figure.savefig(path, dpi=220, bbox_inches="tight")
print_saved(path)

# Cross-polarization diagnostic: Ex has nodal rings where Ey can dominate
# locally despite the low total intensity.
ax, _ = plot_field_polarization(
    E_focal,
    half_size=propagated_half_size,
    background="cross_fraction",
    cross_fraction_min_intensity=1e-7,
    glyph="quiver",
    min_intensity_fraction=1e-7,
    min_cross_fraction=0.50,
    target_arrows=120,
)
ax.set_title(r"Ey-dominant polarization: $|E_y|^2 / |E|^2 \geq 0.5$")
ax.set_xlabel(r"$x/\lambda$")
ax.set_ylabel(r"$y/\lambda$")
path = output_dir / "propagated_cross_polarization.png"
ax.figure.savefig(path, dpi=220, bbox_inches="tight")
print_saved(path)
plt.show()
