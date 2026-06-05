import numpy as np
import matplotlib.pyplot as plt

from vecdiff.CartesianSurfaces import CartesianSurface
from vecdiff.fields import Field
from vecdiff.grid import Grid
from vecdiff.propagation import HT, propagate_to_focal_plane_through_diopter
from vecdiff.coordinate_transformation import polar_grid_to_cartesian_grid


# Length units are mm
n0 = 1.0
ni = 1.5
z0 = -10.0
zi = 6.0
lam = 532e-6

EX0_AMP = 1.0
EY0_AMP = 1.0
R = 2.6

N_R = 1000
N_Q = 1000
N_PHI = 256
N_XY = 700


def crossing_radius(r, y, level):
    for i in range(1, len(y)):
        if y[i] <= level < y[i - 1]:
            r0, r1 = r[i - 1], r[i]
            y0, y1 = y[i - 1], y[i]
            return r0 + (level - y0) * (r1 - r0) / (y1 - y0)
    return np.nan


def first_minimum_radius(r, y):
    d = np.diff(y)
    for i in range(1, len(d)):
        if d[i - 1] < 0 and d[i] > 0:
            return r[i]
    return np.nan


def radial_cut(I2d, xs):
    mid = I2d.shape[0] // 2
    r = xs[mid:] - xs[mid]
    y = I2d[mid, mid:]
    y = y / (y[0] + 1e-30)
    return r, y


def add_circle(ax, rad_mm, color, label, lw=1.6):
    if np.isfinite(rad_mm):
        ax.add_patch(plt.Circle((0.0, 0.0), rad_mm, color=color, fill=False, lw=lw, label=label))


def main():
    # Grids
    r = np.linspace(0.0, R, N_R)
    q_max = (ni / (lam * zi)) * R
    q = q_max * np.linspace(0.0, 1.0, N_Q) ** 2.0
    varphi = np.linspace(0.0, 2.0 * np.pi, N_PHI, endpoint=False)

    # Pupil and input field
    a = 0.95 * R
    P = (r <= a).astype(float)
    Ex0 = P * EX0_AMP
    Ey0 = P * EY0_AMP

    # Propagation (vector)
    grid = Grid.from_polar(r, varphi)
    E0 = Field.from_cartesian(Ex0, Ey0, grid, symmetric=True)
    Sigma = CartesianSurface(n0=n0, ni=ni, z0=z0, zi=zi)
    E = propagate_to_focal_plane_through_diopter(E0, Sigma, q)

    # Cartesian plotting grid in physical x,y
    q_view = 0.012 * q_max
    x_view = (R / q_max) * q_view
    xs = np.linspace(-x_view, x_view, N_XY)
    xx, yy = np.meshgrid(xs, xs)
    xx_q = (q_max / R) * xx
    yy_q = (q_max / R) * yy

    Ex_xy = polar_grid_to_cartesian_grid(E.x, q, varphi, xx_q, yy_q, fill_value=0.0)
    Ey_xy = polar_grid_to_cartesian_grid(E.y, q, varphi, xx_q, yy_q, fill_value=0.0)
    I_vec = np.abs(Ex_xy) ** 2 + np.abs(Ey_xy) ** 2

    # Scalar reference ts=tp=1
    H0 = HT(0, r, P, q)
    amp2 = EX0_AMP**2 + EY0_AMP**2
    I_scalar_q = 4.0 * amp2 * (np.abs(H0) ** 2)
    rho_q = np.sqrt(xx_q**2 + yy_q**2)
    I_scalar = np.interp(rho_q, q, I_scalar_q, left=I_scalar_q[0], right=0.0)

    # Metrics
    rho_vec, prof_vec = radial_cut(I_vec, xs)
    rho_sca, prof_sca = radial_cut(I_scalar, xs)

    fwhm_vec = crossing_radius(rho_vec, prof_vec, 0.5)
    first_min_vec = first_minimum_radius(rho_vec, prof_vec)
    first_min_sca = first_minimum_radius(rho_sca, prof_sca)

    NA = ni * (R / np.sqrt(R**2 + zi**2))
    rayleigh_q = 3.8317059702075125 * lam / NA

    print(f"NA = {NA:.6f}")
    print(f"Rayleigh q-convention = {rayleigh_q*1000:.4f} um")
    print(f"Vector FWHM radius = {fwhm_vec*1000:.4f} um")
    print(f"Vector first minimum = {first_min_vec*1000:.4f} um")
    print(f"Scalar first minimum = {first_min_sca*1000:.4f} um")

    # Figure 1: vector map + bar markers
    fig1, axs = plt.subplots(1, 2, figsize=(12, 5))
    extent = [float(xs[0]), float(xs[-1]), float(xs[0]), float(xs[-1])]

    im = axs[0].imshow(I_vec, extent=extent, origin="lower", cmap="hot", aspect="equal")
    plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04, label="|E|^2")

    add_circle(axs[0], rayleigh_q, "deepskyblue", "Rayleigh (q-convention)", lw=1.8)
    add_circle(axs[0], fwhm_vec, "lime", "Vector FWHM radius", lw=1.8)
    add_circle(axs[0], first_min_vec, "magenta", "Vector 1st minimum", lw=1.8)
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].set_title("Vector spot")
    axs[0].set_xlabel("x [mm]")
    axs[0].set_ylabel("y [mm]")

    labels = ["Rayleigh q-conv", "Vector FWHM", "Vector 1st min"]
    values_um = [rayleigh_q * 1000, fwhm_vec * 1000, first_min_vec * 1000]
    colors = ["deepskyblue", "lime", "magenta"]
    x = np.arange(len(labels))
    bars = axs[1].bar(x, values_um, color=colors, alpha=0.85)
    for b in bars:
        h = b.get_height()
        axs[1].text(b.get_x() + b.get_width() / 2.0, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, rotation=10)
    axs[1].set_ylabel("radius [um]")
    axs[1].set_title("Vector size markers")
    axs[1].grid(axis="y", alpha=0.25)

    # Figure 2: vector vs scalar maps with consistent circles
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))

    I_vec_n = I_vec / (np.max(I_vec) + 1e-30)
    I_sca_n = I_scalar / (np.max(I_scalar) + 1e-30)

    im_vec = ax2[0].imshow(I_vec_n, extent=extent, origin="lower", cmap="magma", aspect="equal", vmin=0.0, vmax=1.0)
    plt.colorbar(im_vec, ax=ax2[0], fraction=0.046, pad=0.04, label="Vector |E|^2 (normalized)")
    ax2[0].set_title("Vector pattern")
    ax2[0].set_xlabel("x [mm]")
    ax2[0].set_ylabel("y [mm]")

    im_sca = ax2[1].imshow(I_sca_n, extent=extent, origin="lower", cmap="viridis", aspect="equal", vmin=0.0, vmax=1.0)
    plt.colorbar(im_sca, ax=ax2[1], fraction=0.046, pad=0.04, label="Scalar |E|^2 (normalized, ts=tp=1)")
    ax2[1].set_title("Scalar pattern")
    ax2[1].set_xlabel("x [mm]")
    ax2[1].set_ylabel("y [mm]")

    add_circle(ax2[0], rayleigh_q, "deepskyblue", "Rayleigh (q-convention)")
    add_circle(ax2[0], fwhm_vec, "lime", "Vector FWHM radius")
    add_circle(ax2[0], first_min_vec, "magenta", "Vector 1st minimum")
    ax2[0].legend(loc="upper right", fontsize=8)

    add_circle(ax2[1], rayleigh_q, "deepskyblue", "Rayleigh (q-convention)")
    add_circle(ax2[1], first_min_sca, "orange", "Scalar 1st minimum")
    add_circle(ax2[1], first_min_vec, "magenta", "Vector 1st minimum")
    ax2[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
