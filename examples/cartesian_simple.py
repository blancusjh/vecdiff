import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

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

print(f"θ = {np.rad2deg(np.arctan(R/zi))}")

N_R = 1000
N_Q = 1000
N_PHI = 256
N_XY = 513
AIRY_VIEW_RADII = 4.0
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
ANIMATION_R_VALUES = np.linspace(0.8, 4.0, 25)
ANIMATION_N_R = 320
ANIMATION_N_Q = 320
ANIMATION_N_PHI = 128
ANIMATION_N_XY = 257
ANIMATION_FPS = 8


def compute_cartesian_intensity(R_value, n_r=N_R, n_q=N_Q, n_phi=N_PHI, n_xy=N_XY, x_view=None):
    r = np.linspace(0.0, R_value, n_r)  # radial source coordinate [same unit as R]
    q_max = (ni / (lam * zi)) * R_value
    q = q_max * np.linspace(0.0, 1.0, n_q) ** 2.0
    varphi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    a = 0.95 * R_value
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

    # 2D visualization on Cartesian mesh in physical coordinates (x, y).
    # Keep the focal spot visible as R changes by viewing a fixed number of
    # first-Airy-zero radii instead of a fixed fraction of q_max.
    if x_view is None:
        first_airy_zero_q = 3.8317059702075125 / a
        q_view = AIRY_VIEW_RADII * first_airy_zero_q
        x_view = (R_value / q_max) * q_view  # convert q-window to physical x-window [same unit as R]
    xs = np.linspace(-x_view, x_view, n_xy)
    xx, yy = np.meshgrid(xs, xs)

    # Convert physical coordinates to q-coordinates for interpolation:
    # q = (q_max / R) * rho, with rho = sqrt(x^2 + y^2)
    xx_q = (q_max / R_value) * xx
    yy_q = (q_max / R_value) * yy

    Ex_xy = polar_grid_to_cartesian_grid(Ex, q, varphi, xx_q, yy_q, fill_value=0.0)
    Ey_xy = polar_grid_to_cartesian_grid(Ey, q, varphi, xx_q, yy_q, fill_value=0.0)

    I_Ex = np.abs(Ex_xy) ** 2
    I_Ey = np.abs(Ey_xy) ** 2
    I_total = I_Ex + I_Ey

    return {
        "R": R_value,
        "q": q,
        "varphi": varphi,
        "xs": xs,
        "Ex": Ex,
        "Ey": Ey,
        "Ex_xy": Ex_xy,
        "Ey_xy": Ey_xy,
        "I_Ex": I_Ex,
        "I_Ey": I_Ey,
        "I_total": I_total,
    }


def save_R_animation(output_dir):
    min_R = float(np.min(ANIMATION_R_VALUES))
    min_a = 0.95 * min_R
    min_q_max = (ni / (lam * zi)) * min_R
    q_view = AIRY_VIEW_RADII * 3.8317059702075125 / min_a
    x_view = (min_R / min_q_max) * q_view

    first = compute_cartesian_intensity(
        ANIMATION_R_VALUES[0],
        n_r=ANIMATION_N_R,
        n_q=ANIMATION_N_Q,
        n_phi=ANIMATION_N_PHI,
        n_xy=ANIMATION_N_XY,
        x_view=x_view,
    )
    xs_lambda_anim = first["xs"] / lam
    extent_anim = [
        float(xs_lambda_anim[0]),
        float(xs_lambda_anim[-1]),
        float(xs_lambda_anim[0]),
        float(xs_lambda_anim[-1]),
    ]

    fig_anim, axs_anim = plt.subplots(1, 3, figsize=(12, 4))
    arrays = [first["I_Ex"], first["I_Ey"], first["I_total"]]
    titles = [r"$|E_x|^2$", r"$|E_y|^2$", r"$|E_x|^2 + |E_y|^2$"]
    images = []
    for ax, arr, title in zip(axs_anim, arrays, titles):
        im = ax.imshow(arr, extent=extent_anim, origin="lower", aspect="equal", cmap="hot", vmin=0)
        ax.set_title(title)
        ax.set_xlabel(r"$x / \lambda$")
        ax.set_ylabel(r"$y / \lambda$")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        images.append(im)

    title = fig_anim.suptitle("")
    fig_anim.tight_layout()

    def update(frame_index):
        result = compute_cartesian_intensity(
            ANIMATION_R_VALUES[frame_index],
            n_r=ANIMATION_N_R,
            n_q=ANIMATION_N_Q,
            n_phi=ANIMATION_N_PHI,
            n_xy=ANIMATION_N_XY,
            x_view=x_view,
        )
        for im, arr in zip(images, [result["I_Ex"], result["I_Ey"], result["I_total"]]):
            im.set_data(arr)
            im.set_clim(0.0, float(np.max(arr)))
        theta = np.rad2deg(np.arctan(result["R"] / zi))
        title.set_text(f"Cartesian focal intensity, R = {result['R']:.2f} mm, θ = {theta:.2f}°")
        return [*images, title]

    update(0)
    animation = FuncAnimation(
        fig_anim,
        update,
        frames=len(ANIMATION_R_VALUES),
        interval=1000 / ANIMATION_FPS,
        blit=False,
    )
    animation.save(
        output_dir / "cartesian_simple_R_sweep.gif",
        writer=PillowWriter(fps=ANIMATION_FPS),
    )
    plt.close(fig_anim)


result = compute_cartesian_intensity(R)
q = result["q"]
varphi = result["varphi"]
xs = result["xs"]
Ex = result["Ex"]
Ey = result["Ey"]
Ex_xy = result["Ex_xy"]
Ey_xy = result["Ey_xy"]
I_Ex = result["I_Ex"]
I_Ey = result["I_Ey"]
I_total = result["I_total"]

OUTPUT_DIR.mkdir(exist_ok=True)

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
fig.savefig(OUTPUT_DIR / "cartesian_simple_intensity.png", dpi=200)
save_R_animation(OUTPUT_DIR)

center = N_XY // 2
half_window = 12
sl = slice(center - half_window, center + half_window + 1)

fig_zoom, axs_zoom = plt.subplots(1, 3, figsize=(12, 4))
zoom_extent = [
    float(xs_lambda[sl][0]),
    float(xs_lambda[sl][-1]),
    float(xs_lambda[sl][0]),
    float(xs_lambda[sl][-1]),
]

zoom_arrays = [I_Ex[sl, sl], I_Ey[sl, sl], I_total[sl, sl]]
zoom_titles = ["Zoom |Ex|^2", "Zoom |Ey|^2", "Zoom |Ex|^2 + |Ey|^2"]
for ax, arr, title in zip(axs_zoom, zoom_arrays, zoom_titles):
    im = ax.imshow(arr, extent=zoom_extent, origin="lower", aspect="equal", cmap="hot", vmin=0)
    ax.set_title(title)
    ax.set_xlabel(r"$x / \lambda$")
    ax.set_ylabel(r"$y / \lambda$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
fig_zoom.savefig(OUTPUT_DIR / "cartesian_simple_center_zoom.png", dpi=200)

fig_profile, ax_profile = plt.subplots(figsize=(6, 4))
ax_profile.plot(xs_lambda, I_Ex[center, :], label=r"$|E_x|^2$")
ax_profile.plot(xs_lambda, I_Ey[center, :], label=r"$|E_y|^2$")
ax_profile.plot(xs_lambda, I_total[center, :], label=r"$|E_x|^2 + |E_y|^2$")
ax_profile.axvline(0.0, color="black", linewidth=0.8)
ax_profile.set_xlabel(r"$x / \lambda$")
ax_profile.set_ylabel("Intensity")
ax_profile.legend()
fig_profile.tight_layout()
fig_profile.savefig(OUTPUT_DIR / "cartesian_simple_center_profile.png", dpi=200)

np.savez(
    OUTPUT_DIR / "cartesian_simple_arrays.npz",
    q=q,
    varphi=varphi,
    xs=xs,
    xs_lambda=xs_lambda,
    Ex_polar_grid=Ex,
    Ey_polar_grid=Ey,
    Ex_xy=Ex_xy,
    Ey_xy=Ey_xy,
    I_Ex=I_Ex,
    I_Ey=I_Ey,
    I_total=I_total,
)

max_index = np.unravel_index(np.argmax(I_total), I_total.shape)
center_report = [
    "cartesian_simple diagnostics",
    f"N_XY = {N_XY}",
    f"center index = {center}",
    f"xs[center] = {xs[center]:.18e}",
    f"xs_lambda[center] = {xs_lambda[center]:.18e}",
    f"Ex(q=0, first angle) = {Ex[0, 0].real:.18e} + {Ex[0, 0].imag:.18e}j",
    f"Ey(q=0, first angle) = {Ey[0, 0].real:.18e} + {Ey[0, 0].imag:.18e}j",
    f"std_angle Ex(q=0) = {np.std(Ex[:, 0]):.18e}",
    f"std_angle Ey(q=0) = {np.std(Ey[:, 0]):.18e}",
    f"Ex_xy center = {Ex_xy[center, center].real:.18e} + {Ex_xy[center, center].imag:.18e}j",
    f"Ey_xy center = {Ey_xy[center, center].real:.18e} + {Ey_xy[center, center].imag:.18e}j",
    f"I_Ex center = {I_Ex[center, center]:.18e}",
    f"I_Ey center = {I_Ey[center, center]:.18e}",
    f"I_total center = {I_total[center, center]:.18e}",
    f"I_total max = {I_total[max_index]:.18e}",
    f"I_total max index = {max_index}",
    f"I_total center / max = {I_total[center, center] / I_total[max_index]:.18e}",
    "I_total 5x5 around center:",
    np.array2string(I_total[center - 2:center + 3, center - 2:center + 3], precision=8),
]

(OUTPUT_DIR / "cartesian_simple_diagnostics.txt").write_text("\n".join(center_report) + "\n")

print(f"Saved diagnostics to {OUTPUT_DIR}")
plt.close("all")
