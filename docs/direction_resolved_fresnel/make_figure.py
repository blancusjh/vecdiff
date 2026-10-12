"""Generate the manuscript's Fresnel-only figure from saved complex fields."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "validation/reference_data"
with np.load(DATA / "focal_pipeline.npz") as reference:
    exact = reference["exact_plane_p1"]
    local = reference["N0_grid280_plane_E"]
    plane = reference["plane"]
with np.load(DATA / "focal_spectral_N280_a18_c1e-07_t3.npz") as spectral:
    resolved = spectral["order0_plane_E"]

mid = len(plane) // 2
x = plane[mid, :, 0]
peak = np.linalg.norm(exact, axis=-1).max()
fields = [local, resolved]
labels = ["Local-direction Fresnel", "Direction-resolved Fresnel"]
errors = [np.linalg.norm(v - exact) / np.linalg.norm(exact) for v in fields]
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), layout="constrained")
for value, label in zip(fields, labels):
    axes[0].semilogy(
        x,
        np.linalg.norm(value - exact, axis=-1)[mid] / peak,
        label=label,
        linewidth=1.5,
    )
axes[0].set(
    xlabel=r"$x/\lambda_0$",
    ylabel=r"$|\Delta E|/\max |E_{\rm ref}|$",
    title="Complex field error at y = 0",
)
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.2)
axes[1].bar([0, 1], 100 * np.array(errors), color=["C0", "C1"], width=0.55)
axes[1].set(
    yscale="log",
    ylim=(1e-3, 1),
    ylabel="Relative focal-field error (%)",
    title="Whole detector, no phase or amplitude fit",
)
axes[1].set_xticks([0, 1], ["Local direction", "Direction resolved"])
for index, error in enumerate(errors):
    axes[1].text(
        index, 100 * error * 1.2, f"{100 * error:.4g}%", ha="center", fontsize=9
    )
axes[1].grid(axis="y", alpha=0.2)
for suffix in ["pdf", "png"]:
    fig.savefig(HERE / f"fresnel_comparison.{suffix}", dpi=200)
plt.close(fig)
print(f"Local/resolved error ratio: {errors[0] / errors[1]:.6f}")
