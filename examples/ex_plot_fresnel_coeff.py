import matplotlib.pyplot as plt
import numpy as np

from vecdiff.fresnel import FresnelOvoid, FresnelOvoidParax


# Ovoid/system parameters
n0 = 1.0
ni = 1.5
z0 = 5.0
zi = 10.0

# Sampling
L = 1.0
N = 800
rho = np.linspace(0.0, L, N)


def sanitize_at_origin(arr, r):
    arr = np.asarray(arr, dtype=float)
    bad = ~np.isfinite(arr)
    if np.any(bad):
        good = ~bad
        arr[bad] = np.interp(r[bad], r[good], arr[good])
    return arr


# Paraxial coefficients
f_parax = FresnelOvoidParax(n0=n0, ni=ni, z0=z0, zi=zi)
ts_parax, tp_parax = f_parax.coeffs(rho)
ts_parax = np.asarray(ts_parax, dtype=float)
tp_parax = np.asarray(tp_parax, dtype=float)

# Exact coefficients
f_exact = FresnelOvoid(n0=n0, ni=ni, z0=z0, zi=zi)
with np.errstate(divide="ignore", invalid="ignore"):
    ts_exact = sanitize_at_origin(f_exact.ts(rho), rho)
    tp_exact = sanitize_at_origin(f_exact.tp(rho), rho)


# Shared y-range for direct visual comparison
ymin = float(min(np.min(ts_parax), np.min(tp_parax), np.min(ts_exact), np.min(tp_exact)))
ymax = float(max(np.max(ts_parax), np.max(tp_parax), np.max(ts_exact), np.max(tp_exact)))
pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
ymin -= pad
ymax += pad


fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, sharey=True)

# Left: paraxial
axes[0].plot(rho, ts_parax, label=r"$t_s$ paraxial", linewidth=2)
axes[0].plot(rho, tp_parax, label=r"$t_p$ paraxial", linewidth=2)
axes[0].set_title("Paraxial Fresnel Coefficients")
axes[0].set_xlabel(r"$r$")
axes[0].set_ylabel("Coefficient")
axes[0].set_ylim(ymin, ymax)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Right: exact
axes[1].plot(rho, ts_exact, label=r"$t_s$ exact", linewidth=2)
axes[1].plot(rho, tp_exact, label=r"$t_p$ exact", linewidth=2)
axes[1].set_title("Exact Fresnel Coefficients")
axes[1].set_xlabel(r"$r$")
axes[1].set_ylim(ymin, ymax)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.suptitle("Fresnel Coefficients: Left Paraxial | Right Exact")
plt.show()

