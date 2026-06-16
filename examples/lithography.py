
"""
Circular-pupil lithography-pattern example using vecdiff.

This example propagates a synthetic lithography diagnostic mask through two
diopters, comparing the scalar approximation tp=ts=1 against the vectorial
transverse diopter operator.

Repository API assumed:
    github.com/blancusjh/vecdiff

Core imports:
    from vecdiff import Grid, FieldCartesian, CartesianSurface
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vecdiff import Grid, FieldCartesian, CartesianSurface


# ---------------------------------------------------------------------
# Optical configuration
# ---------------------------------------------------------------------

lam = 532e-6

# High-contrast diagnostic configuration: the large index jump in D2
# deliberately increases tp - ts and therefore the cross-polarized component.
# The diopters are conjugate: D2["z0"] = -D1["zi"].
D1 = dict(n0=1.0, ni=1.01, z0=-10.0, zi=6.0)
D2 = dict(n0=D1["ni"], ni=20.0, z0=-D1["zi"], zi=0.300)

Kc = 900.0
rA = 3.8317059702075125 / Kc
dA = 2.0 * rA

# Near-resolution feature scale for the polarization-transfer diagnostic.
Pattern_sep = 0.82 * dA
Pattern_theta = 0.25 * np.pi
Incident_polarization = "lineal x"

out_dir = os.path.join("output", "lithography_pattern")
os.makedirs(out_dir, exist_ok=True)


# ---------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------

def angular_axis(N, d):
    return 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=d))


def kgrid_from_spacing(N, dx, dy):
    kx = angular_axis(N, dx)
    ky = angular_axis(N, dy)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    return Grid.from_cartesian(KX, KY, domain="k")


def crop_by_axes(A, ax, ay, xmax):
    ix = (ax >= -xmax) & (ax <= xmax)
    iy = (ay >= -xmax) & (ay <= xmax)
    return np.asarray(A)[np.ix_(iy, ix)], ax[ix], ay[iy]


def apply_circular_pupil(Fx, Fy, kx, ky, Kc):
    """
    Apply the physical circular pupil

        P(kx, ky) = 1,  kx^2 + ky^2 <= Kc^2,

    using a square crop only as the containing computational box.
    """
    Fx, kx_c, ky_c = crop_by_axes(Fx, kx, ky, Kc)
    Fy, _, _ = crop_by_axes(Fy, kx, ky, Kc)

    KX, KY = np.meshgrid(kx_c, ky_c)
    pupil = (KX**2 + KY**2) <= Kc**2

    return Fx * pupil, Fy * pupil, kx_c, ky_c, pupil


def mean_normalize(I):
    I = np.asarray(I, dtype=float)
    positive = I[I > 0]
    return I / positive.mean() if positive.size else I


def image_extent(x, y):
    return [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]


def image_extent_over_lambda(x, y):
    return [
        float(x[0] / lam),
        float(x[-1] / lam),
        float(y[0] / lam),
        float(y[-1] / lam),
    ]


def display_positive(I, percentile=99.6, gamma=0.85):
    I = np.asarray(I, dtype=float)
    vals = I[np.isfinite(I) & (I > 0)]

    vmax = np.percentile(vals, percentile) if vals.size else 1.0
    vmax = max(vmax, 1.0e-14)

    return np.clip(I / vmax, 0.0, 1.0) ** gamma


def display_signed(A, percentile=99.2):
    A = np.asarray(A, dtype=float)
    vals = np.abs(A[np.isfinite(A)])

    vmax = np.percentile(vals, percentile) if vals.size else 1.0
    vmax = max(vmax, 1.0e-14)

    return 0.5 + 0.5 * np.clip(A / vmax, -1.0, 1.0)


def pair_resolution_ratio(I, xr, yr, sep, theta):
    Xr, Yr = np.meshgrid(xr, yr)
    ux = np.cos(theta)
    uy = np.sin(theta)

    coord = Xr * ux + Yr * uy
    perp = -Xr * uy + Yr * ux

    mid = (np.abs(coord) <= 0.12 * sep) & (np.abs(perp) <= 0.35 * dA)
    left = (np.abs(coord + 0.5 * sep) <= 0.30 * sep) & (np.abs(perp) <= 0.45 * dA)
    right = (np.abs(coord - 0.5 * sep) <= 0.30 * sep) & (np.abs(perp) <= 0.45 * dA)

    peak = min(float(np.max(I[left])), float(np.max(I[right])))
    valley = float(np.mean(I[mid]))
    return valley / (peak + 1.0e-15)


def add_airy_radius(ax, x0, y0):
    ax.plot([x0], [y0], marker="o", color="white", mec="black", mew=0.6, ms=3.5, zorder=5)
    ax.plot([x0, x0 + rA / lam], [y0, y0], color="black", lw=4.0, zorder=4)
    ax.plot([x0, x0 + rA / lam], [y0, y0], color="white", lw=2.2, zorder=5)
    ax.text(
        x0,
        y0 + 1.4 * rA / lam,
        f"Radio de Airy\n$r_A/\\lambda={rA / lam:.3g}$",
        color="white",
        fontsize=8,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="black", alpha=0.50, edgecolor="none", pad=2.0),
    )


# ---------------------------------------------------------------------
# Propagation through two diopters
# ---------------------------------------------------------------------

def propagate_two_diopters_circular_pupil(field, L, *, vectorial, Npad1=768, Npad2=768):
    """
    Propagate a transverse Cartesian field through two finite-conjugate diopters.

    Scalar mode:
        propagates with identity transmission, equivalent to tp=ts=1 in this test.

    Vectorial mode:
        uses Field.propagate_through_diopter with vectorial Fresnel transmission.

    The circular pupil is imposed after the first Fourier map.
    """
    dx = float(np.mean(np.diff(field.grid.X[0, :])))
    dy = float(np.mean(np.diff(field.grid.Y[:, 0])))
    transmission = "vectorial" if vectorial else "identity"

    diopter1 = CartesianSurface(**D1)
    propagated1 = field.propagate_through_diopter(
        diopter1.zi,
        diopter1,
        method="fft",
        kgrid=kgrid_from_spacing(Npad1, dx, dy),
        transmission=transmission,
    )
    kx1 = propagated1.grid.X[0, :]
    ky1 = propagated1.grid.Y[:, 0]

    Ex1, Ey1, kx_c, ky_c, pupil = apply_circular_pupil(
        propagated1.x,
        propagated1.y,
        kx1,
        ky1,
        Kc,
    )

    alpha1 = lam * D1["zi"] / (2.0 * np.pi * D1["ni"])

    xf = alpha1 * kx_c
    yf = alpha1 * ky_c

    dxf = float(np.mean(np.diff(xf)))
    dyf = float(np.mean(np.diff(yf)))

    XF, YF = np.meshgrid(xf, yf)

    focal_field = FieldCartesian(
        Ex1,
        Ey1,
        grid=Grid.from_cartesian(XF, YF),
        symmetric=False,
    )

    diopter2 = CartesianSurface(**D2)
    propagated2 = focal_field.propagate_through_diopter(
        diopter2.zi,
        diopter2,
        method="fft",
        kgrid=kgrid_from_spacing(Npad2, dxf, dyf),
        transmission=transmission,
    )

    Ex2 = propagated2.x
    Ey2 = propagated2.y
    kx2 = propagated2.grid.X[0, :]
    ky2 = propagated2.grid.Y[:, 0]

    xrec = -alpha1 * kx2
    yrec = -alpha1 * ky2

    if xrec[0] > xrec[-1]:
        xrec = xrec[::-1]
        Ex2 = Ex2[:, ::-1]
        Ey2 = Ey2[:, ::-1]

    if yrec[0] > yrec[-1]:
        yrec = yrec[::-1]
        Ex2 = Ex2[::-1, :]
        Ey2 = Ey2[::-1, :]

    Ex2, xr, yr = crop_by_axes(Ex2, xrec, yrec, L / 2.0)
    Ey2, _, _ = crop_by_axes(Ey2, xrec, yrec, L / 2.0)

    return Ex2, Ey2, xr, yr


# ---------------------------------------------------------------------
# Lithography diagnostic mask
# ---------------------------------------------------------------------

def build_lithography_mask(X, Y, scale_sep, theta=Pattern_theta):
    """
    Build a separated diagnostic lithography pattern.

    It includes:
        1. A calibrated two-contact pair at the origin.
        2. A line-space grating.
        3. A 3x3 contact array.
        4. Two L-shaped corner / line-end structures.

    The full pattern is scaled by scale_sep, so relative distances are preserved.
    """
    s = scale_sep
    cr = 0.18 * dA
    w = 2.0 * cr
    ux = np.cos(theta)
    uy = np.sin(theta)
    vx = -uy
    vy = ux

    T = np.zeros_like(X, dtype=float)
    boxes = {}

    def add_rect(xmin, xmax, ymin, ymax):
        T[(X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)] = 1.0

    def add_rotated_rect(cx, cy, length, width, angle):
        ca = np.cos(angle)
        sa = np.sin(angle)
        u = (X - cx) * ca + (Y - cy) * sa
        v = -(X - cx) * sa + (Y - cy) * ca
        T[(np.abs(u) <= 0.5 * length) & (np.abs(v) <= 0.5 * width)] = 1.0

    # 1. Diagonal contact pair, placed where sin(2 phi) maximizes cross-polarization.
    for sign in (-1.0, +1.0):
        cx = sign * 0.5 * s * ux
        cy = sign * 0.5 * s * uy
        T[(X - cx)**2 + (Y - cy)**2 <= cr**2] = 1.0

    boxes["diagonal_pair"] = (-0.5 * s - cr, 0.5 * s + cr, -0.5 * s - cr, 0.5 * s + cr)

    # 2. Diagonal line-space grating with near-resolution pitch.
    gcx = -5.8 * s * ux + 2.3 * s * vx
    gcy = -5.8 * s * uy + 2.3 * s * vy

    for m in range(4):
        cx = gcx + (m - 1.5) * s * ux
        cy = gcy + (m - 1.5) * s * uy
        add_rotated_rect(cx, cy, 3.2 * s, w, theta + 0.5 * np.pi)

    boxes["diagonal_grating"] = (gcx - 3.0 * s, gcx + 3.0 * s, gcy - 3.0 * s, gcy + 3.0 * s)

    # 3. Diagonal contact array.
    cx0 = -1.0 * s * ux - 4.6 * s * vx
    cy0 = -1.0 * s * uy - 4.6 * s * vy

    for i in range(3):
        for j in range(3):
            cx = cx0 + i * s * ux + j * s * vx
            cy = cy0 + i * s * uy + j * s * vy
            T[(X - cx)**2 + (Y - cy)**2 <= cr**2] = 1.0

    boxes["diagonal_contact_array"] = (cx0 - 2.4 * s, cx0 + 2.4 * s, cy0 - 2.4 * s, cy0 + 2.4 * s)

    # 4. Diagonal L-shaped corner / line-end structures.
    lx = 5.4 * s * ux + 1.8 * s * vx
    ly = 5.4 * s * uy + 1.8 * s * vy

    add_rotated_rect(lx + 1.35 * s * ux, ly + 1.35 * s * uy, 2.7 * s, w, theta)
    add_rotated_rect(lx + 1.35 * s * vx, ly + 1.35 * s * vy, 2.7 * s, w, theta + 0.5 * np.pi)

    boxes["diagonal_L"] = (lx - 0.6 * s, lx + 2.8 * s, ly - 0.6 * s, ly + 2.8 * s)

    mx = 5.8 * s * ux - 4.4 * s * vx
    my = 5.8 * s * uy - 4.4 * s * vy

    add_rotated_rect(mx - 1.25 * s * ux, my - 1.25 * s * uy, 2.5 * s, w, theta)
    add_rotated_rect(mx + 1.35 * s * vx, my + 1.35 * s * vy, 2.7 * s, w, theta + 0.5 * np.pi)

    boxes["mirrored_diagonal_L"] = (mx - 2.8 * s, mx + 0.7 * s, my - 2.8 * s, my + 0.7 * s)

    return T, boxes


def bbox_union(boxes):
    xs, ys = [], []

    for xmin, xmax, ymin, ymax in boxes.values():
        xs.extend([xmin, xmax])
        ys.extend([ymin, ymax])

    return min(xs), max(xs), min(ys), max(ys)


def run_lithography_example(*, L=1.60, N=513, Npad=768):
    x = np.linspace(-L / 2.0, L / 2.0, N)
    y = np.linspace(-L / 2.0, L / 2.0, N)

    X, Y = np.meshgrid(x, y)
    grid = Grid.from_cartesian(X, Y)

    T, boxes = build_lithography_mask(X, Y, Pattern_sep)

    field = FieldCartesian(
        T.astype(complex),
        np.zeros_like(T, dtype=complex),
        grid=grid,
        symmetric=False,
    )

    Ex_s, Ey_s, xr, yr = propagate_two_diopters_circular_pupil(
        field,
        L,
        vectorial=False,
        Npad1=Npad,
        Npad2=Npad,
    )

    Ex_v, Ey_v, _, _ = propagate_two_diopters_circular_pupil(
        field,
        L,
        vectorial=True,
        Npad1=Npad,
        Npad2=Npad,
    )

    Is = mean_normalize(np.abs(Ex_s)**2 + np.abs(Ey_s)**2)
    Iv = mean_normalize(np.abs(Ex_v)**2 + np.abs(Ey_v)**2)
    Icross = mean_normalize(np.abs(Ey_v)**2)
    Id = Iv - Is
    Id_abs = np.abs(Id)
    cross_fraction = float(
        np.sum(np.abs(Ey_v)**2) / (np.sum(np.abs(Ex_v)**2 + np.abs(Ey_v)**2) + 1.0e-30)
    )
    scalar_pair_ratio = pair_resolution_ratio(Is, xr, yr, Pattern_sep, Pattern_theta)
    vectorial_pair_ratio = pair_resolution_ratio(Iv, xr, yr, Pattern_sep, Pattern_theta)

    xmin, xmax, ymin, ymax = bbox_union(boxes)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    span = max(xmax - xmin, ymax - ymin)

    half_window = 0.5 * (span / 0.80)

    zoom = {
        "xmin": cx - half_window,
        "xmax": cx + half_window,
        "ymin": cy - half_window,
        "ymax": cy + half_window,
    }

    return {
        "x": x,
        "y": y,
        "T": T,
        "boxes": boxes,
        "xr": xr,
        "yr": yr,
        "Is": Is,
        "Iv": Iv,
        "Icross": Icross,
        "Id": Id,
        "Id_abs": Id_abs,
        "zoom": zoom,
        "cross_fraction": cross_fraction,
        "scalar_pair_ratio": scalar_pair_ratio,
        "vectorial_pair_ratio": vectorial_pair_ratio,
    }


def plot_lithography_result(result):
    fig, axes = plt.subplots(1, 4, figsize=(15.4, 4.5), constrained_layout=True)

    panels = [
        (result["T"], result["x"], result["y"], "Máscara de entrada", False),
        (result["Is"], result["xr"], result["yr"], "Intensidad - Escalar", False),
        (result["Iv"], result["xr"], result["yr"], "Intensidad - Vectorial", False),
        (result["Id_abs"], result["xr"], result["yr"], r"$|\mathrm{Vectorial} - \mathrm{Escalar}|$", False),
    ]

    z = result["zoom"]

    for j, (A, xx, yy, title, signed) in enumerate(panels):
        shown = display_signed(A) if signed else display_positive(A)

        im = axes[j].imshow(
            shown,
            origin="lower",
            extent=image_extent_over_lambda(xx, yy),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        fig.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04)
        axes[j].set_xlim(z["xmin"] / lam, z["xmax"] / lam)
        axes[j].set_ylim(z["ymin"] / lam, z["ymax"] / lam)

        axes[j].set_title(title)
        axes[j].set_xlabel(r"$x/\lambda$")
        axes[j].set_ylabel(r"$y/\lambda$")
        axes[j].set_aspect("equal")

        if j == 1:
            add_airy_radius(
                axes[j],
                (z["xmin"] + 0.08 * (z["xmax"] - z["xmin"])) / lam,
                (z["ymin"] + 0.09 * (z["ymax"] - z["ymin"])) / lam,
            )

    fig.suptitle(
        rf"Patrón de litografía a través de lente"
        "\n"
        rf"$z_{{0,1}}={D1['z0']} \,\mathrm{{mm}}$, "
        rf"$z_{{i,1}}={D1['zi']} \,\mathrm{{mm}}$, "
        rf"$z_{{i,2}}={D2['zi']} \,\mathrm{{mm}}$",
    )

    fig.savefig(
        os.path.join(out_dir, "lithography_pattern_check.png"),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


if __name__ == "__main__":
    result = run_lithography_example(L=0.32, N=769, Npad=1024)
    plot_lithography_result(result)

    Is = result["Is"]
    Iv = result["Iv"]
    Id = result["Id"]

    print("OK")
    print(f"rA={rA:.10f}")
    print(f"dA={dA:.10f}")
    print(f"Pattern_sep={Pattern_sep:.10f} ({Pattern_sep / dA:.4f} dA)")
    print(f"conjugacy_error=D1_zi_plus_D2_z0={D1['zi'] + D2['z0']:.10e}")
    print(f"scalar_shape={Is.shape}")
    print(f"vectorial_shape={Iv.shape}")
    print(f"scalar_finite={np.isfinite(Is).all()}")
    print(f"vectorial_finite={np.isfinite(Iv).all()}")
    print(f"cross_pol_fraction={result['cross_fraction']:.10e}")
    print(f"scalar_pair_ratio={result['scalar_pair_ratio']:.10e}")
    print(f"vectorial_pair_ratio={result['vectorial_pair_ratio']:.10e}")
    print(f"difference_abs_max={np.max(np.abs(Id)):.10e}")
    print(f"difference_abs_p99={np.percentile(np.abs(Id), 99):.10e}")
    print(os.path.join(out_dir, "lithography_pattern_check.png"))
