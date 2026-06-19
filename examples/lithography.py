
"""
Circular-pupil lithography-pattern example using vecdiff.

This example propagates a synthetic lithography diagnostic mask through two
diopters, comparing the scalar approximation tp=ts=1 against the vectorial
transverse diopter operator.

Repository API assumed:
    github.com/blancusjh/vecdiff

Core imports:
    from vecdiff import CartesianSurface, FieldCartesian, Grid
"""

from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vecdiff import Grid, FieldCartesian, CartesianSurface
from _output import example_output_dir, print_saved


# ---------------------------------------------------------------------
# Optical configuration
# ---------------------------------------------------------------------

lam = 532e-6

# Light propagates from vacuum into the lens medium through D1, then exits
# back into vacuum through D2.  The second object distance is measured from
# the second vertex, so it depends on the lens thickness xi.
D1 = dict(n0=1.0, ni=1.5, z0=-10.0, zi=2.0)
xi = 0.500
D2 = dict(n0=D1["ni"], ni=1.0, z0=D1["zi"] - xi, zi=2.00)

r_a = 1.5
alpha_max = np.arctan(r_a / abs(D1["z0"]))
Kc = D1["n0"] * (2.0 * np.pi / lam) * np.sin(alpha_max)
r_Airy = 3.8317059702075125 / Kc
d_Airy = 2.0 * r_Airy

# Near-resolution feature scale for the polarization-transfer diagnostic.
Pattern_sep = 0.79 * d_Airy
Pattern_theta = 0.25 * np.pi

out_dir = example_output_dir(__file__)


@dataclass(frozen=True)
class Lens:
    radius: float
    first: dict
    second: dict
    xi: float

    @property
    def distance_from_first_focus_to_second_vertex(self):
        return self.xi - self.first["zi"]


# ---------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------


def orient_field_axes(Ex, Ey, x, y):
    Ex = np.asarray(Ex)
    Ey = np.asarray(Ey)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x[0] > x[-1]:
        x = x[::-1]
        Ex = Ex[:, ::-1]
        Ey = Ey[:, ::-1]

    if y[0] > y[-1]:
        y = y[::-1]
        Ex = Ex[::-1, :]
        Ey = Ey[::-1, :]

    return Ex, Ey, x, y


def mean_normalize(I):
    I = np.asarray(I, dtype=float)
    positive = I[I > 0]
    return I / positive.mean() if positive.size else I


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


# ---------------------------------------------------------------------
# Propagation through a lens
# ---------------------------------------------------------------------

def propagate_through_lens(field, lens, *, vectorial, Npad1=768, Npad2=768):
    transmission = "vectorial" if vectorial else "identity"

    diopter1 = CartesianSurface(**lens.first)
    diopter2 = CartesianSurface(**lens.second)
    pupil_radius = (
        lam * lens.first["zi"] * Kc / (2.0 * np.pi * lens.first["ni"])
    )

    focal_field = field.propagate_through_diopter(
        diopter1.zi,
        diopter1,
        method="fft",
        output="focal",
        wavelength=lam,
        kgrid=field.grid.kgrid(Npad1),
        transmission=transmission,
    ).with_circular_aperture(
        pupil_radius
    )

    second_tangent_field = focal_field.propagate_in_medium(
        lens.distance_from_first_focus_to_second_vertex,
        wavelength=lam,
        n=lens.first["ni"],
    )

    image_field = second_tangent_field.propagate_through_diopter(
        diopter2.zi,
        diopter2,
        method="fft",
        output="focal",
        wavelength=lam,
        kgrid=second_tangent_field.grid.kgrid(Npad2),
        transmission=transmission,
    )

    Ex2 = image_field.x
    Ey2 = image_field.y
    xr = image_field.grid.X[0, :]
    yr = image_field.grid.Y[:, 0]

    Ex2, Ey2, xr, yr = orient_field_axes(Ex2, Ey2, xr, yr)

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
    cr = 0.18 * d_Airy
    w = 2.0 * cr
    ux = np.cos(theta)
    uy = np.sin(theta)
    vx = -uy
    vy = ux

    T = np.zeros_like(X, dtype=float)
    boxes = {}

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


def padded_window_from_box(xmin, xmax, ymin, ymax, padding=0.18):
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    span = max(xmax - xmin, ymax - ymin)
    half_window = 0.5 * span * (1.0 + padding)
    return {
        "xmin": cx - half_window,
        "xmax": cx + half_window,
        "ymin": cy - half_window,
        "ymax": cy + half_window,
    }


def clamp_window_to_axes(window, x, y):
    return {
        "xmin": max(float(window["xmin"]), float(min(x[0], x[-1]))),
        "xmax": min(float(window["xmax"]), float(max(x[0], x[-1]))),
        "ymin": max(float(window["ymin"]), float(min(y[0], y[-1]))),
        "ymax": min(float(window["ymax"]), float(max(y[0], y[-1]))),
    }


def signal_window(arrays, x, y, threshold=0.035, padding=0.18):
    combined = np.zeros_like(np.asarray(arrays[0], dtype=float))

    for A in arrays:
        A = np.asarray(A, dtype=float)
        scale = float(np.nanmax(np.abs(A)))
        if scale > 0.0:
            combined = np.maximum(combined, np.abs(A) / scale)

    active = combined >= threshold
    if not np.any(active):
        return clamp_window_to_axes({
            "xmin": float(x[0]),
            "xmax": float(x[-1]),
            "ymin": float(y[0]),
            "ymax": float(y[-1]),
        }, x, y)

    yy, xx = np.nonzero(active)
    xmin = float(x[max(int(xx.min()) - 1, 0)])
    xmax = float(x[min(int(xx.max()) + 1, len(x) - 1)])
    ymin = float(y[max(int(yy.min()) - 1, 0)])
    ymax = float(y[min(int(yy.max()) + 1, len(y) - 1)])
    return clamp_window_to_axes(
        padded_window_from_box(xmin, xmax, ymin, ymax, padding=padding),
        x,
        y,
    )


def run_lithography_example(*, L=1.60, N=513, Npad=768):
    x = np.linspace(-L / 2.0, L / 2.0, N)
    y = np.linspace(-L / 2.0, L / 2.0, N)
    grid = Grid.from_axes(x, y)
    X, Y = grid.X, grid.Y

    T, boxes = build_lithography_mask(X, Y, Pattern_sep)

    field = FieldCartesian(
        T.astype(complex),
        np.zeros_like(T, dtype=complex),
        grid=grid,
        symmetric=False,
    )

    lens = Lens(radius=r_a, first=D1, second=D2, xi=xi)

    Ex_s, Ey_s, xr, yr = propagate_through_lens(
        field,
        lens,
        vectorial=False,
        Npad1=Npad,
        Npad2=Npad,
    )

    Ex_v, Ey_v, _, _ = propagate_through_lens(
        field,
        lens,
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
    xmin, xmax, ymin, ymax = bbox_union(boxes)
    input_zoom = clamp_window_to_axes(
        padded_window_from_box(xmin, xmax, ymin, ymax, padding=0.22),
        x,
        y,
    )
    output_zoom = signal_window([Is, Iv, Id_abs], xr, yr, threshold=0.035, padding=0.18)

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
        "input_zoom": input_zoom,
        "output_zoom": output_zoom,
        "cross_fraction": cross_fraction,
        "lens": lens,
    }


def plot_lithography_result(result):
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.5), constrained_layout=False)

    panels = [
        (result["T"], result["x"], result["y"], "Máscara de entrada", result["input_zoom"]),
        (result["Is"], result["xr"], result["yr"], "Intensidad - Escalar", result["output_zoom"]),
        (result["Iv"], result["xr"], result["yr"], "Intensidad - Vectorial", result["output_zoom"]),
        (
            result["Id_abs"],
            result["xr"],
            result["yr"],
            r"$|\mathrm{Vectorial} - \mathrm{Escalar}|$",
            result["output_zoom"],
        ),
    ]

    for j, (A, xx, yy, title, zoom) in enumerate(panels):
        im = axes[j].imshow(
            display_positive(A),
            origin="lower",
            extent=image_extent_over_lambda(xx, yy),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        fig.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04)
        axes[j].set_xlim(zoom["xmin"] / lam, zoom["xmax"] / lam)
        axes[j].set_ylim(zoom["ymin"] / lam, zoom["ymax"] / lam)

        axes[j].set_title(title)
        axes[j].set_xlabel(r"$x/\lambda$")
        axes[j].set_ylabel(r"$y/\lambda$")
        axes[j].set_aspect("equal")

    fig.suptitle(
        rf"Patrón de litografía a través de lente"
        "\n"
        rf"$z_{{0,1}}={D1['z0']} \,\mathrm{{mm}}$, "
        rf"$z_{{i,1}}={D1['zi']} \,\mathrm{{mm}}$, "
        rf"$\xi={xi:g}\,\mathrm{{mm}}$, "
        rf"$z_{{i,2}}={D2['zi']} \,\mathrm{{mm}}$, "
        rf"$r_a={r_a:g}\,\mathrm{{mm}}$, "
        rf"$\alpha_\mathrm{{max}}={np.degrees(alpha_max):.2f}^\circ$"
        "\n"
        rf"Campo incidente: $\mathbf{{E}}_0 = T\,\hat{{\mathbf{{x}}}}$",
    )
    fig.subplots_adjust(left=0.045, right=0.965, bottom=0.16, top=0.72, wspace=0.50)

    output_path = out_dir / "lithography_pattern_check.png"
    fig.savefig(output_path, dpi=150)
    print_saved(output_path)

    plt.close(fig)


if __name__ == "__main__":
    result = run_lithography_example(L=0.32, N=769, Npad=1024)
    plot_lithography_result(result)

    Is = result["Is"]
    Iv = result["Iv"]
    Id = result["Id"]

    print("OK")
    print(f"r_a={r_a:.10f}")
    print(f"alpha_max_deg={np.degrees(alpha_max):.10f}")
    print(f"Kc={Kc:.10f}")
    print(f"xi={xi:.10f}")
    print(f"first_focus_to_second_vertex={xi - D1['zi']:.10f}")
    print(f"r_Airy={r_Airy:.10f}")
    print(f"d_Airy={d_Airy:.10f}")
    print(f"Pattern_sep={Pattern_sep:.10f} ({Pattern_sep / d_Airy:.4f} d_Airy)")
    print(f"second_diopter_z0={D2['z0']:.10f}")
    print(f"conjugacy_error=xi_minus_D1_zi_plus_D2_z0={xi - D1['zi'] + D2['z0']:.10e}")
    print(f"scalar_shape={Is.shape}")
    print(f"vectorial_shape={Iv.shape}")
    print(f"scalar_finite={np.isfinite(Is).all()}")
    print(f"vectorial_finite={np.isfinite(Iv).all()}")
    print(f"cross_pol_fraction={result['cross_fraction']:.10e}")
    print(f"difference_abs_max={np.max(np.abs(Id)):.10e}")
    print(f"difference_abs_p99={np.percentile(np.abs(Id), 99):.10e}")
    print(out_dir / "lithography_pattern_check.png")
