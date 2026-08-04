
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
from vecdiff.polarization import polarization_from_components
from vecdiff.polarization_visualization import plot_polarization_map
from _output import example_output_dir, print_saved


# ---------------------------------------------------------------------
# Optical configuration
# ---------------------------------------------------------------------

lam = 193e-6

# Synthetic fused silica is used in 193 nm ArF projection optics.  Its index
# is approximately 1.560 at this wavelength; losses are neglected because the
# diopter model currently accepts real refractive indices only.
# Light propagates from vacuum through fused silica, then exits back to vacuum.
D1 = dict(n0=1.0, ni=1.5602, z0=-4.0, zi=2.0)
xi = 5.000
D2 = dict(n0=D1["ni"], ni=1.0, z0=D1["zi"] - xi, zi=0.25 * D1["zi"] / D1["ni"])

r_a = 4.0 * np.tan(np.deg2rad(70.0))
alpha_max = np.arctan(r_a / abs(D1["z0"]))
NA = D1["n0"] * np.sin(alpha_max)
Kc = (2.0 * np.pi / lam) * NA
r_Airy = 3.8317059702075125 / Kc
d_Airy = 2.0 * r_Airy

# Near-resolution feature scale for the polarization-transfer diagnostic.
# This separation keeps the scalar contacts and grating resolved, while the
# high-NA vectorial transfer mixes a substantial cross-polarized component.
Pattern_sep = 0.95 * d_Airy
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

    @property
    def magnification(self):
        """Transverse scale of the two focal-plane Fourier maps."""
        return -(
            self.first["ni"] * self.second["zi"]
            / (self.second["ni"] * self.first["zi"])
        )


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

def propagate_through_lens(field, lens, *, vectorial, Npad1=768, Npad2=768,
                           geometry="full"):
    """Relay the field through the two dioptres.

    ``geometry`` selects the transfer weighting; ``"none"`` reproduces what the
    package computed before the geometric factor and the pupil mapping went in,
    which is what the before/after comparison uses.
    """
    transmission = "vectorial" if vectorial else "identity"

    diopter1 = CartesianSurface(**lens.first)
    diopter2 = CartesianSurface(**lens.second)
    # The stigmatic oval has a finite usable aperture (its grazing radius).
    # With this geometry the design stop overfills the second dioptre, so the
    # effective NA is below the nominal one: light cannot pass through surface
    # that does not exist.  Clamping here makes that explicit instead of
    # letting the transfer operator silently zero the excess.
    pupil_radius_design = (
        lam * lens.first["zi"] * Kc / (2.0 * np.pi * lens.first["ni"])
    )
    pupil_radius = min(pupil_radius_design, diopter2.aperture_limit)

    focal_field = field.propagate_through_diopter(
        diopter1.zi,
        diopter1,
        method="fft",
        output="focal",
        wavelength=lam,
        kgrid=field.grid.kgrid(Npad1),
        transmission=transmission,
        geometry=geometry,
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
        geometry=geometry,
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


def mask_window_size():
    return 18.0 * Pattern_sep


def run_lithography_example(*, L=None, N=1025, Npad=1024, geometry="full"):
    if L is None:
        L = mask_window_size()

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
        geometry=geometry,
    )

    Ex_v, Ey_v, _, _ = propagate_through_lens(
        field,
        lens,
        vectorial=True,
        Npad1=Npad,
        Npad2=Npad,
        geometry=geometry,
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
        "Ex_in": field.x,
        "Ey_in": field.y,
        "boxes": boxes,
        "xr": xr,
        "yr": yr,
        "Ex_v": Ex_v,
        "Ey_v": Ey_v,
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
    fig, axes = plt.subplots(1, 5, figsize=(20.3, 5.1), constrained_layout=False)

    panels = [
        (result["T"], result["x"], result["y"], "Máscara de entrada", result["input_zoom"]),
        (result["Is"], result["xr"], result["yr"], "Intensidad - Escalar", result["output_zoom"]),
        (result["Iv"], result["xr"], result["yr"], "Intensidad - Vectorial", result["output_zoom"]),
        (
            result["Icross"],
            result["xr"],
            result["yr"],
            r"Intensidad - $E_y$ cruzado",
            result["output_zoom"],
        ),
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
        "Patrón de litografía a través de lente"
        "\n"
        rf"$\lambda={lam * 1.0e6:.0f}\,\mathrm{{nm}}$: vacío $\rightarrow$ sílice fundida $\rightarrow$ vacío"
        "\n"
        rf"$n_0={D1['n0']:.2f}$, $n_1={D1['ni']:.3f}$, $n_2={D2['ni']:.2f}$  |  "
        rf"$z_{{0,1}}={D1['z0']:.1f}\,\mathrm{{mm}}$, $z_{{i,1}}={D1['zi']:.1f}\,\mathrm{{mm}}$, "
        rf"$\xi={xi:.1f}\,\mathrm{{mm}}$, $z_{{i,2}}={D2['zi']:.3f}\,\mathrm{{mm}}$"
        "\n"
        rf"$r_a={r_a:.2f}\,\mathrm{{mm}}$, $\alpha_\mathrm{{max}}={np.degrees(alpha_max):.1f}^\circ$, $\mathrm{{NA}}={NA:.2f}$, "
        rf"$|M|={abs(result['lens'].magnification):.2f}$  |  "
        rf"$\mathbf{{E}}_0 = T\,\hat{{\mathbf{{x}}}}$",
    )
    fig.subplots_adjust(left=0.038, right=0.975, bottom=0.16, top=0.76, wspace=0.48)

    output_path = out_dir / "lithography_pattern_check.png"
    fig.savefig(output_path, dpi=150)
    print_saved(output_path)

    plt.close(fig)


def plot_lithography_polarization_result(result):
    """Plot input/output intensities with their local polarization ellipses."""

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.3), constrained_layout=False)
    panels = [
        (
            result["T"],
            result["Ex_in"],
            result["Ey_in"],
            result["x"],
            result["y"],
            "Entrada: máscara y polarización incidente",
            result["input_zoom"],
            0.50,
            "black",
        ),
        (
            result["Iv"],
            result["Ex_v"],
            result["Ey_v"],
            result["xr"],
            result["yr"],
            "Salida: intensidad y polarización vectorial",
            result["output_zoom"],
            0.018,
            "white",
        ),
    ]

    for ax, (intensity, ex, ey, xx, yy, title, zoom, threshold, glyph_color) in zip(axes, panels):
        x_plot = np.asarray(xx) / lam
        y_plot = np.asarray(yy) / lam
        X_plot, Y_plot = np.meshgrid(x_plot, y_plot, indexing="xy")

        shown_intensity = display_positive(intensity, percentile=99.7, gamma=0.72)
        im = ax.imshow(
            shown_intensity,
            origin="lower",
            extent=image_extent_over_lambda(xx, yy),
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
        colorbar.set_label("Intensidad (escala de visualización)")

        pol = polarization_from_components(ex, ey)
        plot_polarization_map(
            X_plot,
            Y_plot,
            pol,
            target_ellipses=42,
            max_ellipses=850,
            min_intensity_fraction=threshold,
            scale_by_intensity=True,
            intensity_scale_mode="power",
            intensity_scale_gamma=0.35,
            min_ellipse_scale=0.42,
            ellipse_points=48,
            ellipse_mode="cartesian",
            curve_kwargs={"color": glyph_color, "linewidth": 0.72, "alpha": 0.88},
            arrowhead_kwargs={"color": glyph_color, "linewidth": 0.82, "alpha": 0.92},
            ax=ax,
        )

        ax.set_xlim(zoom["xmin"] / lam, zoom["xmax"] / lam)
        ax.set_ylim(zoom["ymin"] / lam, zoom["ymax"] / lam)
        ax.set_title(title)
        ax.set_xlabel(r"$x/\lambda$")
        ax.set_ylabel(r"$y/\lambda$")
        ax.set_aspect("equal")

    fig.suptitle(
        "Litografía: evolución espacial de la polarización"
        "\n"
        rf"$\lambda={lam * 1.0e6:.0f}\,\mathrm{{nm}}$, "
        rf"$\mathrm{{NA}}={NA:.2f}$, $|M|={abs(result['lens'].magnification):.2f}$, "
        rf"$\mathbf{{E}}_0=T\,\hat{{\mathbf{{x}}}}$",
        y=0.975,
    )
    fig.subplots_adjust(left=0.065, right=0.955, bottom=0.12, top=0.83, wspace=0.34)

    output_path = out_dir / "lithography_polarization_maps.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print_saved(output_path)
    plt.close(fig)


def plot_lithography_ellipticity_map(result):
    """Temporarily plot output ellipticity and major-axis orientation."""

    pol = polarization_from_components(result["Ex_v"], result["Ey_v"])
    intensity = np.asarray(pol.s0, dtype=float)
    visible = intensity >= 0.01 * float(np.nanmax(intensity))
    chi_deg = np.degrees(pol.chi)
    chi_visible = np.ma.masked_where(~visible, chi_deg)
    psi_deg = np.degrees(pol.psi)
    psi_visible = np.ma.masked_where(~visible, psi_deg)

    robust_limit = float(np.percentile(np.abs(chi_deg[visible]), 99.5))
    color_limit = max(robust_limit, 1.0e-3)
    chi_cmap = plt.get_cmap("RdBu_r").copy()
    chi_cmap.set_bad("black")
    psi_cmap = plt.get_cmap("twilight_shifted").copy()
    psi_cmap.set_bad("black")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    im_chi = axes[0].imshow(
        chi_visible,
        origin="lower",
        extent=image_extent_over_lambda(result["xr"], result["yr"]),
        cmap=chi_cmap,
        vmin=-color_limit,
        vmax=color_limit,
        interpolation="nearest",
    )
    im_psi = axes[1].imshow(
        psi_visible,
        origin="lower",
        extent=image_extent_over_lambda(result["xr"], result["yr"]),
        cmap=psi_cmap,
        vmin=-90.0,
        vmax=90.0,
        interpolation="nearest",
    )

    intensity_display = display_positive(intensity, percentile=99.7, gamma=0.72)
    zoom = result["output_zoom"]
    for ax in axes:
        ax.contour(
            result["xr"] / lam,
            result["yr"] / lam,
            intensity_display,
            levels=(0.18, 0.45, 0.75),
            colors="white",
            linewidths=0.45,
            alpha=0.38,
        )
        ax.set_xlim(zoom["xmin"] / lam, zoom["xmax"] / lam)
        ax.set_ylim(zoom["ymin"] / lam, zoom["ymax"] / lam)
        ax.set_xlabel(r"$x/\lambda$")
        ax.set_ylabel(r"$y/\lambda$")
        ax.set_aspect("equal")

    axes[0].set_title(r"Ángulo de elipticidad $\chi$")
    axes[1].set_title(r"Ángulo del eje mayor $\psi$")

    chi_colorbar = fig.colorbar(im_chi, ax=axes[0], fraction=0.046, pad=0.035)
    chi_colorbar.set_label(r"$\chi$ [grados]")
    psi_colorbar = fig.colorbar(im_psi, ax=axes[1], fraction=0.046, pad=0.035)
    psi_colorbar.set_label(r"$\psi$ [grados]")
    psi_colorbar.set_ticks((-90.0, -45.0, 0.0, 45.0, 90.0))

    fig.suptitle(
        "Polarización vectorial de salida\n"
        r"Contornos: intensidad; mapas visibles para $I\geq 1\%\,I_{\max}$"
    )
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.11, top=0.82, wspace=0.32)

    output_path = out_dir / "lithography_ellipticity_map.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print_saved(output_path)
    plt.close(fig)


if __name__ == "__main__":
    result = run_lithography_example()
    plot_lithography_result(result)
    plot_lithography_polarization_result(result)
    plot_lithography_ellipticity_map(result)

    Is = result["Is"]
    Iv = result["Iv"]
    Id = result["Id"]

    print("OK")
    print(f"r_a={r_a:.10f}")
    print(f"alpha_max_deg={np.degrees(alpha_max):.10f}")
    print(f"NA={NA:.10f}")
    print(f"Kc={Kc:.10f}")
    print(f"xi={xi:.10f}")
    print(f"first_focus_to_second_vertex={xi - D1['zi']:.10f}")
    print(f"transverse_magnification={result['lens'].magnification:.10f}")
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
    print(out_dir / "lithography_polarization_maps.png")
    print(out_dir / "lithography_ellipticity_map.png")
