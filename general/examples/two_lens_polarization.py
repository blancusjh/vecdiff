"""A two-lens condenser and the polarization of its focus for four input beams.

The system is a stop and two plano-convex lenses, curved sides first (glass
n = 1.5, radius 25 mm, 3 mm thick, 2 mm apart): five elements, focal length
about 27 mm. A collimated Gaussian beam (3 mm waist, cut by the 3 mm stop,
532 nm) is carried through it with x, circular, radial and azimuthal
polarization, and E is read on the plane of best focus.

Run as a script it prints the system, the report and the focal numbers
(about 40 s). With a directory as argument it also writes the two figures of
the README there; the map of the field through the whole system takes a few
minutes more.
"""

import sys
from pathlib import Path

import numpy as np
import vecdiff as vd

WAVELENGTH = 532e-9
RADIUS, THICKNESS, GAP, STOP = 25e-3, 3e-3, 2e-3, 3e-3
VERTICES = (0.0, THICKNESS + GAP)


def condenser():
    """The path: the stop, then each lens, curved face first."""
    air, glass = vd.media.air, vd.media.Dielectric(1.5, "glass")
    elements = [vd.system.Stop(STOP, z=-0.5e-3)]
    for z in VERTICES:
        curved = vd.surfaces.SphericalCap(RADIUS, vd.geometry.Frame([0, 0, z]))
        flat = vd.surfaces.Plane.z(z + THICKNESS)
        elements += [
            vd.system.Refraction(curved, air, glass),
            vd.system.Refraction(flat, glass, air),
        ]
    return vd.system.OpticalSystem(*elements)


def beams():
    """The same Gaussian beam with four polarizations."""
    states = {
        "linear x": vd.fields.polarization.x,
        "circular": vd.fields.polarization.cartesian(1, 1j),
        "radial": vd.fields.polarization.radial,
        "azimuthal": vd.fields.polarization.azimuthal,
    }
    return {
        name: vd.fields.PlaneBeam(
            vd.fields.profiles.Gaussian(3e-3),
            state,
            wavelength0=WAVELENGTH,
            region=vd.media.air,
            frame=vd.geometry.Frame([0, 0, -1e-3]),
        )
        for name, state in states.items()
    }


def best_focus(solution):
    """The brightest point of the axis behind the last lens."""
    z = np.linspace(26e-3, 31e-3, 1001)
    axis = vd.sampling.PointGrid(np.c_[0 * z, 0 * z, z])
    return float(
        z[np.argmax(vd.observables.irradiance(solution.field(axis), [0, 0, 1]))]
    )


def focal_fields(half_width=12e-6, count=161):
    """Each beam carried through the condenser and read on the focal plane."""
    propagator = vd.propagation.Propagator(condenser())
    window = (-half_width, half_width, -half_width, half_width)
    solutions, fields, focus = {}, {}, None
    for name, beam in beams().items():
        solutions[name] = propagator(beam)
        if focus is None:
            focus = best_focus(solutions[name])
        fields[name] = solutions[name].field(
            z=focus, window=window, shape=(count, count)
        )
    return solutions, fields, focus


def lens_outlines(axis, half_height):
    """Draw the stop and the two lenses on a (z, x) axis in millimetres."""
    x = np.linspace(-half_height, half_height, 201)
    for vertex in VERTICES:
        sag = vertex + RADIUS - np.sqrt(RADIUS**2 - x**2)
        flat = np.full_like(x, vertex + THICKNESS)
        outline_z = np.r_[sag, flat[::-1], sag[:1]] * 1e3
        outline_x = np.r_[x, x[::-1], x[:1]] * 1e3
        axis.plot(outline_z, outline_x, color="white", lw=0.9)
    for sign in (-1, 1):
        axis.plot(
            [-0.5, -0.5],
            [sign * STOP * 1e3, sign * half_height * 1e3],
            color="white",
            lw=2.5,
        )


def figure_system(solution, focus, path):
    """|E|² of the x-polarized beam through the system and through its focus."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, (wide, zoom) = plt.subplots(
        2, 1, figsize=(11, 6.6), gridspec_kw={"height_ratios": [1.25, 1]}
    )
    half = 4.2e-3
    field = solution.field(
        y=0.0, window=(-half, half, -0.9e-3, focus + 3e-3), shape=(420, 221)
    )
    intensity = np.sum(np.abs(field.electric.values) ** 2, axis=-1).T
    image = wide.imshow(
        intensity,
        extent=(-0.9, (focus + 3e-3) * 1e3, -half * 1e3, half * 1e3),
        origin="lower",
        aspect="auto",
        cmap="inferno",
        norm=LogNorm(1e-2, float(intensity.max())),
    )
    lens_outlines(wide, half)
    wide.set(xlabel="z (mm)", ylabel="x (mm)")
    wide.set_title("Through the system: stop, two lenses, focus", loc="left")
    fig.colorbar(image, ax=wide, pad=0.01, label=r"$|\mathbf{E}|^2$ (V²/m²)")

    depth, width = 0.5e-3, 14e-6
    field = solution.field(
        y=0.0, window=(-width, width, focus - depth, focus + depth), shape=(361, 141)
    )
    intensity = np.sum(np.abs(field.electric.values) ** 2, axis=-1).T
    image = zoom.imshow(
        intensity / intensity.max(),
        extent=(
            (focus - depth) * 1e3,
            (focus + depth) * 1e3,
            -width * 1e6,
            width * 1e6,
        ),
        origin="lower",
        aspect="auto",
        cmap="inferno",
        norm=LogNorm(1e-3, 1),
    )
    zoom.axvline(focus * 1e3, color="white", ls="--", lw=0.8)
    zoom.set(xlabel="z (mm)", ylabel="x (µm)")
    zoom.set_title(
        "Through focus (dashes: the plane of the polarization maps)", loc="left"
    )
    fig.colorbar(image, ax=zoom, pad=0.01, label=r"$|\mathbf{E}|^2$ / peak")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_polarization(fields, path):
    """Polarization over transverse intensity, and |E_z|², for each input."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from vecdiff.visualization import plot_polarization

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.4))
    for (name, field), top, bottom in zip(fields.items(), *axes):
        values = field.electric.values
        transverse = np.sum(np.abs(values[..., :2]) ** 2, axis=-1)
        peak = float(transverse.max())
        plot_polarization(
            field.electric,
            axis=top,
            crop=False,
            legend=False,
            title=f"{name} input: polarization",
            norm=LogNorm(1e-4, 1),
            reference_intensity=peak,
            colorbar_label=r"$(|E_x|^2+|E_y|^2)$ / peak",
        )
        top.set(xlabel="x (µm)", ylabel="y (µm)")
        u, v = field.grid.u * 1e6, field.grid.v * 1e6
        longitudinal = np.abs(values[..., 2]) ** 2 / peak
        image = bottom.imshow(
            np.maximum(longitudinal, 1e-12),
            extent=(u[0], u[-1], v[0], v[-1]),
            origin="lower",
            cmap="inferno",
            norm=LogNorm(1e-6, 1e-1),
        )
        share = float(longitudinal.max())
        label = f"peak {share:.1e}" if share > 1e-12 else "zero"
        bottom.set_title(rf"$|E_z|^2$ / transverse peak ({label})")
        bottom.set(xlabel="x (µm)", ylabel="y (µm)")
        fig.colorbar(image, ax=bottom, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(figures=None):
    print(condenser())
    solutions, fields, focus = focal_fields()
    print(solutions["linear x"])
    print(f"best focus at z = {focus * 1e3:.3f} mm")
    for name, field in fields.items():
        peak = np.abs(field.electric.values).max(axis=(0, 1))
        print(
            f"{name:>10}: max |Ex|, |Ey|, |Ez| = "
            f"{peak[0]:7.1f}, {peak[1]:7.1f}, {peak[2]:6.2f} V/m"
        )
    if figures is not None:
        figures = Path(figures)
        figure_polarization(fields, figures / "condenser_focal_polarization.png")
        if "--quick" not in sys.argv:
            figure_system(
                solutions["linear x"], focus, figures / "condenser_system.png"
            )


if __name__ == "__main__":
    main(*sys.argv[1:2])
