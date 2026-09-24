"""Reproducible visual review: python validation/polarization_gallery.py.

Writes standalone PNG/PDF figures and a placement report under output/polarization.
These prescribed phasor patterns stress the renderer; the teaching notebook uses
propagated fields. Native layouts use grid nodes; alternate layouts interpolate
complex phasors at exact display coordinates.
"""

import base64
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j1
from vecdiff_general import CartesianGrid, ElectricField, Frame, PolarGrid
from vecdiff_general.visualization import (
    plot_polarization,
    plot_polarization_scalar,
)


def field_on(grid, kind):
    local = (grid.points - grid.frame.origin) @ grid.frame.basis
    x, y = local[..., 0] * 1e6, local[..., 1] * 1e6
    radius = np.hypot(x, y)
    azimuth = np.arctan2(y, x)
    envelope = np.exp(-(radius**2) / 5)
    first = np.ones(grid.shape, dtype=complex)
    second = np.zeros(grid.shape, dtype=complex)

    if kind == "circular":
        second[:] = 1j
    elif kind == "elliptic":
        second = 0.7j * np.tanh(x)
    elif kind == "radial":
        envelope = radius * np.exp(-(radius**2) / 5)
        first, second = np.cos(azimuth), np.sin(azimuth)
    elif kind == "azimuthal":
        envelope = radius * np.exp(-(radius**2) / 5)
        first, second = -np.sin(azimuth), np.cos(azimuth)
    elif kind == "annular":
        envelope = np.exp(-(((radius - 3) / 0.35) ** 2))
        first, second = np.cos(azimuth), 1j * np.sin(azimuth)
    elif kind == "satellite":
        envelope = np.exp(-((x - 1.8) ** 2 + y**2) / 0.7)
        envelope += 0.08 * np.exp(-((x + 2.2) ** 2 + (y - 0.5) ** 2) / 0.5)
        second[:] = 0.5j
    elif kind == "rings":
        argument = 3 * radius
        envelope = np.divide(
            2 * j1(argument), argument, out=np.ones_like(radius), where=argument > 0
        )
        second = 0.7j * np.tanh(x)
    elif kind == "narrow":
        envelope = np.exp(-(x**2) / 12 - y**2 / 0.15)
        second[:] = -0.6j
    elif kind == "dark":
        envelope[:] = 0
    elif kind == "uniform":
        envelope[:] = 1
        second[:] = 1j

    values = envelope[..., None] * np.stack(
        (first, second, np.zeros_like(first)), axis=-1
    )

    return ElectricField(grid, 532e-9, values @ grid.frame.basis.T)


def main():
    output = Path(__file__).resolve().parents[1] / "output/polarization"
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    movies = []

    def save(figure, name):
        figure.savefig(output / f"{name}.png", dpi=180)
        figure.savefig(output / f"{name}.pdf")
        counts = []

        for axis in figure.axes:
            for item in axis.collections:
                if item.get_gid() == "polarization-ellipses":
                    counts.append(len(item.get_segments()))

        report[name] = {
            "figure_inches": list(figure.get_size_inches()),
            "glyph_counts": counts,
        }
        plt.close(figure)

    coordinates = np.linspace(-6, 6, 241) * 1e-6
    grid = CartesianGrid(coordinates, coordinates)

    for kind in (
        "linear",
        "circular",
        "elliptic",
        "radial",
        "azimuthal",
        "annular",
        "satellite",
        "rings",
        "narrow",
        "dark",
        "uniform",
    ):
        axis = plot_polarization(field_on(grid, kind), title=kind.capitalize())
        save(axis.figure, kind)

    polar = PolarGrid(np.linspace(0, 6e-6, 121), np.arange(240) * 2 * np.pi / 240)
    axis = plot_polarization(field_on(polar, "radial"), title="Radial · polar grid")
    save(axis.figure, "polar_grid")

    annulus = PolarGrid(np.linspace(2e-6, 4e-6, 81), np.arange(160) * 2 * np.pi / 160)
    axis = plot_polarization_scalar(field_on(annulus, "annular"), "orientation")
    save(axis.figure, "polar_annular_orientation")

    basis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    tilted = CartesianGrid(coordinates, coordinates, Frame([2e-3, 0, 0], basis))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.7), layout="constrained")

    for axis, current, title in zip(
        axes,
        (field_on(grid, "elliptic"), field_on(tilted, "elliptic")),
        ("XY plane", "YZ plane · same local field"),
    ):
        plot_polarization(current, axis=axis, title=title)

    save(figure, "rotated_plane")

    figure, axes = plt.subplots(1, 3, figsize=(12, 4.6), layout="constrained")

    for axis, kind in zip(axes, ("radial", "elliptic", "satellite")):
        plot_polarization(field_on(grid, kind), axis=axis, title=kind.capitalize())

    save(figure, "small_panels")

    figure, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")

    for axis, quantity in zip(axes, ("orientation", "ellipticity")):
        plot_polarization_scalar(field_on(grid, "elliptic"), quantity, axis=axis)

    save(figure, "scalar_angles")

    # Shared physical phase advances; markers must travel on fixed ellipses.
    for kind in ("linear", "elliptic"):
        figure, axes = plt.subplots(1, 4, figsize=(15, 4.5), layout="constrained")

        for axis, phase in zip(axes, np.arange(4) * np.pi / 2):
            plot_polarization(
                field_on(grid, kind),
                axis=axis,
                phase=phase,
                density=0.5,
                title=f"ωt = {phase / np.pi:.1f}π",
                colorbar=False,
                legend=False,
            )

        save(figure, f"{kind}_phase")

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")

    for axis, state, phase, title in (
        (axes[0], (1, 0, 0), None, "Linear · continuous outline"),
        (axes[1], (1, np.exp(0.6j), 0), 1.7, "Electric arrow · continuous outline"),
        (axes[2], (1, np.exp(0.6j), 0), None, "Rotation head · follows the curve"),
    ):
        values = np.broadcast_to(np.asarray(state), grid.shape + (3,))
        field = ElectricField(grid, 532e-9, values)
        plot_polarization(
            field,
            axis=axis,
            phase=phase,
            max_glyphs=1,
            background=False,
            colorbar=False,
            legend=False,
            title=title,
        )
        axis.set_facecolor("#fff9bd")

    save(figure, "glyph_detail")

    # Hold the field, axes and density fixed while comparing size encodings.
    # Direct intensity and amplitude scaling expose the loss of weak rings.
    size_cases = (
        ("Uniform sizes", {"scale_by_intensity": False}),
        ("Radius ∝ intensity", {"size_power": 1, "min_size_fraction": 0}),
        ("Radius ∝ amplitude", {"size_power": 0.5, "min_size_fraction": 0}),
        ("Fourth-root compression · readable floor", {}),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 12))
    # Fixed, identical axes prevent layout changes from altering the sampling
    # between panels. Only the glyph size law changes in this comparison.
    figure.subplots_adjust(
        left=0.065, right=0.88, bottom=0.065, top=0.95, wspace=0.48, hspace=0.26
    )

    for axis, (title, options) in zip(axes.flat, size_cases):
        plot_polarization(
            field_on(grid, "rings"),
            axis=axis,
            distribution="polar",
            crop=False,
            density=0.8,
            legend=False,
            title=title,
            **options,
        )

    save(figure, "intensity_size_comparison")

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.7), layout="constrained")

    for axis, source, distribution, title in (
        (axes[0], grid, "auto", "Cartesian · exact rows"),
        (axes[1], grid, "polar", "Cartesian field · exact display rings"),
        (axes[2], polar, "auto", "Polar field · native rings"),
    ):
        plot_polarization(
            field_on(source, "radial"),
            axis=axis,
            distribution=distribution,
            crop=False,
            density=0.8,
            legend=False,
            title=title,
        )

    save(figure, "distribution_comparison")

    figure, axes = plt.subplots(1, 2, figsize=(11, 5.5), layout="constrained")

    for axis, distribution in zip(axes, ("cartesian", "polar")):
        plot_polarization(
            field_on(annulus, "annular"),
            axis=axis,
            distribution=distribution,
            crop=False,
            legend=False,
            title=f"Annular domain · {distribution}",
        )

    save(figure, "annular_layouts")

    relative_intensity = np.array([1, 0.1, 0.01, 0.001, 0.0001])
    sizing = {
        "relative_intensity": relative_intensity.tolist(),
        "radius_proportional_to_intensity": relative_intensity.tolist(),
        "radius_proportional_to_amplitude": np.sqrt(relative_intensity).tolist(),
        "compressed_radius": (0.5 + 0.5 * relative_intensity**0.25).tolist(),
        "note": "Radius relative to the spacing-dependent maximum; display only",
    }
    (output / "intensity_sizing.json").write_text(json.dumps(sizing, indent=2) + "\n")

    # Include the retained physical examples without rerunning their propagation.
    notebooks = output.parents[1] / "general/docs/notebooks"

    for path in notebooks.glob("0[156]_*.ipynb"):
        notebook = json.loads(path.read_text())
        number = 0

        for cell in notebook["cells"]:
            if path.name.startswith("01") and not cell["id"].startswith(
                "polarization-"
            ):
                continue

            if path.name.startswith("05") and cell["id"] not in (
                "matched-polarization-map",
                "stigmatic-phase-animation",
            ):
                continue

            for item in cell.get("outputs", []):
                html = item.get("data", {}).get("text/html")

                if html:
                    html = "".join(html)
                    frames = re.findall(
                        r'frames\[\d+\] = "data:image/png;base64,(.*?)"',
                        html,
                        flags=re.DOTALL,
                    )

                    if frames:
                        movie_name = f"notebook_{path.name[:2]}_{cell['id']}"
                        (output / f"{movie_name}.html").write_text(html)
                        movies.append(movie_name)

                        # Remove obsolete frame exports when frame counts change.
                        for previous in output.glob(f"{movie_name}_frame_*.png"):
                            previous.unlink()

                        for frame in (0, len(frames) // 4, len(frames) // 2):
                            name = f"{movie_name}_frame_{frame:02d}"
                            encoded = frames[frame].replace("\\\n", "")
                            (output / f"{name}.png").write_bytes(
                                base64.b64decode(encoded)
                            )
                            report[name] = {
                                "source_notebook": path.name,
                                "animation_frame": frame,
                            }

                encoded = item.get("data", {}).get("image/png")

                if encoded is None:
                    continue

                number += 1
                name = f"notebook_{path.name[:2]}_{number}"
                (output / f"{name}.png").write_bytes(base64.b64decode(encoded))
                report[name] = {"source_notebook": path.name}

    cards = []

    for name in report:
        pdf = (
            f'<a href="{name}.pdf">PDF</a>' if (output / f"{name}.pdf").exists() else ""
        )
        cards.append(
            f"<article><h2>{name.replace('_', ' ')}</h2>"
            f'<a href="{name}.png"><img src="{name}.png" alt="{name}"></a>{pdf}</article>'
        )

    movie_links = "".join(
        f'<li><a href="{name}.html">{name.replace("_", " ")}</a></li>'
        for name in movies
    )
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Polarization visual review</title><style>"
        "body{font:16px system-ui;margin:2rem;background:#f4f5f7;color:#202733}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:2rem}"
        "article{background:white;padding:1rem;border-radius:8px}img{width:100%;height:auto}"
        "h2{font-size:1rem;text-transform:capitalize}a{color:#126475}"
        "</style><h1>Polarization visual review</h1>"
        f"<h2>Optical-cycle animations</h2><ul>{movie_links}</ul>"
        "<p>Click a figure for its full resolution. Synthetic cases exercise "
        "the renderer; notebook figures show propagation and reflection.</p><main>"
        + "".join(cards)
        + "</main></html>"
    )
    (output / "index.html").write_text(html)
    (output / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(report)} visual cases to {output}")


if __name__ == "__main__":
    main()
