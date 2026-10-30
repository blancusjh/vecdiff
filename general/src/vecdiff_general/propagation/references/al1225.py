"""AL1225 scaled propagation study; numerical work stays outside notebook cells."""

import numpy as np
from scipy.fft import next_fast_len

import vecdiff_general as optics
from vecdiff_general.propagation.references.benchmarks import illumination
from vecdiff_general.propagation.references.pipeline_comparison import (
    compare_closed_body,
)


def al1225_size_study(ratios=(3, 10, 30, 100), *, coverage="complete"):
    """Cold spectral pipeline runs of scaled AL1225 prescriptions at 780 nm.

    The comparison helper explicitly continues each face outside the rim;
    it does not propagate the finite catalog OpticalSystem unchanged.
    BEM, when it fits, uses the closed body at h/lambda=0.20. The focal
    plane is the reference's axial maximum when BEM is available, otherwise
    the pipeline's. No calibration or convergence sweep is performed.
    """
    lens = optics.thorlabs_al1225()
    source, glass = illumination(
        wavelength0=780e-9, index=lens.interfaces[0].plus.material.n
    )
    spacing = 0.9 / (2 * glass.material.n * 0.95)
    results = []
    for ratio in ratios:
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("R/lambda must be finite and positive")
        system = optics.thorlabs_al1225(
            scale=ratio * source.wavelength0 / lens.apertures[0]
        )
        body = optics.VolumeBetweenSurfaces(
            system.interfaces[0].surface,
            system.interfaces[1].surface,
            system.apertures[0],
        )
        width = max(12, 4 * ratio)
        count = next_fast_len(max(128, int(np.ceil(width / spacing))), real=True)
        results.append(
            compare_closed_body(
                "Thorlabs AL1225 (scaled)",
                body,
                source,
                glass,
                count=count,
                width=width,
                threads=2,
                workers=1,
                scale_domain=True,
                adaptive_bem=True,
                coverage=coverage,
                focal_halfwidth=8,
                meridian_shape=(
                    max(151, int(np.ceil(19 * ratio)) + 1),
                    max(89, int(np.ceil(8 * ratio)) + 1),
                ),
            )
        )
    return results


def save_al1225_study(series, path):
    """Save measured fields and metadata without serializing Python objects."""
    import json

    arrays, metadata = {}, []
    coordinates = ("x", "z", "focal_axis", "scan_z", "coverage")
    domains = ("focal", "bulk", "axial")
    for i, result in enumerate(series):
        record = {
            k: v for k, v in result.items() if k not in (*coordinates, *domains, "body")
        }
        record["methods"] = list(result["focal"])
        metadata.append(record)
        for name in coordinates:
            arrays[f"{i}/{name}"] = result[name]
        for domain in domains:
            for method, field in result[domain].items():
                arrays[f"{i}/{domain}/{method}/E"] = field.electric.values
                arrays[f"{i}/{domain}/{method}/H"] = field.magnetic
    arrays["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **arrays)


def load_al1225_study(path):
    """Load saved AL1225 measurements; this does not execute or time a solver."""
    import json

    results = []
    with np.load(path, allow_pickle=False) as data:
        for i, record in enumerate(json.loads(str(data["metadata"]))):
            result = dict(record)
            methods = result.pop("methods")
            for name in ("x", "z", "focal_axis", "scan_z", "coverage"):
                result[name] = data[f"{i}/{name}"]
            wavelength = result["wavelength"]
            system = optics.thorlabs_al1225(
                wavelength0=wavelength,
                scale=result["radius_over_wavelength"] * wavelength / 0.00625,
            )
            result["body"] = optics.VolumeBetweenSurfaces(
                system.interfaces[0].surface,
                system.interfaces[1].surface,
                system.apertures[0],
            )
            x, z = np.meshgrid(result["x"], result["z"])
            fx, fy = np.meshgrid(result["focal_axis"], result["focal_axis"])
            scan = result["scan_z"]
            points = {
                "bulk": np.stack((x, np.zeros_like(x), z), axis=-1)[result["coverage"]],
                "focal": np.stack(
                    (fx, fy, np.full_like(fx, result["focus_z"])), axis=-1
                ),
                "axial": np.stack(
                    (np.zeros_like(scan), np.zeros_like(scan), scan), axis=-1
                ),
            }
            for domain, locations in points.items():
                result[domain] = {
                    method: optics.ElectromagneticField(
                        optics.ElectricField(
                            optics.PointGrid(locations),
                            wavelength,
                            data[f"{i}/{domain}/{method}/E"],
                        ),
                        data[f"{i}/{domain}/{method}/H"],
                    )
                    for method in methods
                }
            results.append(result)
    return results


def show_al1225_study(series):
    """Calculated fields first; timings and independent-reference availability next."""
    import matplotlib.pyplot as plt
    from IPython.display import HTML, display
    from matplotlib.colors import PowerNorm

    fig, axes = plt.subplots(
        len(series),
        2,
        figsize=(10, 2.7 * len(series)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, result in enumerate(series):
        ratio, wavelength = result["radius_over_wavelength"], result["wavelength"]
        meridian = np.full((len(result["z"]), len(result["x"])), np.nan)
        meridian[result["coverage"]] = result["bulk"]["Pipeline"].electric.norm2()
        extent = np.array(
            [result["z"][0], result["z"][-1], result["x"][0], result["x"][-1]]
        ) / (ratio * wavelength)
        im = axes[row, 0].imshow(
            meridian.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="hot",
            # Compressed scale: weak lobes stay visible beside the focus.
            norm=PowerNorm(0.3, vmin=0),
        )
        fig.colorbar(im, ax=axes[row, 0], label="|E|² [V²/m², scale I^0.3]", shrink=0.8)
        axes[row, 0].set(
            xlabel="z / R", ylabel="x / R", title=f"Spectral · R/λ = {ratio:g}"
        )
        body = result["body"]
        x = np.linspace(-body.rim_radius, body.rim_radius, 201)
        xy = np.column_stack((x, np.zeros_like(x)))
        for surface in (body.front, body.back):
            axes[row, 0].plot(
                surface.points(xy)[:, 2] / body.rim_radius,
                x / body.rim_radius,
                color="0.65",
                lw=0.7,
            )
        focal_extent = [
            result["focal_axis"][0] / wavelength,
            result["focal_axis"][-1] / wavelength,
        ] * 2
        im = axes[row, 1].imshow(
            result["focal"]["Pipeline"].electric.norm2(),
            origin="lower",
            extent=focal_extent,
            cmap="hot",
            norm=PowerNorm(0.3, vmin=0),
        )
        fig.colorbar(im, ax=axes[row, 1], label="|E|² [V²/m², scale I^0.3]", shrink=0.8)
        axes[row, 1].set(
            xlabel="x / λ",
            ylabel="y / λ",
            title=f"Focal plane · z/λ = {result['focus_z'] / wavelength:.2f}",
        )
    fig.suptitle(
        f"AL1225 scaled models · spectral · {series[0].get('coverage_mode', 'reduced')} coverage\n"
        "Continued faces · incident Ex = 1 V/m"
    )
    plt.show()
    plt.close(fig)
    rows = []
    for result in series:
        timing = next(r for r in result["timing"] if r["method"] == "Pipeline")
        difference = next(
            (r for r in result["accuracy"] if r["reference"] == "BEM"), None
        )
        error = f"{difference['focal E %']:.3g}%" if difference else "Unavailable"
        bem = "Computed" if result["bem_computed"] else "Not computed (dense limit)"
        rows.append(
            f"<tr><td>{result['radius_over_wavelength']:g}</td>"
            f"<td>{result['count']}²</td>"
            f"<td>{timing['full focal calculation s']:.3f}</td>"
            f"<td>{timing['all requested fields s']:.3f}</td>"
            f"<td>{bem}</td><td>{error}</td></tr>"
        )
    display(
        HTML(
            "<table><tr><th>R/λ</th><th>FFT grid</th><th>Full spectral focal run [s]</th>"
            "<th>All requested fields [s]</th><th>BEM</th><th>Complex E difference vs BEM</th>"
            "</tr>" + "".join(rows) + "</table>"
        )
    )
