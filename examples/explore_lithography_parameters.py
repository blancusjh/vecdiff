"""
Explore lithography parameters where scalar features are easier to distinguish
than vectorial features.

The score is intentionally diagnostic rather than definitive.  It favors cases
with high scalar image contrast, lower vectorial contrast, and larger
cross-polarized vectorial power.
"""

from __future__ import annotations

import csv
import itertools
import shutil
from pathlib import Path

import numpy as np

import lithography as litho
from _output import example_output_dir


OUTPUT_DIR = example_output_dir(__file__)


def set_lens_parameters(*, ni, aperture_radius, focus_distance, object_distance, second_focus, gap, sep_factor):
    litho.D1 = dict(n0=1.0, ni=float(ni), z0=-float(object_distance), zi=float(focus_distance))
    litho.xi = litho.D1["zi"] + float(gap)
    litho.D2 = dict(
        n0=litho.D1["ni"],
        ni=1.0,
        z0=litho.D1["zi"] - litho.xi,
        zi=float(second_focus),
    )
    litho.r_a = float(aperture_radius)
    litho.alpha_max = np.arctan(litho.r_a / abs(litho.D1["z0"]))
    litho.Kc = litho.D1["n0"] * (2.0 * np.pi / litho.lam) * np.sin(litho.alpha_max)
    litho.r_Airy = 3.8317059702075125 / litho.Kc
    litho.d_Airy = 2.0 * litho.r_Airy
    litho.Pattern_sep = float(sep_factor) * litho.d_Airy


def contrast_score(I):
    I = np.asarray(I, dtype=float)
    finite = I[np.isfinite(I)]
    if finite.size == 0:
        return 0.0

    high = np.percentile(finite, 99.0)
    middle = np.percentile(finite, 60.0)
    return float((high - middle) / (high + middle + 1.0e-30))


def image_score(result):
    Is = result["Is"]
    Iv = result["Iv"]
    scalar_contrast = contrast_score(Is)
    vectorial_contrast = contrast_score(Iv)
    contrast_drop = scalar_contrast - vectorial_contrast
    cross_fraction = result["cross_fraction"]
    difference = float(np.percentile(result["Id_abs"], 99.0))
    zoom = result["output_zoom"]
    span_x_lam = (zoom["xmax"] - zoom["xmin"]) / litho.lam
    span_y_lam = (zoom["ymax"] - zoom["ymin"]) / litho.lam
    output_span_lam = float(max(span_x_lam, span_y_lam))
    compactness = float(np.exp(-((output_span_lam / 65.0) ** 4)))

    mismatch_score = scalar_contrast * difference * compactness * (1.0 + 25.0 * cross_fraction)
    contrast_drop_score = scalar_contrast * max(contrast_drop, 0.0) * (1.0 + 25.0 * cross_fraction)

    return {
        "scalar_contrast": scalar_contrast,
        "vectorial_contrast": vectorial_contrast,
        "contrast_drop": contrast_drop,
        "cross_fraction": cross_fraction,
        "difference_p99": difference,
        "output_span_lam": output_span_lam,
        "compactness": compactness,
        "contrast_drop_score": contrast_drop_score,
        "score": mismatch_score,
    }


def mask_window_size():
    return litho.mask_window_size()


def sampling_metrics(*, L, N):
    dx = L / (N - 1)
    contact_radius = 0.18 * litho.d_Airy
    return {
        "input_L": L,
        "input_dx": dx,
        "contact_radius_px": contact_radius / dx,
        "feature_sep_px": litho.Pattern_sep / dx,
    }


def evaluate_case(params, *, N, Npad):
    set_lens_parameters(**params)
    L = mask_window_size()
    result = litho.run_lithography_example(L=L, N=N, Npad=Npad)
    metrics = image_score(result)
    sampling = sampling_metrics(L=L, N=N)
    lens = result["lens"]
    return {
        **params,
        "xi": lens.xi,
        "first_focus_to_second_vertex": lens.distance_from_first_focus_to_second_vertex,
        "second_diopter_z0": lens.second["z0"],
        "magnification": lens.magnification,
        "abs_magnification": abs(lens.magnification),
        **sampling,
        **metrics,
    }


def sweep_cases(*, N=641, Npad=768):
    parameter_grid = {
        "ni": [1.5, 2.4, 4.0],
        "aperture_radius": [1.0, 2.0, 4.0],
        "focus_distance": [2.0],
        "object_distance": [6.0, 10.0],
        "second_focus": [0.60, 0.80, 1.00],
        "gap": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
        "sep_factor": [0.70, 0.85],
    }
    keys = list(parameter_grid)
    values = [parameter_grid[key] for key in keys]

    rows = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        try:
            row = evaluate_case(params, N=N, Npad=Npad)
            if row["contact_radius_px"] < 4.0:
                row["error"] = "input mask under-sampled"
            elif row["abs_magnification"] >= 1.0:
                row["error"] = "magnifying case"
            rows.append(row)
        except Exception as exc:
            rows.append({**params, "error": repr(exc)})
    return rows


def write_csv(rows, path):
    keys = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_top_figures(rows, *, count=3):
    saved = []
    for index, row in enumerate(rows[:count], start=1):
        params = {
            key: row[key]
            for key in (
                "ni",
                "aperture_radius",
                "focus_distance",
                "object_distance",
                "second_focus",
                "gap",
                "sep_factor",
            )
        }
        set_lens_parameters(**params)
        result = litho.run_lithography_example(L=mask_window_size(), N=1025, Npad=1024)
        litho.plot_lithography_result(result)

        src = litho.out_dir / "lithography_pattern_check.png"
        dst = OUTPUT_DIR / f"top_{index:02d}_lithography_pattern_check.png"
        shutil.copyfile(src, dst)
        saved.append(dst)
    return saved


def main():
    rows = sweep_cases()
    rows_ok = [row for row in rows if "error" not in row]
    rows_ok.sort(key=lambda row: row["score"], reverse=True)

    csv_path = OUTPUT_DIR / "parameter_sweep.csv"
    write_csv(rows, csv_path)
    figure_paths = save_top_figures(rows_ok)

    print(f"Wrote {csv_path}")
    for path in figure_paths:
        print(f"Wrote {path}")
    print("Top cases: scalar contrast high with strong scalar-vectorial mismatch")
    for row in rows_ok[:12]:
        print(
            "score={score:.4e}  "
            "ni={ni:.2f}  z0=-{object_distance:.1f}  r_a={aperture_radius:.2f}  "
            "gap={gap:.2f}  xi={xi:.2f}  |M|={abs_magnification:.3f}  sep={sep_factor:.2f} d_Airy  "
            "contact_px={contact_radius_px:.1f}  "
            "Cs={scalar_contrast:.3f}  Cv={vectorial_contrast:.3f}  "
            "drop={contrast_drop:.3f}  cross={cross_fraction:.3e}  "
            "diff99={difference_p99:.3f}  span={output_span_lam:.1f}λ".format(**row)
        )


if __name__ == "__main__":
    main()
