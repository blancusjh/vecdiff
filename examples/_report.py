"""Shared example output and provenance, not optical physics."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import matplotlib.pyplot as plt
import numpy as np


def versions():
    result = {"python": platform.python_version()}
    for name in ("vecdiff", "numpy", "scipy", "matplotlib", "finufft", "miepython"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def relative(actual, reference):
    return float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference), 1e-30))


def main(name, run):
    parser = argparse.ArgumentParser(description=run.__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/output")/name)
    args = parser.parse_args()
    figure, report = run()
    report["versions"] = versions()
    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output/"figure.png", dpi=160, bbox_inches="tight")
    (args.output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    plt.close(figure)
    print(json.dumps(report, indent=2, allow_nan=False))


def image(ax, grid, values, title):
    artist = ax.imshow(values, origin="lower", extent=(grid.x[0]-grid.dx/2, grid.x[-1]+grid.dx/2,
                                                      grid.y[0]-grid.dy/2, grid.y[-1]+grid.dy/2),
                       aspect="equal", cmap="magma", interpolation="nearest")
    ax.set(title=title, xlabel="x / wavelength", ylabel="y / wavelength")
    return artist
