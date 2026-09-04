"""Run every curated workflow; save figures, full reports, and a compact manifest."""
import json
from importlib import import_module
from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from examples._report import versions

NAMES = ("field_propagation", "plane_interface", "cavity_resonance", "frustrated_tir",
         "curved_interface", "vector_focus", "sphere_resonance")


def main():
    manifest = dict(versions=versions(), workflows=[])
    destination = Path("benchmarks/results/workflows.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        start = time.perf_counter()
        figure, report = import_module("examples."+name).run()
        output = Path("examples/output")/name
        output.mkdir(parents=True, exist_ok=True)
        figure.savefig(output/"figure.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
        report["versions"] = manifest["versions"]
        (output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        # Full scan rows remain in each example report; retain the principal
        # metrics and the worst-case refinement in the compact committed manifest.
        summary = {k: v for k, v in report.items() if k not in ("cases", "versions", "residual_history")}
        if "cases" in report:
            summary["case_count"] = len(report["cases"])
        manifest["workflows"].append(dict(name=name, status="passed", seconds=time.perf_counter()-start, **summary))
        destination.write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
        print(name, "passed", flush=True)


if __name__ == "__main__":
    main()
