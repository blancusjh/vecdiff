"""Recompute the stored AL1225 study shown in notebook 06.

Run from the repository root with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
VECLIB_MAXIMUM_THREADS=1. This computes scaled models, not the physical-size
catalog lens. Field plots and timings use the same completed runs.
"""

from pathlib import Path

from vecdiff_general.propagation.references.al1225 import (
    al1225_size_study,
    save_al1225_study,
)


def main():
    results = []
    destination = Path(__file__).parent / "reference_data" / "al1225_size_study.npz"
    checkpoint = destination.with_suffix(".partial.npz")
    for ratio in (3, 10, 30, 100):
        print(f"AL1225 scaled R/lambda={ratio}: running spectral pipeline", flush=True)
        results.extend(al1225_size_study((ratio,)))
        save_al1225_study(results, checkpoint)
        print(results[-1]["timing"], flush=True)
    checkpoint.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
