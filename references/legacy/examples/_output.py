"""Shared output locations for runnable examples."""

from __future__ import annotations

from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = EXAMPLES_DIR / "output"


def example_output_dir(example_file: str | Path) -> Path:
    """Return and create examples/output/<example_stem>/ for an example script."""
    path = OUTPUT_ROOT / Path(example_file).stem
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_saved(path: Path) -> None:
    """Print saved paths relative to the repository root when possible."""
    try:
        shown = path.resolve().relative_to(EXAMPLES_DIR.parent)
    except ValueError:
        shown = path
    print(f"Saved: {shown}")
