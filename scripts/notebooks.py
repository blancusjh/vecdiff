"""Build paired percent-script notebooks; optionally execute every notebook.

Generated .ipynb sources are versioned without outputs. Execution artifacts and
the machine-readable report go under build/notebooks, never into source cells.
Run from the repository root: python scripts/notebooks.py --check --execute.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def notebook(source):
    cells = []
    for part in re.split(r"^# %%", source.read_text(), flags=re.MULTILINE)[1:]:
        header, body = part.split("\n", 1)
        body = body.strip()
        if "[markdown]" in header:
            text = "\n".join(line[2:] if line.startswith("# ") else line[1:] if line.startswith("#") else line for line in body.splitlines())
            cell = nbformat.v4.new_markdown_cell(text)
        else:
            cell = nbformat.v4.new_code_cell(body)
        cell["id"] = hashlib.sha256((str(len(cells))+cell.source).encode()).hexdigest()[:12]
        cells.append(cell)
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "source_script": source.name,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if committed notebooks differ from paired scripts")
    parser.add_argument("--execute", action="store_true", help="execute every notebook using this Python interpreter")
    args = parser.parse_args()
    output = ROOT/"build/notebooks"
    report = []
    for source in sorted((ROOT/"docs/notebooks").glob("*.py")):
        nb = notebook(source)
        target = source.with_suffix(".ipynb")
        if args.check:
            if not target.exists() or nbformat.read(target, as_version=4) != nb:
                raise RuntimeError(f"Stale notebook: regenerate {target.relative_to(ROOT)}")
        else:
            nbformat.write(nb, target)
        if args.execute:
            from nbclient import NotebookClient
            from jupyter_client import KernelManager
            from jupyter_client.kernelspec import KernelSpec
            # Avoid accidental use of a different system Python kernel.
            manager = KernelManager(kernel_name="python3")
            manager._kernel_spec = KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                              display_name="vecdiff validation", language="python")
            start = time.perf_counter()
            NotebookClient(nb, km=manager, timeout=600, allow_errors=False,
                           resources={"metadata": {"path": str(ROOT)}}).execute()
            output.mkdir(parents=True, exist_ok=True)
            nbformat.write(nb, output/target.name)
            report.append(dict(notebook=target.name, status="passed", cells=len(nb.cells),
                               seconds=time.perf_counter()-start))
            (output/"execution.json").write_text(json.dumps(report, indent=2)+"\n")
            print(report[-1], flush=True)
    if not report and args.execute:
        raise RuntimeError("No notebooks were executed")


if __name__ == "__main__":
    main()
