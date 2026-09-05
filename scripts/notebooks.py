"""Build paired percent-script notebooks; optionally execute every notebook.

Executed .ipynb files are versioned with outputs. Execution also writes review
artifacts under build/notebooks. Source synchronization ignores outputs but
requires matching source cells; --check alone also requires completed outputs.
Run from the repository root: python scripts/notebooks.py --execute.
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


def execution_fingerprint():
    """Bind outputs to the numerical code, input data, and paired sources.

Notebook output bytes are excluded to avoid a circular hash. A change in any
maintained numerical dependency requires re-executing the suite before merge.
    """
    paths = set()
    for pattern in ('vecdiff/**/*.py', 'references/**/*.py', 'examples/*.py',
                    'benchmarks/**/*.py', 'benchmarks/results/*.json',
                    'docs/notebooks/*.py', 'scripts/*.py'):
        paths.update(ROOT.glob(pattern))
    for name in ('pyproject.toml', 'requirements-validation.txt'):
        if (ROOT/name).exists(): paths.add(ROOT/name)
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode()+b'\0')
        digest.update(path.read_bytes()+b'\0')
    return digest.hexdigest()


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
    parser.add_argument("--check", action="store_true", help="check sources and, unless executing now, completed outputs")
    parser.add_argument("--execute", action="store_true", help="execute every notebook using this Python interpreter")
    args = parser.parse_args()
    output = ROOT/"build/notebooks"
    report = []
    fingerprint = execution_fingerprint()
    for source in sorted((ROOT/"docs/notebooks").glob("*.py")):
        nb = notebook(source)
        target = source.with_suffix(".ipynb")
        existing = nbformat.read(target, as_version=4) if target.exists() else None
        def source_signature(book):
            return [(cell.cell_type, cell.source, cell.id) for cell in book.cells]
        same_source = existing is not None and source_signature(existing) == source_signature(nb)
        if args.check and not same_source:
            raise RuntimeError(f"Stale source: execute {target.relative_to(ROOT)} from its paired script")
        if not args.execute:
            if not same_source:
                raise RuntimeError(f"Source changed: use --execute to update {target.relative_to(ROOT)} with its results")
            code = [cell for cell in existing.cells if cell.cell_type == 'code']
            figures = sum('image/png' in item.get('data', {}) for cell in code for item in cell.outputs)
            if (not code or any(cell.execution_count is None for cell in code)
                or any(item.output_type == 'error' for cell in code for item in cell.outputs)
                or not figures):
                raise RuntimeError(f"Unexecuted notebook: run with --execute for {target.relative_to(ROOT)}")
            if existing.metadata.get('vecdiff_execution', {}).get('input_sha256') != fingerprint:
                raise RuntimeError(f"Stale results: code or inputs changed; execute {target.relative_to(ROOT)}")
            continue
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
            figures = sum("image/png" in item.get("data", {})
                          for cell in nb.cells for item in cell.get("outputs", []))
            if figures == 0:
                raise RuntimeError(f"{target.name} executed without rendering its scientific figures")
            output.mkdir(parents=True, exist_ok=True)
            nb.metadata['vecdiff_execution'] = dict(input_sha256=fingerprint, python=sys.version.split()[0])
            nbformat.write(nb, output/target.name)
            # Commit the same executed notebook users open, not a blank source copy.
            nbformat.write(nb, target)
            report.append(dict(notebook=target.name, status="passed", cells=len(nb.cells), figures=figures,
                               seconds=time.perf_counter()-start))
            (output/"execution.json").write_text(json.dumps(report, indent=2)+"\n")
            print(report[-1], flush=True)
    if not report and args.execute:
        raise RuntimeError("No notebooks were executed")


if __name__ == "__main__":
    main()
