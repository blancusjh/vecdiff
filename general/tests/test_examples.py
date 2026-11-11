"""The scripts in general/examples: every name they use exists, and the cheap
ones run to the end at a reduced size."""

import ast
import importlib.util
from pathlib import Path

import pytest
import vecdiff

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCRIPTS = sorted(EXAMPLES.glob("*.py"))
# Reduced sizes of the examples that run here; the others take tens of
# seconds or several GB and are only checked for their names.
RUNS = {
    "beams_and_pupils": {"scale": 0.3},
    "exact_references": {"radius": 4.0},
    "observables_of_light": {"count": 24},
    "one_transport_leg": {"count": 100},
    "plate_ghost": {},
    "wave_transport": {},
    "paraboloidal_mirror": {},
}


def load(path):
    spec = importlib.util.spec_from_file_location(f"example_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def names(tree, aliases):
    """Every dotted name read from the package under one of ``aliases``."""
    for node in ast.walk(tree):
        chain = []
        while isinstance(node, ast.Attribute):
            chain.append(node.attr)
            node = node.value
        if chain and isinstance(node, ast.Name) and node.id in aliases:
            yield node.id, chain[::-1]


def test_every_example_is_listed():
    assert SCRIPTS, "no examples found"
    assert {p.stem for p in SCRIPTS} >= set(RUNS)


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_example_names_exist(path):
    tree = ast.parse(path.read_text(), str(path))
    aliases = {"vecdiff"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases |= {
                a.asname for a in node.names if a.name == "vecdiff" and a.asname
            }
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("vecdiff")
        ):
            module = importlib.import_module(node.module)
            for alias in node.names:
                assert hasattr(module, alias.name), f"{node.module}.{alias.name}"
    for alias, chain in names(tree, aliases):
        obj = vecdiff
        for attribute in chain:
            if not hasattr(obj, attribute):
                # Attributes of returned objects are not module names.
                assert not isinstance(obj, type(vecdiff)), (
                    f"{alias}.{'.'.join(chain)} does not exist"
                )
                break
            obj = getattr(obj, attribute)
    assert callable(getattr(load(path), "main", None))


@pytest.mark.parametrize("name", sorted(RUNS))
def test_example_runs(name, capsys):
    pytest.importorskip("finufft")
    load(EXAMPLES / f"{name}.py").main(**RUNS[name])
    assert capsys.readouterr().out.strip()
