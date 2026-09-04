import ast
from pathlib import Path
import vecdiff


def test_core_has_no_reference_dependency_and_only_root_api():
    root = Path(vecdiff.__file__).parent
    assert [p.name for p in root.glob("*.py")] == ["__init__.py"]
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom): names = [node.module or ""]
            else: continue
            assert not any(n.startswith(("references", "miepython")) for n in names), path
    assert (root/"interfaces/fresnel.py").is_file()
    assert (root/"fourier/nufft.py").is_file()
