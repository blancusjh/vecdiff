"""Committed notebooks must preserve results and reject stale/unexecuted copies."""
import nbformat
import pytest
from scripts import notebooks


def setup_notebook(tmp_path,monkeypatch):
    directory=tmp_path/'docs/notebooks';directory.mkdir(parents=True)
    source=directory/'example.py';source.write_text('# %%\n1 + 1\n')
    nb=notebooks.notebook(source)
    nb.cells[0].execution_count=1
    nb.cells[0].outputs=[nbformat.v4.new_output('display_data',data={'image/png':'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII='})]
    monkeypatch.setattr(notebooks,'ROOT',tmp_path)
    nb.metadata['vecdiff_execution']={'input_sha256':notebooks.execution_fingerprint()}
    target=source.with_suffix('.ipynb');nbformat.write(nb,target)
    monkeypatch.setattr('sys.argv',['notebooks.py','--check'])
    return source,target,nb


def test_check_preserves_completed_outputs(tmp_path,monkeypatch):
    _,target,_=setup_notebook(tmp_path,monkeypatch)
    before=target.read_bytes()
    notebooks.main()
    assert target.read_bytes()==before


def test_changed_source_cannot_reuse_stale_outputs(tmp_path,monkeypatch):
    source,target,_=setup_notebook(tmp_path,monkeypatch)
    before=target.read_bytes();source.write_text('# %%\n2 + 2\n')
    with pytest.raises(RuntimeError,match='Stale source'):
        notebooks.main()
    assert target.read_bytes()==before


def test_unexecuted_copy_is_rejected(tmp_path,monkeypatch):
    _,target,nb=setup_notebook(tmp_path,monkeypatch)
    nb.cells[0].execution_count=None;nbformat.write(nb,target)
    with pytest.raises(RuntimeError,match='Unexecuted notebook'):
        notebooks.main()


def test_numerical_code_changes_invalidate_existing_results(tmp_path,monkeypatch):
    _,_,_=setup_notebook(tmp_path,monkeypatch)
    core=tmp_path/'vecdiff';core.mkdir()
    (core/'changed.py').write_text('new_physics = True\n')
    with pytest.raises(RuntimeError,match='Stale results'):
        notebooks.main()
