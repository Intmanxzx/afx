import sys
import types

import pytest

from autofusex.fuzz.environment import collect


@pytest.mark.parametrize('loaded', [False, True])
@pytest.mark.parametrize('contents,expected', [
    ('Version=9.3.0-beta.1\n', '9.3.0-beta.1'),
    ('Other=ignored\n Version = 9.3.0-beta.1 \n', '9.3.0-beta.1'),
    ('Version=\n', None), ('Other=9.3.0\n', None), (None, None),
])
def test_cann_always_reads_version_info(tmp_path, monkeypatch, loaded, contents, expected):
    monkeypatch.setattr('autofusex.fuzz.environment.importlib.util.find_spec', lambda name: None)
    monkeypatch.setenv('ASCEND_HOME_PATH', str(tmp_path))
    if contents is not None:
        (tmp_path/'compiler').mkdir()
        (tmp_path/'compiler/version.info').write_text(contents)
    for name in ('torch_npu', 'torch_npu.npu', 'torch_npu.npu.utils'):
        monkeypatch.delitem(sys.modules, name, raising=False)
        if loaded:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    if loaded:
        def forbidden():
            pytest.fail('CANN runtime accessor must not be used')
        sys.modules['torch_npu.npu.utils'].get_cann_version = forbidden
    assert collect(tmp_path/'home')['cann'] == expected
    if not loaded:
        assert 'torch_npu' not in sys.modules


def test_no_selected_toolkit_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr('autofusex.fuzz.environment.importlib.util.find_spec', lambda name: None)
    monkeypatch.delenv('ASCEND_HOME_PATH', raising=False)
    assert collect(tmp_path)['cann'] is None


def test_graph_autofusion_run_version_has_precedence(tmp_path, monkeypatch):
    root = tmp_path/'cann-9.3.0'
    monkeypatch.setenv('ASCEND_HOME_PATH', str(root))
    (root/'compiler').mkdir(parents=True)
    (root/'compiler/version.info').write_text('Version=9.3.0\n')
    package = root/'python/site-packages/autofuse'
    package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    graph_version = root/'share/info/graph_autofusion/version.info'
    graph_version.parent.mkdir(parents=True)
    graph_version.write_text('Version=9.2.0\n')
    monkeypatch.setattr('autofusex.fuzz.environment.importlib.util.find_spec',
                        lambda name: type('Spec', (), {'origin': str(package/'__init__.py')})())
    assert collect(tmp_path/'home')['cann'] == '9.2.0'
