import subprocess
import sys

import pytest


def probe(paths, succeeds):
    code = f'''
import sys
sys.path[:0] = {list(map(str, paths))!r}
from autofusex.execution.backends.mlir_capability import verify_compile_adapter
from autofusex.execution.backends.base import UnsupportedBackend
try:
    result = verify_compile_adapter()
except UnsupportedBackend as exc:
    assert not {succeeds!r}, str(exc)
    assert 'selected package:' in str(exc)
    assert 'shadows later PYTHONPATH' in str(exc)
    print(str(exc))
else:
    assert {succeeds!r}, result
    print(result)
'''
    return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10)


def package(root, adapter=None):
    p = root/'autofuse'
    p.mkdir(parents=True)
    (p/'__init__.py').write_text('')
    if adapter is not None:
        (p/'compile_adapter.py').write_text(adapter)
    return root


COMPLETE = '\n'.join(f'def {name}(*args, **kwargs): pass' for name in
                      ('build_tiling_obj', 'build_kernel_obj', 'build_kernel_so'))


def test_partial_package_shadows_complete_package(tmp_path):
    partial = package(tmp_path/'partial')
    complete = package(tmp_path/'complete', COMPLETE)
    result = probe([partial, complete], False)
    assert result.returncode == 0, result.stderr
    assert str(partial/'autofuse/__init__.py') in result.stdout
    result = probe([complete, partial], True)
    assert result.returncode == 0, result.stderr
    assert str(complete/'autofuse/compile_adapter.py') in result.stdout


@pytest.mark.parametrize('adapter', ['', 'build_tiling_obj = 1', 'import nonexistent_runtime_dependency'])
def test_incomplete_api_and_transitive_dependency_fail_early(tmp_path, adapter):
    result = probe([package(tmp_path/'runtime', adapter)], False)
    assert result.returncode == 0, result.stderr
